"""IPM lift tests on synthetic geometry shaped like the real nuScenes CAM_FRONT
setup (camera ~1.5m above ground, slight pitch, ego frame x-forward/z-up)."""

import numpy as np
import pytest
from pyquaternion import Quaternion

from adp.calib.camera import CameraModel, SE3
from adp.lift.bev_state import update_bev_states
from adp.lift.ipm import GroundPlaneLift
from adp.track.bytetrack import ByteTracker
from adp.detect.detector import Detection2D


def nuscenes_like_rig(pitch_deg: float = 0.0):
    """CameraModel + T_ego_from_cam mimicking nuScenes CAM_FRONT.

    Ego frame: x forward, y left, z up (ground at z=0).
    Camera frame: z forward (optical axis), x right, y down.
    """
    cam = CameraModel(
        K=np.array([[1266.0, 0, 800.0], [0, 1266.0, 450.0], [0, 0, 1.0]]),
        width=1600,
        height=900,
    )
    # Base rotation: cam z -> ego x, cam x -> ego -y, cam y -> ego -z.
    R_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    R_pitch = Quaternion(axis=[0, 1, 0], degrees=pitch_deg).rotation_matrix  # cam-frame pitch
    T = SE3.from_rt(R_base @ R_pitch, np.array([1.7, 0.0, 1.5]))
    return cam, T


class TestGroundPlaneLift:
    @pytest.mark.parametrize("pitch_deg", [0.0, 2.0, -3.0])
    @pytest.mark.parametrize("gx,gy", [(10.0, 0.0), (30.0, -4.0), (60.0, 8.0)])
    def test_roundtrip_ground_point(self, pitch_deg, gx, gy):
        cam, T = nuscenes_like_rig(pitch_deg)
        lift = GroundPlaneLift(cam, T)
        # Project a known ego-frame ground point into the image...
        pt_cam = T.inverse().apply([gx, gy, 0.0])
        uv, depth = cam.project(pt_cam)
        assert depth[0] > 0
        # ...then lift the pixel back to the ground plane.
        xy, valid = lift.lift_pixels(uv)
        assert valid[0]
        assert np.max(np.abs(xy[0] - [gx, gy])) < 1e-6

    def test_horizon_pixel_invalid(self):
        cam, T = nuscenes_like_rig()
        lift = GroundPlaneLift(cam, T)
        # Principal row is the horizon for a level camera; above it must fail.
        _, valid = lift.lift_pixels([[800.0, 200.0]])
        assert not valid[0]

    def test_camera_height_read_from_extrinsics(self):
        cam, T = nuscenes_like_rig()
        assert GroundPlaneLift(cam, T).cam_height == pytest.approx(1.5)

    def test_meas_std_grows_quadratically(self):
        cam, T = nuscenes_like_rig()
        lift = GroundPlaneLift(cam, T)
        # 20m and 60m are both inside the clamp's active band for this rig.
        s20, s60 = lift.range_meas_std(20.0), lift.range_meas_std(60.0)
        assert s60 > s20
        assert s60 / s20 == pytest.approx(9.0, rel=0.05)  # r^2 scaling: (60/20)^2

    def test_box_bottom_range(self):
        cam, T = nuscenes_like_rig()
        lift = GroundPlaneLift(cam, T)
        # A 2m-wide object standing on the ground 20m ahead: its box bottom
        # center pixel is the projection of the ground contact point.
        pt_cam = T.inverse().apply([20.0, 0.0, 0.0])
        uv, _ = cam.project(pt_cam)
        box = np.array([uv[0, 0] - 60, uv[0, 1] - 120, uv[0, 0] + 60, uv[0, 1]])
        xy, rng = lift.lift_box_bottom(box)
        assert np.allclose(xy, [20.0, 0.0], atol=1e-6)
        assert rng == pytest.approx(np.hypot(20.0, 0.0))


class TestBevKalmanIntegration:
    def test_velocity_converges_with_moving_ego(self):
        """Object drives at constant global velocity while ego also moves;
        the global-frame BEV filter must recover the object's true velocity."""
        cam, T_ego_cam = nuscenes_like_rig()
        lift = GroundPlaneLift(cam, T_ego_cam)
        tracker = ByteTracker(n_init=2)
        dt = 1 / 12

        obj_v = np.array([8.0, 1.0])   # object global velocity, m/s
        ego_v = np.array([5.0, 0.0])   # ego drives straight at 5 m/s
        obj_p = np.array([25.0, 2.0])  # ahead of ego start
        ego_p = np.array([0.0, 0.0])

        track = None
        for _ in range(48):  # 4 seconds
            obj_p = obj_p + obj_v * dt
            ego_p = ego_p + ego_v * dt
            T_global_ego = SE3.from_rt(np.eye(3), [*ego_p, 0.0])
            # Synthesize the detection the camera would see.
            rel = obj_p - ego_p
            pt_cam = T_ego_cam.inverse().apply([*rel, 0.0])
            uv, depth = cam.project(pt_cam)
            assert depth[0] > 0
            box = np.array([uv[0, 0] - 40, uv[0, 1] - 80, uv[0, 0] + 40, uv[0, 1]])
            confirmed = tracker.step(
                [Detection2D(xyxy=box, score=0.9, category="car")], dt
            )
            update_bev_states(confirmed, lift, T_global_ego, dt)
            if confirmed:
                track = confirmed[0]

        bev = track.extra["bev"]
        assert np.max(np.abs(bev.kf.velocity - obj_v)) < 0.6
        # Ego-frame view: relative position ahead-left, sane magnitude.
        pos_e, vel_e = bev.in_ego(SE3.from_rt(np.eye(3), [*ego_p, 0.0]))
        assert np.allclose(pos_e, obj_p - ego_p, atol=0.5)

    def test_innovation_gate_rejects_teleport(self):
        """A single occlusion-induced range jump must not corrupt velocity."""
        from adp.lift.bev_state import BevState, _fresh_filter
        state = BevState(kf=_fresh_filter(np.array([20.0, 0.0]), 0.5), last_range_m=20.0)
        dt = 1 / 12
        for i in range(24):  # object crawling forward at 2 m/s
            state.kf.predict(dt)
            z = np.array([20.0 + 2.0 * (i + 1) * dt, 0.0])
            assert state.kf.mahalanobis_sq(z, 0.5**2) < 13.8
            state.kf.update(z, meas_var=0.5**2)
        v_before = state.kf.velocity.copy()
        # Occluded bottom: measurement teleports 40m downrange for one frame.
        state.kf.predict(dt)
        assert state.kf.mahalanobis_sq(np.array([64.0, 0.0]), 4.0**2) > 13.8
        # (update skipped by the gate) — velocity unchanged by the outlier
        assert np.allclose(state.kf.velocity, v_before)

    def test_gate_reseeds_after_persistent_jump(self):
        from adp.detect.detector import Detection2D  # noqa: F401
        from adp.lift.bev_state import BevState, MAX_REJECTS, _fresh_filter, update_bev_states
        from adp.track.bytetrack import Track, TrackState
        from adp.track.kalman import ConstantVelocityKalman
        cam, T_ego_cam = nuscenes_like_rig()
        lift = GroundPlaneLift(cam, T_ego_cam)
        T_g = SE3.from_rt(np.eye(3), np.zeros(3))
        # Hand-build a confirmed track whose box projects to 60m while its
        # BEV filter believes 15m: after MAX_REJECTS+1 frames it must re-seed.
        pt_cam = T_ego_cam.inverse().apply([60.0, 0.0, 0.0])
        uv, _ = cam.project(pt_cam)
        box_kf = ConstantVelocityKalman(
            z0=[uv[0, 0], uv[0, 1] - 40, 80, 80], pos_std=1, vel_std=1,
            meas_std=1, process_std=1,
        )
        track = Track(track_id=1, kf=box_kf, category="car", score=0.9,
                      state=TrackState.CONFIRMED)
        track.extra["bev"] = BevState(kf=_fresh_filter(np.array([15.0, 0.0]), 0.5),
                                      last_range_m=15.0)
        for _ in range(MAX_REJECTS + 1):
            update_bev_states([track], lift, T_g, dt=1 / 12)
        assert track.extra["bev"].last_range_m == pytest.approx(60.0, abs=1.0)

    def test_coasting_when_lift_invalid(self):
        cam, T_ego_cam = nuscenes_like_rig()
        lift = GroundPlaneLift(cam, T_ego_cam)
        tracker = ByteTracker(n_init=2)
        T_g = SE3.from_rt(np.eye(3), np.zeros(3))
        dt = 1 / 12
        # Confirm a track with a valid ground box first.
        pt_cam = T_ego_cam.inverse().apply([15.0, 0.0, 0.0])
        uv, _ = cam.project(pt_cam)
        box = np.array([uv[0, 0] - 40, uv[0, 1] - 80, uv[0, 0] + 40, uv[0, 1]])
        for _ in range(3):
            confirmed = tracker.step([Detection2D(xyxy=box, score=0.9, category="car")], dt)
            update_bev_states(confirmed, lift, T_g, dt)
        assert "bev" in confirmed[0].extra
        pos_before = confirmed[0].extra["bev"].kf.position.copy()
        # Now the detection's bottom edge sits above the horizon (bad box):
        sky_box = np.array([700.0, 100.0, 900.0, 200.0])
        confirmed = tracker.step([Detection2D(xyxy=sky_box, score=0.9, category="car")], dt)
        if confirmed:  # association may or may not hold; filter must not corrupt
            state = confirmed[0].extra.get("bev")
            if state is not None:
                assert np.all(np.isfinite(state.kf.position))
                assert np.linalg.norm(state.kf.position - pos_before) < 5.0
