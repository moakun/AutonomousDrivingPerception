"""M9: width-prior depth estimation and IPM+width fusion, on synthetic
geometry matching the nuScenes CAM_FRONT rig."""

import numpy as np
import pytest
from pyquaternion import Quaternion

from adp.calib.camera import CameraModel, SE3
from adp.lift.fused import FusedLift
from adp.lift.ipm import GroundPlaneLift
from adp.lift.width_prior import WidthPriorLift

PRIORS = {"car": {"width_median_m": 2.0, "rel_spread": 0.20},
          "pedestrian": {"width_median_m": 0.7, "rel_spread": 0.30}}


def rig():
    cam = CameraModel(K=np.array([[1266.0, 0, 800.0], [0, 1266.0, 450.0], [0, 0, 1.0]]),
                      width=1600, height=900)
    R_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    T = SE3.from_rt(R_base @ Quaternion(axis=[0, 1, 0], degrees=0).rotation_matrix,
                    np.array([1.7, 0.0, 1.5]))
    return cam, T


def box_for(cam, T, x, y=0.0, width_m=2.0, height_m=1.5, bottom_lift_px=0.0):
    """2D box of an object standing on the ground at ego (x, y).
    bottom_lift_px raises the bottom edge to simulate an occluding barrier."""
    pts_ego = np.array([[x, y - width_m / 2, 0.0], [x, y + width_m / 2, 0.0],
                        [x, y, height_m]])
    uv, depth = cam.project(T.inverse().apply(pts_ego))
    assert np.all(depth > 0)
    x0, x1 = sorted([uv[0, 0], uv[1, 0]])
    y_bottom = max(uv[0, 1], uv[1, 1]) - bottom_lift_px
    return np.array([x0, uv[2, 1], x1, y_bottom])


class TestWidthPriorLift:
    def test_depth_from_known_width(self):
        lift = WidthPriorLift(priors=PRIORS)
        # z = f * W / w_px = 1000 * 2.0 / 100 = 20
        z, _ = lift.estimate_depth(np.array([0, 0, 100, 80]), "car", f_x=1000.0)
        assert z == pytest.approx(20.0)

    def test_relative_uncertainty(self):
        lift = WidthPriorLift(priors=PRIORS)
        z, s = lift.estimate_depth(np.array([0, 0, 100, 80]), "car", f_x=1000.0)
        expect = z * np.hypot(0.20, 4.0 / 100)  # prior spread + box noise
        assert s == pytest.approx(expect)

    def test_error_is_scale_free(self):
        """Unlike IPM's r^2 growth, relative width error is constant with range."""
        lift = WidthPriorLift(priors=PRIORS)
        z1, s1 = lift.estimate_depth(np.array([0, 0, 200, 160]), "car", 1000.0)
        z2, s2 = lift.estimate_depth(np.array([0, 0, 50, 40]), "car", 1000.0)
        assert z2 > z1
        assert (s2 / z2) == pytest.approx(s1 / z1, rel=0.25)

    def test_unknown_category(self):
        lift = WidthPriorLift(priors=PRIORS)
        z, _ = lift.estimate_depth(np.array([0, 0, 100, 80]), "spaceship", 1000.0)
        assert np.isnan(z)

    def test_tiny_box_rejected(self):
        lift = WidthPriorLift(priors=PRIORS)
        z, _ = lift.estimate_depth(np.array([0, 0, 6, 5]), "car", 1000.0)
        assert np.isnan(z)

    def test_lift_box_recovers_position(self):
        cam, T = rig()
        lift = WidthPriorLift(priors=PRIORS)
        box = box_for(cam, T, x=25.0, y=2.0, width_m=2.0)
        xy, rng, sigma = lift.lift_box(box, "car", cam, T)
        assert xy[0] == pytest.approx(25.0, abs=0.8)
        assert xy[1] == pytest.approx(2.0, abs=0.5)
        assert sigma > 0

    def test_occluded_bottom_does_not_affect_width(self):
        """The whole premise: a truncated bottom edge leaves width untouched."""
        cam, T = rig()
        lift = WidthPriorLift(priors=PRIORS)
        clean = box_for(cam, T, x=20.0)
        occluded = box_for(cam, T, x=20.0, bottom_lift_px=25.0)
        z_clean, _ = lift.estimate_depth(clean, "car", cam.K[0, 0])
        z_occ, _ = lift.estimate_depth(occluded, "car", cam.K[0, 0])
        assert z_occ == pytest.approx(z_clean)

    def test_aspect_conditioning(self):
        priors = {"car": {"width_median_m": 2.0, "rel_spread": 0.25,
                          "linear_a": 1.0, "linear_b": 1.0,
                          "rel_spread_conditional": 0.15}}
        lift = WidthPriorLift(priors=priors, use_aspect=True)
        # aspect 1.0 -> W=2.0 ; aspect 2.0 (side-on) -> W=3.0
        w1, rel1 = lift.effective_width("car", 1.0)
        w2, _ = lift.effective_width("car", 2.0)
        assert w1 == pytest.approx(2.0) and w2 == pytest.approx(3.0)
        assert rel1 == 0.15
        # Extrapolation guard: absurd aspect cannot produce absurd width.
        w_big, _ = lift.effective_width("car", 20.0)
        assert w_big <= 2.5 * 2.0


class TestFusedLift:
    def make(self):
        cam, T = rig()
        return FusedLift(GroundPlaneLift(cam, T), WidthPriorLift(priors=PRIORS)), cam, T

    def test_near_field_fuses_and_tightens(self):
        fused, cam, T = self.make()
        box = box_for(cam, T, x=8.0, width_m=2.0)   # inside FUSE_RANGE_M
        _, r = fused.lift_box_bottom(box, "car")
        assert r == pytest.approx(8.0, abs=1.0)
        assert fused.last_source == "fused"
        # Fusing two independent cues must beat either alone.
        s_ipm = fused.ipm.range_meas_std(8.0)
        _, _, s_w = fused.width.lift_box(box, "car", cam, T)
        assert fused.range_meas_std(r) < min(s_ipm, s_w)

    def test_far_field_drops_ipm(self):
        """Past the switch point IPM is biased, so width is used alone."""
        fused, cam, T = self.make()
        box = box_for(cam, T, x=30.0, width_m=2.0)
        _, r = fused.lift_box_bottom(box, "car")
        assert fused.last_source == "width"
        assert r == pytest.approx(30.0, rel=0.15)

    def test_severe_occlusion_recovered(self):
        """M9's reason for existing: barrier-truncated box biases IPM long.

        With honest (measured) IPM variance the recovery comes mostly from
        weighting rather than the occlusion gate — at that range IPM's own
        sigma is large, so width dominates automatically.
        """
        fused, cam, T = self.make()
        box = box_for(cam, T, x=20.0, bottom_lift_px=60.0)
        _, r_ipm = fused.ipm.lift_box_bottom(box)
        assert r_ipm > 35.0  # IPM badly fooled
        _, r_fused = fused.lift_box_bottom(box, "car")
        assert abs(r_fused - 20.0) < abs(r_ipm - 20.0) / 4  # error cut >=4x

    def test_moderate_occlusion_needs_temporal_gate(self):
        """A 30px truncation is below single-frame significance but persists,
        so the per-track EMA must catch what one frame cannot."""
        fused, cam, T = self.make()
        box = box_for(cam, T, x=20.0, bottom_lift_px=30.0)
        _, r_ipm = fused.ipm.lift_box_bottom(box)

        # Single frame, no track state: correctly NOT flagged (too noisy).
        fused.lift_box_bottom(box, "car")
        assert fused.last_occluded is False

        # Same box over consecutive frames with track state: flagged.
        state = {}
        for _ in range(12):
            _, r_fused = fused.lift_box_bottom(box, "car", state)
        assert fused.last_occluded is True
        assert state["occl_ema"] > 0.8
        assert abs(r_fused - 20.0) < abs(r_ipm - 20.0)

    def test_clean_object_never_flagged_over_time(self):
        """The temporal gate must not drift into false positives on good boxes."""
        fused, cam, T = self.make()
        state = {}
        for x in np.linspace(30.0, 12.0, 25):  # approaching, unoccluded
            box = box_for(cam, T, x=float(x))
            fused.lift_box_bottom(box, "car", state)
            assert fused.last_occluded is False
        assert state["occl_ema"] < 0.8

    def test_asymmetry_width_long_keeps_ipm(self):
        """Width reading long (narrow/side-on box) must NOT trip the gate,
        which exists only for the IPM-reads-long occlusion signature."""
        fused, cam, T = self.make()
        box = box_for(cam, T, x=8.0, width_m=1.0)
        fused.lift_box_bottom(box, "car")
        assert fused.last_occluded is False
        assert fused.n_occluded == 0

    def test_no_category_uses_ipm_only(self):
        fused, cam, T = self.make()
        box = box_for(cam, T, x=20.0)
        _, r = fused.lift_box_bottom(box, None)
        assert fused.last_source == "ipm"
        assert r == pytest.approx(20.0, abs=0.5)

    def test_width_only_when_ipm_invalid(self):
        fused, cam, T = self.make()
        # Box bottom above the horizon: IPM cannot lift it at all.
        box = np.array([780.0, 150.0, 880.0, 300.0])
        xy, r = fused.lift_box_bottom(box, "car")
        assert fused.last_source == "width"
        assert xy is not None and np.isfinite(r)

    def test_bearing_preserved(self):
        """In the fused (near-field) band the bearing comes from IPM, whose
        horizontal centre survives vertical truncation."""
        fused, cam, T = self.make()
        box = box_for(cam, T, x=9.0, y=2.0)
        xy_ipm, _ = fused.ipm.lift_box_bottom(box)
        xy, _ = fused.lift_box_bottom(box, "car")
        # Same direction from ego, different magnitude.
        cos = np.dot(xy, xy_ipm) / (np.linalg.norm(xy) * np.linalg.norm(xy_ipm))
        assert cos == pytest.approx(1.0, abs=1e-6)

    def test_unusable_input_returns_none(self):
        fused, cam, T = self.make()
        xy, r = fused.lift_box_bottom(np.array([780.0, 150.0, 785.0, 156.0]), "car")
        assert xy is None and np.isnan(r)
