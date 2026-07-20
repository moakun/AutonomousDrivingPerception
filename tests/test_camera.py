"""Geometry round-trip tests — M0 exit criteria.

These use synthetic cameras/points only; real-data verification lives in
scripts/check_data.py (renders GT boxes into images for visual inspection).
"""

import numpy as np
import pytest

from adp.calib.camera import CameraModel, SE3, ray_plane_intersection

RNG = np.random.default_rng(7)


def random_camera() -> CameraModel:
    fx, fy = RNG.uniform(800, 1600, size=2)
    return CameraModel(
        K=np.array([[fx, 0, 810.0], [0, fy, 460.0], [0, 0, 1.0]]),
        width=1600,
        height=900,
    )


class TestProjection:
    def test_project_backproject_roundtrip(self):
        cam = random_camera()
        pts = RNG.uniform([-20, -5, 1], [20, 5, 80], size=(500, 3))
        uv, depth = cam.project(pts)
        restored = cam.backproject(uv, depth)
        assert np.max(np.abs(restored - pts)) < 1e-9

    def test_backproject_project_pixel_roundtrip(self):
        cam = random_camera()
        uv = RNG.uniform([0, 0], [cam.width, cam.height], size=(500, 2))
        depth = RNG.uniform(1, 100, size=500)
        uv2, depth2 = cam.project(cam.backproject(uv, depth))
        assert np.max(np.abs(uv2 - uv)) < 1e-6  # M0 target: < 1px; achieved: ~machine eps
        assert np.max(np.abs(depth2 - depth)) < 1e-9

    def test_principal_point_projects_to_center(self):
        cam = random_camera()
        uv, depth = cam.project(np.array([[0.0, 0.0, 10.0]]))
        assert np.allclose(uv[0], [cam.K[0, 2], cam.K[1, 2]])
        assert depth[0] == 10.0

    def test_behind_camera_flagged(self):
        cam = random_camera()
        uv, depth = cam.project(np.array([[0.0, 0.0, -5.0]]))
        assert depth[0] < 0
        assert not cam.in_image(uv, depth)[0]


class TestSE3:
    def test_inverse_roundtrip(self):
        T = SE3.from_quat_trans(RNG.normal(size=4), RNG.normal(size=3) * 10)
        pts = RNG.normal(size=(100, 3)) * 50
        restored = T.inverse().apply(T.apply(pts))
        assert np.max(np.abs(restored - pts)) < 1e-9

    def test_compose_matches_sequential_apply(self):
        A = SE3.from_quat_trans(RNG.normal(size=4), RNG.normal(size=3))
        B = SE3.from_quat_trans(RNG.normal(size=4), RNG.normal(size=3))
        pts = RNG.normal(size=(50, 3))
        assert np.allclose(A.compose(B).apply(pts), A.apply(B.apply(pts)))

    def test_identity(self):
        T = SE3.from_rt(np.eye(3), np.zeros(3))
        pts = RNG.normal(size=(10, 3))
        assert np.allclose(T.apply(pts), pts)


class TestRayPlane:
    def test_known_ground_intersection(self):
        # Camera 1.5m above ground (cam frame: +y down, ground plane y = +1.5),
        # ray pitched down so it hits the ground at exactly z = 30m ahead.
        h = 1.5
        z_hit = 30.0
        direction = np.array([0.0, h / z_hit, 1.0])
        pts, valid = ray_plane_intersection(
            origins=np.zeros(3), directions=direction,
            plane_point=[0, h, 0], plane_normal=[0, 1, 0],
        )
        assert valid[0]
        assert np.allclose(pts[0], [0.0, h, z_hit])

    def test_parallel_ray_invalid(self):
        pts, valid = ray_plane_intersection(
            origins=np.zeros(3), directions=[0, 0, 1],
            plane_point=[0, 1.5, 0], plane_normal=[0, 1, 0],
        )
        assert not valid[0]
        assert np.all(np.isnan(pts[0]))

    def test_horizon_ray_invalid(self):
        # Ray pointing above the horizon never hits the ground (t <= 0).
        pts, valid = ray_plane_intersection(
            origins=np.zeros(3), directions=[0, -0.1, 1],
            plane_point=[0, 1.5, 0], plane_normal=[0, 1, 0],
        )
        assert not valid[0]

    def test_full_ipm_chain_synthetic(self):
        # Project a known ground point, then recover it from the pixel alone
        # via ray-ground intersection — the exact mechanism M3 (IPM) will use.
        cam = random_camera()
        h = 1.6
        ground_pt = np.array([[3.0, h, 25.0]])  # on ground, 25m ahead, 3m right
        uv, _ = cam.project(ground_pt)
        rays = cam.pixel_rays(uv)
        recovered, valid = ray_plane_intersection(
            origins=np.zeros(3), directions=rays,
            plane_point=[0, h, 0], plane_normal=[0, 1, 0],
        )
        assert valid[0]
        assert np.max(np.abs(recovered - ground_pt)) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
