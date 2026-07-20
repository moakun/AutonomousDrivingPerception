"""Metric lift v1: inverse perspective mapping (ground-plane homography).

Assumes locally flat road: the nuScenes ego frame has z=0 at road level
(CAM_FRONT sits ~1.5m above it), so a pixel ray intersected with the ego-frame
plane z=0 yields a metric ground point. Known failure modes, by design:
hills/banked curves (plane assumption breaks) and occluded box bottoms (the
observed bottom edge is not the object's ground contact).

Also provides the analytic range-dependent measurement noise used by the BEV
Kalman filters: a pixel error e on the box bottom maps to roughly
e * r^2 / (f * h) meters of range error at range r.
"""

from __future__ import annotations

import numpy as np

from adp.calib.camera import CameraModel, SE3, ray_plane_intersection


class GroundPlaneLift:
    def __init__(self, camera: CameraModel, T_ego_from_cam: SE3):
        self.camera = camera
        self.T_ego_from_cam = T_ego_from_cam
        self.cam_origin_ego = T_ego_from_cam.translation  # camera center, ego frame
        self.cam_height = float(self.cam_origin_ego[2])   # ~1.5m for CAM_FRONT

    def lift_pixels(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels -> (x, y) ground points in the EGO frame.

        Returns (xy (N,2), valid (N,)). Invalid where the ray points at or
        above the horizon (never meets the ground ahead).
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        rays_cam = self.camera.pixel_rays(uv)
        rays_ego = rays_cam @ self.T_ego_from_cam.rotation.T
        pts, valid = ray_plane_intersection(
            origins=self.cam_origin_ego,
            directions=rays_ego,
            plane_point=[0.0, 0.0, 0.0],
            plane_normal=[0.0, 0.0, 1.0],
        )
        return pts[:, :2], valid

    def lift_box_bottom(self, xyxy: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Box bottom-center pixel -> ego-frame (x, y) ground point.

        Returns (xy or None, range_m). The bottom-center is the object's
        ground-contact estimate under the flat-road assumption.
        """
        u = (xyxy[0] + xyxy[2]) / 2
        v = xyxy[3]
        xy, valid = self.lift_pixels([[u, v]])
        if not valid[0]:
            return None, float("nan")
        return xy[0], float(np.linalg.norm(xy[0]))

    def range_meas_std(self, range_m: float, pixel_noise: float = 3.0) -> float:
        """Expected metric noise of an IPM measurement at a given range.

        Derived from d(range)/d(pixel) = r^2 / (f * h); clamped so nearby
        objects keep a sane floor and far ones don't blow up the filter.
        """
        f = self.camera.K[1, 1]
        std = pixel_noise * range_m**2 / (f * max(self.cam_height, 0.1))
        return float(np.clip(std, 0.3, 8.0))
