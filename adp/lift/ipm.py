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

# Empirical IPM range-noise model sigma(r) = C * r^P, fitted to nuScenes GT by
# scripts/calibrate_lift_noise.py (n=1480 matched pairs). The exponent lands at
# 2.51 rather than the theoretical 2.0 because occlusion-driven bias grows
# faster than pixel jitter.
MEASURED_C = 0.001451
MEASURED_P = 2.51


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

    def lift_box_bottom(self, xyxy: np.ndarray, category: str | None = None,
                        state: dict | None = None) -> tuple[np.ndarray | None, float]:
        """Box bottom-center pixel -> ego-frame (x, y) ground point.

        Returns (xy or None, range_m). The bottom-center is the object's
        ground-contact estimate under the flat-road assumption. `category` is
        accepted for interface parity with the fused/hybrid lifts and ignored.
        """
        u = (xyxy[0] + xyxy[2]) / 2
        v = xyxy[3]
        xy, valid = self.lift_pixels([[u, v]])
        if not valid[0]:
            return None, float("nan")
        return xy[0], float(np.linalg.norm(xy[0]))

    def range_meas_std(self, range_m: float, pixel_noise: float = 3.0,
                       measured: bool = True) -> float:
        """Expected metric noise of an IPM measurement at a given range.

        The analytic model (measured=False) is d(range)/d(pixel) = r^2/(f*h),
        i.e. pure pixel jitter. It is the right SHAPE but badly optimistic in
        practice, because the dominant error is systematic, not jitter: box
        bottoms sit above the true contact point under occlusion, and the road
        is not exactly planar. Measured against nuScenes GT
        (scripts/calibrate_lift_noise.py, n=1480) the real spread is 4-6x
        larger and grows as r^2.5:

            r=20m: analytic 0.63m vs measured 2.64m
            r=50m: analytic 3.95m vs measured 26.2m

        Using the analytic value made inverse-variance fusion over-trust IPM by
        more than an order of magnitude in variance. The measured model is the
        default; it is what makes FusedLift weight the two cues correctly.
        """
        if measured:
            std = MEASURED_C * range_m**MEASURED_P
            return float(np.clip(std, 0.3, 40.0))
        f = self.camera.K[1, 1]
        std = pixel_noise * range_m**2 / (f * max(self.cam_height, 0.1))
        return float(np.clip(std, 0.3, 8.0))
