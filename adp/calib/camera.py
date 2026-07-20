"""Camera geometry: pinhole projection, rigid transforms, ray-plane intersection.

Conventions:
- Points are float64 arrays of shape (N, 3); pixels are (N, 2) in (u, v).
- Camera frame is OpenCV-style: +x right, +y down, +z forward (optical axis).
- SE3 maps points FROM its source frame TO its target frame: p_dst = T.apply(p_src).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion


@dataclass
class SE3:
    """Rigid transform as a 4x4 homogeneous matrix."""

    matrix: np.ndarray  # (4, 4)

    @classmethod
    def from_rt(cls, rotation: np.ndarray, translation: np.ndarray) -> "SE3":
        m = np.eye(4)
        m[:3, :3] = rotation
        m[:3, 3] = translation
        return cls(m)

    @classmethod
    def from_quat_trans(cls, quat_wxyz, translation) -> "SE3":
        """From nuScenes-style rotation quaternion [w, x, y, z] + translation."""
        return cls.from_rt(Quaternion(quat_wxyz).rotation_matrix, np.asarray(translation, dtype=float))

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def inverse(self) -> "SE3":
        r_inv = self.rotation.T
        return SE3.from_rt(r_inv, -r_inv @ self.translation)

    def compose(self, other: "SE3") -> "SE3":
        """self ∘ other: apply `other` first, then `self`."""
        return SE3(self.matrix @ other.matrix)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return pts @ self.rotation.T + self.translation


@dataclass
class CameraModel:
    """Pinhole camera. nuScenes images are already undistorted; K is the full model."""

    K: np.ndarray  # (3, 3)
    width: int
    height: int

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points to pixels.

        Returns (uv (N,2), depth (N,)). Points with depth <= 0 are behind the
        camera; their uv values are invalid and must be masked by the caller.
        """
        pts = np.atleast_2d(np.asarray(pts_cam, dtype=float))
        depth = pts[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uvw = pts @ self.K.T
            uv = uvw[:, :2] / uvw[:, 2:3]
        return uv, depth

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Pixels + depths -> camera-frame 3D points."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        depth = np.asarray(depth, dtype=float).reshape(-1, 1)
        ones = np.ones((uv.shape[0], 1))
        rays = np.hstack([uv, ones]) @ np.linalg.inv(self.K).T
        return rays * depth

    def pixel_rays(self, uv: np.ndarray) -> np.ndarray:
        """Camera-frame ray directions (unnormalized, z=1) through pixels."""
        return self.backproject(uv, np.ones(np.atleast_2d(uv).shape[0]))

    def in_image(self, uv: np.ndarray, depth: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask: point is in front of the camera and inside the image."""
        uv = np.atleast_2d(uv)
        return (
            (depth > 0)
            & (uv[:, 0] >= -margin)
            & (uv[:, 0] < self.width + margin)
            & (uv[:, 1] >= -margin)
            & (uv[:, 1] < self.height + margin)
        )


def ray_plane_intersection(
    origins: np.ndarray,
    directions: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect rays with a plane.

    Returns (points (N,3), valid (N,)). valid is False where the ray is parallel
    to the plane or the intersection is behind the ray origin (t <= 0); those
    rows contain NaN.
    """
    origins = np.atleast_2d(np.asarray(origins, dtype=float))
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    n = np.asarray(plane_normal, dtype=float)
    p0 = np.asarray(plane_point, dtype=float)

    denom = directions @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((p0 - origins) @ n) / denom
    valid = (np.abs(denom) > 1e-12) & (t > 0)
    points = origins + np.where(valid, t, 0.0)[:, None] * directions
    points[~valid] = np.nan
    return points, valid
