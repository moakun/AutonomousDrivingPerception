"""Calibration for footage that arrives without any: estimate intrinsics and
the camera's height/pitch from the video itself.

Stage 0 of the plan flagged this as the price of leaving nuScenes: "you need
intrinsics — either checkerboard-calibrate the camera or estimate focal length
from vanishing points."

Two estimators:

1. Focal length from the horizontal field of view. Dashcams cluster in a
   narrow band (~55-75 deg HFOV), so f_x = (W/2) / tan(HFOV/2) lands within
   ~15% without any scene analysis. Crude, but honest and bounded.

2. Pitch from the vanishing point of lane lines. Lane markings are parallel in
   the world, so their image lines meet at the vanishing point, whose vertical
   offset from the principal point gives camera pitch directly:
   pitch = atan((v_vp - c_y) / f_y). This is worth having because M10 measured
   pitch as the error IPM is most sensitive to (4.9x degradation at 2 deg).

What CANNOT be recovered from monocular video alone is metric scale: camera
height must be supplied or assumed. After M9 that matters much less than it
used to — the width prior carries range from apparent size and f_x, and only
the sub-12m IPM band depends on height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DASHCAM_HFOV_DEG = 65.0     # typical consumer dashcam
DASHCAM_HEIGHT_M = 1.30     # windscreen mount, saloon/hatchback


@dataclass
class EstimatedCalibration:
    K: np.ndarray
    height_m: float
    pitch_deg: float
    hfov_deg: float
    source: str          # how each part was obtained
    pitch_confidence: int = 0   # number of frames that voted


def intrinsics_from_hfov(width: int, height: int,
                         hfov_deg: float = DASHCAM_HFOV_DEG) -> np.ndarray:
    """Pinhole K from an assumed horizontal field of view."""
    f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


def vanishing_point(lines: np.ndarray) -> np.ndarray | None:
    """Least-squares vanishing point of image lines.

    `lines` is (N, 4) as (x0, y0, x1, y1). Each line contributes the constraint
    n . p = 0 where n is its normal in homogeneous coordinates; the vanishing
    point is the least-squares intersection.
    """
    if len(lines) < 2:
        return None
    p0 = np.stack([lines[:, 0], lines[:, 1], np.ones(len(lines))], axis=1)
    p1 = np.stack([lines[:, 2], lines[:, 3], np.ones(len(lines))], axis=1)
    normals = np.cross(p0, p1)                      # line in homogeneous form
    norm = np.linalg.norm(normals[:, :2], axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-9)
    # Solve min |A v|^2 with |v| = 1 -> smallest singular vector.
    _, _, vh = np.linalg.svd(normals)
    v = vh[-1]
    if abs(v[2]) < 1e-9:
        return None                                 # vanishing point at infinity
    return v[:2] / v[2]


def pitch_from_vanishing_point(v_vp: float, c_y: float, f_y: float) -> float:
    """Camera pitch (degrees, positive = nose up) from vanishing-point height.

    For a camera looking along the road, the vanishing point of road-parallel
    lines sits on the horizon; its offset from the principal point is the pitch.
    """
    return float(np.degrees(np.arctan((c_y - v_vp) / f_y)))


def lane_mask_to_lines(mask: np.ndarray, min_len: int = 60,
                       max_lines: int = 40) -> np.ndarray:
    """Hough segments from a lane-line mask, filtered to plausible lane
    directions (near-horizontal segments are lane-crossing clutter)."""
    import cv2
    seg = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, np.pi / 180,
                          threshold=40, minLineLength=min_len, maxLineGap=20)
    if seg is None:
        return np.empty((0, 4))
    seg = seg[:, 0, :].astype(float)
    dx = seg[:, 2] - seg[:, 0]
    dy = seg[:, 3] - seg[:, 1]
    angle = np.abs(np.degrees(np.arctan2(dy, dx)))
    keep = (angle > 20.0) & (angle < 80.0)   # exclude horizontal + vertical
    seg = seg[keep]
    if len(seg) > max_lines:
        length = np.hypot(dx[keep], dy[keep])
        seg = seg[np.argsort(-length)[:max_lines]]
    return seg


def estimate_from_frames(frames_masks, width: int, height: int,
                         hfov_deg: float = DASHCAM_HFOV_DEG,
                         camera_height_m: float = DASHCAM_HEIGHT_M
                         ) -> EstimatedCalibration:
    """Estimate K (from HFOV) and pitch (from lane vanishing points).

    `frames_masks` is an iterable of boolean lane masks. The pitch estimate is
    the median over frames that yielded a usable vanishing point, which rejects
    the frames where lane detection produced junk.
    """
    K = intrinsics_from_hfov(width, height, hfov_deg)
    pitches = []
    for mask in frames_masks:
        lines = lane_mask_to_lines(mask)
        vp = vanishing_point(lines)
        if vp is None:
            continue
        # A vanishing point far outside the frame means the fit is junk.
        if not (0 <= vp[0] < width and -height < vp[1] < 2 * height):
            continue
        pitches.append(pitch_from_vanishing_point(vp[1], K[1, 2], K[1, 1]))
    pitch = float(np.median(pitches)) if pitches else 0.0
    return EstimatedCalibration(
        K=K, height_m=camera_height_m, pitch_deg=pitch, hfov_deg=hfov_deg,
        source=(f"f from assumed {hfov_deg:.0f} deg HFOV; "
                + (f"pitch from {len(pitches)} vanishing points"
                   if pitches else "pitch assumed 0 (no vanishing point found)")
                + f"; height assumed {camera_height_m:.2f}m"),
        pitch_confidence=len(pitches),
    )
