"""BEV canvas: ego-frame top-down view of tracked objects.

This view is what proves the metric lift works — objects at their estimated
metric positions with velocity vectors, optionally with GT ghost markers for
validation overlays.
"""

from __future__ import annotations

import cv2
import numpy as np

CLASS_COLORS = {  # BGR, matches scripts/check_data.py
    "car": (255, 160, 0),
    "truck": (255, 80, 80),
    "bus": (0, 200, 255),
    "pedestrian": (0, 80, 255),
    "cyclist": (0, 220, 120),
}


class BevCanvas:
    """Ego at bottom-center, x (forward) up, y (left) to the left."""

    def __init__(self, size_px: int = 540, x_range: tuple = (-8.0, 64.0),
                 y_half: float = 27.0):
        self.size = size_px
        self.x_range = x_range
        self.y_half = y_half
        self.scale = size_px / (x_range[1] - x_range[0])  # px per meter

    def to_px(self, x_m: float, y_m: float) -> tuple[int, int]:
        u = int(self.size / 2 - y_m * self.scale)
        v = int(self.size - (x_m - self.x_range[0]) * self.scale)
        return u, v


def render_bev(
    tracks: list[dict],
    gt: list[dict] | None = None,
    canvas: BevCanvas | None = None,
) -> np.ndarray:
    """tracks: [{id, category, pos (2,), vel (2,), range_std}] in EGO frame.
    gt: [{pos (2,)}] ghost markers. Returns BGR image."""
    c = canvas or BevCanvas()
    img = np.full((c.size, c.size, 3), 30, dtype=np.uint8)

    # Range rings every 10m + faint lane-width corridor for orientation.
    for r in range(10, int(c.x_range[1]) + 1, 10):
        cv2.circle(img, c.to_px(0, 0), int(r * c.scale), (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r}", c.to_px(r * 0.02 - 0.5, -r * 0.99),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1, cv2.LINE_AA)
    for y in (-1.85, 1.85):
        cv2.line(img, c.to_px(0, y), c.to_px(c.x_range[1], y), (45, 45, 60), 1)

    # Ego marker.
    ex, ey = c.to_px(0, 0)
    cv2.fillPoly(img, [np.array([[ex, ey - 10], [ex - 6, ey + 6], [ex + 6, ey + 6]])],
                 (255, 255, 255))

    if gt:
        for g in gt:
            u, v = c.to_px(*g["pos"])
            cv2.drawMarker(img, (u, v), (140, 140, 140), cv2.MARKER_TILTED_CROSS, 10, 1)

    for t in tracks:
        u, v = c.to_px(*t["pos"])
        if not (0 <= u < c.size and 0 <= v < c.size):
            continue
        color = CLASS_COLORS.get(t["category"], (160, 160, 160))
        # Position uncertainty ring (1 sigma).
        cv2.circle(img, (u, v), max(int(t.get("range_std", 0) * c.scale), 2),
                   color, 1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 3, color, -1, cv2.LINE_AA)
        # Velocity vector: 1s lookahead.
        vel = t.get("vel")
        if vel is not None and np.linalg.norm(vel) > 0.5:
            u2, v2 = c.to_px(t["pos"][0] + vel[0], t["pos"][1] + vel[1])
            cv2.arrowedLine(img, (u, v), (u2, v2), color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(img, f"#{t['id']}", (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img
