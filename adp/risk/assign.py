"""Lane assignment and pedestrian crossing intent, in BEV.

Lane zones turn "there's a car" into "there's a car in my path". When M4
found no corridor (unmarked road, intersection, night), a fixed-width
straight corridor is the documented fallback — the assignment result records
which one was used so downstream consumers know the confidence differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from adp.lanes.bev_lanes import EgoCorridor, LaneLine

FALLBACK_HALF_WIDTH = 1.85  # straight ego corridor when no lanes detected
DEFAULT_LANE_WIDTH = 3.7


class LaneZone(Enum):
    EGO = "ego"
    ADJACENT = "adjacent"
    OFF_PATH = "off_path"
    BEHIND = "behind"


def fallback_corridor() -> EgoCorridor:
    return EgoCorridor(
        left=LaneLine(np.array([FALLBACK_HALF_WIDTH, 0.0, 0.0]), (0.0, 60.0), 0),
        right=LaneLine(np.array([-FALLBACK_HALF_WIDTH, 0.0, 0.0]), (0.0, 60.0), 0),
        width=2 * FALLBACK_HALF_WIDTH,
    )


@dataclass
class LaneAssignment:
    zone: LaneZone
    corridor_source: str  # "detected" | "fallback"
    lateral_offset_m: float  # signed distance outside corridor (0 if inside)


def assign_lane(pos_ego_xy: np.ndarray, corridor: EgoCorridor | None) -> LaneAssignment:
    x, y = float(pos_ego_xy[0]), float(pos_ego_xy[1])
    source = "detected" if corridor is not None and corridor.valid else "fallback"
    corr = corridor if source == "detected" else fallback_corridor()

    if x < 0:
        return LaneAssignment(LaneZone.BEHIND, source, 0.0)

    y_left = float(corr.left.y_at(x))
    y_right = float(corr.right.y_at(x))
    width = corr.width or DEFAULT_LANE_WIDTH

    if y_right <= y <= y_left:
        return LaneAssignment(LaneZone.EGO, source, 0.0)
    offset = y - y_left if y > y_left else y - y_right  # signed: + left, - right
    if abs(offset) <= width:
        return LaneAssignment(LaneZone.ADJACENT, source, offset)
    return LaneAssignment(LaneZone.OFF_PATH, source, offset)


def crossing_intent(
    pos_ego_xy: np.ndarray,
    vel_ego_frame_xy: np.ndarray,
    assignment: LaneAssignment,
    corridor: EgoCorridor | None,
    horizon_s: float = 3.0,
    x_max: float = 40.0,
) -> float | None:
    """Cheap pedestrian/cyclist intent: seconds until the object's lateral
    motion carries it into the ego corridor, if within horizon. None = no
    crossing indicated. (Plan: 'is the BEV velocity vector pointed into the
    ego path within N seconds?')"""
    if assignment.zone not in (LaneZone.ADJACENT, LaneZone.OFF_PATH):
        return None
    x, y = float(pos_ego_xy[0]), float(pos_ego_xy[1])
    vy = float(vel_ego_frame_xy[1])
    if x < 0 or x > x_max or abs(vy) < 0.2:
        return None
    # Lateral gap to the near corridor boundary, sign-aware.
    corr = corridor if corridor is not None and corridor.valid else fallback_corridor()
    if y > 0:  # object left of corridor: needs vy < 0 to enter
        gap = y - float(corr.left.y_at(x))
        t = gap / -vy if vy < 0 else None
    else:
        gap = float(corr.right.y_at(x)) - y
        t = gap / vy if vy > 0 else None
    if t is not None and 0 <= t <= horizon_s:
        return float(t)
    return None
