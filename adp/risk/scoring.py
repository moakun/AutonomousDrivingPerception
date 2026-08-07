"""Per-object risk records and the scalar score.

Design rule from the plan: never a bare color-coded box. Every record carries
the computed quantities it was scored from, and `reasons` lists the specific
contributions in human-readable form ("TTC 1.8±0.4s", "same lane, closing
12.0 m/s"). The score is a transparent weighted sum, not a learned blob:

    score = zone_base
          + ttc_term        (0..0.45, ramps up as trustworthy TTC drops below 5s)
          + rss_term        (0.25 if RSS gap violated, ego/adjacent zones only)
          + intent_term     (0.2 if pedestrian/cyclist crossing within 3s)
    clamped to [0, 1]; buckets: <0.3 low, <0.6 caution, >=0.6 high.

Scalar ego risk = max over object scores (a single imminent collision must
dominate any number of parked cars).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adp.risk.assign import LaneAssignment, LaneZone
from adp.risk.ttc import TtcResult

from adp.config import CONFIG

ZONE_BASE = {z: CONFIG.risk.zone_base[z.value] for z in LaneZone}
TTC_CRITICAL_S = CONFIG.risk.ttc_critical_s

# Category-plausible top speeds (m/s). An estimated object speed above these
# is a corrupted filter (occlusion drift, re-seed), not physics — such records
# keep their zone but have TTC/RSS scoring suppressed, with the reason stated.
PLAUSIBLE_SPEED = CONFIG.risk.plausible_speed


@dataclass
class RiskRecord:
    track_id: int
    category: str
    range_m: float
    zone: LaneZone
    corridor_source: str
    closing_ms: float | None
    ttc: TtcResult | None
    rss_min_gap_m: float | None
    rss_violated: bool
    intent_cross_s: float | None
    object_speed_ms: float = 0.0  # estimated GLOBAL speed (for plausibility)
    score: float = 0.0
    bucket: str = "low"
    reasons: list[str] = field(default_factory=list)

    @property
    def velocity_plausible(self) -> bool:
        limit = PLAUSIBLE_SPEED.get(self.category, 45.0)
        return self.object_speed_ms <= limit


def score_object(record: RiskRecord) -> RiskRecord:
    """Fill score/bucket/reasons from the record's computed quantities."""
    score = ZONE_BASE[record.zone]
    reasons = [f"{record.zone.value} lane"
               + (" (fallback corridor)" if record.corridor_source == "fallback" else "")]

    if not record.velocity_plausible:
        reasons.append(
            f"velocity implausible for {record.category} "
            f"({record.object_speed_ms:.1f} m/s) — TTC/RSS suppressed")
        record.score = float(np.clip(score, 0.0, 1.0))
        record.bucket = ("high" if record.score >= 0.6
                         else "caution" if record.score >= 0.3 else "low")
        record.reasons = reasons
        return record

    if record.ttc is not None and record.zone in (LaneZone.EGO, LaneZone.ADJACENT):
        reasons.append(str(record.ttc)
                       + f", closing {record.ttc.closing_ms:.1f} m/s")
        if record.ttc.trustworthy and record.ttc.ttc_s < TTC_CRITICAL_S:
            score += CONFIG.risk.ttc_weight * (1.0 - record.ttc.ttc_s / TTC_CRITICAL_S)

    if record.rss_violated and record.zone in (LaneZone.EGO, LaneZone.ADJACENT):
        score += CONFIG.risk.rss_weight
        gap = record.ttc.gap_m if record.ttc else record.range_m
        reasons.append(f"RSS violated: gap {gap:.1f}m < {record.rss_min_gap_m:.1f}m required")

    if record.intent_cross_s is not None:
        score += CONFIG.risk.intent_weight
        reasons.append(f"crossing into path in ~{record.intent_cross_s:.1f}s")

    record.score = float(np.clip(score, 0.0, 1.0))
    record.bucket = bucket_for(record.score)
    record.reasons = reasons
    return record


def bucket_for(score: float) -> str:
    enter = CONFIG.risk.bucket_enter
    return ("high" if score >= enter["high"]
            else "caution" if score >= enter["caution"] else "low")


class BucketHysteresis:
    """Debounce bucket transitions (improvement #9).

    Without this an object hovering at a threshold flips label every frame:
    visually noisy, and worse for any consumer that acts on transitions. A
    bucket is entered at its enter threshold but only left below a lower exit
    threshold, and a promotion must persist for `confirm_frames` — except
    escalation to 'high', which is applied immediately, because delaying a
    collision warning to look tidy is the wrong trade.
    """

    ORDER = {"low": 0, "caution": 1, "high": 2}

    def __init__(self):
        self._state: dict[int, tuple[str, int]] = {}  # id -> (bucket, pending)

    def apply(self, track_id: int, raw_bucket: str, score: float) -> str:
        prev, pending = self._state.get(track_id, (raw_bucket, 0))
        exit_thr = CONFIG.risk.bucket_exit
        keep = prev
        if self.ORDER[raw_bucket] > self.ORDER[prev]:
            if raw_bucket == "high" or pending + 1 >= CONFIG.risk.confirm_frames:
                keep, pending = raw_bucket, 0
            else:
                pending += 1
        elif self.ORDER[raw_bucket] < self.ORDER[prev]:
            # Only demote once the score clears the lower exit threshold.
            if score < exit_thr.get(prev, 0.0):
                keep, pending = raw_bucket, 0
            else:
                pending = 0
        else:
            pending = 0
        self._state[track_id] = (keep, pending)
        return keep

    def forget(self, live_ids) -> None:
        live = set(live_ids)
        self._state = {k: v for k, v in self._state.items() if k in live}


def ego_risk(records: list[RiskRecord]) -> float:
    return max((r.score for r in records), default=0.0)
