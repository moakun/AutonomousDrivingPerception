"""Per-object risk records and the scalar score.

Design rule from the plan: never a bare color-coded box. Every record carries
the computed quantities it was scored from, and `reasons` lists the specific
contributions in human-readable form ("TTC 1.8±0.4s", "same lane, closing
12.0 m/s").

M8b STRUCTURE — zone MULTIPLIES a continuous hazard:

    score  = zone_factor * hazard
    hazard = max(noisy_or(ttc, rss, intent), presence_floor)

The previous version ADDED a flat +0.30 for anything in the ego lane on top of
the dynamic terms. M8 measured what that cost: 33% of frames landed on exactly
0.30, and the composite scored worse than its own inverse-TTC component
(AUC 0.655 vs 0.737) — the constant swamped the signal it was meant to
contextualise. Multiplying instead of adding means a stationary car 40m ahead
and a car closing at 12 m/s no longer sit next to each other.

Each component is continuous in [0, 1]:
  ttc_hazard    ramps as TTC falls below `ttc_critical_s`, discounted by the
                Kalman-derived confidence interval — a wide CI counts for
                less, never for nothing
  rss_hazard    1 - gap/rss_min_gap, a continuous proximity ratio replacing
                the old binary flag that could only be 0 or 0.25
  intent_hazard ramps as a pedestrian's predicted crossing time shortens

They combine with a noisy-OR (1 - prod(1 - h)): monotone in every component,
smoothly saturating, and — the point — introducing no invented weights.

The M8 measurement was NOT used to fit any constant. 25 positive events across
10 scenes would overfit; only the structure changed, and the result is
re-measured on the same harness afterwards.

`presence_floor` stops an in-lane object scoring zero when its dynamics are
unknown or untrustworthy: absence of evidence must not read as evidence of
absence.

Scalar ego risk = max over object scores (a single imminent collision must
dominate any number of parked cars).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adp.risk.assign import LaneAssignment, LaneZone
from adp.risk.ttc import TtcResult

from adp.config import CONFIG

ZONE_FACTOR = {z: CONFIG.risk.zone_factor[z.value] for z in LaneZone}
ZONE_BASE = {z: CONFIG.risk.zone_base[z.value] for z in LaneZone}  # legacy
TTC_CRITICAL_S = CONFIG.risk.ttc_critical_s
PRESENCE_FLOOR = CONFIG.risk.presence_floor

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
    gap_m: float | None = None    # bumper-to-object gap, for the RSS ratio
    hazard: float = 0.0           # dynamic hazard before the zone factor
    score: float = 0.0
    bucket: str = "low"
    reasons: list[str] = field(default_factory=list)

    @property
    def velocity_plausible(self) -> bool:
        limit = PLAUSIBLE_SPEED.get(self.category, 45.0)
        return self.object_speed_ms <= limit


def ttc_hazard(ttc: TtcResult | None) -> float:
    """Ramps as TTC falls below the critical horizon, discounted by its CI.

    The confidence weight uses the interval the Kalman filters already produce
    — sigma comparable to the estimate means the number is nearly meaningless,
    but it is floored rather than zeroed so a noisy imminent TTC still speaks.
    """
    if ttc is None or not ttc.trustworthy or ttc.ttc_s >= TTC_CRITICAL_S:
        return 0.0
    conf = np.clip(1.0 - ttc.sigma_s / max(ttc.ttc_s, 1e-6),
                   CONFIG.risk.ttc_conf_floor, 1.0)
    return float((1.0 - ttc.ttc_s / TTC_CRITICAL_S) * conf)


def rss_hazard(gap_m: float | None, rss_min_gap_m: float | None) -> float:
    """Continuous proximity ratio: 0 at the RSS safe distance, 1 at contact."""
    if gap_m is None or rss_min_gap_m is None or rss_min_gap_m <= 0:
        return 0.0
    return float(np.clip(1.0 - gap_m / rss_min_gap_m, 0.0, 1.0))


def intent_hazard(cross_s: float | None) -> float:
    """Ramps as a predicted crossing gets nearer in time."""
    if cross_s is None:
        return 0.0
    horizon = CONFIG.risk.intent_horizon_s
    return float(np.clip(1.0 - cross_s / horizon, 0.0, 1.0))


def noisy_or(*hazards: float) -> float:
    """1 - prod(1 - h): monotone, saturating, and weight-free."""
    out = 1.0
    for h in hazards:
        out *= (1.0 - float(np.clip(h, 0.0, 1.0)))
    return float(1.0 - out)


def score_object(record: RiskRecord) -> RiskRecord:
    """Fill hazard/score/bucket/reasons from the record's computed quantities."""
    reasons = [f"{record.zone.value} lane"
               + (" (fallback corridor)" if record.corridor_source == "fallback" else "")]
    zone_factor = ZONE_FACTOR[record.zone]

    if not record.velocity_plausible:
        # Dynamics are corrupted; presence is still real, so fall back to the
        # floor rather than either trusting or ignoring the object.
        reasons.append(
            f"velocity implausible for {record.category} "
            f"({record.object_speed_ms:.1f} m/s) — TTC/RSS suppressed")
        record.hazard = PRESENCE_FLOOR
        record.score = float(np.clip(zone_factor * PRESENCE_FLOOR, 0.0, 1.0))
        record.bucket = bucket_for(record.score)
        record.reasons = reasons
        return record

    h_ttc = h_rss = h_intent = 0.0
    if record.zone in (LaneZone.EGO, LaneZone.ADJACENT):
        if record.ttc is not None:
            reasons.append(str(record.ttc)
                           + f", closing {record.ttc.closing_ms:.1f} m/s")
            h_ttc = ttc_hazard(record.ttc)
        gap = record.gap_m if record.gap_m is not None else (
            record.ttc.gap_m if record.ttc else None)
        h_rss = rss_hazard(gap, record.rss_min_gap_m)
        if record.rss_violated and gap is not None:
            reasons.append(f"RSS violated: gap {gap:.1f}m < "
                           f"{record.rss_min_gap_m:.1f}m required")

    if record.intent_cross_s is not None:
        h_intent = intent_hazard(record.intent_cross_s)
        reasons.append(f"crossing into path in ~{record.intent_cross_s:.1f}s")

    record.hazard = max(noisy_or(h_ttc, h_rss, h_intent),
                        PRESENCE_FLOOR if zone_factor > 0 else 0.0)
    record.score = float(np.clip(zone_factor * record.hazard, 0.0, 1.0))
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
