"""Time-to-collision with uncertainty.

TTC = longitudinal gap / closing speed, both in the ego frame. The 1-sigma
interval comes from the delta method on the ratio of two noisy quantities:

    Var(g/c) ~= (sigma_g / c)^2 + (g * sigma_c / c^2)^2

Position/velocity sigmas are read from the BEV Kalman covariance (plus the ego
velocity filter's) — the plan rule that every metric estimate carries the
uncertainty it inherited.

Trust gating: TTC beyond MAX_TRUST_RANGE is flagged low-confidence rather than
hidden. The threshold tracks the MEASURED lift error curve and moved once that
curve improved:
  - 30m, when the IPM-only lift gave 12% range / 2.3 m/s velocity error at
    10-30m but 24% / 4.7 m/s beyond;
  - 50m now, after the M9 fused lift cut 30-50m error to 8.6% range /
    1.5 m/s velocity — better than the 10-30m band was when the gate was set.
Never set this by intuition; re-derive it from out/eval/ipm_validation.json.
"""

from __future__ import annotations

from dataclasses import dataclass

# Longitudinal offset from the nuScenes ego-frame origin (rear-axle area) to
# the front bumper. Approximation for the Renault Zoe data-collection car.
from adp.config import CONFIG

EGO_FRONT_M = CONFIG.risk.ego_front_m
MIN_CLOSING_MS = CONFIG.risk.min_closing_ms  # below this, "not closing"
MAX_TRUST_RANGE = CONFIG.risk.max_trust_range  # from the M9-refreshed curve


@dataclass
class TtcResult:
    ttc_s: float
    sigma_s: float
    gap_m: float          # bumper-to-object longitudinal gap
    closing_ms: float     # positive = approaching
    trustworthy: bool     # inside validated range and CI not degenerate

    def __str__(self) -> str:
        flag = "" if self.trustworthy else " (low conf)"
        return f"TTC {self.ttc_s:.1f}±{self.sigma_s:.1f}s{flag}"


def compute_ttc(
    x_forward_m: float,
    v_rel_forward_ms: float,
    pos_sigma_m: float,
    vel_sigma_ms: float,
) -> TtcResult | None:
    """TTC for an object at longitudinal ego-frame position x, with relative
    forward velocity v (object minus ego; negative = approaching).

    Returns None when the object is not meaningfully closing — an explicit
    "no TTC" rather than a huge number.
    """
    gap = x_forward_m - EGO_FRONT_M
    closing = -v_rel_forward_ms
    if gap <= 0.0 or closing < MIN_CLOSING_MS:
        return None

    ttc = gap / closing
    sigma = ((pos_sigma_m / closing) ** 2
             + (gap * vel_sigma_ms / closing**2) ** 2) ** 0.5
    trustworthy = (x_forward_m <= MAX_TRUST_RANGE) and (sigma < ttc)
    return TtcResult(ttc_s=ttc, sigma_s=sigma, gap_m=gap,
                     closing_ms=closing, trustworthy=trustworthy)
