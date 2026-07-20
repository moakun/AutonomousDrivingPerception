"""Time-to-collision with uncertainty.

TTC = longitudinal gap / closing speed, both in the ego frame. The 1-sigma
interval comes from the delta method on the ratio of two noisy quantities:

    Var(g/c) ~= (sigma_g / c)^2 + (g * sigma_c / c^2)^2

Position/velocity sigmas are read from the BEV Kalman covariance (plus the ego
velocity filter's) — the plan rule that every metric estimate carries the
uncertainty it inherited.

Trust gating: M3 measured range error ~12% at 10-30m but 24%+ beyond, so TTC
beyond MAX_TRUST_RANGE is flagged low-confidence rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

# Longitudinal offset from the nuScenes ego-frame origin (rear-axle area) to
# the front bumper. Approximation for the Renault Zoe data-collection car.
EGO_FRONT_M = 3.5
MIN_CLOSING_MS = 0.3   # below this, "not closing" (avoids TTC -> infinity noise)
MAX_TRUST_RANGE = 30.0  # from the M3 validation error curve


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
