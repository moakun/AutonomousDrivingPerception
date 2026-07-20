"""RSS minimum safe longitudinal distance.

Implements the longitudinal safe-distance formula from Shalev-Shwartz,
Shammah, Shashua (Mobileye), "On a Formal Model of Safe and Scalable
Self-driving Cars", arXiv:1708.06374 (2017), Definition 1:

    d_min = v_r*rho + 0.5*a_accel*rho^2
            + (v_r + rho*a_accel)^2 / (2*b_min)
            - v_f^2 / (2*b_max)          , clamped at 0

where the rear car (ego) may accelerate at up to a_accel during its response
time rho, then brakes at least at b_min, while the front car brakes at most
at b_max. A gap below d_min means: if the front car brakes hard right now,
collision is not provably avoidable.

Lateral RSS is not implemented (documented scope cut) — lane assignment
covers the lateral dimension qualitatively.
"""

from __future__ import annotations

# Common parameter choices from the RSS literature (all m/s^2 except rho).
RHO_S = 0.5        # response time
A_ACCEL = 3.5      # max ego acceleration during response
B_MIN = 4.0        # min guaranteed ego braking
B_MAX = 8.0        # max assumed front-vehicle braking


def rss_min_gap(v_ego_ms: float, v_front_ms: float) -> float:
    """Minimum safe longitudinal gap (m). v_front is the front object's speed
    along the ego travel direction (0 for stationary obstacles/pedestrians)."""
    v_r = max(v_ego_ms, 0.0)
    v_f = max(v_front_ms, 0.0)
    d = (v_r * RHO_S
         + 0.5 * A_ACCEL * RHO_S**2
         + (v_r + RHO_S * A_ACCEL) ** 2 / (2 * B_MIN)
         - v_f**2 / (2 * B_MAX))
    return max(d, 0.0)
