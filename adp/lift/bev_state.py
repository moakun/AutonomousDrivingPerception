"""Per-track metric (BEV) state: a 2D constant-velocity Kalman filter fed by
IPM lifts, maintained in the GLOBAL frame.

Why global: the ego frame rotates and translates with the car, which would
corrupt a constant-velocity model (a parked car would appear to accelerate
whenever ego turns). In the global frame parked cars have ~zero velocity and
moving ones have their true velocity; anything ego-relative (display, closing
speed) is computed by transforming state back through the current ego pose.

Velocity rule: BEV velocity comes ONLY from this filter (plan cross-cutting
rule) — never from differencing raw lifts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adp.calib.camera import SE3
from adp.lift.ipm import GroundPlaneLift
from adp.track.bytetrack import Track
from adp.track.kalman import ConstantVelocityKalman


# Chi-square gate for 2 dof at ~99.9%: measurements farther than this from the
# prediction (in Mahalanobis distance^2) are rejected as physically implausible
# (typically an occluded box bottom lifting to the wrong ground point).
GATE_MD2 = 13.8
# After this many consecutive rejections, believe the measurements: the object
# genuinely moved (or the track re-associated) — re-seed the filter.
MAX_REJECTS = 6


@dataclass
class BevState:
    kf: ConstantVelocityKalman  # state: (x, y, vx, vy) in GLOBAL frame
    last_range_m: float
    consecutive_rejects: int = 0

    def in_ego(self, T_global_from_ego: SE3) -> tuple[np.ndarray, np.ndarray]:
        """(position_xy, velocity_xy) expressed in the current ego frame."""
        T_ego_from_global = T_global_from_ego.inverse()
        pos_g = np.array([*self.kf.position, 0.0])
        pos_e = T_ego_from_global.apply(pos_g)[0][:2]
        vel_e = (T_ego_from_global.rotation[:2, :2] @ self.kf.velocity)
        return pos_e, vel_e

    def range_std(self) -> float:
        return float(np.linalg.norm(self.kf.position_std()))

    def speed_std(self) -> float:
        return float(np.linalg.norm(self.kf.velocity_std()))


def update_bev_states(
    tracks: list[Track],
    lift: GroundPlaneLift,
    T_global_from_ego: SE3,
    dt: float,
) -> None:
    """Advance/refresh the BEV filter of each confirmed track for this frame.

    Every live filter predicts by dt; tracks whose image box lifted validly get
    a measurement update with range-dependent noise. Objects whose box bottom
    is at/above the horizon (or clipped) simply coast.
    """
    for track in tracks:
        state: BevState | None = track.extra.get("bev")
        if state is not None:
            state.kf.predict(dt)

        xy_ego, range_m = lift.lift_box_bottom(track.xyxy)
        if xy_ego is None:
            continue
        xy_global = T_global_from_ego.apply([*xy_ego, 0.0])[0][:2]
        meas_std = lift.range_meas_std(range_m)

        if state is None:
            track.extra["bev"] = BevState(kf=_fresh_filter(xy_global, meas_std),
                                          last_range_m=range_m)
        elif state.kf.mahalanobis_sq(xy_global, meas_std**2) > GATE_MD2:
            state.consecutive_rejects += 1
            if state.consecutive_rejects > MAX_REJECTS:
                # The "implausible" position persisted: trust it, start over.
                track.extra["bev"] = BevState(kf=_fresh_filter(xy_global, meas_std),
                                              last_range_m=range_m)
        else:
            state.kf.update(xy_global, meas_var=meas_std**2)
            state.last_range_m = range_m
            state.consecutive_rejects = 0


def _fresh_filter(xy_global: np.ndarray, meas_std: float) -> ConstantVelocityKalman:
    return ConstantVelocityKalman(
        z0=xy_global,
        pos_std=meas_std,
        vel_std=10.0,     # generous: urban objects are 0..20 m/s
        meas_std=meas_std,
        process_std=3.0,  # ~vehicle acceleration magnitude
    )
