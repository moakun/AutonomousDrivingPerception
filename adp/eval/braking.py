"""Hard-braking proxy labels from ego pose — the M8 validation signal.

The risk layer has no ground truth: nuScenes labels objects, not danger. But
the data does record what a human driver did, and hard braking is a
behavioural proxy for "the driver judged the situation risky". Differentiating
the ego trajectory gives longitudinal acceleration; a keyframe is positive if
hard braking occurs within a horizon.

Label generation is offline and acausal (it uses the whole scene, including
the future) — correct for labels, and strictly separate from the causal
perception path, which never sees this.

Honest caveats, carried into the report: drivers also brake for traffic
lights, stop signs and turns; this measures correlation with human behaviour,
not ground-truth danger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class EgoKinematics:
    t_s: np.ndarray        # seconds since scene start
    speed_ms: np.ndarray   # smoothed speed
    accel_ms2: np.ndarray  # d(speed)/dt; negative = decelerating


def ego_kinematics(
    timestamps_us: np.ndarray,
    positions_xy: np.ndarray,
    smooth_window_s: float = 0.9,
    polyorder: int = 2,
) -> EgoKinematics:
    """Ego speed and longitudinal acceleration from a pose trajectory.

    Raw finite differences on position are far too noisy to threshold, so the
    speed series is Savitzky-Golay smoothed before differentiation.
    """
    t = (np.asarray(timestamps_us, dtype=float) - float(timestamps_us[0])) / 1e6
    pos = np.asarray(positions_xy, dtype=float)
    if len(t) < 3:
        z = np.zeros(len(t))
        return EgoKinematics(t_s=t, speed_ms=z, accel_ms2=z)

    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = np.diff(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1 / 12
    raw_speed = np.r_[step[0] / dt[0], step / dt]

    win = int(round(smooth_window_s / float(np.median(dt))))
    win = max(win, polyorder + 2)
    if win % 2 == 0:
        win += 1
    if win > len(raw_speed):
        win = len(raw_speed) if len(raw_speed) % 2 == 1 else len(raw_speed) - 1
    speed = (savgol_filter(raw_speed, win, polyorder)
             if win > polyorder else raw_speed.copy())
    speed = np.clip(speed, 0.0, None)
    accel = np.gradient(speed, t)  # handles the mild timestamp jitter
    return EgoKinematics(t_s=t, speed_ms=speed, accel_ms2=accel)


def label_hard_braking(
    kin: EgoKinematics,
    horizon_s: float = 3.0,
    accel_thresh: float = -2.0,
    min_speed_ms: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample (labels, valid).

    label[i] = 1 if acceleration drops below `accel_thresh` anywhere in
    [t_i, t_i + horizon] — "already braking hard" counts, since the perception
    system should be flagging risk then too.

    valid[i] excludes samples where the ego is below `min_speed_ms` (braking
    is undefined when stopped, and those frames would trivially inflate the
    negative class) and samples too near the scene end to observe a full
    horizon.
    """
    t, a, v = kin.t_s, kin.accel_ms2, kin.speed_ms
    n = len(t)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n):
        j_end = np.searchsorted(t, t[i] + horizon_s, side="right")
        if np.any(a[i:j_end] <= accel_thresh):
            labels[i] = 1
    valid = (v >= min_speed_ms) & (t + horizon_s <= t[-1])
    return labels, valid
