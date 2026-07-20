"""Temporal smoothing of the ego corridor: lanes don't teleport.

EMA on boundary polynomial coefficients while detections agree; hold the last
valid corridor briefly through dropouts; report no-corridor (rather than a
stale guess) once the hold expires. A large single-frame jump in either
boundary is treated as a dropout, not a measurement — the road doesn't move
half a lane in 80ms, but a misdetection does.
"""

from __future__ import annotations

import numpy as np

from adp.lanes.bev_lanes import EgoCorridor, LaneLine


class CorridorSmoother:
    def __init__(self, alpha: float = 0.3, hold_s: float = 1.0,
                 jump_m: float = 1.0, probe_x: float = 8.0):
        self.alpha = alpha
        self.hold_s = hold_s
        self.jump_m = jump_m
        self.probe_x = probe_x
        self._state: EgoCorridor | None = None
        self._stale_s = 0.0

    def update(self, measured: EgoCorridor, dt: float) -> EgoCorridor:
        if not measured.valid:
            return self._coast(dt)

        if self._state is None:
            self._state = measured
            self._stale_s = 0.0
            return self._state

        # Jump gate on either boundary's lateral position at the probe point.
        for old, new in ((self._state.left, measured.left),
                         (self._state.right, measured.right)):
            if abs(float(new.y_at(self.probe_x)) - float(old.y_at(self.probe_x))) > self.jump_m:
                return self._coast(dt)

        a = self.alpha
        self._state = EgoCorridor(
            left=self._blend(self._state.left, measured.left, a),
            right=self._blend(self._state.right, measured.right, a),
            width=(1 - a) * self._state.width + a * measured.width,
        )
        self._stale_s = 0.0
        return self._state

    def _coast(self, dt: float) -> EgoCorridor:
        self._stale_s += dt
        if self._state is not None and self._stale_s <= self.hold_s:
            return self._state
        self._state = None
        return EgoCorridor(None, None, None)

    @staticmethod
    def _blend(old: LaneLine, new: LaneLine, a: float) -> LaneLine:
        return LaneLine(
            coeffs=(1 - a) * old.coeffs + a * new.coeffs,
            x_range=new.x_range,
            n_points=new.n_points,
        )
