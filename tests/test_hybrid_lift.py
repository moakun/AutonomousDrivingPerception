"""Hybrid lift selection logic with stubbed IPM/depth sources."""

import numpy as np

from adp.lift.hybrid import SWITCH_RANGE_M, HybridLift


class StubIpm:
    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=float) if xy is not None else None
        self.camera = None
        self.T_ego_from_cam = None

    def lift_box_bottom(self, xyxy):
        if self.xy is None:
            return None, float("nan")
        return self.xy, float(np.linalg.norm(self.xy))

    def range_meas_std(self, r, pixel_noise=3.0):
        return 0.5


class StubDepth:
    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=float) if xy is not None else None

    def range_from_box(self, xyxy, camera, T):
        if self.xy is None:
            return None, float("nan")
        return self.xy, float(np.linalg.norm(self.xy))


BOX = np.array([700.0, 400.0, 800.0, 500.0])


def test_near_range_uses_ipm():
    h = HybridLift(StubIpm([15.0, 0.0]), StubDepth([18.0, 0.0]))
    xy, rng = h.lift_box_bottom(BOX)
    assert rng == 15.0 and h._last_source == "ipm"
    assert h.range_meas_std(15.0) == 0.5  # IPM's noise model


def test_far_range_uses_depth():
    h = HybridLift(StubIpm([40.0, 0.0]), StubDepth([36.0, 0.0]))
    xy, rng = h.lift_box_bottom(BOX)
    assert rng == 36.0 and h._last_source == "depth"
    assert h.range_meas_std(36.0) == 36.0 * 0.15  # proportional depth noise


def test_switch_exactly_at_boundary():
    just_under = SWITCH_RANGE_M - 0.01
    h = HybridLift(StubIpm([just_under, 0.0]), StubDepth([50.0, 0.0]))
    assert h.lift_box_bottom(BOX)[1] == just_under


def test_depth_unavailable_falls_back_to_ipm():
    h = HybridLift(StubIpm([40.0, 0.0]), StubDepth(None))
    xy, rng = h.lift_box_bottom(BOX)
    assert rng == 40.0 and h._last_source == "ipm"


def test_ipm_invalid_uses_depth():
    # Horizon-clipped box: IPM fails entirely, depth still answers.
    h = HybridLift(StubIpm(None), StubDepth([25.0, 0.0]))
    xy, rng = h.lift_box_bottom(BOX)
    assert rng == 25.0 and h._last_source == "depth"
