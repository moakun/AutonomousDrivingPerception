"""Hybrid metric lift: IPM inside its validated range, monocular depth beyond.

M6 finding (out/eval/depth_vs_ipm.json): IPM median error 4.1%/11.0% at
0-10/10-30m vs depth's 58%/17.7% (depth has a strong nonlinear near-field
bias — gt/depth ratio 0.63->1.06 across bins, so not calibratable by a scale
factor). Beyond 30m the situation inverts: depth 12.7%/15.4% vs IPM
22.8%/39.3%. Switching on the IPM range estimate at 30m gives the best of
both: 4.1% / 10.6% / 14.6% / 15.4%.

Depth measurement noise is modeled as proportional (~15% of range, from the
measured error curve) rather than IPM's r^2 pixel-noise law.
"""

from __future__ import annotations

import numpy as np

from adp.lift.depth import DepthLift
from adp.lift.ipm import GroundPlaneLift

SWITCH_RANGE_M = 30.0
DEPTH_REL_STD = 0.15


class HybridLift:
    """Drop-in for GroundPlaneLift in update_bev_states. Requires
    depth.compute(img) to have been called for the current frame."""

    def __init__(self, ipm: GroundPlaneLift, depth: DepthLift):
        self.ipm = ipm
        self.depth = depth
        self._last_source = "ipm"

    def lift_box_bottom(self, xyxy: np.ndarray):
        xy, rng = self.ipm.lift_box_bottom(xyxy)
        if xy is not None and rng < SWITCH_RANGE_M:
            self._last_source = "ipm"
            return xy, rng
        xy_d, rng_d = self.depth.range_from_box(
            xyxy, self.ipm.camera, self.ipm.T_ego_from_cam)
        if xy_d is not None:
            self._last_source = "depth"
            return xy_d, rng_d
        self._last_source = "ipm"
        return xy, rng  # depth unavailable: fall back to IPM (may be None)

    def range_meas_std(self, range_m: float, pixel_noise: float = 3.0) -> float:
        if self._last_source == "depth":
            return float(np.clip(DEPTH_REL_STD * range_m, 1.0, 12.0))
        return self.ipm.range_meas_std(range_m, pixel_noise)
