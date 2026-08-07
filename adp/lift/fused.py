"""Fusion of the ground-plane (IPM) and width-prior range cues.

The two cues fail differently, which is the whole point:
- IPM is precise near and degrades as r^2, and is biased LONG when the box
  bottom is occluded (barrier, lead vehicle) — measured in M3.
- Width-prior is scale-free (constant relative error at any range) and is
  immune to bottom occlusion, but noisier and yaw-sensitive.

Fusion is inverse-variance on RANGE only; the bearing comes from IPM, whose
horizontal centre is unaffected by vertical truncation.

Occlusion handling is deliberately ASYMMETRIC. If IPM reads much longer than
width predicts, that is the signature of a truncated bottom edge, so IPM's
variance is inflated and width dominates. The reverse disagreement (width much
longer) is more likely a width anomaly — a side-on vehicle, a clipped box — so
IPM is left alone. Physics justifies the asymmetry; symmetric gating would
throw away good IPM measurements.

Two width estimates are used, for the reason documented in width_prior.py:
- unconditioned (immune to bottom truncation, ~33% spread for cars) — the only
  honest basis for DETECTING occlusion, and the estimate trusted once it is
  detected;
- aspect-conditioned (~12% spread, but 72-82% height-derived, so it inherits
  the same truncation bias) — used for accuracy only when the box looks clean.
"""

from __future__ import annotations

import numpy as np

from adp.lift.ipm import GroundPlaneLift
from adp.lift.width_prior import WidthPriorLift

from adp.config import CONFIG

OCCLUSION_K = CONFIG.lift.occlusion_k        # single-frame gate: sigmas
IPM_INFLATE = CONFIG.lift.ipm_inflate        # variance x when suspected
EMA_ALPHA = CONFIG.lift.occlusion_ema_alpha  # ~10-frame memory
EMA_K = CONFIG.lift.occlusion_ema_k          # persistent disagreement to trip

# Beyond this IPM reading, IPM is dropped entirely rather than fused.
# Inverse-variance fusion is optimal for UNBIASED estimators, and IPM is not:
# measured bias is +1.7m at 30-40m and +11.5m at 40-60m, so any non-zero weight
# drags the estimate long no matter how honest the variance. Swept offline over
# the M9 records (n=1480): fusing below 12m and using width alone above gives
# 3.2/6.2/7.6/9.4% median error by range bin — better than IPM alone
# (4.1/11.0/22.8/39.3), width alone (5.1/6.3/7.6/9.4) and full fusion
# (3.2/8.9/11.0/13.8) in every bin.
FUSE_RANGE_M = CONFIG.lift.fuse_range_m


class FusedLift:
    """Drop-in for GroundPlaneLift in update_bev_states.

    Occlusion detection is TEMPORAL where possible. Single-frame disagreement
    is a weak signal — the immune width estimate carries ~33% spread, so one
    frame cannot separate a 7m bias from noise at any sane threshold. But a
    barrier does not come and go: the disagreement persists while the noise
    averages out, so an EMA of the normalised disagreement (kept per track,
    in `state`) detects what a per-frame gate cannot. The strict single-frame
    gate remains as a fallback when no track state is supplied.
    """

    def __init__(self, ipm: GroundPlaneLift, width: WidthPriorLift,
                 occlusion_k: float = OCCLUSION_K):
        self.ipm = ipm
        self.width = width
        self.occlusion_k = occlusion_k
        self._last_std = 1.0
        self.last_source = "ipm"      # which cue(s) produced the value
        self.last_occluded = False    # did the truncation gate fire?
        self.n_occluded = 0           # diagnostic counter

    def lift_box_bottom(self, xyxy: np.ndarray, category: str | None = None,
                        state: dict | None = None):
        xy_i, r_i = self.ipm.lift_box_bottom(xyxy)
        s_i = self.ipm.range_meas_std(r_i) if xy_i is not None else float("nan")

        xy_u = xy_c = None
        r_u = s_u = r_c = s_c = float("nan")
        if category is not None:
            cam, T = self.ipm.camera, self.ipm.T_ego_from_cam
            # Immune estimate (detection + occluded case) and tight estimate.
            xy_u, r_u, s_u = self.width.lift_box(xyxy, category, cam, T,
                                                 conditioned=False)
            xy_c, r_c, s_c = self.width.lift_box(xyxy, category, cam, T,
                                                 conditioned=True)

        ipm_ok = xy_i is not None and np.isfinite(r_i)
        width_ok = xy_u is not None and np.isfinite(r_u) and r_u > 0

        if not ipm_ok and not width_ok:
            self.last_source = "none"
            return None, float("nan")
        if not width_ok:
            self.last_source = "ipm"
            self._last_std = s_i
            return xy_i, r_i
        if not ipm_ok:
            # No ground contact at all: prefer the tight estimate if available.
            use_c = np.isfinite(r_c) and r_c > 0
            self.last_source = "width"
            self._last_std = s_c if use_c else s_u
            return (xy_c, r_c) if use_c else (xy_u, r_u)

        # Detect truncation using ONLY the occlusion-immune estimate.
        d_norm = (r_i - r_u) / max(np.hypot(s_i, s_u), 1e-9)
        if state is not None:
            ema = (1 - EMA_ALPHA) * state.get("occl_ema", 0.0) + EMA_ALPHA * d_norm
            state["occl_ema"] = float(ema)
            occluded = ema > EMA_K
        else:
            occluded = d_norm > self.occlusion_k
        self.last_occluded = bool(occluded)
        if occluded:
            self.n_occluded += 1

        # The conditioned estimate is used even when truncation is suspected.
        # Substituting the immune (unconditioned) estimate there was tried and
        # MEASURED WORSE: it costs 11.7% -> 33% spread on every sample the gate
        # touches, and the gate fires on ~30% of samples, which outweighs the
        # ~20% truncation bias it avoids (10-30m bin: 6.3% -> 10.4%). The immune
        # estimate remains the detector — its job — and the flag still
        # down-weights IPM in the near field.
        use_c = np.isfinite(r_c) and r_c > 0
        r_w, var_w, xy_w = (r_c, s_c**2, xy_c) if use_c else (r_u, s_u**2, xy_u)

        if r_i >= FUSE_RANGE_M:
            # Far field: IPM contributes bias, not information — drop it.
            self.last_source = "width"
            self._last_std = float(np.sqrt(var_w))
            return xy_w, float(r_w)

        var_i = s_i**2 * (IPM_INFLATE if occluded else 1.0)
        w_i, w_w = 1.0 / var_i, 1.0 / var_w
        r_fused = (r_i * w_i + r_w * w_w) / (w_i + w_w)
        self._last_std = float(np.sqrt(1.0 / (w_i + w_w)))
        self.last_source = "fused"
        # Keep IPM's bearing, rescale to the fused range.
        direction = xy_i / max(np.linalg.norm(xy_i), 1e-9)
        return direction * r_fused, float(r_fused)

    def range_meas_std(self, range_m: float, pixel_noise: float = 3.0) -> float:
        """Uncertainty of the measurement returned by the last lift call."""
        return float(np.clip(self._last_std, 0.3, 12.0))
