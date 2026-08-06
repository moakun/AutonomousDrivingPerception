"""Range from apparent box width + a class size prior.

Why width and not height: the M3 failure mode is a box whose BOTTOM is
truncated by a barrier or a vehicle in front, which biases the ground-plane
lift long. A height-based estimate fails the same way (a truncated box looks
shorter, hence farther). Horizontal extent survives that occlusion, so width
is the cue that is *independent* of the exact error IPM makes.

Depth from the pinhole relation w_px = f_x * W / z:

    z = f_x * W_eff / w_px

with W_eff and its spread calibrated from GT by
scripts/calibrate_width_prior.py. Uncertainty combines the prior's spread
(relative, dominant) with detector box noise (also relative, via dz/dw_px =
-z/w_px), in quadrature.

The cue's weakness is yaw: a car seen from behind spans ~1.8m, from the side
~4.5m. Box aspect ratio proxies yaw, and calibration on nuScenes shows it
tightens the estimate dramatically (car 32.6% -> 11.7% relative spread,
aspect correlation 0.89).

BUT that tightening comes at a cost that matters here. Substituting
W = a + b*(w_px/h_px) into z = f_x*W/w_px gives

    z = f_x*a/w_px + f_x*b/h_px

so the conditioned estimator is partly a HEIGHT estimator — measured at 72-82%
of the estimate for vehicles. Height is exactly what bottom-occlusion
truncates: a 20% truncation biases the conditioned estimate ~+20%, while the
unconditioned one is unaffected. Both are therefore exposed:
`conditioned=True` for accuracy in the clean case, `conditioned=False` for the
occlusion-immune estimate that FusedLift needs. Never conflate them.
"""

from __future__ import annotations

import json
import os

import numpy as np

from adp.calib.camera import CameraModel, SE3

# Fallback priors (metres) if no calibration file is present. Replaced by
# measured values from out/eval/width_prior.json when available.
DEFAULT_PRIORS = {
    "car":        {"width_median_m": 1.90, "rel_spread": 0.25},
    "truck":      {"width_median_m": 2.40, "rel_spread": 0.30},
    "bus":        {"width_median_m": 2.70, "rel_spread": 0.30},
    "pedestrian": {"width_median_m": 0.70, "rel_spread": 0.30},
    "cyclist":    {"width_median_m": 0.90, "rel_spread": 0.35},
}

CALIB_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                          "out", "eval", "width_prior.json")
PIXEL_NOISE = 4.0   # detector box-edge noise, px (two edges -> width noise)
MIN_WIDTH_PX = 10.0  # below this the relative width error explodes


class WidthPriorLift:
    def __init__(self, priors: dict | None = None, use_aspect: bool = True):
        self.priors = priors or DEFAULT_PRIORS
        self.use_aspect = use_aspect

    @classmethod
    def from_calibration(cls, path: str = CALIB_PATH, use_aspect: bool = True):
        """Load measured priors; fall back to defaults if not calibrated yet."""
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(priors=data["priors"], use_aspect=use_aspect)
        except (FileNotFoundError, KeyError):
            return cls(use_aspect=use_aspect)

    def effective_width(self, category: str, aspect: float,
                        conditioned: bool | None = None) -> tuple[float, float]:
        """(W_eff metres, relative spread) for this class and box shape.

        conditioned=None follows `use_aspect`; pass False to force the
        occlusion-immune unconditioned prior.
        """
        p = self.priors.get(category)
        if p is None:
            return (float("nan"), float("nan"))
        if conditioned is None:
            conditioned = self.use_aspect
        if conditioned and "linear_a" in p:
            w = p["linear_a"] + p["linear_b"] * aspect
            rel = p.get("rel_spread_conditional", p["rel_spread"])
            # Guard the extrapolation: never let the fit go non-physical.
            w = float(np.clip(w, 0.4 * p["width_median_m"], 2.5 * p["width_median_m"]))
            return w, rel
        return p["width_median_m"], p["rel_spread"]

    def estimate_depth(self, xyxy: np.ndarray, category: str, f_x: float,
                       conditioned: bool | None = None) -> tuple[float, float]:
        """(depth z along the optical axis, 1-sigma) — NaN if unusable."""
        w_px = float(xyxy[2] - xyxy[0])
        h_px = float(xyxy[3] - xyxy[1])
        if w_px < MIN_WIDTH_PX or h_px <= 0:
            return float("nan"), float("nan")
        w_eff, rel = self.effective_width(category, w_px / h_px, conditioned)
        if not np.isfinite(w_eff):
            return float("nan"), float("nan")
        z = f_x * w_eff / w_px
        # Relative errors add in quadrature: prior spread + box-width noise.
        rel_total = float(np.hypot(rel, PIXEL_NOISE / w_px))
        return z, z * rel_total

    def lift_box(self, xyxy: np.ndarray, category: str, camera: CameraModel,
                 T_ego_from_cam: SE3, conditioned: bool | None = None
                 ) -> tuple[np.ndarray | None, float, float]:
        """(ego-frame ground xy, range, sigma_range). Uses the box CENTRE pixel:
        the object's 3-D centre back-projected at the estimated depth, then
        flattened to the ground — deliberately never touching the bottom edge."""
        z, sigma_z = self.estimate_depth(xyxy, category, float(camera.K[0, 0]),
                                         conditioned)
        if not np.isfinite(z):
            return None, float("nan"), float("nan")
        uv = np.array([[(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]])
        p_cam = camera.backproject(uv, np.array([z]))
        p_ego = T_ego_from_cam.apply(p_cam)[0]
        xy = p_ego[:2]
        rng = float(np.linalg.norm(xy))
        # Range scales with depth, so the relative uncertainty carries over.
        return xy, rng, rng * (sigma_z / z)
