"""Central configuration (improvement #5).

Before this, tuned constants lived in eight modules — tracker thresholds in
bytetrack.py, gates in bev_state.py, the lift switch in fused.py, RSS params
and scoring weights in risk/, lane geometry in bev_lanes.py. That made sweeps
impossible and provenance meaningless: a results file could not say what
produced it.

Every value here is the CURRENT behaviour, so loading the defaults changes
nothing. Modules read these at import time; `configs/default.yaml` overrides
them when present. Values that are physical facts (not tuning knobs) stay in
their modules.

This module must not import anything from `adp` — everything imports it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5        # primary association band
    low_thresh: float = 0.1         # ByteTrack recovery band
    new_track_thresh: float = 0.6   # min score to birth a track
    match_iou: float = 0.2
    match_iou_low: float = 0.4
    n_init: int = 3                 # hits before a track is confirmed
    max_misses: int = 12            # ~1s at 12Hz


@dataclass
class LiftConfig:
    # IPM measured noise model sigma(r) = c * r^p (calibrate_lift_noise.py)
    ipm_sigma_c: float = 0.001451
    ipm_sigma_p: float = 2.51
    ipm_sigma_clip: tuple = (0.3, 40.0)
    fuse_range_m: float = 12.0      # above this, IPM is dropped (M9)
    ipm_inflate: float = 4.0        # variance x when truncation suspected
    occlusion_k: float = 2.5        # single-frame gate
    occlusion_ema_alpha: float = 0.25
    occlusion_ema_k: float = 0.8
    width_pixel_noise: float = 4.0
    min_width_px: float = 10.0
    depth_switch_range_m: float = 30.0   # M6 hybrid (superseded)
    depth_rel_std: float = 0.15
    # BEV Kalman
    bev_vel_std: float = 10.0
    bev_process_std: float = 3.0
    gate_md2: float = 13.8          # chi-square 2 dof @ ~99.9%
    max_rejects: int = 6


@dataclass
class LaneConfig:
    x_min: float = 4.0
    x_max: float = 45.0
    y_half: float = 12.0
    slice_m: float = 2.0
    gap_m: float = 1.0
    link_m: float = 1.2
    max_link_slices: int = 4        # bridges dashed-marking gaps
    max_link_lateral: float = 2.5
    min_span_slices: int = 5
    min_chain: int = 3
    lane_width_range: tuple = (2.4, 5.0)
    smoother_alpha: float = 0.3
    hold_s: float = 1.0
    jump_m: float = 1.0


@dataclass
class RiskConfig:
    ego_front_m: float = 3.5
    min_closing_ms: float = 0.3
    max_trust_range: float = 50.0   # re-derived from the M9 error curve
    # RSS (Shalev-Shwartz et al., arXiv:1708.06374)
    rho_s: float = 0.5
    a_accel: float = 3.5
    b_min: float = 4.0
    b_max: float = 8.0
    # scoring
    zone_base: dict = field(default_factory=lambda: {
        "ego": 0.30, "adjacent": 0.10, "off_path": 0.0, "behind": 0.0})
    ttc_critical_s: float = 5.0
    rss_weight: float = 0.25
    intent_weight: float = 0.20
    ttc_weight: float = 0.45
    plausible_speed: dict = field(default_factory=lambda: {
        "pedestrian": 4.5, "cyclist": 15.0, "car": 45.0,
        "truck": 40.0, "bus": 35.0})
    fallback_half_width: float = 1.85
    # bucket hysteresis (improvement #9)
    bucket_enter: dict = field(default_factory=lambda: {
        "caution": 0.30, "high": 0.60})
    bucket_exit: dict = field(default_factory=lambda: {
        "caution": 0.22, "high": 0.50})
    confirm_frames: int = 2


@dataclass
class AdpConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    lift: LiftConfig = field(default_factory=LiftConfig)
    lanes: LaneConfig = field(default_factory=LaneConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def as_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Short stable hash — goes into every results file (improvement #6)."""
        blob = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "configs",
                            "default.yaml")


def load_config(path: str | None = None) -> AdpConfig:
    """Load config, overriding defaults with a YAML file when available."""
    path = path or DEFAULT_PATH
    cfg = AdpConfig()
    if not os.path.isfile(path):
        return cfg
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return cfg
    for section, values in data.items():
        target = getattr(cfg, section, None)
        if target is None or not isinstance(values, dict):
            continue
        for k, v in values.items():
            if hasattr(target, k):
                setattr(target, k, v)
    return cfg


CONFIG = load_config()
