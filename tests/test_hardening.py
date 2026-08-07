"""M11 tests: config plumbing (#5), provenance (#6), the lane-stride hold-timer
bug (#8), and risk-bucket hysteresis (#9)."""

import numpy as np
import pytest

from adp.config import AdpConfig, CONFIG, load_config
from adp.eval.provenance import provenance, stamp
from adp.lanes.bev_lanes import EgoCorridor, LaneLine, find_ego_corridor
from adp.lanes.smoothing import CorridorSmoother
from adp.risk.scoring import BucketHysteresis, bucket_for


class TestConfig:
    def test_yaml_matches_code_defaults(self):
        """configs/default.yaml must reproduce committed behaviour exactly —
        otherwise loading 'defaults' silently changes results."""
        coded, loaded = AdpConfig(), load_config()
        assert coded.lift.fuse_range_m == loaded.lift.fuse_range_m
        assert coded.risk.max_trust_range == loaded.risk.max_trust_range
        assert coded.tracker.high_thresh == loaded.tracker.high_thresh
        assert coded.lift.ipm_sigma_p == loaded.lift.ipm_sigma_p

    def test_modules_read_from_config(self):
        from adp.lift.fused import FUSE_RANGE_M
        from adp.lift.ipm import MEASURED_P
        from adp.risk.ttc import MAX_TRUST_RANGE
        assert FUSE_RANGE_M == CONFIG.lift.fuse_range_m
        assert MEASURED_P == CONFIG.lift.ipm_sigma_p
        assert MAX_TRUST_RANGE == CONFIG.risk.max_trust_range

    def test_fingerprint_is_stable_and_sensitive(self):
        a, b = AdpConfig(), AdpConfig()
        assert a.fingerprint() == b.fingerprint()
        b.lift.fuse_range_m = 15.0
        assert a.fingerprint() != b.fingerprint()

    def test_missing_file_falls_back(self):
        cfg = load_config("does/not/exist.yaml")
        assert cfg.lift.fuse_range_m == AdpConfig().lift.fuse_range_m


class TestProvenance:
    def test_fields_present(self):
        p = provenance()
        for key in ("git_sha", "git_dirty", "generated_utc", "host",
                    "python", "config_fingerprint"):
            assert key in p
        assert p["config_fingerprint"] == CONFIG.fingerprint()

    def test_stamp_is_chainable_and_in_place(self):
        report = {"result": 1}
        out = stamp(report)
        assert out is report and "_provenance" in report


class TestLaneHoldTimer:
    """#8: the hold window must be measured in wall time, not update count."""

    def corridor(self):
        return find_ego_corridor([
            LaneLine(np.array([1.85, 0, 0]), (4.0, 40.0), 100),
            LaneLine(np.array([-1.85, 0, 0]), (4.0, 40.0), 100)])

    def test_hold_expires_on_wall_time(self):
        sm = CorridorSmoother(hold_s=1.0)
        sm.update(self.corridor(), dt=1 / 12)
        none = EgoCorridor(None, None, None)
        # 12 skip frames at 12Hz = 1.0s of real time: still inside the hold.
        for _ in range(11):
            assert sm.update(none, dt=1 / 12).valid
        # Past 1s it must report honestly rather than hold a stale corridor.
        sm.update(none, dt=1 / 12)
        assert not sm.update(none, dt=1 / 12).valid

    def test_zero_dt_would_never_expire(self):
        """Documents the old bug: charging 0s per skip frame froze the timer."""
        sm = CorridorSmoother(hold_s=1.0)
        sm.update(self.corridor(), dt=1 / 12)
        none = EgoCorridor(None, None, None)
        for _ in range(100):
            assert sm.update(none, dt=0.0).valid  # stale forever


class TestBucketHysteresis:
    def test_stable_score_keeps_bucket(self):
        h = BucketHysteresis()
        for _ in range(5):
            assert h.apply(1, "caution", 0.40) == "caution"

    def test_promotion_needs_confirmation(self):
        h = BucketHysteresis()
        h.apply(1, "low", 0.10)
        # First frame of caution is provisional; second confirms.
        assert h.apply(1, "caution", 0.31) == "low"
        assert h.apply(1, "caution", 0.31) == "caution"

    def test_escalation_to_high_is_immediate(self):
        """Safety asymmetry: never delay a collision warning to look tidy."""
        h = BucketHysteresis()
        h.apply(1, "low", 0.10)
        assert h.apply(1, "high", 0.85) == "high"

    def test_demotion_requires_clearing_exit_threshold(self):
        h = BucketHysteresis()
        for _ in range(3):
            h.apply(1, "caution", 0.45)
        # Dip to 0.28: below the 0.30 enter threshold but above the 0.22 exit.
        assert h.apply(1, "low", 0.28) == "caution"
        assert h.apply(1, "low", 0.15) == "low"

    def test_flicker_is_suppressed(self):
        h = BucketHysteresis()
        raw = [0.31, 0.29, 0.31, 0.28, 0.32, 0.29]  # hovering at threshold
        out = [h.apply(1, bucket_for(s), s) for s in raw]
        assert len(set(out)) == 1  # no oscillation at all

    def test_tracks_are_independent_and_forgotten(self):
        h = BucketHysteresis()
        h.apply(1, "high", 0.9)
        h.apply(2, "low", 0.1)
        assert h.apply(1, "high", 0.9) == "high"
        h.forget([2])
        assert 1 not in h._state and 2 in h._state
