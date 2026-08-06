"""M8 support tests: ranking metrics and hard-braking label generation."""

import numpy as np
import pytest

from adp.eval.braking import EgoKinematics, ego_kinematics, label_hard_braking
from adp.eval.ranking import average_precision, bootstrap_ci, roc_auc


class TestRankingMetrics:
    def test_perfect_separation(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        assert roc_auc(scores, labels) == 1.0
        assert average_precision(scores, labels) == 1.0

    def test_inverted_predictor(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([1, 1, 0, 0])
        assert roc_auc(scores, labels) == 0.0

    def test_all_ties_is_chance(self):
        scores = np.ones(8)
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        assert roc_auc(scores, labels) == 0.5
        # AP with all ties = base rate (one PR point at full recall).
        assert average_precision(scores, labels) == pytest.approx(0.5)

    def test_known_auc_value(self):
        # 2 positives, 2 negatives; one negative outranks one positive -> 0.75
        scores = np.array([0.9, 0.6, 0.7, 0.1])
        labels = np.array([1, 1, 0, 0])
        assert roc_auc(scores, labels) == pytest.approx(0.75)

    def test_average_precision_known_value(self):
        # Ranked: P(1), N, P(1) -> precisions 1.0 and 2/3 at the two positives
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 0, 1])
        assert average_precision(scores, labels) == pytest.approx((1.0 + 2 / 3) / 2)

    def test_monotone_transform_invariance(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(1, 50, 200)
        labels = (raw < 20).astype(int)
        # -x and 1/x are both monotone-decreasing in x: same ranking, same AUC.
        assert roc_auc(-raw, labels) == pytest.approx(roc_auc(1 / raw, labels))

    def test_degenerate_single_class(self):
        assert np.isnan(roc_auc(np.array([1.0, 2.0]), np.array([1, 1])))
        assert np.isnan(average_precision(np.array([1.0, 2.0]), np.array([0, 0])))

    def test_bootstrap_ci_brackets_estimate(self):
        rng = np.random.default_rng(1)
        scores = np.r_[rng.normal(1.0, 1.0, 150), rng.normal(0.0, 1.0, 150)]
        labels = np.r_[np.ones(150), np.zeros(150)].astype(int)
        point = roc_auc(scores, labels)
        lo, hi = bootstrap_ci(roc_auc, scores, labels, n_boot=200, seed=2)
        assert lo < point < hi
        assert 0.0 <= lo and hi <= 1.0


class TestEgoKinematics:
    def test_constant_speed(self):
        t_us = (np.arange(60) / 12 * 1e6).astype(np.int64)
        pos = np.stack([10.0 * np.arange(60) / 12, np.zeros(60)], axis=1)
        kin = ego_kinematics(t_us, pos)
        assert np.allclose(kin.speed_ms[5:-5], 10.0, atol=0.1)
        assert np.max(np.abs(kin.accel_ms2[5:-5])) < 0.1

    def test_constant_deceleration_recovered(self):
        # 12 m/s decelerating at -3 m/s^2 for 3s.
        dt = 1 / 12
        n = 40
        t = np.arange(n) * dt
        speed = 12.0 - 3.0 * t
        x = np.cumsum(speed * dt)
        kin = ego_kinematics((t * 1e6).astype(np.int64),
                            np.stack([x, np.zeros(n)], axis=1))
        assert np.median(kin.accel_ms2[8:-8]) == pytest.approx(-3.0, abs=0.4)

    def test_short_trajectory_no_crash(self):
        kin = ego_kinematics(np.array([0, 83333]), np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert len(kin.speed_ms) == 2


class TestBrakingLabels:
    def make_kin(self, accel, speed=10.0, hz=12):
        n = len(accel)
        t = np.arange(n) / hz
        return EgoKinematics(t_s=t, speed_ms=np.full(n, speed),
                             accel_ms2=np.asarray(accel, dtype=float))

    def test_positive_window_before_event(self):
        # Braking spike at index 48 (t=4.0s); horizon 3s at 12Hz.
        accel = np.zeros(80)
        accel[48] = -4.0
        kin = self.make_kin(accel)
        labels, valid = label_hard_braking(kin, horizon_s=3.0, accel_thresh=-2.0)
        assert labels[48] == 1   # already braking counts
        assert labels[24] == 1   # t=2.0s, event 2.0s ahead: inside horizon
        assert labels[12] == 1   # t=1.0s, event exactly at the horizon edge (inclusive)
        assert labels[0] == 0    # t=0.0s, horizon ends at 3.0s: event not yet visible
        assert labels[50] == 0   # after the event

    def test_below_threshold_not_labelled(self):
        accel = np.full(60, -1.0)  # gentle deceleration only
        labels, _ = label_hard_braking(self.make_kin(accel), accel_thresh=-2.0)
        assert labels.sum() == 0

    def test_stopped_frames_invalid(self):
        accel = np.zeros(60)
        kin = self.make_kin(accel, speed=0.5)
        _, valid = label_hard_braking(kin, min_speed_ms=2.0)
        assert not valid.any()

    def test_scene_end_invalid(self):
        accel = np.zeros(60)
        _, valid = label_hard_braking(self.make_kin(accel), horizon_s=3.0)
        # Last 3s (36 samples at 12Hz) lack a full horizon.
        assert valid[:23].all()
        assert not valid[-36:].any()

    def test_threshold_monotonicity(self):
        accel = np.zeros(60)
        accel[30] = -2.5
        kin = self.make_kin(accel)
        strict, _ = label_hard_braking(kin, accel_thresh=-3.0)
        loose, _ = label_hard_braking(kin, accel_thresh=-2.0)
        assert strict.sum() == 0 and loose.sum() > 0
