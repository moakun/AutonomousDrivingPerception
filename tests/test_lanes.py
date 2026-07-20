"""Lane clustering/fitting/corridor tests on synthetic BEV points — no model,
pure geometry."""

import numpy as np
import pytest

from adp.lanes.bev_lanes import (
    EgoCorridor, LaneLine, cluster_and_fit, find_ego_corridor,
)
from adp.lanes.smoothing import CorridorSmoother

RNG = np.random.default_rng(3)


def lane_points(c0, c1=0.0, c2=0.0, x_lo=4.0, x_hi=40.0, n=200, noise=0.08):
    """Points along y = c0 + c1 x + c2 x^2 with lateral noise."""
    x = RNG.uniform(x_lo, x_hi, n)
    y = c0 + c1 * x + c2 * x**2 + RNG.normal(0, noise, n)
    return np.stack([x, y], axis=1)


class TestClusterAndFit:
    def test_two_straight_lines_recovered(self):
        pts = np.vstack([lane_points(1.85), lane_points(-1.85)])
        lines = cluster_and_fit(pts)
        assert len(lines) == 2
        offsets = sorted(float(l.y_at(10.0)) for l in lines)
        assert offsets[0] == pytest.approx(-1.85, abs=0.15)
        assert offsets[1] == pytest.approx(1.85, abs=0.15)

    def test_curved_lane_curvature_sign(self):
        # Road curving left: y grows with x^2.
        lines = cluster_and_fit(lane_points(1.85, c2=0.004))
        assert len(lines) == 1
        assert lines[0].coeffs[2] > 0.001

    def test_adjacent_lane_kept_separate(self):
        pts = np.vstack([
            lane_points(1.85), lane_points(-1.85), lane_points(5.55),  # adjacent-left line
        ])
        lines = cluster_and_fit(pts)
        assert len(lines) == 3

    def test_dashed_line_chained_across_gaps(self):
        # Dashed marking: 3m dashes with 6m gaps — must fit as ONE line.
        dashes = []
        x = 5.0
        while x < 40.0:
            dashes.append(lane_points(1.85, x_lo=x, x_hi=x + 3.0, n=25))
            x += 9.0
        lines = cluster_and_fit(np.vstack(dashes))
        assert len(lines) == 1
        assert float(lines[0].y_at(20.0)) == pytest.approx(1.85, abs=0.2)

    def test_sparse_blob_rejected(self):
        # A short blob (< MIN_SLICES span) must not become a lane line.
        pts = lane_points(1.85, x_lo=10.0, x_hi=14.0, n=40)
        assert cluster_and_fit(pts) == []

    def test_empty_input(self):
        assert cluster_and_fit(np.empty((0, 2))) == []


class TestEgoCorridor:
    def make_lines(self, offsets):
        return [
            LaneLine(coeffs=np.array([o, 0.0, 0.0]), x_range=(4.0, 40.0), n_points=100)
            for o in offsets
        ]

    def test_bracketing_pair_selected(self):
        corr = find_ego_corridor(self.make_lines([-5.55, -1.85, 1.85, 5.55]))
        assert corr.valid
        assert corr.width == pytest.approx(3.7, abs=0.01)
        assert float(corr.left.y_at(8.0)) == pytest.approx(1.85)

    def test_one_side_missing_invalid(self):
        corr = find_ego_corridor(self.make_lines([1.85, 5.55]))  # nothing right of ego
        assert not corr.valid

    def test_absurd_width_invalid(self):
        corr = find_ego_corridor(self.make_lines([-0.4, 0.5]))  # 0.9m "lane"
        assert not corr.valid

    def test_contains(self):
        corr = find_ego_corridor(self.make_lines([-1.85, 1.85]))
        assert corr.contains(10.0, 0.0) is True
        assert corr.contains(10.0, 3.0) is False
        assert EgoCorridor(None, None, None).contains(10.0, 0.0) is None


class TestCorridorSmoother:
    def measured(self, left_y=1.85, right_y=-1.85):
        lines = [
            LaneLine(np.array([left_y, 0, 0]), (4.0, 40.0), 100),
            LaneLine(np.array([right_y, 0, 0]), (4.0, 40.0), 100),
        ]
        return find_ego_corridor(lines)

    def test_smooths_noise(self):
        sm = CorridorSmoother(alpha=0.3)
        sm.update(self.measured(), dt=0.08)
        out = sm.update(self.measured(left_y=2.15), dt=0.08)  # 30cm jitter
        # EMA: moved toward 2.15 but not all the way.
        assert 1.85 < float(out.left.y_at(8.0)) < 2.15

    def test_jump_treated_as_dropout(self):
        sm = CorridorSmoother(jump_m=1.0)
        sm.update(self.measured(), dt=0.08)
        out = sm.update(self.measured(left_y=4.0, right_y=0.3), dt=0.08)
        assert float(out.left.y_at(8.0)) == pytest.approx(1.85)  # kept old state

    def test_holds_then_reports_none(self):
        sm = CorridorSmoother(hold_s=0.5)
        sm.update(self.measured(), dt=0.08)
        none_corr = EgoCorridor(None, None, None)
        out = sm.update(none_corr, dt=0.3)
        assert out.valid  # inside hold window
        out = sm.update(none_corr, dt=0.3)
        assert not out.valid  # hold expired: honest "no lane"
