"""Risk layer tests: TTC math + CI, RSS formula, lane assignment, pedestrian
intent, and the plan's sanity scenes (obviously-correct risk ranking)."""

import numpy as np
import pytest

from adp.risk.assign import LaneZone, assign_lane, crossing_intent, fallback_corridor
from adp.risk.rss import rss_min_gap, RHO_S, A_ACCEL, B_MIN, B_MAX
from adp.risk.scoring import RiskRecord, ego_risk, score_object
from adp.risk.ttc import EGO_FRONT_M, compute_ttc


class TestTtc:
    def test_basic_ratio(self):
        # Object 23.5m ahead of origin -> 20m bumper gap, closing at 10 m/s.
        r = compute_ttc(20.0 + EGO_FRONT_M, -10.0, pos_sigma_m=1.0, vel_sigma_ms=0.5)
        assert r.ttc_s == pytest.approx(2.0)
        assert r.closing_ms == 10.0
        assert r.trustworthy

    def test_not_closing_returns_none(self):
        assert compute_ttc(20.0, +2.0, 1.0, 0.5) is None       # receding
        assert compute_ttc(20.0, -0.1, 1.0, 0.5) is None       # below closing floor

    def test_delta_method_ci(self):
        # sigma^2 = (sp/c)^2 + (g*sv/c^2)^2 with g=20, c=10, sp=1, sv=0.5
        r = compute_ttc(20.0 + EGO_FRONT_M, -10.0, 1.0, 0.5)
        expect = np.sqrt((1.0 / 10) ** 2 + (20.0 * 0.5 / 100) ** 2)
        assert r.sigma_s == pytest.approx(expect)

    def test_far_range_flagged_untrustworthy(self):
        r = compute_ttc(45.0, -10.0, 2.0, 1.0)
        assert r is not None and not r.trustworthy

    def test_huge_ci_flagged(self):
        r = compute_ttc(25.0, -1.0, 5.0, 3.0)  # sigma dwarfs the estimate
        assert r is not None and not r.trustworthy


class TestRss:
    def test_formula_stationary_lead(self):
        v = 10.0
        expect = (v * RHO_S + 0.5 * A_ACCEL * RHO_S**2
                  + (v + RHO_S * A_ACCEL) ** 2 / (2 * B_MIN))
        assert rss_min_gap(v, 0.0) == pytest.approx(expect)  # ~22.6m at 10 m/s

    def test_matched_speeds_exact(self):
        # v_r = v_f = 10: full formula including the front's braking credit.
        # Note the gap stays large even at matched speed: RSS assumes ego
        # brakes at only b_min=4 while the lead may brake at b_max=8.
        v = 10.0
        expect = (v * RHO_S + 0.5 * A_ACCEL * RHO_S**2
                  + (v + RHO_S * A_ACCEL) ** 2 / (2 * B_MIN) - v**2 / (2 * B_MAX))
        assert rss_min_gap(v, v) == pytest.approx(expect)  # ~16.4m
        assert rss_min_gap(v, v) < rss_min_gap(v, 0.0)

    def test_never_negative(self):
        assert rss_min_gap(0.0, 20.0) == 0.0

    def test_monotonic_in_ego_speed(self):
        gaps = [rss_min_gap(v, 0.0) for v in (5.0, 10.0, 20.0)]
        assert gaps == sorted(gaps)


class TestLaneAssignment:
    def test_zones_with_fallback(self):
        assert assign_lane(np.array([20.0, 0.0]), None).zone == LaneZone.EGO
        assert assign_lane(np.array([20.0, 3.0]), None).zone == LaneZone.ADJACENT
        assert assign_lane(np.array([20.0, 9.0]), None).zone == LaneZone.OFF_PATH
        assert assign_lane(np.array([-5.0, 0.0]), None).zone == LaneZone.BEHIND

    def test_source_recorded(self):
        a = assign_lane(np.array([20.0, 0.0]), None)
        assert a.corridor_source == "fallback"
        b = assign_lane(np.array([20.0, 0.0]), fallback_corridor())
        assert b.corridor_source == "detected"  # any valid corridor counts

    def test_curved_corridor_followed(self):
        # Corridor curving left; a point that is "straight ahead" at x=30 falls
        # outside the curved ego lane.
        from adp.lanes.bev_lanes import EgoCorridor, LaneLine
        curve = 0.006
        corr = EgoCorridor(
            left=LaneLine(np.array([1.85, 0.0, curve]), (0, 60), 10),
            right=LaneLine(np.array([-1.85, 0.0, curve]), (0, 60), 10),
            width=3.7,
        )
        assert assign_lane(np.array([30.0, 5.4]), corr).zone == LaneZone.EGO
        assert assign_lane(np.array([30.0, 0.0]), corr).zone == LaneZone.ADJACENT


class TestCrossingIntent:
    def test_ped_walking_into_path(self):
        pos = np.array([15.0, 4.0])
        a = assign_lane(pos, None)
        t = crossing_intent(pos, np.array([0.0, -1.5]), a, None)
        # 4.0 - 1.85 = 2.15m gap at 1.5 m/s -> ~1.43s
        assert t == pytest.approx(2.15 / 1.5, abs=0.01)

    def test_ped_walking_away_none(self):
        pos = np.array([15.0, 4.0])
        a = assign_lane(pos, None)
        assert crossing_intent(pos, np.array([0.0, +1.5]), a, None) is None

    def test_ped_already_in_lane_none(self):
        pos = np.array([15.0, 0.0])
        a = assign_lane(pos, None)
        assert crossing_intent(pos, np.array([0.0, -1.5]), a, None) is None

    def test_too_slow_none(self):
        pos = np.array([15.0, 4.0])
        a = assign_lane(pos, None)
        assert crossing_intent(pos, np.array([0.0, -0.1]), a, None) is None


def make_record(**kw):
    from adp.risk.ttc import TtcResult
    defaults = dict(track_id=1, category="car", range_m=20.0, zone=LaneZone.EGO,
                    corridor_source="detected", closing_ms=None, ttc=None,
                    rss_min_gap_m=None, rss_violated=False, intent_cross_s=None)
    defaults.update(kw)
    return score_object(RiskRecord(**defaults))


class TestScoring:
    def test_sanity_ranking(self):
        """Plan sanity scene: braking lead car >> adjacent parked car >> empty."""
        from adp.risk.ttc import TtcResult
        braking_lead = make_record(
            ttc=TtcResult(1.8, 0.4, 16.5, 12.0, True), rss_violated=True,
            rss_min_gap_m=25.0)
        parked_adjacent = make_record(zone=LaneZone.ADJACENT)
        offpath = make_record(zone=LaneZone.OFF_PATH)
        assert braking_lead.score > parked_adjacent.score > offpath.score
        assert braking_lead.bucket == "high"
        assert offpath.bucket == "low"
        assert ego_risk([braking_lead, parked_adjacent, offpath]) == braking_lead.score

    def test_untrustworthy_ttc_does_not_score(self):
        from adp.risk.ttc import TtcResult
        far = make_record(ttc=TtcResult(1.5, 3.0, 40.0, 5.0, False))
        near = make_record(ttc=TtcResult(1.5, 0.3, 15.0, 5.0, True))
        assert near.score > far.score
        # ...but the low-confidence TTC still appears in the reasons.
        assert any("low conf" in r for r in far.reasons)

    def test_crossing_ped_scores(self):
        ped = make_record(category="pedestrian", zone=LaneZone.ADJACENT,
                          intent_cross_s=1.2)
        assert any("crossing" in r for r in ped.reasons)
        assert ped.score > make_record(zone=LaneZone.ADJACENT).score

    def test_reasons_always_present(self):
        assert make_record().reasons  # even a boring record explains itself

    def test_empty_scene_zero_risk(self):
        assert ego_risk([]) == 0.0

    def test_implausible_pedestrian_velocity_suppressed(self):
        from adp.risk.ttc import TtcResult
        # A "35.8 m/s pedestrian" is filter corruption, not a sprinting human:
        # zone base survives, TTC/RSS contributions must not.
        ghost = make_record(
            category="pedestrian", object_speed_ms=35.8,
            ttc=TtcResult(0.1, 0.02, 3.0, 35.8, True),
            rss_violated=True, rss_min_gap_m=4.1)
        assert ghost.score == pytest.approx(0.30)  # zone base only
        assert any("implausible" in r for r in ghost.reasons)
        # The same numbers on a car are plausible and score fully.
        car = make_record(
            category="car", object_speed_ms=35.8,
            ttc=TtcResult(0.1, 0.02, 3.0, 35.8, True),
            rss_violated=True, rss_min_gap_m=4.1)
        assert car.score > 0.9
