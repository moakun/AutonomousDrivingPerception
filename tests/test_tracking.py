"""Tests: Kalman velocity convergence, ByteTrack ID persistence through
occlusion and low-confidence recovery, MOT metric sanity."""

import numpy as np

from adp.detect.detector import Detection2D
from adp.eval.tracking import MotAccumulator
from adp.track.bytetrack import ByteTracker, TrackState
from adp.track.kalman import ConstantVelocityKalman


def det(x0, y0, x1, y1, score=0.9, cls="car"):
    return Detection2D(xyxy=np.array([x0, y0, x1, y1], dtype=float), score=score, category=cls)


def box_at(x, w=60.0, h=40.0, y=300.0):
    return (x, y, x + w, y + h)


class TestKalman:
    def test_velocity_converges(self):
        # Object moving at exactly (10, -4) units/s, observed at 12Hz.
        kf = ConstantVelocityKalman(
            z0=[0.0, 0.0], pos_std=1.0, vel_std=10.0, meas_std=0.5, process_std=1.0
        )
        dt = 1 / 12
        true_v = np.array([10.0, -4.0])
        pos = np.zeros(2)
        for _ in range(48):  # 4 seconds
            pos = pos + true_v * dt
            kf.predict(dt)
            kf.update(pos)
        assert np.max(np.abs(kf.velocity - true_v)) < 0.5
        assert np.max(np.abs(kf.position - pos)) < 0.5

    def test_uncertainty_grows_without_updates(self):
        kf = ConstantVelocityKalman(
            z0=[0.0], pos_std=1.0, vel_std=1.0, meas_std=0.5, process_std=1.0
        )
        s0 = kf.position_std()[0]
        for _ in range(10):
            kf.predict(0.1)
        assert kf.position_std()[0] > s0  # coasting must widen the estimate

    def test_variable_dt(self):
        kf = ConstantVelocityKalman(
            z0=[0.0], pos_std=1.0, vel_std=5.0, meas_std=0.1, process_std=0.5
        )
        t, x = 0.0, 0.0
        for dt in [0.05, 0.1, 0.083, 0.12, 0.083] * 10:
            t += dt
            x = 5.0 * t
            kf.predict(dt)
            kf.update([x])
        assert abs(kf.velocity[0] - 5.0) < 0.3


class TestByteTracker:
    def test_stable_id_moving_object(self):
        tr = ByteTracker(n_init=3)
        ids = []
        for i in range(10):
            tracks = tr.step([det(*box_at(100 + 5 * i))], dt=1 / 12)
            ids.extend(t.track_id for t in tracks)
        assert len(set(ids)) == 1  # one identity throughout

    def test_id_survives_short_occlusion(self):
        tr = ByteTracker(n_init=3, max_misses=12)
        for i in range(6):
            tracks = tr.step([det(*box_at(100 + 5 * i))], dt=1 / 12)
        original_id = tracks[0].track_id
        for _ in range(4):  # fully occluded: no detections
            tr.step([], dt=1 / 12)
        # Reappears roughly where the Kalman prediction coasted to.
        tracks = tr.step([det(*box_at(100 + 5 * 10))], dt=1 / 12)
        assert [t.track_id for t in tracks] == [original_id]

    def test_low_conf_recovery_byte_pass(self):
        # Score drops below high_thresh during partial occlusion; the second
        # (low-confidence) association pass must keep the track alive.
        tr = ByteTracker(n_init=3)
        for i in range(5):
            tracks = tr.step([det(*box_at(100 + 5 * i), score=0.9)], dt=1 / 12)
        original_id = tracks[0].track_id
        for i in range(5, 9):
            tracks = tr.step([det(*box_at(100 + 5 * i), score=0.3)], dt=1 / 12)
        assert [t.track_id for t in tracks] == [original_id]

    def test_low_conf_never_births_track(self):
        tr = ByteTracker()
        for _ in range(10):
            tracks = tr.step([det(*box_at(100), score=0.3)], dt=1 / 12)
        assert tracks == [] and tr.tracks == []

    def test_two_objects_two_ids(self):
        tr = ByteTracker(n_init=3)
        for i in range(6):
            tracks = tr.step(
                [det(*box_at(100 + 5 * i)), det(*box_at(600 - 5 * i))], dt=1 / 12
            )
        assert len(tracks) == 2
        assert tracks[0].track_id != tracks[1].track_id

    def test_class_group_gating(self):
        # A pedestrian detection must not continue a car track, even at high IoU.
        tr = ByteTracker(n_init=2)
        tr.step([det(*box_at(100), cls="car")], dt=1 / 12)
        tr.step([det(*box_at(100), cls="car")], dt=1 / 12)  # car now CONFIRMED
        tr.step([det(*box_at(100), cls="pedestrian")], dt=1 / 12)
        cats = sorted(t.category for t in tr.tracks)
        assert cats == ["car", "pedestrian"]  # separate tracks, no takeover

    def test_car_truck_flicker_keeps_track(self):
        tr = ByteTracker(n_init=2)
        tr.step([det(*box_at(100), cls="car", score=0.9)], dt=1 / 12)
        tracks = tr.step([det(*box_at(102), cls="truck", score=0.6)], dt=1 / 12)
        assert len(tr.tracks) == 1  # same vehicle-group track
        assert tracks[0].category == "car"  # highest-confidence evidence wins

    def test_velocity_from_kalman(self):
        tr = ByteTracker(n_init=3)
        for i in range(24):  # 2s of motion at 60 px/s
            tracks = tr.step([det(*box_at(100 + 5 * i))], dt=1 / 12)
        vx = tracks[0].kf.velocity[0]
        assert abs(vx - 60.0) < 6.0


class TestMotAccumulator:
    def test_perfect_tracking(self):
        acc = MotAccumulator()
        for i in range(5):
            acc.add_keyframe(
                gt=[("inst-a", np.array(box_at(100 + i * 10)))],
                tracks=[(1, np.array(box_at(100 + i * 10)))],
            )
        s = acc.summary()
        assert s.mota == 1.0 and s.idf1 == 1.0 and s.id_switches == 0

    def test_id_switch_counted_and_idf1_drops(self):
        acc = MotAccumulator()
        for i in range(4):
            tid = 1 if i < 2 else 2  # identity swaps mid-sequence
            acc.add_keyframe(
                gt=[("inst-a", np.array(box_at(100)))],
                tracks=[(tid, np.array(box_at(100)))],
            )
        s = acc.summary()
        assert s.id_switches == 1
        assert s.idf1 == 0.5  # only the majority track counts toward identity
        assert s.mota == 1.0 - 1 / 4

    def test_misses_and_false_positives(self):
        acc = MotAccumulator()
        acc.add_keyframe(gt=[("a", np.array(box_at(100)))], tracks=[])
        acc.add_keyframe(gt=[], tracks=[(1, np.array(box_at(500)))])
        s = acc.summary()
        assert s.fn == 1 and s.fp == 1
