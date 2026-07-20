"""Sanity tests for the AP@0.5 evaluator on hand-constructed cases."""

import numpy as np

from adp.detect.detector import Detection2D
from adp.eval.detection2d import DetectionEvaluator, FrameGT, iou_matrix


def det(x0, y0, x1, y1, score, cls="car"):
    return Detection2D(xyxy=np.array([x0, y0, x1, y1], dtype=float), score=score, category=cls)


def gt(boxes_by_class, ignore=None):
    return FrameGT(
        boxes={c: np.array(b, dtype=float).reshape(-1, 4) for c, b in boxes_by_class.items()},
        ignore=np.array(ignore or [], dtype=float).reshape(-1, 4),
    )


class TestIoU:
    def test_identical_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=float)
        assert np.isclose(iou_matrix(a, a)[0, 0], 1.0)

    def test_disjoint_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=float)
        b = np.array([[20, 20, 30, 30]], dtype=float)
        assert iou_matrix(a, b)[0, 0] == 0.0

    def test_half_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=float)
        b = np.array([[5, 0, 15, 10]], dtype=float)
        assert np.isclose(iou_matrix(a, b)[0, 0], 50 / 150)


class TestEvaluator:
    def test_perfect_detection(self):
        ev = DetectionEvaluator()
        ev.add_frame([det(0, 0, 10, 10, 0.9)], gt({"car": [[0, 0, 10, 10]]}))
        assert ev.ap("car") == 1.0

    def test_miss_gives_zero(self):
        ev = DetectionEvaluator()
        ev.add_frame([], gt({"car": [[0, 0, 10, 10]]}))
        assert ev.ap("car") == 0.0

    def test_false_positive_halves_precision(self):
        ev = DetectionEvaluator()
        # High-scoring FP first, then the TP: precision at full recall is 0.5.
        ev.add_frame(
            [det(100, 100, 120, 120, 0.95), det(0, 0, 10, 10, 0.9)],
            gt({"car": [[0, 0, 10, 10]]}),
        )
        assert np.isclose(ev.ap("car"), 0.5)

    def test_ignore_region_absorbs_fp(self):
        ev = DetectionEvaluator()
        # Same FP but it lands on an ignore box -> discarded -> perfect AP.
        ev.add_frame(
            [det(100, 100, 120, 120, 0.95), det(0, 0, 10, 10, 0.9)],
            gt({"car": [[0, 0, 10, 10]]}, ignore=[[100, 100, 120, 120]]),
        )
        assert ev.ap("car") == 1.0

    def test_duplicate_detection_is_fp(self):
        ev = DetectionEvaluator()
        ev.add_frame(
            [det(0, 0, 10, 10, 0.9), det(0.5, 0, 10.5, 10, 0.8)],
            gt({"car": [[0, 0, 10, 10]]}),
        )
        assert np.isclose(ev.ap("car"), 1.0)  # AP unaffected: TP found before FP

    def test_wrong_class_is_missed(self):
        ev = DetectionEvaluator()
        ev.add_frame([det(0, 0, 10, 10, 0.9, cls="truck")], gt({"car": [[0, 0, 10, 10]]}))
        assert ev.ap("car") == 0.0  # car GT never matched
        assert np.isnan(ev.ap("truck"))  # no truck GT anywhere -> class excluded from mAP
