"""2D detection evaluation: per-class AP@0.5 against 2D projections of nuScenes
3D GT boxes.

Ground-truth protocol:
- GT = objects of the 5 target classes with visibility >= 2 (over 40% visible).
- Ignore regions = visibility-1 objects (any class) and all 'other'-category
  objects (barriers, cones, ...). Predictions matching an ignore region are
  discarded rather than counted as false positives; ignore regions are never
  counted as false negatives.

Caveat, stated once and honestly: projected 3D corners give a slightly loose
2D rectangle (it bounds the full 3D extent, including self-occluded faces), so
absolute AP here reads a few points lower than a hand-labeled 2D benchmark
would. It is consistent across models, which is what a model-selection
baseline needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adp.data.nuscenes_source import Frame
from adp.detect.detector import Detection2D

ADP_CLASSES = ["car", "truck", "bus", "pedestrian", "cyclist"]


@dataclass
class FrameGT:
    """2D GT for one image: real boxes per class, plus class-agnostic ignore boxes."""

    boxes: dict[str, np.ndarray] = field(default_factory=dict)  # class -> (N, 4) xyxy
    ignore: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))


def gt_boxes_2d(frame: Frame, min_size_px: float = 4.0) -> FrameGT:
    """Project 3D GT boxes to clipped 2D rectangles and split GT vs ignore."""
    per_class: dict[str, list] = {c: [] for c in ADP_CLASSES}
    ignore: list = []

    for obj in frame.objects:
        uv, depth = frame.camera.project(obj.corners_cam())
        if np.any(depth <= 0.1):
            continue
        x0, y0 = uv.min(axis=0)
        x1, y1 = uv.max(axis=0)
        x0, x1 = np.clip([x0, x1], 0, frame.camera.width)
        y0, y1 = np.clip([y0, y1], 0, frame.camera.height)
        if (x1 - x0) < min_size_px or (y1 - y0) < min_size_px:
            continue
        rect = [x0, y0, x1, y1]
        if obj.category in per_class and obj.visibility >= 2:
            per_class[obj.category].append(rect)
        else:
            ignore.append(rect)

    return FrameGT(
        boxes={c: np.array(b).reshape(-1, 4) for c, b in per_class.items()},
        ignore=np.array(ignore).reshape(-1, 4),
    )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class DetectionEvaluator:
    """Accumulates (detections, GT) per frame; computes per-class AP@0.5."""

    def __init__(self, iou_thr: float = 0.5):
        self.iou_thr = iou_thr
        # per class: list of (score, is_tp); plus total GT count
        self._records: dict[str, list] = {c: [] for c in ADP_CLASSES}
        self._n_gt: dict[str, int] = {c: 0 for c in ADP_CLASSES}

    def add_frame(self, detections: list[Detection2D], gt: FrameGT) -> None:
        for cls in ADP_CLASSES:
            gt_boxes = gt.boxes.get(cls, np.empty((0, 4)))
            self._n_gt[cls] += len(gt_boxes)

            dets = [d for d in detections if d.category == cls]
            dets.sort(key=lambda d: d.score, reverse=True)
            det_boxes = np.array([d.xyxy for d in dets]).reshape(-1, 4)

            ious_gt = iou_matrix(det_boxes, gt_boxes)
            ious_ign = iou_matrix(det_boxes, gt.ignore)
            gt_used = np.zeros(len(gt_boxes), dtype=bool)

            for i, det in enumerate(dets):
                j = -1
                if len(gt_boxes):
                    candidates = np.where(~gt_used & (ious_gt[i] >= self.iou_thr))[0]
                    if len(candidates):
                        j = candidates[np.argmax(ious_gt[i][candidates])]
                if j >= 0:
                    gt_used[j] = True
                    self._records[cls].append((det.score, 1))
                elif len(gt.ignore) and ious_ign[i].max() >= self.iou_thr:
                    continue  # matched an ignore region: discard silently
                else:
                    self._records[cls].append((det.score, 0))

    def ap(self, cls: str) -> float:
        """All-point interpolated AP@iou_thr for one class (NaN if no GT)."""
        n_gt = self._n_gt[cls]
        if n_gt == 0:
            return float("nan")
        records = sorted(self._records[cls], key=lambda r: r[0], reverse=True)
        if not records:
            return 0.0
        tps = np.array([r[1] for r in records])
        cum_tp = np.cumsum(tps)
        recall = cum_tp / n_gt
        precision = cum_tp / (np.arange(len(tps)) + 1)
        # precision envelope, then integrate over recall steps
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        r = np.concatenate([[0.0], recall])
        return float(np.sum((r[1:] - r[:-1]) * precision))

    def summary(self) -> dict:
        aps = {c: self.ap(c) for c in ADP_CLASSES}
        valid = [v for v in aps.values() if not np.isnan(v)]
        return {
            "ap_per_class": aps,
            "n_gt_per_class": dict(self._n_gt),
            "map": float(np.mean(valid)) if valid else float("nan"),
        }
