"""Tracking evaluation at nuScenes keyframes: MOTA, IDF1, ID switches.

GT identities are nuScenes instance tokens (visibility >= 2, target classes
only). Per keyframe, GT boxes are matched to confirmed track boxes by
Hungarian assignment on IoU (gate 0.5).

IDF1 note: computed from the per-frame IoU-gated matches via a global
bipartite assignment of GT identity <-> track id (a standard simplification;
official IDF1 re-matches trajectories globally without the per-frame gate).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from adp.eval.detection2d import iou_matrix


@dataclass
class MotSummary:
    mota: float
    idf1: float
    id_switches: int
    n_gt: int
    fp: int
    fn: int
    recall: float
    precision: float


class MotAccumulator:
    RANGE_BINS = [(0, 30), (30, 50), (50, float("inf"))]

    def __init__(self, iou_thr: float = 0.5):
        self.iou_thr = iou_thr
        self.n_gt = 0
        self.n_pred = 0
        self.fp = 0
        self.fn = 0
        self.idsw = 0
        self._last_match: dict[str, int] = {}  # gt instance -> last track id
        self._pair_counts: dict[tuple[str, int], int] = defaultdict(int)
        # per range bin: [n_gt, n_matched] — recall breakdown by distance
        self._range_counts = {b: [0, 0] for b in self.RANGE_BINS}

    def _bin(self, r: float):
        for lo, hi in self.RANGE_BINS:
            if lo <= r < hi:
                return (lo, hi)
        return self.RANGE_BINS[-1]

    def add_keyframe(
        self,
        gt: list[tuple[str, np.ndarray]],      # (instance_token, xyxy)
        tracks: list[tuple[int, np.ndarray]],  # (track_id, xyxy)
        gt_ranges: list[float] | None = None,  # meters, aligned with gt
    ) -> None:
        self.n_gt += len(gt)
        self.n_pred += len(tracks)

        gt_boxes = np.array([r for _, r in gt]).reshape(-1, 4)
        tr_boxes = np.array([r for _, r in tracks]).reshape(-1, 4)
        iou = iou_matrix(gt_boxes, tr_boxes)

        matched_g, matched_t = set(), set()
        if len(gt) and len(tracks):
            rows, cols = linear_sum_assignment(1.0 - iou)
            for g, t in zip(rows, cols):
                if iou[g, t] < self.iou_thr:
                    continue
                matched_g.add(g)
                matched_t.add(t)
                inst, tid = gt[g][0], tracks[t][0]
                self._pair_counts[(inst, tid)] += 1
                if inst in self._last_match and self._last_match[inst] != tid:
                    self.idsw += 1
                self._last_match[inst] = tid

        self.fn += len(gt) - len(matched_g)
        self.fp += len(tracks) - len(matched_t)

        if gt_ranges is not None:
            for g, r in enumerate(gt_ranges):
                counts = self._range_counts[self._bin(r)]
                counts[0] += 1
                counts[1] += int(g in matched_g)

    def summary(self) -> MotSummary:
        # IDF1 via max-weight bipartite matching of identities.
        insts = sorted({k[0] for k in self._pair_counts})
        tids = sorted({k[1] for k in self._pair_counts})
        idtp = 0
        if insts and tids:
            m = np.zeros((len(insts), len(tids)))
            for (inst, tid), c in self._pair_counts.items():
                m[insts.index(inst), tids.index(tid)] = c
            rows, cols = linear_sum_assignment(-m)
            idtp = int(m[rows, cols].sum())
        idf1 = 2 * idtp / max(self.n_gt + self.n_pred, 1)

        tp = self.n_gt - self.fn
        self.recall_by_range = {
            f"{lo:.0f}-{hi:.0f}m" if np.isfinite(hi) else f"{lo:.0f}m+":
                (m / n if n else float("nan"))
            for (lo, hi), (n, m) in self._range_counts.items()
        }
        return MotSummary(
            mota=1.0 - (self.fn + self.fp + self.idsw) / max(self.n_gt, 1),
            idf1=idf1,
            id_switches=self.idsw,
            n_gt=self.n_gt,
            fp=self.fp,
            fn=self.fn,
            recall=tp / max(self.n_gt, 1),
            precision=tp / max(tp + self.fp, 1),
        )
