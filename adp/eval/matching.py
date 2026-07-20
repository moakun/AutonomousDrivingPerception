"""Shared track<->GT matching helpers for the lift validation harnesses."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from adp.eval.detection2d import ADP_CLASSES, iou_matrix, project_box_2d


def match_tracks_to_gt(confirmed, frame, iou_thr: float = 0.5):
    """IoU-match confirmed tracks to visible target-class GT.
    Returns [(track, obj)] pairs."""
    gt_objs, gt_rects = [], []
    for obj in frame.objects:
        if obj.category not in ADP_CLASSES or obj.visibility < 2:
            continue
        rect = project_box_2d(frame, obj)
        if rect is not None:
            gt_objs.append(obj)
            gt_rects.append(rect)
    if not gt_objs or not confirmed:
        return []
    iou = iou_matrix(np.array([t.xyxy for t in confirmed]),
                     np.array(gt_rects).reshape(-1, 4))
    rows, cols = linear_sum_assignment(1.0 - iou)
    return [(confirmed[i], gt_objs[j]) for i, j in zip(rows, cols)
            if iou[i, j] >= iou_thr]


def gt_ground_pos_ego(frame, obj) -> np.ndarray:
    """GT box center in ego frame, projected to the ground plane (z dropped)."""
    return frame.T_ego_from_cam.apply(obj.center_cam)[0][:2]
