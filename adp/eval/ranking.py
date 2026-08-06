"""Ranking metrics for predictor evaluation: ROC-AUC and average precision.

Implemented directly rather than pulling in scikit-learn — both are a dozen
lines, and having them under test means the M8 validation numbers don't rest
on an unexamined dependency. Both metrics are rank-based, so any monotone
transform of a predictor (1/x, -x) leaves them unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Ties get average ranks (the standard treatment: a predictor that cannot
    separate two samples scores 0.5 between them, not 0 or 1).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation).

    AP = sum_n (R_n - R_{n-1}) * P_n over distinct score thresholds. Preferred
    over ROC-AUC when positives are rare, which they are here.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    # Evaluate only at distinct thresholds, so tied scores form one point.
    idx = np.r_[np.where(np.diff(s))[0], len(y) - 1]
    tp = np.cumsum(y)[idx]
    fp = np.cumsum(1 - y)[idx]
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def bootstrap_ci(
    metric_fn,
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI. With ~10 scenes of data the point estimates
    are noisy; reporting them without an interval would overstate certainty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric_fn(scores[idx], labels[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))
