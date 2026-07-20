"""Lane geometry in BEV: mask pixels -> metric ground points -> clustered
polylines -> quadratic fits -> ego corridor.

All fitting happens in BEV space (plan rule): image-space lane curves are
perspective-distorted and useless for geometry. Curves are y = c0 + c1*x +
c2*x^2 in the ego frame (x forward, y left), i.e. lateral offset as a
function of forward distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adp.lift.ipm import GroundPlaneLift

# Working region for lane geometry, ego frame (meters).
X_MIN, X_MAX = 4.0, 45.0
Y_HALF = 12.0
SLICE_M = 2.0        # x-slice width for clustering
GAP_M = 1.0          # lateral gap that separates two lines within a slice
LINK_M = 1.2         # max lateral jump per slice when linking clusters across x
MAX_LINK_SLICES = 4  # bridge up to ~8m forward gaps (dashed markings: ~3m
                     # dash + ~6m gap, so consecutive dashes must chain)
MAX_LINK_LATERAL = 2.5  # cap on total lateral jump across a bridged gap
MIN_SPAN_SLICES = 5  # a lane line must SPAN at least this many slices (~10m)
MIN_CHAIN = 3        # ...and have at least this many linked clusters
LANE_WIDTH_RANGE = (2.4, 5.0)  # sane ego-lane widths


@dataclass
class LaneLine:
    coeffs: np.ndarray      # [c0, c1, c2]: y = c0 + c1*x + c2*x^2
    x_range: tuple          # (x_min, x_max) where points supported the fit
    n_points: int

    def y_at(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.coeffs[0] + self.coeffs[1] * x + self.coeffs[2] * x**2


@dataclass
class EgoCorridor:
    left: LaneLine | None
    right: LaneLine | None
    width: float | None

    @property
    def valid(self) -> bool:
        return self.left is not None and self.right is not None

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool | None:
        """Is BEV point (x, y) inside the corridor? None if corridor invalid."""
        if not self.valid:
            return None
        return bool(self.right.y_at(x) - margin <= y <= self.left.y_at(x) + margin)


def mask_to_bev_points(mask: np.ndarray, lift: GroundPlaneLift,
                       stride: int = 8) -> np.ndarray:
    """Subsampled lane-mask pixels -> (N, 2) ego-frame ground points in the
    working region."""
    vs, us = np.nonzero(mask[::stride, ::stride])
    if len(us) == 0:
        return np.empty((0, 2))
    uv = np.stack([us * stride, vs * stride], axis=1).astype(float)
    xy, valid = lift.lift_pixels(uv)
    xy = xy[valid]
    keep = (xy[:, 0] >= X_MIN) & (xy[:, 0] <= X_MAX) & (np.abs(xy[:, 1]) <= Y_HALF)
    return xy[keep]


def cluster_and_fit(points: np.ndarray) -> list[LaneLine]:
    """Greedy slice-and-link clustering, then weighted quadratic fit per line.

    Deterministic and dependency-free: slice BEV by forward distance, split
    each slice's points on lateral gaps, then chain slice-clusters whose
    lateral centers stay within LINK_M of the running line.
    """
    if len(points) == 0:
        return []

    # 1) Per-slice lateral clusters: (slice_idx, y_center, n).
    slice_idx = ((points[:, 0] - X_MIN) // SLICE_M).astype(int)
    slice_clusters: dict[int, list[tuple[float, int]]] = {}
    for si in np.unique(slice_idx):
        ys = np.sort(points[slice_idx == si, 1])
        splits = np.where(np.diff(ys) > GAP_M)[0]
        for seg in np.split(ys, splits + 1):
            if len(seg):
                slice_clusters.setdefault(si, []).append((float(seg.mean()), len(seg)))

    # 2) Link across slices, nearest-center chaining.
    chains: list[list[tuple[int, float, int]]] = []  # [(si, y, n), ...]
    for si in sorted(slice_clusters):
        for y, n in slice_clusters[si]:
            best = None
            for chain in chains:
                last_si, last_y, _ = chain[-1]
                gap = si - last_si
                if gap < 1 or gap > MAX_LINK_SLICES:
                    continue
                tol = min(LINK_M * gap, MAX_LINK_LATERAL)
                d = abs(y - last_y)
                if d <= tol and (best is None or d < best[1]):
                    best = (chain, d)
            if best is not None:
                best[0].append((si, y, n))
            else:
                chains.append([(si, y, n)])

    # 3) Fit chains with enough support: must span a real distance (dashed
    #    lines have few clusters but long span) and have >= MIN_CHAIN members.
    lines = []
    for chain in chains:
        span = chain[-1][0] - chain[0][0] + 1
        if span < MIN_SPAN_SLICES or len(chain) < MIN_CHAIN:
            continue
        xs = np.array([X_MIN + (si + 0.5) * SLICE_M for si, _, _ in chain])
        ys = np.array([y for _, y, _ in chain])
        w = np.sqrt([n for _, _, n in chain])
        coeffs = np.polyfit(xs, ys, deg=2, w=w)[::-1]  # -> [c0, c1, c2]
        lines.append(LaneLine(
            coeffs=coeffs,
            x_range=(float(xs.min()), float(xs.max())),
            n_points=int(sum(n for _, _, n in chain)),
        ))
    return lines


def find_ego_corridor(lines: list[LaneLine], probe_x: float = 8.0) -> EgoCorridor:
    """Ego lane = the pair of lines bracketing y=0 at probe_x with sane width.
    Degrades to invalid (None sides) rather than guessing."""
    left, right = None, None
    for line in lines:
        if not (line.x_range[0] <= probe_x <= line.x_range[1] + SLICE_M):
            continue
        y = float(line.y_at(probe_x))
        if y > 0 and (left is None or y < left[0]):
            left = (y, line)
        elif y < 0 and (right is None or y > right[0]):
            right = (y, line)
    if left is None or right is None:
        return EgoCorridor(None, None, None)
    width = left[0] - right[0]
    if not (LANE_WIDTH_RANGE[0] <= width <= LANE_WIDTH_RANGE[1]):
        return EgoCorridor(None, None, None)
    return EgoCorridor(left=left[1], right=right[1], width=float(width))
