"""ByteTrack-style multi-object tracker (Zhang et al. 2022), implemented
directly so the Kalman state stays ours (M3 attaches BEV filters per track).

Core idea: associate high-confidence detections first (Hungarian on IoU), then
give unmatched tracks a second chance against LOW-confidence detections —
recovering occluded/blurred objects that a single threshold would drop.

Class handling: association is gated by coarse group (vehicle / pedestrian /
cyclist) so car<->truck flicker doesn't fragment tracks; the reported category
is the highest-confidence recent detection's class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from adp.detect.detector import Detection2D
from adp.eval.detection2d import iou_matrix
from adp.track.kalman import ConstantVelocityKalman

COARSE_GROUP = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "pedestrian": "pedestrian", "cyclist": "cyclist",
}


class TrackState(Enum):
    TENTATIVE = 1  # seen, not yet confirmed
    CONFIRMED = 2
    LOST = 3       # missed recently, still eligible for re-association


def _xyxy_to_cxcywh(xyxy: np.ndarray) -> np.ndarray:
    return np.array([
        (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2,
        xyxy[2] - xyxy[0], xyxy[3] - xyxy[1],
    ])


def _cxcywh_to_xyxy(c: np.ndarray) -> np.ndarray:
    return np.array([c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2])


@dataclass
class Track:
    track_id: int
    kf: ConstantVelocityKalman  # image-space (cx, cy, w, h)
    category: str
    score: float
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    misses: int = 0
    age_s: float = 0.0
    extra: dict = field(default_factory=dict)  # M3+ attaches BEV state here

    @property
    def xyxy(self) -> np.ndarray:
        return _cxcywh_to_xyxy(self.kf.position)

    @property
    def group(self) -> str:
        return COARSE_GROUP[self.category]


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.5,   # detections above: primary association
        low_thresh: float = 0.1,    # detections in [low, high): recovery pass
        new_track_thresh: float = 0.6,  # min score to birth a track
        match_iou: float = 0.2,     # min IoU for primary match
        match_iou_low: float = 0.4, # stricter gate for low-conf recovery
        n_init: int = 3,            # consecutive hits to confirm
        max_misses: int = 12,       # frames before a lost track is dropped (~1s @ 12Hz)
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_iou = match_iou
        self.match_iou_low = match_iou_low
        self.n_init = n_init
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 1

    # -- association helpers -------------------------------------------------

    @staticmethod
    def _match(tracks: list[Track], dets: list[Detection2D], iou_gate: float):
        """Hungarian on (1 - IoU), gated by IoU and coarse class group.
        Returns (pairs, unmatched_track_idx, unmatched_det_idx)."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        t_boxes = np.array([t.xyxy for t in tracks])
        d_boxes = np.array([d.xyxy for d in dets])
        iou = iou_matrix(t_boxes, d_boxes)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                if COARSE_GROUP[d.category] != t.group:
                    iou[i, j] = 0.0
        rows, cols = linear_sum_assignment(1.0 - iou)
        pairs, um_t, um_d = [], set(range(len(tracks))), set(range(len(dets)))
        for i, j in zip(rows, cols):
            if iou[i, j] >= iou_gate:
                pairs.append((i, j))
                um_t.discard(i)
                um_d.discard(j)
        return pairs, sorted(um_t), sorted(um_d)

    def _apply_match(self, track: Track, det: Detection2D) -> None:
        track.kf.update(_xyxy_to_cxcywh(det.xyxy))
        track.hits += 1
        track.misses = 0
        if det.score >= track.score:  # adopt class of the most confident evidence
            track.score = det.score
            track.category = det.category
        if track.state != TrackState.CONFIRMED and track.hits >= self.n_init:
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED

    def _new_track(self, det: Detection2D) -> Track:
        c = _xyxy_to_cxcywh(det.xyxy)
        h = max(c[3], 1.0)
        # Noise scales follow ByteTrack convention: proportional to box height.
        kf = ConstantVelocityKalman(
            z0=c, pos_std=h / 10, vel_std=h / 4, meas_std=h / 20, process_std=h / 2,
        )
        track = Track(track_id=self._next_id, kf=kf, category=det.category, score=det.score)
        self._next_id += 1
        return track

    # -- main step -----------------------------------------------------------

    def step(self, detections: list[Detection2D], dt: float) -> list[Track]:
        """Advance one frame. Returns currently CONFIRMED tracks."""
        for t in self.tracks:
            t.kf.predict(dt)
            t.age_s += dt

        high = [d for d in detections if d.score >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        # Pass 1: high-confidence detections vs all live tracks.
        pairs, um_t, um_d = self._match(self.tracks, high, self.match_iou)
        for i, j in pairs:
            self._apply_match(self.tracks[i], high[j])

        # Pass 2 (the "byte" pass): remaining tracks vs low-confidence dets.
        remaining = [self.tracks[i] for i in um_t]
        pairs2, um_t2, _ = self._match(remaining, low, self.match_iou_low)
        for i, j in pairs2:
            self._apply_match(remaining[i], low[j])

        # Unmatched tracks: mark missed.
        for i in um_t2:
            t = remaining[i]
            t.misses += 1
            if t.state == TrackState.CONFIRMED:
                t.state = TrackState.LOST
        self.tracks = [
            t for t in self.tracks
            if not (t.misses > self.max_misses
                    or (t.state == TrackState.TENTATIVE and t.misses > 0))
        ]

        # Unmatched high-confidence detections: birth new tracks.
        for j in um_d:
            if high[j].score >= self.new_track_thresh:
                self.tracks.append(self._new_track(high[j]))

        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]
