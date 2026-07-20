"""The ADP perception pipeline: frame -> detection -> tracking -> BEV lift ->
lanes -> per-object risk.

Single stateful object per scene; scripts should use this rather than wiring
modules together themselves. Lane segmentation optionally runs at a reduced
rate (lane_stride) — the corridor smoother holds between updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adp.calib.camera import SE3
from adp.data.nuscenes_source import CamFrame
from adp.detect.detector import Detector
from adp.lanes.bev_lanes import EgoCorridor, cluster_and_fit, find_ego_corridor, \
    mask_to_bev_points
from adp.lanes.detector import LaneSegmenter
from adp.lanes.smoothing import CorridorSmoother
from adp.lift.bev_state import update_bev_states
from adp.lift.ipm import GroundPlaneLift
from adp.risk.assign import assign_lane, crossing_intent
from adp.risk.rss import rss_min_gap
from adp.risk.scoring import RiskRecord, ego_risk, score_object
from adp.risk.ttc import compute_ttc
from adp.track.bytetrack import ByteTracker, Track
from adp.track.kalman import ConstantVelocityKalman


@dataclass
class PipelineOutput:
    tracks: list[Track]
    corridor: EgoCorridor          # smoothed; may be invalid
    lane_lines: list
    risk: list[RiskRecord]
    ego_risk: float
    ego_speed_ms: float
    timings_ms: dict = field(default_factory=dict)


class PerceptionPipeline:
    def __init__(self, detector: Detector | None = None,
                 segmenter: LaneSegmenter | None = None,
                 lane_stride: int = 2,
                 use_depth: bool = False):
        """use_depth: enable the M6 hybrid lift (IPM <30m, monocular depth
        beyond). Costs ~39ms/frame; improves far-field BEV accuracy only —
        the TTC/RSS working range (<30m) is IPM's regardless. Default off."""
        self.detector = detector or Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)
        self.segmenter = segmenter or LaneSegmenter()
        self.lane_stride = lane_stride
        self.depth = None
        if use_depth:
            from adp.lift.depth import DepthLift
            self.depth = DepthLift()
        self.tracker = ByteTracker()
        self.corridor_smoother = CorridorSmoother()
        self.ego_kf: ConstantVelocityKalman | None = None
        self._frame_i = 0
        self._prev_ts: int | None = None
        self._lane_lines: list = []

    def step(self, cam: CamFrame, img_bgr: np.ndarray) -> PipelineOutput:
        import time
        timings = {}
        dt = (1 / 12 if self._prev_ts is None
              else (cam.timestamp_us - self._prev_ts) / 1e6)
        self._prev_ts = cam.timestamp_us

        # Ego velocity filter (global frame) — pose is localization input, but
        # velocity still comes from a Kalman filter like everything else.
        ego_xy = cam.T_global_from_ego.translation[:2]
        if self.ego_kf is None:
            self.ego_kf = ConstantVelocityKalman(
                z0=ego_xy, pos_std=0.5, vel_std=5.0, meas_std=0.05, process_std=2.0)
        else:
            self.ego_kf.predict(dt)
            self.ego_kf.update(ego_xy)

        t0 = time.perf_counter()
        dets = self.detector(img_bgr)
        timings["detect"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        confirmed = self.tracker.step(dets, dt)
        lift = GroundPlaneLift(cam.camera, cam.T_ego_from_cam)
        if self.depth is not None:
            from adp.lift.hybrid import HybridLift
            self.depth.compute(img_bgr)
            lift = HybridLift(lift, self.depth)
        update_bev_states(confirmed, lift, cam.T_global_from_ego, dt)
        timings["track_lift"] = (time.perf_counter() - t0) * 1e3

        if self._frame_i % self.lane_stride == 0:
            t0 = time.perf_counter()
            mask = self.segmenter.lane_mask(img_bgr)
            self._lane_lines = cluster_and_fit(mask_to_bev_points(mask, lift))
            measured = find_ego_corridor(self._lane_lines)
            corridor = self.corridor_smoother.update(measured, dt * self.lane_stride)
            timings["lanes"] = (time.perf_counter() - t0) * 1e3
        else:
            corridor = self.corridor_smoother.update(
                EgoCorridor(None, None, None), 0.0)  # hold, no time charged
        self._frame_i += 1

        risk = self._risk_records(confirmed, corridor, cam.T_global_from_ego)
        return PipelineOutput(
            tracks=confirmed,
            corridor=corridor,
            lane_lines=self._lane_lines,
            risk=risk,
            ego_risk=ego_risk(risk),
            ego_speed_ms=float(np.linalg.norm(self.ego_kf.velocity)),
            timings_ms=timings,
        )

    def _risk_records(self, tracks, corridor, T_global_from_ego: SE3):
        records = []
        R_eg = T_global_from_ego.inverse().rotation[:2, :2]
        v_ego_g = self.ego_kf.velocity
        ego_vel_sigma = float(np.linalg.norm(self.ego_kf.velocity_std()))
        corr = corridor if corridor.valid else None

        for t in tracks:
            state = t.extra.get("bev")
            if state is None:
                continue
            pos_e, _ = state.in_ego(T_global_from_ego)
            vel_rel_e = R_eg @ (state.kf.velocity - v_ego_g)
            rng = float(np.linalg.norm(pos_e))

            assignment = assign_lane(pos_e, corr)
            ttc = compute_ttc(
                x_forward_m=float(pos_e[0]),
                v_rel_forward_ms=float(vel_rel_e[0]),
                pos_sigma_m=float(np.linalg.norm(state.kf.position_std())),
                vel_sigma_ms=float(np.linalg.norm(state.kf.velocity_std())) + ego_vel_sigma,
            )

            # RSS: front-object speed along ego travel direction.
            v_front_along = float((R_eg @ state.kf.velocity)[0])
            rss_gap = rss_min_gap(float(np.linalg.norm(v_ego_g)), v_front_along)
            gap = ttc.gap_m if ttc is not None else float(pos_e[0]) - 3.5
            rss_violated = (assignment.zone.value == "ego" and gap > 0
                            and gap < rss_gap)

            intent = None
            if t.category in ("pedestrian", "cyclist"):
                intent = crossing_intent(pos_e, R_eg @ state.kf.velocity,
                                         assignment, corr)

            records.append(score_object(RiskRecord(
                track_id=t.track_id,
                category=t.category,
                range_m=rng,
                zone=assignment.zone,
                corridor_source=assignment.corridor_source,
                closing_ms=float(-vel_rel_e[0]),
                ttc=ttc,
                rss_min_gap_m=rss_gap,
                rss_violated=rss_violated,
                intent_cross_s=intent,
                object_speed_ms=float(np.linalg.norm(state.kf.velocity)),
            )))
        records.sort(key=lambda r: -r.score)
        return records
