"""Replay the tracker + lift + BEV Kalman from a cached detection stream.

Lets M10 sweep camera-extrinsic and ego-pose perturbations without touching a
GPU. Perturbations are applied to what the SYSTEM BELIEVES (assumed extrinsics
/ assumed ego pose); ground truth is untouched, which is what makes the
resulting error curves meaningful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment

from adp.calib.camera import CameraModel, SE3
from adp.detect.detector import Detection2D
from adp.eval.detection2d import iou_matrix
from adp.lift.bev_state import update_bev_states
from adp.lift.fused import FusedLift
from adp.lift.ipm import GroundPlaneLift
from adp.lift.width_prior import WidthPriorLift
from adp.track.bytetrack import ByteTracker


def load_cache(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pitch_perturbed(T_ego_cam: SE3, degrees: float) -> SE3:
    """Rotate the ASSUMED camera extrinsics about the camera's lateral axis.

    Models both miscalibration and real vehicle pitch (braking, acceleration,
    a crest or dip). The camera's x axis is lateral, so the rotation is applied
    on the right in camera coordinates.
    """
    if degrees == 0.0:
        return T_ego_cam
    R_pitch = Quaternion(axis=[1.0, 0.0, 0.0], degrees=degrees).rotation_matrix
    return SE3.from_rt(T_ego_cam.rotation @ R_pitch, T_ego_cam.translation)


def noisy_pose(T_glob_ego: SE3, pos_sigma_m: float, yaw_sigma_deg: float,
               rng: np.random.Generator) -> SE3:
    """Degrade the ASSUMED ego pose — simulates weaker localisation."""
    if pos_sigma_m == 0.0 and yaw_sigma_deg == 0.0:
        return T_glob_ego
    dyaw = rng.normal(0.0, np.deg2rad(yaw_sigma_deg))
    R = Quaternion(axis=[0.0, 0.0, 1.0], radians=dyaw).rotation_matrix
    t = T_glob_ego.translation + np.r_[rng.normal(0, pos_sigma_m, 2), 0.0]
    return SE3.from_rt(R @ T_glob_ego.rotation, t)


@dataclass
class ReplayResult:
    range_rel_err: list = field(default_factory=list)   # (gt_range, rel_err)
    vel_err: list = field(default_factory=list)         # (gt_range, m/s error)
    n_pairs: int = 0

    def median_range_err(self, lo=0.0, hi=np.inf) -> float:
        e = [r for g, r in self.range_rel_err if lo <= g < hi]
        return float(np.median(e)) if e else float("nan")

    def median_vel_err(self, lo=0.0, hi=np.inf) -> float:
        e = [v for g, v in self.vel_err if lo <= g < hi]
        return float(np.median(e)) if e else float("nan")


def replay_scene(
    scene: dict,
    width_prior: WidthPriorLift | None,
    pitch_deg: float = 0.0,
    pos_sigma_m: float = 0.0,
    yaw_sigma_deg: float = 0.0,
    use_localization: bool = True,
    gt_velocity: dict | None = None,
    seed: int = 0,
) -> ReplayResult:
    """Replay one scene under a perturbation. gt_velocity maps annotation
    token -> global velocity (m/s) when velocity validation is wanted."""
    rng = np.random.default_rng(seed)
    tracker = ByteTracker()
    result = ReplayResult()
    prev_ts = None

    for fr in scene["frames"]:
        cam_model = CameraModel(K=np.array(fr["K"]), width=fr["wh"][0],
                                height=fr["wh"][1])
        T_ego_cam = pitch_perturbed(SE3(np.array(fr["T_ego_cam"])), pitch_deg)
        T_glob_ego_true = SE3(np.array(fr["T_glob_ego"]))
        if use_localization:
            T_glob_ego = noisy_pose(T_glob_ego_true, pos_sigma_m,
                                    yaw_sigma_deg, rng)
        else:
            # No localisation at all: track in the ego frame. Objects then
            # carry ego-relative motion, which is what M3 chose to avoid.
            T_glob_ego = SE3.from_rt(np.eye(3), np.zeros(3))

        dets = [Detection2D(xyxy=np.array(d[:4], dtype=float), score=d[4],
                            category=d[5]) for d in fr["dets"]]
        dt = 1 / 12 if prev_ts is None else (fr["ts"] - prev_ts) / 1e6
        prev_ts = fr["ts"]
        confirmed = tracker.step(dets, dt)

        ground = GroundPlaneLift(cam_model, T_ego_cam)
        lift = FusedLift(ground, width_prior) if width_prior else ground
        update_bev_states(confirmed, lift, T_glob_ego, dt)

        rows = scene["gt"].get(fr["sample"] or "", []) if fr["key"] else []
        if not rows or not confirmed:
            continue
        iou = iou_matrix(np.array([t.xyxy for t in confirmed]),
                         np.array([r["xyxy"] for r in rows], dtype=float))
        ri, ci = linear_sum_assignment(1.0 - iou)
        for i, j in zip(ri, ci):
            if iou[i, j] < 0.5:
                continue
            track, gt = confirmed[i], rows[j]
            state = track.extra.get("bev")
            if state is None:
                continue
            pos_e, _ = state.in_ego(T_glob_ego)
            # GT stays in the TRUE ego frame; the perturbation lives in the
            # system's beliefs, so the comparison stays honest.
            gt_range = gt["range"]
            result.range_rel_err.append(
                (gt_range, abs(float(np.linalg.norm(pos_e)) - gt_range) / gt_range))
            result.n_pairs += 1
            if gt_velocity is not None:
                v_gt = gt_velocity.get(gt["token"])
                if v_gt is not None and np.all(np.isfinite(v_gt)):
                    v_est = state.kf.velocity
                    if not use_localization:
                        # Ego-frame tracking yields RELATIVE velocity; compare
                        # against relative GT so the variant is judged fairly.
                        continue
                    result.vel_err.append(
                        (gt_range, float(np.linalg.norm(v_est - np.array(v_gt)))))
    return result
