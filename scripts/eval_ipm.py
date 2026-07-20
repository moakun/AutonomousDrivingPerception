"""M3 validation harness — the load-bearing script of the project.

Runs detector + tracker + IPM lift + BEV Kalman over every scene, matches
tracked objects to nuScenes 3D GT at keyframes (IoU on 2D boxes), and reports
range error binned by GT distance, for BOTH the raw single-frame IPM lift and
the Kalman-smoothed estimate. Also validates BEV velocity against nuScenes
annotation velocities (GT computed by the devkit from neighboring keyframes).

M3 exit target: median relative range error <= 15% in the 10-30m bin.

Usage: python scripts/eval_ipm.py [--demo-scene scene-0061]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector
from adp.eval.detection2d import ADP_CLASSES, iou_matrix, project_box_2d
from adp.lift.bev_state import update_bev_states
from adp.lift.ipm import GroundPlaneLift
from adp.track.bytetrack import ByteTracker
from adp.viz.bev import BevCanvas, render_bev

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "out", "eval", "ipm_validation.json")
OUT_DEMO = os.path.join(os.path.dirname(__file__), "..", "out", "demo")

RANGE_BINS = [(0, 10), (10, 30), (30, 50), (50, np.inf)]


def match_tracks_to_gt(confirmed, frame):
    """IoU-match confirmed tracks to visible target-class GT. Returns
    [(track, obj)] pairs."""
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
    return [(confirmed[i], gt_objs[j]) for i, j in zip(rows, cols) if iou[i, j] >= 0.5]


def gt_ground_pos_ego(frame, obj):
    """GT box center in ego frame, projected to the ground plane (z dropped)."""
    center_ego = frame.T_ego_from_cam.apply(obj.center_cam)[0]
    return center_ego[:2]


def bin_name(lo, hi):
    return f"{lo:.0f}-{hi:.0f}m" if np.isfinite(hi) else f"{lo:.0f}m+"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-scene", default="scene-0061")
    args = parser.parse_args()

    source = NuScenesSource(verbose=False)
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)
    os.makedirs(OUT_DEMO, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    records = []  # one row per matched (track, GT) pair at a keyframe

    for scene_name in source.scene_names():
        gt_frames = {f.sample_token: f for f in source.frames(scene_name, min_visibility=1)}
        tracker = ByteTracker()
        prev_ts = None

        writer = None
        if scene_name == args.demo_scene:
            demo_path = os.path.join(OUT_DEMO, f"{scene_name}_bev.mp4")
            writer = cv2.VideoWriter(demo_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     12, (960 + 540, 540))

        for cam in source.camera_frames(scene_name):
            img = cv2.imread(cam.image_path)
            dets = detector(img)
            dt = 1 / 12 if prev_ts is None else (cam.timestamp_us - prev_ts) / 1e6
            prev_ts = cam.timestamp_us

            confirmed = tracker.step(dets, dt)
            lift = GroundPlaneLift(cam.camera, cam.T_ego_from_cam)
            update_bev_states(confirmed, lift, cam.T_global_from_ego, dt)

            if cam.is_keyframe and cam.sample_token in gt_frames:
                frame = gt_frames[cam.sample_token]
                for track, obj in match_tracks_to_gt(confirmed, frame):
                    state = track.extra.get("bev")
                    if state is None:
                        continue
                    gt_xy = gt_ground_pos_ego(frame, obj)
                    gt_range = float(np.linalg.norm(gt_xy))
                    pos_e, vel_e = state.in_ego(cam.T_global_from_ego)
                    kf_range = float(np.linalg.norm(pos_e))
                    raw_xy, raw_range = lift.lift_box_bottom(track.xyxy)

                    row = {
                        "scene": scene_name,
                        "category": obj.category,
                        "gt_range": gt_range,
                        "kf_range": kf_range,
                        "raw_range": raw_range,
                        "kf_pos_err": float(np.linalg.norm(pos_e - gt_xy)),
                        "range_std": state.range_std(),
                    }
                    gt_vel = source.nusc.box_velocity(obj.token)[:2]  # global m/s
                    if np.all(np.isfinite(gt_vel)):
                        row["gt_speed"] = float(np.linalg.norm(gt_vel))
                        row["kf_speed"] = float(np.linalg.norm(state.kf.velocity))
                        row["vel_err"] = float(np.linalg.norm(state.kf.velocity - gt_vel))
                    records.append(row)

            if writer is not None:
                frame_gt = gt_frames.get(cam.sample_token) if cam.is_keyframe else None
                bev_tracks = []
                for t in confirmed:
                    s = t.extra.get("bev")
                    if s is None:
                        continue
                    pos_e, vel_e = s.in_ego(cam.T_global_from_ego)
                    bev_tracks.append({"id": t.track_id, "category": t.category,
                                       "pos": pos_e, "vel": vel_e,
                                       "range_std": s.range_std()})
                gt_markers = ([{"pos": gt_ground_pos_ego(frame_gt, o)}
                               for o in frame_gt.objects
                               if o.category in ADP_CLASSES and o.visibility >= 2]
                              if frame_gt else None)
                bev_img = render_bev(bev_tracks, gt_markers, BevCanvas())
                cam_img = cv2.resize(img, (960, 540))
                for t in confirmed:
                    x0, y0, x1, y1 = t.xyxy.astype(int)
                    sx = 960 / 1600
                    s = t.extra.get("bev")
                    label = f"#{t.track_id}"
                    if s is not None:
                        pos_e, _ = s.in_ego(cam.T_global_from_ego)
                        label += f" {np.linalg.norm(pos_e):.0f}m"
                    cv2.rectangle(cam_img, (int(x0 * sx), int(y0 * sx)),
                                  (int(x1 * sx), int(y1 * sx)), (0, 200, 255), 2)
                    cv2.putText(cam_img, label, (int(x0 * sx), int(y0 * sx) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                                cv2.LINE_AA)
                writer.write(np.hstack([cam_img, bev_img]))

        if writer is not None:
            writer.release()
            print(f"demo video: {demo_path}")
        print(f"{scene_name}: {sum(r['scene'] == scene_name for r in records)} "
              f"matched pairs", flush=True)

    # ---- aggregate ----------------------------------------------------------
    def rel_errs(rows, key):
        return np.array([abs(r[key] - r["gt_range"]) / r["gt_range"] for r in rows
                         if np.isfinite(r[key])])

    report = {"n_pairs": len(records), "bins": {}}
    print(f"\n{'bin':>8} | {'n':>4} | {'raw med':>8} {'raw p90':>8} | "
          f"{'kf med':>8} {'kf p90':>8} | {'vel med':>8}")
    for lo, hi in RANGE_BINS:
        rows = [r for r in records if lo <= r["gt_range"] < hi]
        if not rows:
            continue
        raw, kf = rel_errs(rows, "raw_range"), rel_errs(rows, "kf_range")
        vels = np.array([r["vel_err"] for r in rows if "vel_err" in r])
        b = {
            "n": len(rows),
            "raw_rel_median": float(np.median(raw)),
            "raw_rel_p90": float(np.percentile(raw, 90)),
            "kf_rel_median": float(np.median(kf)),
            "kf_rel_p90": float(np.percentile(kf, 90)),
            "vel_err_median": float(np.median(vels)) if len(vels) else None,
        }
        report["bins"][bin_name(lo, hi)] = b
        vel_s = f"{b['vel_err_median']:8.2f}" if b["vel_err_median"] is not None else "     n/a"
        print(f"{bin_name(lo, hi):>8} | {b['n']:>4} | {b['raw_rel_median']:8.1%} "
              f"{b['raw_rel_p90']:8.1%} | {b['kf_rel_median']:8.1%} "
              f"{b['kf_rel_p90']:8.1%} | {vel_s}")

    # Per-scene medians (mini has no real hills; slope breakdown deferred).
    report["per_scene_kf_median"] = {
        s: float(np.median(rel_errs([r for r in records if r["scene"] == s], "kf_range")))
        for s in sorted({r["scene"] for r in records})
    }

    target = report["bins"].get("10-30m", {}).get("kf_rel_median")
    verdict = "MET" if target is not None and target <= 0.15 else "NOT MET"
    report["m3_target"] = {"kf_rel_median_10_30m": target, "verdict": verdict}
    print(f"\nM3 target (median rel err <= 15% @ 10-30m): {target:.1%} -> {verdict}")

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved: {os.path.abspath(OUT_JSON)}")


if __name__ == "__main__":
    main()
