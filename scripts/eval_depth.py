"""M6 decision harness: raw IPM vs Depth Anything V2 metric lift, head to head
on identical matched (track, GT) pairs — the same protocol as M3.

Breakdowns:
- by GT range bin (as M3), and
- by nuScenes visibility level (2-3 = partially occluded vs 4 = fully
  visible), which directly tests the hypothesis that depth fixes IPM's
  occluded-box-bottom failure mode.

Also measures depth latency. Ends with the M6 decision-gate verdict.

Usage: python scripts/eval_depth.py
"""

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector
from adp.eval.matching import gt_ground_pos_ego, match_tracks_to_gt
from adp.lift.depth import DepthLift
from adp.lift.ipm import GroundPlaneLift
from adp.track.bytetrack import ByteTracker

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "out", "eval", "depth_vs_ipm.json")
RANGE_BINS = [(0, 10), (10, 30), (30, 50), (50, np.inf)]


def bin_name(lo, hi):
    return f"{lo:.0f}-{hi:.0f}m" if np.isfinite(hi) else f"{lo:.0f}m+"


def main():
    source = NuScenesSource(verbose=False)
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)
    depth = DepthLift()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    records, depth_ms = [], []
    for scene_name in source.scene_names():
        gt_frames = {f.sample_token: f for f in source.frames(scene_name, min_visibility=1)}
        tracker = ByteTracker()
        prev_ts = None
        for cam in source.camera_frames(scene_name):
            img = cv2.imread(cam.image_path)
            dets = detector(img)
            dt = 1 / 12 if prev_ts is None else (cam.timestamp_us - prev_ts) / 1e6
            prev_ts = cam.timestamp_us
            confirmed = tracker.step(dets, dt)

            if not (cam.is_keyframe and cam.sample_token in gt_frames):
                continue
            frame = gt_frames[cam.sample_token]
            pairs = match_tracks_to_gt(confirmed, frame)
            if not pairs:
                continue

            lift = GroundPlaneLift(cam.camera, cam.T_ego_from_cam)
            t0 = time.perf_counter()
            depth.compute(img)
            depth_ms.append((time.perf_counter() - t0) * 1e3)

            for track, obj in pairs:
                gt_range = float(np.linalg.norm(gt_ground_pos_ego(frame, obj)))
                _, r_ipm = lift.lift_box_bottom(track.xyxy)
                _, r_dep = depth.range_from_box(track.xyxy, cam.camera,
                                               cam.T_ego_from_cam)
                if not (np.isfinite(r_ipm) and np.isfinite(r_dep)):
                    continue
                records.append({
                    "scene": scene_name,
                    "gt_range": gt_range,
                    "ipm": r_ipm,
                    "depth": r_dep,
                    "visibility": obj.visibility,
                    "category": obj.category,
                })
        print(f"{scene_name}: cumulative pairs={len(records)}", flush=True)

    def stats(rows, key):
        errs = np.array([abs(r[key] - r["gt_range"]) / r["gt_range"] for r in rows])
        return {"median": float(np.median(errs)), "p90": float(np.percentile(errs, 90))}

    # Hybrid lift: IPM inside its validated range, depth beyond.
    HYBRID_SWITCH_M = 30.0
    for r in records:
        r["hybrid"] = r["ipm"] if r["ipm"] < HYBRID_SWITCH_M else r["depth"]

    report = {"n_pairs": len(records),
              "depth_ms_p50": float(np.percentile(depth_ms, 50)),
              "hybrid_switch_m": HYBRID_SWITCH_M,
              "by_range": {}, "by_visibility": {}}

    print(f"\n{'bin':>10} | {'n':>4} | {'ipm med':>8} {'ipm p90':>8} | "
          f"{'dep med':>8} {'dep p90':>8} | {'hyb med':>8} | {'gt/dep':>6}")
    for lo, hi in RANGE_BINS:
        rows = [r for r in records if lo <= r["gt_range"] < hi]
        if not rows:
            continue
        # Scale diagnostic: if gt/depth is a constant across bins, the depth
        # error is a calibratable per-camera scale factor (focal mismatch).
        ratio = float(np.median([r["gt_range"] / r["depth"] for r in rows]))
        s = {"n": len(rows), "ipm": stats(rows, "ipm"), "depth": stats(rows, "depth"),
             "hybrid": stats(rows, "hybrid"), "gt_over_depth_median": ratio}
        report["by_range"][bin_name(lo, hi)] = s
        print(f"{bin_name(lo, hi):>10} | {s['n']:>4} | {s['ipm']['median']:8.1%} "
              f"{s['ipm']['p90']:8.1%} | {s['depth']['median']:8.1%} "
              f"{s['depth']['p90']:8.1%} | {s['hybrid']['median']:8.1%} | {ratio:6.2f}")

    print()
    for label, lo_v, hi_v in [("vis 2-3 (occluded)", 2, 3), ("vis 4 (clear)", 4, 4)]:
        rows = [r for r in records
                if lo_v <= r["visibility"] <= hi_v and r["gt_range"] < 50]
        if not rows:
            continue
        s = {"n": len(rows), "ipm": stats(rows, "ipm"), "depth": stats(rows, "depth")}
        report["by_visibility"][label] = s
        print(f"{label:>20} (<50m) | n={s['n']:>4} | ipm {s['ipm']['median']:.1%} "
              f"| depth {s['depth']['median']:.1%}")

    # Decision gate: does depth beat IPM inside the working range?
    key_bins = ["10-30m", "30-50m"]
    wins = sum(report["by_range"][b]["depth"]["median"]
               < report["by_range"][b]["ipm"]["median"] for b in key_bins
               if b in report["by_range"])
    report["verdict"] = {
        "depth_wins_key_bins": int(wins),
        "depth_ms_p50": report["depth_ms_p50"],
        "recommendation": ("adopt-hybrid" if wins == 2 else
                           "adopt-partial" if wins == 1 else "keep-ipm"),
    }
    print(f"\ndepth latency p50: {report['depth_ms_p50']:.0f}ms | "
          f"depth wins {wins}/2 key bins -> {report['verdict']['recommendation']}")

    report["records"] = records  # per-pair dump for later analysis
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved: {os.path.abspath(OUT_JSON)}")


if __name__ == "__main__":
    main()
