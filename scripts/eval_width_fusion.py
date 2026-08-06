"""M9 decision harness: raw IPM vs width-prior vs fused, on identical matched
(track, GT) pairs — same protocol as M3/M6 so the numbers are comparable.

The breakdown that decides it is by nuScenes visibility: the fused lift exists
to rescue partially-occluded objects (visibility 2-3), and must not damage
fully-visible ones (visibility 4).

Lifts run on every 12Hz frame so the fused lift's per-track occlusion EMA
accumulates exactly as it would in deployment; comparisons are recorded at
keyframes where GT exists.

Usage: python scripts/eval_width_fusion.py
"""

import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector
from adp.eval.matching import gt_ground_pos_ego, match_tracks_to_gt
from adp.lift.fused import FusedLift
from adp.lift.ipm import GroundPlaneLift
from adp.lift.width_prior import WidthPriorLift
from adp.track.bytetrack import ByteTracker

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                        "width_fusion.json")
RANGE_BINS = [(0, 10), (10, 30), (30, 50), (50, np.inf)]


def bin_name(lo, hi):
    return f"{lo:.0f}-{hi:.0f}m" if np.isfinite(hi) else f"{lo:.0f}m+"


def rel_err(rows, key):
    e = [abs(r[key] - r["gt_range"]) / r["gt_range"] for r in rows
         if r[key] is not None and np.isfinite(r[key])]
    return (float(np.median(e)), float(np.percentile(e, 90)), len(e)) if e else (
        float("nan"), float("nan"), 0)


def main():
    source = NuScenesSource(verbose=False)
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)
    width = WidthPriorLift.from_calibration(use_aspect=True)
    print(f"priors: { {k: round(v['width_median_m'], 2) for k, v in width.priors.items()} }")

    records, sources = [], Counter()
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

            ipm = GroundPlaneLift(cam.camera, cam.T_ego_from_cam)
            fused = FusedLift(ipm, width)
            # Run every frame so the per-track occlusion EMA behaves as deployed.
            for t in confirmed:
                xy_f, r_f = fused.lift_box_bottom(t.xyxy, t.category, t.extra)
                _, r_w, _ = width.lift_box(t.xyxy, t.category, cam.camera,
                                           cam.T_ego_from_cam)
                t.extra["_last"] = (r_f, fused.last_source, r_w)

            if not (cam.is_keyframe and cam.sample_token in gt_frames):
                continue
            frame = gt_frames[cam.sample_token]
            for track, obj in match_tracks_to_gt(confirmed, frame):
                last = track.extra.get("_last")
                if last is None:
                    continue
                r_fused, src, r_width = last
                _, r_ipm = ipm.lift_box_bottom(track.xyxy)
                if not np.isfinite(r_ipm):
                    continue
                sources[src] += 1
                records.append({
                    "scene": scene_name, "category": obj.category,
                    "visibility": obj.visibility,
                    "gt_range": float(np.linalg.norm(gt_ground_pos_ego(frame, obj))),
                    "ipm": float(r_ipm),
                    "width": float(r_width) if np.isfinite(r_width) else None,
                    "fused": float(r_fused) if np.isfinite(r_fused) else None,
                    "source": src,
                })
        print(f"{scene_name}: {len(records)} pairs cumulative", flush=True)

    report = {"n_pairs": len(records), "source_counts": dict(sources),
              "by_range": {}, "by_visibility": {}}

    print(f"\nlift source usage: {dict(sources)}")
    print(f"\n{'bin':>10} | {'n':>4} | {'ipm med':>8} {'p90':>7} | "
          f"{'width med':>9} {'p90':>7} | {'fused med':>9} {'p90':>7}")
    for lo, hi in RANGE_BINS:
        rows = [r for r in records if lo <= r["gt_range"] < hi]
        if not rows:
            continue
        i_m, i_p, _ = rel_err(rows, "ipm")
        w_m, w_p, _ = rel_err(rows, "width")
        f_m, f_p, _ = rel_err(rows, "fused")
        report["by_range"][bin_name(lo, hi)] = {
            "n": len(rows), "ipm": [i_m, i_p], "width": [w_m, w_p],
            "fused": [f_m, f_p]}
        print(f"{bin_name(lo, hi):>10} | {len(rows):>4} | {i_m:7.1%} {i_p:6.1%} | "
              f"{w_m:8.1%} {w_p:6.1%} | {f_m:8.1%} {f_p:6.1%}")

    print()
    for label, vis in (("occluded (vis 2-3)", (2, 3)), ("clear (vis 4)", (4, 4))):
        rows = [r for r in records
                if vis[0] <= r["visibility"] <= vis[1] and r["gt_range"] < 50]
        if not rows:
            continue
        i_m, i_p, _ = rel_err(rows, "ipm")
        w_m, _, _ = rel_err(rows, "width")
        f_m, f_p, _ = rel_err(rows, "fused")
        report["by_visibility"][label] = {
            "n": len(rows), "ipm_median": i_m, "width_median": w_m,
            "fused_median": f_m, "ipm_p90": i_p, "fused_p90": f_p}
        delta = (f_m - i_m) / i_m
        print(f"{label:>20} (<50m) n={len(rows):>4} | ipm {i_m:.1%} | "
              f"width {w_m:.1%} | fused {f_m:.1%} ({delta:+.0%} vs ipm)")

    occ = report["by_visibility"].get("occluded (vis 2-3)", {})
    clear = report["by_visibility"].get("clear (vis 4)", {})
    helps = occ and occ["fused_median"] < occ["ipm_median"]
    harms = clear and clear["fused_median"] > clear["ipm_median"] * 1.05
    report["verdict"] = {
        "improves_occluded": bool(helps),
        "damages_clear": bool(harms),
        "recommendation": ("adopt" if helps and not harms else
                           "adopt-occluded-only" if helps else "keep-ipm"),
    }
    print(f"\nverdict: {report['verdict']['recommendation']}")

    report["records"] = records
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved: {os.path.abspath(OUT_JSON)}")


if __name__ == "__main__":
    main()
