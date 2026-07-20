"""M2 evaluation: yolo11m@960 + ByteTracker at 12Hz, scored at 2Hz keyframes.

Reports MOTA / IDF1 / ID switches / recall / precision per scene and overall,
saves out/eval/tracking_baseline.json, and renders a demo video with track IDs
and motion trails for one scene.

Usage: python scripts/eval_tracking.py [--demo-scene scene-0061]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector
from adp.eval.detection2d import ADP_CLASSES, project_box_2d
from adp.eval.tracking import MotAccumulator
from adp.track.bytetrack import ByteTracker

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "out", "eval", "tracking_baseline.json")
OUT_DEMO = os.path.join(os.path.dirname(__file__), "..", "out", "demo")

ID_COLORS = [(255, 160, 0), (0, 200, 255), (0, 220, 120), (200, 100, 255),
             (0, 80, 255), (255, 80, 80), (255, 220, 0), (120, 255, 255)]


def keyframe_gt(frame):
    """((instance_token, xyxy) list, range list) for visible target-class GT."""
    out, ranges = [], []
    for obj in frame.objects:
        if obj.category not in ADP_CLASSES or obj.visibility < 2:
            continue
        rect = project_box_2d(frame, obj)
        if rect is not None:
            out.append((obj.instance_token, rect))
            ranges.append(obj.range_m)
    return out, ranges


def run_scene(source, detector, scene_name, gt_frames, tracker_kwargs, writer=None):
    tracker = ByteTracker(**tracker_kwargs)
    acc = MotAccumulator()
    trails: dict[int, list] = {}
    prev_ts = None
    track_ms = []

    for cam in source.camera_frames(scene_name):
        img = cv2.imread(cam.image_path)
        dets = detector(img)
        dt = 1 / 12 if prev_ts is None else (cam.timestamp_us - prev_ts) / 1e6
        prev_ts = cam.timestamp_us

        t0 = time.perf_counter()
        confirmed = tracker.step(dets, dt)
        track_ms.append((time.perf_counter() - t0) * 1e3)

        if cam.is_keyframe and cam.sample_token in gt_frames:
            gt, ranges = keyframe_gt(gt_frames[cam.sample_token])
            acc.add_keyframe(
                gt=gt,
                tracks=[(t.track_id, t.xyxy) for t in confirmed],
                gt_ranges=ranges,
            )

        if writer is not None:
            for t in confirmed:
                color = ID_COLORS[t.track_id % len(ID_COLORS)]
                x0, y0, x1, y1 = t.xyxy.astype(int)
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                cv2.putText(img, f"#{t.track_id} {t.category}", (x0, y0 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                center = (int((x0 + x1) / 2), int(y1))
                trails.setdefault(t.track_id, []).append(center)
                pts = trails[t.track_id][-24:]  # ~2s trail
                for a, b in zip(pts, pts[1:]):
                    cv2.line(img, a, b, color, 2)
            writer.write(img)

    summary = acc.summary()
    return summary, acc.recall_by_range, float(np.percentile(track_ms, 50))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-scene", default=None,
                        help="scene name to render as demo video (slower)")
    parser.add_argument("--high-thresh", type=float, default=0.5)
    parser.add_argument("--new-track-thresh", type=float, default=0.6)
    parser.add_argument("--max-misses", type=int, default=12)
    parser.add_argument("--tag", default="baseline", help="output filename suffix")
    args = parser.parse_args()
    tracker_kwargs = dict(
        high_thresh=args.high_thresh,
        new_track_thresh=args.new_track_thresh,
        max_misses=args.max_misses,
    )

    source = NuScenesSource(verbose=False)
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)

    os.makedirs(OUT_DEMO, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    results = {}
    for scene_name in source.scene_names():
        gt_frames = {f.sample_token: f for f in source.frames(scene_name, min_visibility=1)}

        writer = None
        if scene_name == args.demo_scene:
            demo_path = os.path.join(OUT_DEMO, f"{scene_name}_tracking.mp4")
            writer = cv2.VideoWriter(
                demo_path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (1600, 900)
            )

        summary, recall_rng, track_p50 = run_scene(
            source, detector, scene_name, gt_frames, tracker_kwargs, writer
        )
        if writer is not None:
            writer.release()
            print(f"demo video: {demo_path}")

        results[scene_name] = {**summary.__dict__, "recall_by_range": recall_rng,
                               "tracker_ms_p50": track_p50}
        print(f"{scene_name}: MOTA={summary.mota:.3f} IDF1={summary.idf1:.3f} "
              f"IDsw={summary.id_switches} recall={summary.recall:.3f} "
              f"precision={summary.precision:.3f} "
              f"recall<30m={recall_rng['0-30m']:.3f}", flush=True)

    # Overall = GT-weighted aggregate of per-scene counts.
    n_gt = sum(r["n_gt"] for r in results.values())
    n_fp = sum(r["fp"] for r in results.values())
    n_fn = sum(r["fn"] for r in results.values())
    n_sw = sum(r["id_switches"] for r in results.values())
    idf1_w = sum(r["idf1"] * r["n_gt"] for r in results.values()) / n_gt
    # GT-weighted mean of per-scene range recalls (NaN bins excluded).
    recall_rng_overall = {}
    for bin_name in results[next(iter(results))]["recall_by_range"]:
        vals = [(r["recall_by_range"][bin_name], r["n_gt"]) for r in results.values()
                if not np.isnan(r["recall_by_range"][bin_name])]
        recall_rng_overall[bin_name] = sum(v * w for v, w in vals) / sum(w for _, w in vals)
    overall_row = {
        "mota": 1.0 - (n_fn + n_fp + n_sw) / n_gt,
        "idf1_gt_weighted": idf1_w,
        "id_switches": n_sw,
        "n_gt": n_gt, "fp": n_fp, "fn": n_fn,
        "recall": (n_gt - n_fn) / n_gt,
        "recall_by_range": recall_rng_overall,
        "config": tracker_kwargs,
    }
    results["OVERALL"] = overall_row
    print(f"\nOVERALL: MOTA={overall_row['mota']:.3f} "
          f"IDF1(w)={idf1_w:.3f} IDsw={n_sw} recall={overall_row['recall']:.3f}")
    print("recall by range: " + "  ".join(
        f"{k}={v:.3f}" for k, v in recall_rng_overall.items()))

    out_path = OUT_JSON.replace("baseline", args.tag)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
