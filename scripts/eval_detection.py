"""M1 baseline: pretrained detectors on nuScenes mini CAM_FRONT keyframes.

Per model: per-class AP@0.5 (vs projected 3D GT, see adp/eval/detection2d.py),
mAP over the 5 target classes, and latency percentiles on this machine's GPU.
Results are printed as a table and saved to out/eval/detection_baseline.json.

Usage: python scripts/eval_detection.py [--models yolov8s.pt rtdetr-l.pt ...]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector
from adp.eval.detection2d import ADP_CLASSES, DetectionEvaluator, gt_boxes_2d

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "out", "eval", "detection_baseline.json")

# (weights, inference imgsz) — imgsz matters: 1600px-wide frames downscaled to
# 640 lose small/distant objects, so the strongest candidate also runs at 960.
DEFAULT_CONFIGS = [
    ("yolov8s.pt", 640),
    ("yolov8m.pt", 640),
    ("yolo11m.pt", 640),
    ("yolo11m.pt", 960),
    ("rtdetr-l.pt", 640),
]


def run_config(source, weights: str, imgsz: int) -> dict:
    name = f"{weights.replace('.pt', '')}@{imgsz}"
    detector = Detector(weights=weights, imgsz=imgsz, conf=0.05)
    evaluator = DetectionEvaluator(iou_thr=0.5)
    inference_ms, total_ms = [], []
    n_frames = 0

    for scene_name in source.scene_names():
        for frame in source.frames(scene_name, min_visibility=1):
            img = cv2.imread(frame.image_path)
            detections = detector(img)
            evaluator.add_frame(detections, gt_boxes_2d(frame))
            if n_frames > 0:  # skip warmup frame in latency stats
                inference_ms.append(detector.last_speed["inference"])
                total_ms.append(sum(detector.last_speed.values()))
            n_frames += 1

    summary = evaluator.summary()
    summary.update(
        model=name,
        n_frames=n_frames,
        latency_ms={
            "inference_p50": float(np.percentile(inference_ms, 50)),
            "inference_p95": float(np.percentile(inference_ms, 95)),
            "total_p50": float(np.percentile(total_ms, 50)),
            "total_p95": float(np.percentile(total_ms, 95)),
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                        help="weights files; default runs the standard candidate set")
    args = parser.parse_args()
    configs = [(m, 640) for m in args.models] if args.models else DEFAULT_CONFIGS

    source = NuScenesSource(verbose=False)
    results = []
    for weights, imgsz in configs:
        print(f"=== {weights} @ imgsz={imgsz} ===", flush=True)
        summary = run_config(source, weights, imgsz)
        results.append(summary)
        print(f"  mAP@0.5={summary['map']:.3f}  "
              f"inference p50={summary['latency_ms']['inference_p50']:.1f}ms", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    header = ["model", *ADP_CLASSES, "mAP", "inf p50", "inf p95", "total p50"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in results:
        aps = [f"{r['ap_per_class'][c]:.3f}" for c in ADP_CLASSES]
        lat = r["latency_ms"]
        print(f"| {r['model']} | " + " | ".join(aps) +
              f" | {r['map']:.3f} | {lat['inference_p50']:.1f} | "
              f"{lat['inference_p95']:.1f} | {lat['total_p50']:.1f} |")
    print(f"\nGT counts: {results[0]['n_gt_per_class']}")
    print(f"Saved: {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
