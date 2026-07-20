"""M1: export a detector to ONNX and verify it against the PyTorch model.

Checks two things on real nuScenes frames:
1. Numeric parity — boxes/scores from ONNX Runtime match PyTorch outputs.
2. Latency — ONNX Runtime (CPU EP here; TensorRT/CUDA EP is a later
   optimization) vs PyTorch GPU, so the deployment cost is a measured number.

Order matters: the ultralytics exporter sets CUDA_VISIBLE_DEVICES=-1 for the
export pass, killing CUDA for the rest of the process — so all PyTorch GPU
inference runs BEFORE the export, and the ONNX (CPU) pass runs after.

Usage: python scripts/export_onnx.py --weights yolo11m.pt --imgsz 960
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector


def collect(detector, images):
    """Run detector over images; return per-image detections + inference ms."""
    all_dets, ms = [], []
    for i, img in enumerate(images):
        all_dets.append(detector(img))
        if i > 0:  # skip warmup frame in latency stats
            ms.append(detector.last_speed["inference"])
    return all_dets, ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--n-frames", type=int, default=20)
    args = parser.parse_args()

    source = NuScenesSource(verbose=False)
    images = []
    for scene_name in source.scene_names()[:4]:
        it = source.frames(scene_name)
        images.extend(cv2.imread(next(it).image_path) for _ in range(args.n_frames // 4))

    # Rectangular inference shape for 1600x900 frames at this imgsz (stride 32):
    # export must use the SAME shape PyTorch predicts at, or preprocessing
    # differs (square export letterboxes differently) and parity is meaningless.
    rect = (round(args.imgsz * 900 / 1600 / 32) * 32, args.imgsz)  # (h, w), e.g. (544, 960)

    # 1) PyTorch on GPU (must run before export — see module docstring).
    pytorch_det = Detector(weights=args.weights, imgsz=rect, conf=0.25)
    dets_pt, pt_ms = collect(pytorch_det, images)

    # 2) Export.
    onnx_path = pytorch_det.model.export(format="onnx", imgsz=list(rect), dynamic=False)
    print(f"exported: {onnx_path} (input {rect[0]}x{rect[1]})")

    # 3) Same ultralytics predict API over the ONNX file: identical pre/post-
    #    processing, only the backend differs.
    onnx_det = Detector(weights=onnx_path, imgsz=rect, conf=0.25, device="cpu")
    dets_ort, ort_ms = collect(onnx_det, images)

    # Compare via IoU matching at a working threshold, so near-conf-threshold
    # flicker (expected across fp32 backends) doesn't scramble the pairing.
    WORK_CONF = 0.35
    n_pairs, n_unmatched = 0, 0
    box_errs, score_errs = [], []
    for a_list, b_list in zip(dets_pt, dets_ort):
        a_list = [d for d in a_list if d.score >= WORK_CONF]
        b_list = [d for d in b_list if d.score >= WORK_CONF]
        used = np.zeros(len(b_list), dtype=bool)
        for a in sorted(a_list, key=lambda d: -d.score):
            best_j, best_iou = -1, 0.5
            for j, b in enumerate(b_list):
                if used[j] or b.category != a.category:
                    continue
                lt = np.maximum(a.xyxy[:2], b.xyxy[:2])
                rb = np.minimum(a.xyxy[2:], b.xyxy[2:])
                inter = np.prod(np.clip(rb - lt, 0, None))
                union = (np.prod(a.xyxy[2:] - a.xyxy[:2])
                         + np.prod(b.xyxy[2:] - b.xyxy[:2]) - inter)
                iou = inter / (union + 1e-9)
                if iou > best_iou:
                    best_j, best_iou = j, iou
            if best_j >= 0:
                used[best_j] = True
                b = b_list[best_j]
                n_pairs += 1
                box_errs.append(float(np.max(np.abs(a.xyxy - b.xyxy))))
                score_errs.append(abs(a.score - b.score))
            else:
                n_unmatched += 1
        n_unmatched += int((~used).sum())

    unmatched_frac = n_unmatched / max(n_pairs + n_unmatched, 1)
    print(f"\nframes compared: {len(images)} (working conf >= {WORK_CONF})")
    print(f"matched pairs: {n_pairs}, unmatched either side: {n_unmatched} "
          f"({unmatched_frac:.1%})")
    print(f"box diff px: median {np.median(box_errs):.3f}, max {np.max(box_errs):.3f}")
    print(f"score diff: median {np.median(score_errs):.5f}, max {np.max(score_errs):.5f}")
    print(f"latency p50: pytorch-gpu {np.percentile(pt_ms, 50):.1f} ms | "
          f"onnxruntime-cpu {np.percentile(ort_ms, 50):.1f} ms")

    ok = unmatched_frac < 0.05 and np.median(box_errs) < 1.0 and np.median(score_errs) < 0.01
    print("PARITY:", "OK" if ok else "FAILED — investigate before adopting ONNX")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
