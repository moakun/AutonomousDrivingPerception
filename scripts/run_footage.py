"""Run the full pipeline on ordinary video footage — no nuScenes, no
calibration file.

What this costs, stated up front because the overlay will present metres:
- Intrinsics are ESTIMATED from an assumed field of view (or --hfov / --fx).
- Camera height is ASSUMED; monocular video cannot recover metric scale.
- Pitch is estimated from lane vanishing points when lanes are visible.
- There is no localisation, so the BEV filters run in the ego frame: TTC and
  closing speeds remain valid (both are relative), but absolute object speed
  and RSS are unavailable and are suppressed rather than faked.

Usage:
  python scripts/run_footage.py --video path/to/clip.mp4
  python scripts/run_footage.py --video clip.mp4 --hfov 70 --height 1.4
  python scripts/run_footage.py --video clip.mp4 --fx 1266   # known focal
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.calib.camera import CameraModel, SE3
from adp.calib.estimate import (DASHCAM_HEIGHT_M, DASHCAM_HFOV_DEG,
                                estimate_from_frames, intrinsics_from_hfov)
from adp.data.nuscenes_source import CamFrame
from adp.detect.detector import Detector
from adp.lanes.detector import LaneSegmenter
from adp.pipeline import PerceptionPipeline
from adp.viz.compose import CAM_H, CAM_W, compose_frame
from pyquaternion import Quaternion

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out", "footage")


def extrinsics(height_m: float, pitch_deg: float) -> SE3:
    """T_ego_from_cam for a forward-facing camera at `height_m`.

    Ego frame: x forward, y left, z up. Camera: x right, y down, z forward.
    """
    R_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    R_pitch = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-pitch_deg).rotation_matrix
    return SE3.from_rt(R_base @ R_pitch, np.array([0.0, 0.0, height_m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hfov", type=float, default=DASHCAM_HFOV_DEG)
    ap.add_argument("--fx", type=float, default=None,
                    help="known focal length in px; skips the HFOV assumption")
    ap.add_argument("--height", type=float, default=DASHCAM_HEIGHT_M)
    ap.add_argument("--pitch", type=float, default=None,
                    help="known pitch in degrees; skips vanishing-point search")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{os.path.basename(args.video)}: {W}x{H} @ {fps:.1f}fps, "
          f"{n_total} frames")

    segmenter = LaneSegmenter()
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)

    # ---- calibration -------------------------------------------------------
    if args.fx is not None:
        K = np.array([[args.fx, 0, W / 2], [0, args.fx, H / 2], [0, 0, 1.0]])
        hfov = float(np.degrees(2 * np.arctan((W / 2) / args.fx)))
        calib_note = f"f_x supplied ({args.fx:.0f}px -> {hfov:.0f} deg HFOV)"
        pitch = args.pitch or 0.0
    else:
        K = intrinsics_from_hfov(W, H, args.hfov)
        calib_note = f"f_x from assumed {args.hfov:.0f} deg HFOV"
        pitch = args.pitch
    if pitch is None:
        probe = []
        for i in range(0, min(n_total, 240), 24):   # sample ~10 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if ok:
                probe.append(segmenter.lane_mask(img))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        est = estimate_from_frames(probe, W, H, args.hfov, args.height)
        pitch = est.pitch_deg
        calib_note += f"; {est.source.split(';')[1].strip()}"
    print(f"calibration: {calib_note}; height assumed {args.height:.2f}m; "
          f"pitch {pitch:+.2f} deg")

    camera = CameraModel(K=K, width=W, height=H)
    T_ego_cam = extrinsics(args.height, pitch)
    T_glob_ego = SE3.from_rt(np.eye(3), np.zeros(3))   # no localisation

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        OUT_DIR, os.path.splitext(os.path.basename(args.video))[0] + "_adp.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (CAM_W + 540, CAM_H))

    pipeline = PerceptionPipeline(detector=detector, segmenter=segmenter,
                                  localization=False)
    n, ts_us, ranges = 0, 0, []
    while n < args.max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if args.stride > 1 and n % args.stride:
            n += 1
            continue
        cam = CamFrame(scene_name=os.path.basename(args.video),
                       timestamp_us=int(ts_us), image_path="",
                       camera=camera, T_ego_from_cam=T_ego_cam,
                       T_global_from_ego=T_glob_ego,
                       is_keyframe=False, sample_token=None)
        out = pipeline.step(cam, img)
        ranges += [r.range_m for r in out.risk]
        writer.write(compose_frame(img, out, cam))
        ts_us += 1e6 / fps * args.stride
        n += 1
        if n % 50 == 0:
            print(f"  {n} frames, {len(out.risk)} objects, "
                  f"ego risk {out.ego_risk:.2f}", flush=True)
    writer.release()
    cap.release()

    summary = {
        "video": os.path.abspath(args.video), "frames": n,
        "resolution": [W, H], "fps": fps,
        "calibration": {"note": calib_note, "f_x": float(K[0, 0]),
                        "height_m": args.height, "pitch_deg": float(pitch),
                        "localization": False},
        "objects_seen": len(ranges),
        "range_median_m": float(np.median(ranges)) if ranges else None,
        "caveat": ("Ranges rest on an assumed camera height and estimated "
                   "focal length; they are indicative, not validated. "
                   "Absolute speeds and RSS are unavailable without "
                   "localisation."),
        "output_video": os.path.abspath(out_path),
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{n} frames -> {out_path}")
    print(f"objects: {len(ranges)}, median range "
          f"{summary['range_median_m'] or float('nan'):.1f}m")


if __name__ == "__main__":
    main()
