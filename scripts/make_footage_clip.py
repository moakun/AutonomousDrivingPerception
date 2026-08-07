"""Encode a nuScenes scene as an ordinary video file, so the footage path can
be tested on something with ground truth.

This is the only honest way to measure what running on an uncalibrated dashcam
costs: run_footage.py sees nothing but pixels — no intrinsics, no extrinsics,
no ego pose — and its output can still be scored against nuScenes 3D GT.

Usage: python scripts/make_footage_clip.py --scene scene-0103
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out", "footage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()

    source = NuScenesSource(verbose=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.scene}_raw.mp4")

    writer, n = None, 0
    true_f = None
    for cam in source.camera_frames(args.scene):
        img = cv2.imread(cam.image_path)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
            true_f = cam.camera.K[0, 0]
            true_height = cam.T_ego_from_cam.translation[2]
        writer.write(img)
        n += 1
    writer.release()

    print(f"{n} frames -> {out_path}")
    print(f"(withheld ground truth: f_x = {true_f:.1f}px, "
          f"camera height = {true_height:.2f}m)")


if __name__ == "__main__":
    main()
