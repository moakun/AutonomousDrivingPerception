"""M7 end-to-end runner: nuScenes scene -> fully annotated demo video.

Renders the composite view (camera overlay + BEV panel) for any scene using
the complete pipeline. This is the deliverable demo.

Usage:
  python scripts/run_demo.py --scenes scene-0103 scene-1100
  python scripts/run_demo.py --scenes scene-0061 --use-depth
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector
from adp.lanes.detector import LaneSegmenter
from adp.pipeline import PerceptionPipeline
from adp.viz.compose import CAM_H, CAM_W, compose_frame

OUT_DEMO = os.path.join(os.path.dirname(__file__), "..", "out", "demo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=["scene-0103", "scene-1100"])
    parser.add_argument("--use-depth", action="store_true")
    args = parser.parse_args()

    source = NuScenesSource(verbose=False)
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)
    segmenter = LaneSegmenter()
    os.makedirs(OUT_DEMO, exist_ok=True)

    for scene_name in args.scenes:
        pipeline = PerceptionPipeline(detector=detector, segmenter=segmenter,
                                      use_depth=args.use_depth)
        suffix = "_depth" if args.use_depth else ""
        path = os.path.join(OUT_DEMO, f"{scene_name}_demo{suffix}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12,
                                 (CAM_W + 540, CAM_H))
        for cam in source.camera_frames(scene_name):
            img = cv2.imread(cam.image_path)
            out = pipeline.step(cam, img)
            writer.write(compose_frame(img, out, cam))
        writer.release()
        print(f"{scene_name} -> {path}", flush=True)


if __name__ == "__main__":
    main()
