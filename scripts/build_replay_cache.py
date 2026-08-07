"""M10 support: cache detections + calibration + GT once, so robustness
experiments can replay the tracker/lift/Kalman under many variants without
re-running the detector.

Detection and tracking are unaffected by the things M10 perturbs (camera
extrinsics, ego pose), so one GPU pass supports every variant and each replay
is CPU-only and fast.

Usage: python scripts/build_replay_cache.py
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector
from adp.eval.detection2d import ADP_CLASSES, project_box_2d
from adp.eval.matching import gt_ground_pos_ego

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                   "replay_cache.json")


def main():
    source = NuScenesSource(verbose=False)
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)
    cache = {"scenes": {}}

    for scene_name in source.scene_names():
        gt_frames = {f.sample_token: f
                     for f in source.frames(scene_name, min_visibility=1)}
        frames, gt_out = [], {}

        for cam in source.camera_frames(scene_name):
            img = cv2.imread(cam.image_path)
            dets = detector(img)
            frames.append({
                "ts": cam.timestamp_us,
                "key": bool(cam.is_keyframe),
                "sample": cam.sample_token,
                "K": np.asarray(cam.camera.K).round(4).tolist(),
                "wh": [cam.camera.width, cam.camera.height],
                "T_ego_cam": cam.T_ego_from_cam.matrix.round(6).tolist(),
                "T_glob_ego": cam.T_global_from_ego.matrix.round(6).tolist(),
                "dets": [[*np.round(d.xyxy, 2).tolist(), round(d.score, 4),
                          d.category] for d in dets],
            })
            if not (cam.is_keyframe and cam.sample_token in gt_frames):
                continue
            frame = gt_frames[cam.sample_token]
            rows = []
            for obj in frame.objects:
                if obj.category not in ADP_CLASSES or obj.visibility < 2:
                    continue
                rect = project_box_2d(frame, obj)
                if rect is None:
                    continue
                xy = gt_ground_pos_ego(frame, obj)
                rows.append({
                    "inst": obj.instance_token,
                    "xyxy": np.round(rect, 2).tolist(),
                    "xy_ego": np.round(xy, 3).tolist(),
                    "range": round(float(np.linalg.norm(xy)), 3),
                    "vis": obj.visibility,
                    "cat": obj.category,
                    "token": obj.token,
                })
            gt_out[cam.sample_token] = rows

        cache["scenes"][scene_name] = {"frames": frames, "gt": gt_out}
        print(f"{scene_name}: {len(frames)} frames, {len(gt_out)} keyframes",
              flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(cache, f)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"\nsaved: {os.path.abspath(OUT)} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
