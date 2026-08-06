"""M9 step 1: measure how tight the apparent-width cue actually is.

For a pinhole camera, an object of apparent metric width W at depth z spans
w_px = f_x * W / z, so W = w_px * z / f_x. Running that over matched
(detection, GT) pairs gives the real distribution of apparent width per class
— the prior AND its spread, which is the noise model.

The complication is yaw: a car seen from behind spans its width (~1.8m), from
the side its length (~4.5m). A 2D detector doesn't know yaw, but box aspect
ratio is a proxy for it, so this also tests whether conditioning on aspect
tightens the estimate enough to be worth the complexity.

Calibration hygiene: only fully-visible (visibility 4) objects, and boxes not
touching the image border, since both occlusion and clipping truncate width.

Usage: python scripts/calibrate_width_prior.py
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector
from adp.eval.matching import gt_ground_pos_ego, match_tracks_to_gt
from adp.lift.ipm import GroundPlaneLift
from adp.track.bytetrack import ByteTracker

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "eval", "width_prior.json")
EDGE_PX = 8


def main():
    source = NuScenesSource(verbose=False)
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)
    records = []

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
            lift = GroundPlaneLift(cam.camera, cam.T_ego_from_cam)
            for track, obj in match_tracks_to_gt(confirmed, frame):
                x0, y0, x1, y1 = track.xyxy
                w_px, h_px = x1 - x0, y1 - y0
                if w_px < 8 or h_px < 8:
                    continue
                clipped = (x0 < EDGE_PX or y0 < EDGE_PX
                           or x1 > cam.camera.width - EDGE_PX
                           or y1 > cam.camera.height - EDGE_PX)
                gt_z = float(obj.center_cam[2])          # depth along optical axis
                f_x = float(cam.camera.K[0, 0])
                _, r_ipm = lift.lift_box_bottom(track.xyxy)
                records.append({
                    "scene": scene_name,
                    "category": obj.category,
                    "visibility": obj.visibility,
                    "clipped": bool(clipped),
                    "w_px": float(w_px),
                    "aspect": float(w_px / h_px),
                    "gt_z": gt_z,
                    "gt_range": float(np.linalg.norm(gt_ground_pos_ego(frame, obj))),
                    "apparent_width_m": float(w_px * gt_z / f_x),
                    "f_x": f_x,
                    "ipm_range": float(r_ipm) if np.isfinite(r_ipm) else None,
                })
        print(f"{scene_name}: {len(records)} pairs cumulative", flush=True)

    clean = [r for r in records
             if r["visibility"] == 4 and not r["clipped"] and r["gt_z"] > 3.0]
    print(f"\n{len(records)} pairs, {len(clean)} clean (vis 4, unclipped)\n")

    priors = {}
    print(f"{'class':>11} | {'n':>4} | {'width m':>18} | {'rel spread':>10} | "
          f"{'aspect corr':>11}")
    for cat in ("car", "truck", "bus", "pedestrian", "cyclist"):
        rows = [r for r in clean if r["category"] == cat]
        if len(rows) < 20:
            print(f"{cat:>11} | {len(rows):>4} | too few for a prior")
            continue
        w = np.array([r["apparent_width_m"] for r in rows])
        asp = np.array([r["aspect"] for r in rows])
        med = float(np.median(w))
        p25, p75 = np.percentile(w, [25, 75])
        # Robust relative spread: IQR-based sigma / median.
        sigma = float((p75 - p25) / 1.349)
        corr = float(np.corrcoef(asp, w)[0, 1])
        # Linear model apparent_width ~ a + b*aspect, fit robustly-ish on the
        # central 90% to test whether aspect conditioning is worth it.
        keep = (w > np.percentile(w, 5)) & (w < np.percentile(w, 95))
        b, a = np.polyfit(asp[keep], w[keep], 1)
        resid = w[keep] - (a + b * asp[keep])
        sigma_cond = float(np.std(resid))
        priors[cat] = {
            "n": len(rows), "width_median_m": med, "sigma_m": sigma,
            "rel_spread": sigma / med, "aspect_corr": corr,
            "linear_a": float(a), "linear_b": float(b),
            "sigma_conditional_m": sigma_cond,
            "rel_spread_conditional": sigma_cond / med,
        }
        print(f"{cat:>11} | {len(rows):>4} | {med:6.2f} "
              f"(sd {sigma:.2f}) | {sigma / med:9.1%} | {corr:11.2f}")
        print(f"{'':>11} |      | conditional on aspect: sd {sigma_cond:.2f} "
              f"-> {sigma_cond / med:.1%}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"priors": priors, "n_pairs": len(records),
                   "n_clean": len(clean), "records": records}, f, indent=2)
    print(f"\nsaved: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
