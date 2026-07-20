"""M4 evaluation: YOLOP lane mask -> BEV fit -> ego corridor, over all scenes.

No lane GT exists in nuScenes mini (map expansion not included), so per the
plan's exit criteria this reports degradation statistics + qualitative demo:
- valid-ego-corridor fraction per scene (raw and after temporal smoothing)
- lane-width distribution of valid corridors (sanity: ~3.5m urban lanes)
- lane-line count distribution
- demo video: image overlay + BEV with corridor shading

Usage: python scripts/eval_lanes.py [--demo-scene scene-0061]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.lanes.bev_lanes import cluster_and_fit, find_ego_corridor, mask_to_bev_points
from adp.lanes.detector import LaneSegmenter
from adp.lanes.smoothing import CorridorSmoother
from adp.lift.ipm import GroundPlaneLift
from adp.viz.bev import BevCanvas, render_bev

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "out", "eval", "lanes_report.json")
OUT_DEMO = os.path.join(os.path.dirname(__file__), "..", "out", "demo")


def sample_line(line, n=30):
    xs = np.linspace(line.x_range[0], min(line.x_range[1], 45.0), n)
    return np.stack([xs, line.y_at(xs)], axis=1)


def project_polyline_to_image(poly_ego, cam_frame):
    """(N,2) ego ground polyline -> (N,2) pixel points (may include off-image)."""
    pts_ego = np.hstack([poly_ego, np.zeros((len(poly_ego), 1))])
    pts_cam = cam_frame.T_ego_from_cam.inverse().apply(pts_ego)
    uv, depth = cam_frame.camera.project(pts_cam)
    return uv[depth > 0.5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-scene", default="scene-0061")
    args = parser.parse_args()

    source = NuScenesSource(verbose=False)
    segmenter = LaneSegmenter()
    os.makedirs(OUT_DEMO, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    report = {}
    for scene_name in source.scene_names():
        smoother = CorridorSmoother()
        n_frames = raw_valid = smooth_valid = 0
        widths, line_counts, seg_ms = [], [], []
        prev_ts = None

        writer = None
        if scene_name == args.demo_scene:
            demo_path = os.path.join(OUT_DEMO, f"{scene_name}_lanes.mp4")
            writer = cv2.VideoWriter(demo_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     12, (960 + 540, 540))

        for cam in source.camera_frames(scene_name):
            # Stats at keyframes only; demo renders every frame.
            if not cam.is_keyframe and writer is None:
                continue
            img = cv2.imread(cam.image_path)
            dt = 1 / 12 if prev_ts is None else (cam.timestamp_us - prev_ts) / 1e6
            prev_ts = cam.timestamp_us

            t0 = cv2.getTickCount()
            mask = segmenter.lane_mask(img)
            seg_ms.append((cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1e3)

            lift = GroundPlaneLift(cam.camera, cam.T_ego_from_cam)
            points = mask_to_bev_points(mask, lift)
            lines = cluster_and_fit(points)
            corridor = find_ego_corridor(lines)
            smoothed = smoother.update(corridor, dt)

            if cam.is_keyframe:
                n_frames += 1
                raw_valid += corridor.valid
                smooth_valid += smoothed.valid
                line_counts.append(len(lines))
                if corridor.valid:
                    widths.append(corridor.width)

            if writer is not None:
                cam_img = cv2.resize(img, (960, 540))
                cam_img[cv2.resize(mask.astype(np.uint8), (960, 540)) > 0] = (60, 200, 60)
                shown = smoothed if smoothed.valid else corridor
                if shown.valid:
                    for line in (shown.left, shown.right):
                        uv = project_polyline_to_image(sample_line(line), cam) * (960 / 1600)
                        for a, b in zip(uv[:-1].astype(int), uv[1:].astype(int)):
                            cv2.line(cam_img, tuple(a), tuple(b), (0, 220, 220), 3,
                                     cv2.LINE_AA)
                bev = render_bev(
                    [], canvas=BevCanvas(),
                    lanes=[sample_line(l) for l in lines],
                    ego_corridor=((sample_line(shown.left), sample_line(shown.right))
                                  if shown.valid else None),
                )
                status = "corridor OK" if shown.valid else "NO CORRIDOR"
                cv2.putText(bev, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 220, 220) if shown.valid else (0, 0, 255), 2)
                writer.write(np.hstack([cam_img, bev]))

        if writer is not None:
            writer.release()
            print(f"demo video: {demo_path}")

        report[scene_name] = {
            "n_keyframes": n_frames,
            "raw_valid_frac": raw_valid / max(n_frames, 1),
            "smooth_valid_frac": smooth_valid / max(n_frames, 1),
            "width_median": float(np.median(widths)) if widths else None,
            "width_iqr": [float(np.percentile(widths, 25)),
                          float(np.percentile(widths, 75))] if widths else None,
            "lines_median": float(np.median(line_counts)) if line_counts else 0,
            "seg_ms_p50": float(np.median(seg_ms)),
        }
        r = report[scene_name]
        wm = f"{r['width_median']:.2f}m" if r["width_median"] else "n/a"
        print(f"{scene_name}: corridor raw={r['raw_valid_frac']:.0%} "
              f"smoothed={r['smooth_valid_frac']:.0%} width_med={wm} "
              f"lines_med={r['lines_median']:.0f} seg={r['seg_ms_p50']:.0f}ms",
              flush=True)

    valid_fracs = [r["smooth_valid_frac"] for r in report.values()]
    all_widths = [r["width_median"] for r in report.values() if r["width_median"]]
    report["OVERALL"] = {
        "smooth_valid_frac_mean": float(np.mean(valid_fracs)),
        "width_median_of_medians": float(np.median(all_widths)) if all_widths else None,
    }
    print(f"\nOVERALL: smoothed corridor valid {report['OVERALL']['smooth_valid_frac_mean']:.0%} "
          f"of keyframes; width median {report['OVERALL']['width_median_of_medians']:.2f}m")

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved: {os.path.abspath(OUT_JSON)}")


if __name__ == "__main__":
    main()
