"""M5 evaluation: full pipeline (detect -> track -> lift -> lanes -> risk)
over all scenes.

Reports: lane-zone distribution, TTC availability/trustworthiness, RSS
violations, ego-risk stats, and the top-N highest-risk moments with their
reason strings (the plan's sanity review: are the high-risk moments the
obviously-dangerous ones?). Renders a demo video with risk-colored boxes and
the why-text, plus the BEV panel.

Usage: python scripts/eval_risk.py [--demo-scene scene-0103]
"""

import argparse
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.detect.detector import Detector
from adp.lanes.detector import LaneSegmenter
from adp.pipeline import PerceptionPipeline
from adp.viz.bev import BevCanvas, render_bev

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "out", "eval", "risk_report.json")
OUT_DEMO = os.path.join(os.path.dirname(__file__), "..", "out", "demo")

BUCKET_COLORS = {"low": (90, 200, 90), "caution": (0, 200, 255), "high": (0, 60, 255)}


def sample_line(line, n=24):
    xs = np.linspace(line.x_range[0], min(line.x_range[1], 45.0), n)
    return np.stack([xs, line.y_at(xs)], axis=1)


def render_frame(img, out, cam):
    sx = 960 / 1600
    cam_img = cv2.resize(img, (960, 540))
    by_id = {r.track_id: r for r in out.risk}
    for t in out.tracks:
        r = by_id.get(t.track_id)
        color = BUCKET_COLORS[r.bucket] if r else (150, 150, 150)
        x0, y0, x1, y1 = (t.xyxy * sx).astype(int)
        cv2.rectangle(cam_img, (x0, y0), (x1, y1), color, 2)
        label = f"#{t.track_id}"
        if r:
            label += f" {r.range_m:.0f}m {r.score:.2f}"
        cv2.putText(cam_img, label, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    # Why-text for the top-risk object.
    if out.risk and out.risk[0].score >= 0.3:
        top = out.risk[0]
        why = (f"#{top.track_id} {top.category}: " + "; ".join(top.reasons)
               ).replace("±", "+-")  # cv2 putText has no unicode glyphs
        cv2.putText(cam_img, why[:110], (8, 528), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, BUCKET_COLORS[top.bucket], 1, cv2.LINE_AA)
    cv2.putText(cam_img, f"ego risk {out.ego_risk:.2f}  v={out.ego_speed_ms:.1f} m/s",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    bev_tracks = []
    for t in out.tracks:
        s = t.extra.get("bev")
        if s is None:
            continue
        pos_e, vel_e = s.in_ego(cam.T_global_from_ego)
        r = by_id.get(t.track_id)
        bev_tracks.append({"id": t.track_id, "category": t.category, "pos": pos_e,
                           "vel": vel_e, "range_std": s.range_std()})
    corr = out.corridor if out.corridor.valid else None
    bev = render_bev(
        bev_tracks, canvas=BevCanvas(),
        lanes=[sample_line(l) for l in out.lane_lines],
        ego_corridor=((sample_line(corr.left), sample_line(corr.right))
                      if corr else None),
    )
    return np.hstack([cam_img, bev])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-scene", default="scene-0103")
    args = parser.parse_args()

    source = NuScenesSource(verbose=False)
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)
    segmenter = LaneSegmenter()
    os.makedirs(OUT_DEMO, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    zone_counts = Counter()
    corridor_sources = Counter()
    ttc_total = ttc_trust = rss_viol = intent_events = 0
    ego_risks, timings = [], []
    top_moments = []

    for scene_name in source.scene_names():
        pipeline = PerceptionPipeline(detector=detector, segmenter=segmenter)
        writer = None
        if scene_name == args.demo_scene:
            demo_path = os.path.join(OUT_DEMO, f"{scene_name}_risk.mp4")
            writer = cv2.VideoWriter(demo_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     12, (960 + 540, 540))

        for cam in source.camera_frames(scene_name):
            img = cv2.imread(cam.image_path)
            out = pipeline.step(cam, img)
            timings.append(sum(out.timings_ms.values()))

            if cam.is_keyframe:
                ego_risks.append(out.ego_risk)
                for r in out.risk:
                    zone_counts[r.zone.value] += 1
                    corridor_sources[r.corridor_source] += 1
                    if r.ttc is not None:
                        ttc_total += 1
                        ttc_trust += r.ttc.trustworthy
                    rss_viol += r.rss_violated
                    intent_events += r.intent_cross_s is not None
                if out.risk and out.risk[0].score >= 0.3:
                    top = out.risk[0]
                    top_moments.append({
                        "scene": scene_name, "ts": cam.timestamp_us,
                        "track": top.track_id, "category": top.category,
                        "score": round(top.score, 3), "reasons": top.reasons,
                    })

            if writer is not None:
                writer.write(render_frame(img, out, cam))

        if writer is not None:
            writer.release()
            print(f"demo video: {demo_path}")
        print(f"{scene_name} done", flush=True)

    top_moments.sort(key=lambda m: -m["score"])
    report = {
        "zone_distribution": dict(zone_counts),
        "corridor_source": dict(corridor_sources),
        "ttc": {"computed": ttc_total, "trustworthy": ttc_trust},
        "rss_violations": rss_viol,
        "crossing_intent_events": intent_events,
        "ego_risk": {"mean": float(np.mean(ego_risks)),
                     "p95": float(np.percentile(ego_risks, 95)),
                     "frames_high": int(sum(r >= 0.6 for r in ego_risks)),
                     "n_keyframes": len(ego_risks)},
        "pipeline_ms_p50": float(np.percentile(timings, 50)),
        "pipeline_ms_p95": float(np.percentile(timings, 95)),
        "top_moments": top_moments[:10],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nzones: {dict(zone_counts)}")
    print(f"corridor source: {dict(corridor_sources)}")
    print(f"TTC: {ttc_total} computed, {ttc_trust} trustworthy")
    print(f"RSS violations: {rss_viol} | crossing-intent events: {intent_events}")
    print(f"ego risk mean {report['ego_risk']['mean']:.2f}, "
          f"p95 {report['ego_risk']['p95']:.2f}, "
          f"high frames {report['ego_risk']['frames_high']}/{report['ego_risk']['n_keyframes']}")
    print(f"pipeline p50 {report['pipeline_ms_p50']:.0f}ms p95 {report['pipeline_ms_p95']:.0f}ms")
    print("\ntop moments:")
    for m in report["top_moments"][:5]:
        print(f"  [{m['score']:.2f}] {m['scene']} #{m['track']} {m['category']}: "
              + "; ".join(m["reasons"]))
    print(f"\nsaved: {os.path.abspath(OUT_JSON)}")


if __name__ == "__main__":
    main()
