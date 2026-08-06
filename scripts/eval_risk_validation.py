"""M8: does the risk score predict what a human driver did?

The risk layer has no ground truth, so this uses a behavioural proxy: hard
braking within a horizon, derived offline from ego pose. The question that
matters for the project's thesis is not just "is the score predictive" but
"does metric grounding beat the pixel-area heuristic the plan called
worthless" — so the risk score is scored against four baselines.

Stage 1 (default) runs the pipeline once and caches per-keyframe predictor
values plus each scene's ego kinematics. Stage 2 computes labels + metrics
from that cache, so threshold/horizon sweeps cost nothing (--from-cache).

Usage:
  python scripts/eval_risk_validation.py
  python scripts/eval_risk_validation.py --from-cache --accel-thresh -3.0
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
from adp.eval.braking import EgoKinematics, ego_kinematics, label_hard_braking
from adp.eval.ranking import average_precision, bootstrap_ci, roc_auc
from adp.lanes.detector import LaneSegmenter
from adp.pipeline import PerceptionPipeline

CACHE = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                     "risk_validation_cache.json")
OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                        "risk_validation.json")
IMG_AREA = 1600 * 900
FAR_M = 200.0

# name -> (row -> score), higher = riskier. All rank-based, so monotone
# transforms are irrelevant; the negations below are only for readability.
PREDICTORS = {
    "risk score (ours)": lambda r: r["ego_risk"],
    "box area (strawman)": lambda r: r["max_box_area_frac"],
    "nearest object range": lambda r: -r["nearest_range"],
    "inverse TTC only": lambda r: r["max_inv_ttc"],
    "ego speed only": lambda r: r["ego_speed_ms"],
}


def collect(args) -> dict:
    source = NuScenesSource(verbose=False)
    detector = Detector(weights="yolo11m.pt", imgsz=960, conf=0.1)
    segmenter = LaneSegmenter()
    cache = {"scenes": {}}

    for scene_name in source.scene_names():
        pipeline = PerceptionPipeline(detector=detector, segmenter=segmenter)
        ts, pos, keyframes = [], [], []

        for i, cam in enumerate(source.camera_frames(scene_name)):
            img = cv2.imread(cam.image_path)
            out = pipeline.step(cam, img)
            ts.append(cam.timestamp_us)
            pos.append(cam.T_global_from_ego.translation[:2].tolist())

            if not cam.is_keyframe:
                continue
            areas = [float(np.prod(t.xyxy[2:] - t.xyxy[:2]) / IMG_AREA)
                     for t in out.tracks]
            ranges = [r.range_m for r in out.risk]
            inv_ttcs = [1.0 / r.ttc.ttc_s for r in out.risk
                        if r.ttc is not None and r.ttc.trustworthy and r.ttc.ttc_s > 0]
            keyframes.append({
                "idx": i,
                "ego_risk": out.ego_risk,
                "max_box_area_frac": max(areas, default=0.0),
                "nearest_range": min(ranges, default=FAR_M),
                "max_inv_ttc": max(inv_ttcs, default=0.0),
                "ego_speed_ms": out.ego_speed_ms,
                "n_objects": len(out.risk),
            })

        kin = ego_kinematics(np.array(ts), np.array(pos))
        cache["scenes"][scene_name] = {
            "t_s": kin.t_s.tolist(),
            "speed_ms": kin.speed_ms.tolist(),
            "accel_ms2": kin.accel_ms2.tolist(),
            "keyframes": keyframes,
        }
        print(f"{scene_name}: {len(keyframes)} keyframes, "
              f"min accel {kin.accel_ms2.min():.2f} m/s^2", flush=True)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f)
    print(f"cached: {os.path.abspath(CACHE)}")
    return cache


def build_dataset(cache: dict, horizon_s: float, accel_thresh: float,
                  min_speed_ms: float):
    """-> (rows, labels) over valid keyframes across all scenes."""
    rows, labels = [], []
    for scene_name, sc in cache["scenes"].items():
        kin = EgoKinematics(t_s=np.array(sc["t_s"]),
                            speed_ms=np.array(sc["speed_ms"]),
                            accel_ms2=np.array(sc["accel_ms2"]))
        lab, valid = label_hard_braking(kin, horizon_s=horizon_s,
                                        accel_thresh=accel_thresh,
                                        min_speed_ms=min_speed_ms)
        for kf in sc["keyframes"]:
            i = kf["idx"]
            if i < len(lab) and valid[i]:
                rows.append({**kf, "scene": scene_name})
                labels.append(int(lab[i]))
    return rows, np.array(labels, dtype=int)


def evaluate(cache, horizon_s, accel_thresh, min_speed_ms, n_boot=1000):
    rows, labels = build_dataset(cache, horizon_s, accel_thresh, min_speed_ms)
    n, n_pos = len(labels), int(labels.sum())
    base_rate = n_pos / n if n else float("nan")
    result = {
        "config": {"horizon_s": horizon_s, "accel_thresh_ms2": accel_thresh,
                   "min_speed_ms": min_speed_ms},
        "n_valid_keyframes": n, "n_positive": n_pos, "base_rate": base_rate,
        "predictors": {},
    }
    if n_pos == 0 or n_pos == n:
        return result, rows

    for name, fn in PREDICTORS.items():
        scores = np.array([fn(r) for r in rows], dtype=float)
        auc = roc_auc(scores, labels)
        ap = average_precision(scores, labels)
        result["predictors"][name] = {
            "roc_auc": auc,
            "roc_auc_ci": bootstrap_ci(roc_auc, scores, labels, n_boot=n_boot),
            "avg_precision": ap,
            "ap_lift_over_base_rate": ap / base_rate,
        }
    return result, rows


def print_table(res):
    print(f"\n--- braking within {res['config']['horizon_s']:.0f}s at "
          f"{res['config']['accel_thresh_ms2']:.1f} m/s^2 "
          f"(ego >= {res['config']['min_speed_ms']:.0f} m/s) ---")
    print(f"{res['n_valid_keyframes']} valid keyframes, {res['n_positive']} positive "
          f"(base rate {res['base_rate']:.1%})")
    if not res["predictors"]:
        print("  degenerate labels — no metrics")
        return
    print(f"\n{'predictor':>22} | {'ROC-AUC':>8} | {'95% CI':>14} | "
          f"{'AP':>6} | {'lift':>5}")
    for name, m in sorted(res["predictors"].items(),
                          key=lambda kv: -kv[1]["roc_auc"]):
        lo, hi = m["roc_auc_ci"]
        print(f"{name:>22} | {m['roc_auc']:8.3f} | [{lo:.2f}, {hi:.2f}]".ljust(56)
              + f" | {m['avg_precision']:6.3f} | {m['ap_lift_over_base_rate']:5.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true",
                        help="skip the pipeline run, recompute metrics from cache")
    parser.add_argument("--horizon-s", type=float, default=3.0)
    parser.add_argument("--accel-thresh", type=float, default=-2.0)
    parser.add_argument("--min-speed", type=float, default=2.0)
    args = parser.parse_args()

    if args.from_cache:
        with open(CACHE) as f:
            cache = json.load(f)
    else:
        cache = collect(args)

    primary, rows = evaluate(cache, args.horizon_s, args.accel_thresh, args.min_speed)
    print_table(primary)

    # Sensitivity: the headline must not be an artifact of one threshold/horizon.
    sweep = []
    for thresh in (-1.5, -2.0, -3.0):
        for horizon in (2.0, 3.0, 5.0):
            res, _ = evaluate(cache, horizon, thresh, args.min_speed, n_boot=200)
            sweep.append(res)
            if not res["predictors"]:
                continue
            ours = res["predictors"]["risk score (ours)"]["roc_auc"]
            straw = res["predictors"]["box area (strawman)"]["roc_auc"]
            print(f"  thresh {thresh:>4} m/s^2, horizon {horizon:.0f}s: "
                  f"base {res['base_rate']:5.1%} | ours {ours:.3f} | "
                  f"strawman {straw:.3f}")

    report = {"primary": primary, "sensitivity": sweep,
              "caveats": [
                  "Hard braking is a behavioural proxy, not ground-truth danger: "
                  "drivers also brake for lights, stop signs and turns.",
                  "10 scenes is a small sample — see bootstrap CIs.",
                  "Labels are acausal (offline smoothing over the whole scene); "
                  "the perception path never sees them.",
              ]}
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {os.path.abspath(OUT_JSON)}")


if __name__ == "__main__":
    main()
