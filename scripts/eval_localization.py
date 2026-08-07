"""M10 / improvement #4: how much does the system lean on dataset localisation?

The BEV Kalman filters live in the GLOBAL frame (M3), using nuScenes
`ego_pose` — ground-truth localisation a camera-only system would not have.
That makes the headline velocity numbers partly a measure of nuScenes' SLAM,
which is not an honest basis for claiming camera-only performance.

Two questions, both answered by replaying the same cached detections:

1. How good does localisation have to be? Sweep position and yaw noise on the
   ASSUMED pose and watch velocity error respond. Yaw is expected to dominate:
   a small heading error rotates every object's global position by
   range * sin(dyaw), and velocity is the time-derivative of that.
2. What does the global frame buy at all? The `no localisation` variant tracks
   in the ego frame instead — the design M3 rejected because ego rotation
   makes parked cars appear to accelerate.

Usage: python scripts/eval_localization.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource
from adp.eval.replay import ReplayResult, load_cache, replay_scene
from adp.lift.width_prior import WidthPriorLift

CACHE = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                     "replay_cache.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                   "localization.json")

# (label, position sigma m, yaw sigma deg) — roughly: perfect, good RTK-GNSS,
# consumer GNSS + IMU, poor / urban-canyon multipath.
VARIANTS = [
    ("GT pose (current)", 0.0, 0.0),
    ("pos 0.1m, yaw 0.2deg", 0.1, 0.2),
    ("pos 0.3m, yaw 0.5deg", 0.3, 0.5),
    ("pos 1.0m, yaw 1.0deg", 1.0, 1.0),
    ("pos 0.0m, yaw 2.0deg", 0.0, 2.0),   # isolates heading error
    ("pos 3.0m, yaw 0.0deg", 3.0, 0.0),   # isolates position error
]
BINS = ((0, 10), (10, 30), (30, 50))


def merge(results):
    out = ReplayResult()
    for r in results:
        out.range_rel_err += r.range_rel_err
        out.vel_err += r.vel_err
        out.n_pairs += r.n_pairs
    return out


def main():
    cache = load_cache(CACHE)
    width = WidthPriorLift.from_calibration(use_aspect=True)

    # GT annotation velocities, for the velocity comparison.
    source = NuScenesSource(verbose=False)
    gt_vel = {}
    for scene in cache["scenes"].values():
        for rows in scene["gt"].values():
            for r in rows:
                v = source.nusc.box_velocity(r["token"])[:2]
                if np.all(np.isfinite(v)):
                    gt_vel[r["token"]] = v.tolist()
    print(f"{len(gt_vel)} annotations with GT velocity\n")

    report = {"variants": []}
    print(f"{'localisation':>22} | {'range err':>9} | {'vel err':>8} | "
          f"{'vel <30m':>8} | {'vel 30-50m':>10}")
    for label, pos_s, yaw_s in VARIANTS:
        res = merge([replay_scene(sc, width, pos_sigma_m=pos_s,
                                  yaw_sigma_deg=yaw_s, gt_velocity=gt_vel,
                                  seed=abs(hash(label)) % 2**31)
                     for sc in cache["scenes"].values()])
        row = {"label": label, "pos_sigma_m": pos_s, "yaw_sigma_deg": yaw_s,
               "range_err": res.median_range_err(),
               "vel_err": res.median_vel_err(),
               "vel_err_by_bin": {f"{lo}-{hi}m": res.median_vel_err(lo, hi)
                                  for lo, hi in BINS},
               "n": res.n_pairs}
        report["variants"].append(row)
        print(f"{label:>22} | {row['range_err']:8.1%} | {row['vel_err']:7.2f} | "
              f"{row['vel_err_by_bin']['10-30m']:7.2f} | "
              f"{row['vel_err_by_bin']['30-50m']:9.2f}")

    # What the global frame buys: ego-frame tracking cannot produce absolute
    # velocity at all, so only range is comparable.
    res_ego = merge([replay_scene(sc, width, use_localization=False)
                     for sc in cache["scenes"].values()])
    report["no_localization"] = {"range_err": res_ego.median_range_err(),
                                 "n": res_ego.n_pairs,
                                 "note": "ego-frame tracking: velocity is "
                                         "relative, not absolute"}
    print(f"\n{'no localisation':>22} | {res_ego.median_range_err():8.1%} | "
          "velocity is ego-relative only (absolute speed unrecoverable)")

    base = report["variants"][0]
    print("\nvelocity error vs the GT-pose baseline:")
    for row in report["variants"][1:]:
        print(f"  {row['label']:>22}: {row['vel_err']:.2f} m/s "
              f"({row['vel_err'] / base['vel_err']:.2f}x)")

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
