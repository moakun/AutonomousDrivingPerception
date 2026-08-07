"""What does running on uncalibrated footage actually cost?

`run_footage.py` works on any video, but it must ASSUME a field of view and a
camera height, and it has no localisation. Those assumptions are usually
unmeasurable — except here, because the test clip was encoded from nuScenes,
so the true calibration is known and 3D GT exists.

The error is decomposed rather than lumped together:
  A  true calibration + localisation      -- the normal system
  B  true calibration, no localisation    -- cost of having no ego pose
  C  ESTIMATED calibration, no localisation -- the real footage condition
  D  C, but with the true camera height   -- isolates the height assumption

Usage: python scripts/eval_footage_calibration.py
"""

import json
import os
import sys

import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.calib.camera import SE3
from adp.calib.estimate import (DASHCAM_HEIGHT_M, DASHCAM_HFOV_DEG,
                                intrinsics_from_hfov)
from adp.eval.provenance import stamp
from adp.eval.replay import ReplayResult, load_cache, replay_scene
from adp.lift.width_prior import WidthPriorLift

CACHE = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                     "replay_cache.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                   "footage_calibration.json")
SCENE = "scene-0103"
EST_PITCH_DEG = -3.93   # what run_footage.py estimated from vanishing points
BINS = ((0, 10), (10, 30), (30, 50), (50, np.inf))


def bin_label(lo, hi):
    return f"{lo:.0f}-{hi:.0f}m" if np.isfinite(hi) else f"{lo:.0f}m+"


def footage_extrinsics(height_m: float, pitch_deg: float) -> SE3:
    """Same construction run_footage.py uses for an unknown camera."""
    R_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    R_pitch = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-pitch_deg).rotation_matrix
    return SE3.from_rt(R_base @ R_pitch, np.array([0.0, 0.0, height_m]))


def main():
    cache = load_cache(CACHE)
    scene = cache["scenes"][SCENE]
    width = WidthPriorLift.from_calibration(use_aspect=True)

    fr0 = scene["frames"][0]
    W, H = fr0["wh"]
    K_true = np.array(fr0["K"])
    T_true = SE3(np.array(fr0["T_ego_cam"]))
    true_height = float(T_true.translation[2])
    K_est = intrinsics_from_hfov(W, H, DASHCAM_HFOV_DEG)

    print(f"scene {SCENE} — calibration truth vs estimate")
    print(f"  f_x   true {K_true[0, 0]:7.1f} px | estimated "
          f"{K_est[0, 0]:7.1f} px  ({K_est[0, 0] / K_true[0, 0] - 1:+.1%})")
    print(f"  height true {true_height:6.2f} m  | assumed   "
          f"{DASHCAM_HEIGHT_M:6.2f} m   "
          f"({DASHCAM_HEIGHT_M / true_height - 1:+.1%})")

    variants = {
        "A true calib + localisation": dict(),
        "B true calib, no localisation": dict(use_localization=False),
        "C estimated calib, no localisation": dict(
            use_localization=False, K_override=K_est,
            T_ego_cam_override=footage_extrinsics(DASHCAM_HEIGHT_M,
                                                  EST_PITCH_DEG)),
        "D estimated calib + true height": dict(
            use_localization=False, K_override=K_est,
            T_ego_cam_override=footage_extrinsics(true_height, EST_PITCH_DEG)),
    }

    report = {"scene": SCENE,
              "calibration": {
                  "f_x_true": float(K_true[0, 0]),
                  "f_x_estimated": float(K_est[0, 0]),
                  "height_true_m": true_height,
                  "height_assumed_m": DASHCAM_HEIGHT_M,
                  "pitch_estimated_deg": EST_PITCH_DEG,
                  "hfov_assumed_deg": DASHCAM_HFOV_DEG},
              "variants": {}}

    print(f"\n{'variant':>36} | {'overall':>8} | " +
          " | ".join(f"{bin_label(lo, hi):>9}" for lo, hi in BINS))
    for label, kwargs in variants.items():
        res: ReplayResult = replay_scene(scene, width, **kwargs)
        row = {"overall": res.median_range_err(),
               "bins": {bin_label(lo, hi): res.median_range_err(lo, hi)
                        for lo, hi in BINS},
               "n": res.n_pairs}
        report["variants"][label] = row
        print(f"{label:>36} | {row['overall']:7.1%} | " +
              " | ".join(f"{row['bins'][bin_label(lo, hi)]:8.1%} "
                         for lo, hi in BINS))

    a = report["variants"]["A true calib + localisation"]["overall"]
    b = report["variants"]["B true calib, no localisation"]["overall"]
    c = report["variants"]["C estimated calib, no localisation"]["overall"]
    d = report["variants"]["D estimated calib + true height"]["overall"]
    report["decomposition"] = {
        "localisation_cost": b - a,
        "calibration_cost": c - b,
        "height_assumption_cost": c - d,
        "total_cost": c - a,
    }
    print(f"\nerror decomposition (median range error, percentage points):")
    print(f"  losing localisation : {100 * (b - a):+.1f} pp")
    print(f"  estimating calib    : {100 * (c - b):+.1f} pp")
    print(f"    of which height   : {100 * (c - d):+.1f} pp")
    print(f"  TOTAL vs calibrated : {100 * (c - a):+.1f} pp "
          f"({a:.1%} -> {c:.1%})")

    # The focal estimate above was lucky: nuScenes CAM_FRONT is ~64 deg HFOV,
    # almost exactly the dashcam assumption. Width-based range scales as
    # 1/f_x, so a wrong FOV maps ~1:1 into range error. Quantify it rather
    # than hoping every camera is this close.
    print("\nsensitivity to the assumed field of view (real dashcams: 55-75 deg):")
    print(f"{'assumed HFOV':>14} | {'f_x':>8} | {'f_x err':>8} | "
          f"{'range err':>9}")
    sweep = []
    for hfov in (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 85.0):
        K = intrinsics_from_hfov(W, H, hfov)
        res = replay_scene(scene, width, use_localization=False, K_override=K,
                           T_ego_cam_override=footage_extrinsics(
                               DASHCAM_HEIGHT_M, EST_PITCH_DEG))
        row = {"hfov_deg": hfov, "f_x": float(K[0, 0]),
               "f_x_rel_err": float(K[0, 0] / K_true[0, 0] - 1),
               "range_err": res.median_range_err()}
        sweep.append(row)
        print(f"{hfov:13.0f}° | {row['f_x']:8.1f} | {row['f_x_rel_err']:+7.1%} | "
              f"{row['range_err']:8.1%}")
    report["hfov_sensitivity"] = sweep

    with open(OUT, "w") as f:
        json.dump(stamp(report), f, indent=2)
    print(f"\nsaved: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
