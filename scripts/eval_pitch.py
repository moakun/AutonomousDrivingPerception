"""M10 / improvement #3: how much does the flat-ground assumption cost?

mini contains no hill scenes, but the failure mode can be measured directly by
perturbing the ASSUMED camera pitch and re-running the lift. A pitch error is
exactly what a crest, a dip, a banked curve, or braking-induced nose-dive
produces, and it is also what a stale calibration looks like.

The comparison that matters after M9: IPM-only vs the fused lift. IPM's range
comes from where a ray meets an assumed ground plane, so it is directly
exposed to pitch; the width prior gets range from apparent size and camera
intrinsics, so it should barely care. This quantifies how much the M9 lift
incidentally hardened the system against non-flat ground.

Usage: python scripts/eval_pitch.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.eval.replay import ReplayResult, load_cache, replay_scene
from adp.lift.width_prior import WidthPriorLift

CACHE = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                     "replay_cache.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                   "pitch_sensitivity.json")
PITCHES = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
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
    report = {"pitch_deg": list(PITCHES), "variants": {}}

    for label, wp in (("IPM only", None), ("fused (M9 default)", width)):
        rows = []
        print(f"\n{label}")
        print(f"{'pitch':>7} | {'overall':>8} | " +
              " | ".join(f"{f'{lo}-{hi}m':>9}" for lo, hi in BINS))
        for pitch in PITCHES:
            res = merge([replay_scene(sc, wp, pitch_deg=pitch)
                         for sc in cache["scenes"].values()])
            row = {"pitch": pitch, "overall": res.median_range_err(),
                   "bins": {f"{lo}-{hi}m": res.median_range_err(lo, hi)
                            for lo, hi in BINS},
                   "n": res.n_pairs}
            rows.append(row)
            print(f"{pitch:+6.1f}° | {row['overall']:7.1%} | " +
                  " | ".join(f"{row['bins'][f'{lo}-{hi}m']:8.1%} "
                             for lo, hi in BINS))
        report["variants"][label] = rows

    # Degradation summary: error at +/-2 deg relative to nominal.
    print("\ndegradation at +/-2° vs nominal (overall median error):")
    for label, rows in report["variants"].items():
        base = next(r["overall"] for r in rows if r["pitch"] == 0.0)
        worst = max(r["overall"] for r in rows if abs(r["pitch"]) <= 2.0)
        print(f"  {label:>20}: {base:.1%} -> {worst:.1%} "
              f"({worst / base:.1f}x)")
        report["variants"][label] = {"rows": rows, "nominal": base,
                                     "worst_2deg": worst,
                                     "degradation_factor": worst / base}

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
