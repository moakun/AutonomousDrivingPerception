"""M9 step 3: honest noise models for both lift cues, computed offline from
the cached calibration records (no GPU, no re-run).

Two jobs:

1. LEAVE-ONE-SCENE-OUT validation of the width prior. The priors in
   width_prior.json were fitted on the same scenes the M9 harness evaluated
   on, so its width numbers are optimistic by an unknown amount. Refitting per
   held-out scene measures that optimism. Comparison is on depth z (what the
   cue predicts directly); relative error in z is relative error in range.

2. EMPIRICAL IPM noise model. `GroundPlaneLift.range_meas_std` implements the
   r^2 pixel-noise law, which is the right SHAPE but only counts pixel jitter
   — it ignores the systematic bottom-edge error (occlusion, non-flat ground,
   box bottom not at the true contact point). That optimism is why
   inverse-variance fusion over-trusted IPM. Fitting sigma(r) to the observed
   errors makes the fusion weights honest.

Usage: python scripts/calibrate_lift_noise.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                       "width_prior.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "out", "eval",
                        "lift_noise.json")
CLASSES = ("car", "truck", "bus", "pedestrian", "cyclist")


def fit_priors(rows):
    """Same fit as calibrate_width_prior.py, on an arbitrary subset."""
    priors = {}
    for cat in CLASSES:
        sub = [r for r in rows if r["category"] == cat]
        if len(sub) < 20:
            continue
        w = np.array([r["apparent_width_m"] for r in sub])
        asp = np.array([r["aspect"] for r in sub])
        keep = (w > np.percentile(w, 5)) & (w < np.percentile(w, 95))
        b, a = np.polyfit(asp[keep], w[keep], 1)
        resid = w[keep] - (a + b * asp[keep])
        priors[cat] = {"width_median_m": float(np.median(w)),
                       "linear_a": float(a), "linear_b": float(b),
                       "sigma_conditional_m": float(np.std(resid))}
    return priors


def predict_z(rec, priors, conditioned=True):
    p = priors.get(rec["category"])
    if p is None:
        return None
    if conditioned:
        w_eff = p["linear_a"] + p["linear_b"] * rec["aspect"]
        w_eff = float(np.clip(w_eff, 0.4 * p["width_median_m"],
                              2.5 * p["width_median_m"]))
    else:
        w_eff = p["width_median_m"]
    return rec["f_x"] * w_eff / rec["w_px"]


def main():
    with open(IN_PATH) as f:
        data = json.load(f)
    records = data["records"]
    clean = [r for r in records
             if r["visibility"] == 4 and not r["clipped"] and r["gt_z"] > 3.0]
    scenes = sorted({r["scene"] for r in records})

    # ---- 1. leave-one-scene-out width validation ---------------------------
    print("Leave-one-scene-out validation of the width prior")
    print(f"{'held-out scene':>16} | {'n':>4} | {'in-sample':>10} | {'held-out':>9}")
    in_all, out_all = [], []
    priors_full = fit_priors(clean)
    for scene in scenes:
        train = [r for r in clean if r["scene"] != scene]
        test = [r for r in records
                if r["scene"] == scene and not r["clipped"] and r["gt_z"] > 3.0]
        if len(test) < 10:
            continue
        priors_loo = fit_priors(train)
        e_in, e_out = [], []
        for rec in test:
            z_in = predict_z(rec, priors_full)
            z_out = predict_z(rec, priors_loo)
            if z_in:
                e_in.append(abs(z_in - rec["gt_z"]) / rec["gt_z"])
            if z_out:
                e_out.append(abs(z_out - rec["gt_z"]) / rec["gt_z"])
        if not e_out:
            continue
        in_all += e_in
        out_all += e_out
        print(f"{scene:>16} | {len(e_out):>4} | {np.median(e_in):9.1%} | "
              f"{np.median(e_out):8.1%}")
    optimism = np.median(out_all) / np.median(in_all) - 1
    print(f"\npooled: in-sample {np.median(in_all):.1%}, "
          f"held-out {np.median(out_all):.1%} ({optimism:+.0%} optimism)")

    # ---- 2. empirical IPM noise model --------------------------------------
    ipm_rows = [r for r in records
                if r.get("ipm_range") and np.isfinite(r["ipm_range"])]
    print(f"\nEmpirical IPM error vs range (n={len(ipm_rows)})")
    print(f"{'range bin':>12} | {'n':>4} | {'bias':>7} | {'robust sd':>9} | "
          f"{'sd/r':>6}")
    bins, sds, centres = [], [], []
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 100)]:
        sub = [r for r in ipm_rows if lo <= r["gt_range"] < hi]
        if len(sub) < 15:
            continue
        err = np.array([r["ipm_range"] - r["gt_range"] for r in sub])
        med = float(np.median(err))
        # Robust scale (MAD-based); the innovation gate handles the tail.
        sd = float(1.4826 * np.median(np.abs(err - med)))
        centre = float(np.median([r["gt_range"] for r in sub]))
        bins.append({"lo": lo, "hi": hi, "n": len(sub), "bias_m": med,
                     "robust_sd_m": sd, "centre_m": centre})
        sds.append(sd)
        centres.append(centre)
        print(f"{f'{lo}-{hi}m':>12} | {len(sub):>4} | {med:+6.2f}m | "
              f"{sd:8.2f}m | {sd / centre:5.1%}")

    # sigma(r) = c * r^p, fitted in log space.
    p, logc = np.polyfit(np.log(centres), np.log(sds), 1)
    c = float(np.exp(logc))
    print(f"\nfitted sigma_ipm(r) = {c:.4g} * r^{p:.2f}")
    print("(the analytic pixel-noise model assumes p = 2.0 with "
          f"c = 3/(f*h) = {3 / (1266 * 1.5):.4g})")
    for r in (10, 20, 30, 50):
        analytic = np.clip(3 * r**2 / (1266 * 1.5), 0.3, 8.0)
        print(f"   r={r:>2}m: analytic {analytic:5.2f}m vs measured "
              f"{c * r**p:5.2f}m")

    out = {"width_loo": {"in_sample_median": float(np.median(in_all)),
                         "held_out_median": float(np.median(out_all)),
                         "optimism_frac": float(optimism)},
           "ipm_noise": {"bins": bins, "model_c": c, "model_p": float(p)}}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
