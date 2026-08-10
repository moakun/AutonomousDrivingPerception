"""M7: consolidate every milestone's metrics into REPORT.md — the
per-condition evaluation report a skeptical reader can audit.

Reads out/eval/*.json (produced by the M1-M6 harnesses), classifies scenes
day/night from nuScenes descriptions, and writes REPORT.md at the repo root.

Usage: python scripts/make_report.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource

ROOT = os.path.join(os.path.dirname(__file__), "..")
EVAL = os.path.join(ROOT, "out", "eval")


def load(name):
    with open(os.path.join(EVAL, name)) as f:
        return json.load(f)


def classify_scenes(source):
    cond = {}
    for s in source.nusc.scene:
        d = s["description"].lower()
        cond[s["name"]] = "night" if "night" in d else "day"
    return cond


def wavg(pairs):
    """[(value, weight)] -> weighted mean, or None."""
    pairs = [(v, w) for v, w in pairs if v is not None and np.isfinite(v) and w > 0]
    if not pairs:
        return None
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


def pct(x, digits=1):
    return f"{x:.{digits}%}" if x is not None else "n/a"


def num(x, digits=2):
    return f"{x:.{digits}f}" if x is not None else "n/a"


def main():
    source = NuScenesSource(verbose=False)
    cond = classify_scenes(source)
    day = [s for s, c in cond.items() if c == "day"]
    night = [s for s, c in cond.items() if c == "night"]

    det = load("detection_per_scene.json")
    det_all = load("detection_baseline.json")
    trk = load("tracking_baseline.json")
    ipm = load("ipm_validation.json")
    dep = load("depth_vs_ipm.json")
    lanes = load("lanes_report.json")
    risk = load("risk_report.json")

    L = []
    L.append("# ADP Evaluation Report\n")
    L.append("All numbers measured on nuScenes v1.0-mini (10 scenes, 404 keyframes, "
             "CAM_FRONT), RTX 4060 Laptop GPU. Every table below is reproducible "
             "from the referenced harness script and its JSON in `out/eval/`.\n")
    L.append("**Honest-numbers preamble:** 2D GT boxes are projections of 3D "
             "boxes (systematically loose → absolute AP/MOTA read low); mini has "
             "no rain or hill scenes (slope failure mode untested); detector and "
             "lane model are pretrained only, no fine-tuning (404 frames is too "
             "few). Conditions: "
             f"{len(day)} day scenes, {len(night)} night scenes.\n")

    L.append("## 1. Pipeline & latency\n")
    L.append("```\nframe -> yolo11m@960 (2D det) -> ByteTrack (12Hz) -> IPM lift "
             "-> BEV Kalman (global frame)\n      -> YOLOP lanes -> BEV corridor "
             "  -> risk: TTC+CI | lane zone | ped intent | RSS\n```\n")
    L.append("| module | p50 ms | note |\n|---|---|---|\n")
    L.append("| detection (yolo11m@960, PyTorch GPU) | ~18 | ONNX parity verified (max 0.06px) |\n")
    L.append("| tracking + IPM lift + BEV KF | ~1 | |\n")
    L.append("| lanes (YOLOP + BEV fit) | ~27 | can run at half rate |\n")
    L.append("| depth (optional hybrid, off by default) | ~39 | DA-V2-metric fp16 |\n")
    L.append(f"| **end-to-end** | **{risk['pipeline_ms_p50']:.0f}** "
             f"(p95 {risk['pipeline_ms_p95']:.0f}) | staleness at 30 m/s: "
             f"~{risk['pipeline_ms_p50'] / 1000 * 30:.1f}m (p95 "
             f"{risk['pipeline_ms_p95'] / 1000 * 30:.1f}m) |\n")

    L.append("\n## 2. Detection (AP@0.5 vs projected 3D GT)\n")
    L.append("Model selection table: `out/eval/detection_baseline.json` "
             "(yolo11m@960 selected: mAP 0.419 @ ~18ms).\n")
    L.append("| condition | mAP@0.5 (GT-weighted) | car | pedestrian |\n|---|---|---|---|\n")
    for label, scenes in (("day", day), ("night", night)):
        w = lambda s: sum(det[s]["n_gt_per_class"].values())
        m = wavg([(det[s]["map"], w(s)) for s in scenes])
        car = wavg([(det[s]["ap_per_class"]["car"], det[s]["n_gt_per_class"]["car"])
                    for s in scenes])
        ped = wavg([(det[s]["ap_per_class"]["pedestrian"],
                     det[s]["n_gt_per_class"]["pedestrian"]) for s in scenes])
        L.append(f"| {label} ({len(scenes)} scenes) | {num(m, 3)} | {num(car, 3)} "
                 f"| {num(ped, 3)} |\n")

    L.append("\n## 3. Tracking (12Hz sweeps, scored at 2Hz keyframes)\n")
    L.append("| condition | MOTA | IDF1 | IDsw | recall <30m | 30-50m | 50m+ |\n"
             "|---|---|---|---|---|---|---|\n")
    for label, scenes in (("day", day), ("night", night)):
        rows = [trk[s] for s in scenes]
        n = sum(r["n_gt"] for r in rows)
        mota = 1 - sum(r["fn"] + r["fp"] + r["id_switches"] for r in rows) / n
        idf1 = wavg([(r["idf1"], r["n_gt"]) for r in rows])
        sw = sum(r["id_switches"] for r in rows)
        rec = {b: wavg([(r["recall_by_range"][b], r["n_gt"]) for r in rows])
               for b in ("0-30m", "30-50m", "50m+")}
        L.append(f"| {label} | {num(mota, 3)} | {num(idf1, 3)} | {sw} | "
                 f"{pct(rec['0-30m'])} | {pct(rec['30-50m'])} | {pct(rec['50m+'])} |\n")

    L.append("\n## 4. Metric lift (range error vs 3D GT)\n")
    L.append("Current default lift: **IPM fused with a class width prior** "
             "(M9) — IPM below 12m, width beyond. M3 target: median relative "
             "error <=15% at 10-30m — **met: "
             f"{pct(ipm['m3_target']['kf_rel_median_10_30m'])}** (KF-smoothed).\n")
    L.append("\nEnd-to-end (Kalman-smoothed, current system):\n\n")
    L.append("| GT range | n | range err median | p90 | velocity err median |\n"
             "|---|---|---|---|---|\n")
    for b, s in ipm["bins"].items():
        vel = (f"{s['vel_err_median']:.2f} m/s"
               if s.get("vel_err_median") is not None else "n/a")
        L.append(f"| {b} | {s['n']} | {pct(s['kf_rel_median'])} | "
                 f"{pct(s['kf_rel_p90'])} | {vel} |\n")

    try:
        wf = load("width_fusion.json")
    except FileNotFoundError:
        wf = None
    if wf:
        L.append(f"\n**M9 — why the lift changed** (raw single-frame estimates, "
                 f"identical matched pairs, n={wf['n_pairs']}):\n\n")
        L.append("| GT range | IPM alone (old) | width alone | **fused (new)** |\n"
                 "|---|---|---|---|\n")
        for b, s in wf["by_range"].items():
            L.append(f"| {b} | {pct(s['ipm'][0])} | {pct(s['width'][0])} | "
                     f"**{pct(s['fused'][0])}** |\n")
        L.append("\nBy occlusion level (<50m): ")
        L.append("; ".join(
            f"{k}: IPM {pct(v['ipm_median'])} -> fused {pct(v['fused_median'])}"
            for k, v in wf["by_visibility"].items()) + ".\n")
        L.append("\nThe width cue survives leave-one-scene-out validation "
                 "(8.2% held-out vs 7.5% in-sample), so it is not calibration "
                 "contamination. Two measured corrections made it work: IPM's "
                 "analytic noise model was 4-6x too optimistic (real sigma "
                 "grows as r^2.51 with +11.5m bias at 40-60m), and fusion is "
                 "dropped entirely beyond 12m because inverse-variance fusion "
                 "assumes unbiased inputs.\n")

    L.append(f"\n**M6 (superseded) — monocular depth comparison**, "
             f"n={dep['n_pairs']}. The width prior beats this hybrid beyond "
             "10m at ~39ms/frame lower cost, so depth is retained only for "
             "reproducing the comparison:\n\n")
    L.append("| GT range | IPM med | depth med | depth hybrid | gt/depth ratio |\n"
             "|---|---|---|---|---|\n")
    for b, s in dep["by_range"].items():
        L.append(f"| {b} | {pct(s['ipm']['median'])} | {pct(s['depth']['median'])} "
                 f"| {pct(s['hybrid']['median'])} | {num(s['gt_over_depth_median'])} |\n")
    L.append("\nPer-condition raw-IPM median error (<50m), from per-pair records:\n")
    L.append("| condition | n pairs | IPM med | depth med |\n|---|---|---|---|\n")
    for label, scenes in (("day", day), ("night", night)):
        rows = [r for r in dep["records"] if r["scene"] in scenes and r["gt_range"] < 50]
        if not rows:
            continue
        e = lambda k: float(np.median([abs(r[k] - r["gt_range"]) / r["gt_range"]
                                       for r in rows]))
        L.append(f"| {label} | {len(rows)} | {pct(e('ipm'))} | {pct(e('depth'))} |\n")
    L.append("\n*Note: night IPM error reading better than day is a "
             "survivorship artifact — night detection only yields matched pairs "
             "on close, well-lit objects, so the night sample is skewed toward "
             "easy ranges (see the night recall drop in table 3).*\n")
    L.append("\nVelocity is validated against nuScenes annotation velocities "
             "(table above). The TTC trust gate tracks this curve and was "
             "widened from 30m to 50m after M9, because 30-50m error is now "
             "better than 10-30m was when the gate was first set.\n")

    L.append("\n## 5. Lanes (no vector GT in mini -> degradation stats)\n")
    L.append("| condition | corridor valid (smoothed) | width median |\n|---|---|---|\n")
    for label, scenes in (("day", day), ("night", night)):
        rows = [lanes[s] for s in scenes]
        v = wavg([(r["smooth_valid_frac"], r["n_keyframes"]) for r in rows])
        wd = [r["width_median"] for r in rows if r["width_median"]]
        L.append(f"| {label} | {pct(v, 0)} | "
                 f"{num(float(np.median(wd)) if wd else None)}m |\n")
    L.append("\nZero-validity scenes are parking lots / intersection idling / "
             "dark night — 'no corridor' is the correct output there; the risk "
             "layer falls back to a fixed ±1.85m corridor and records the "
             "fallback per object.\n")

    L.append("\n## 6. Risk layer\n")
    z = risk["zone_distribution"]
    L.append(f"- Records at keyframes: {sum(z.values())} "
             f"(ego {z.get('ego', 0)}, adjacent {z.get('adjacent', 0)}, "
             f"off-path {z.get('off_path', 0)})\n")
    L.append(f"- TTC: {risk['ttc']['computed']} computed, "
             f"{risk['ttc']['trustworthy']} trustworthy (50m gate, re-derived "
             "from the M9 lift error curve; CI from KF covariances via the "
             "delta method)\n")
    L.append(f"- RSS (arXiv:1708.06374) violations: {risk['rss_violations']}; "
             f"pedestrian crossing-intent events: {risk['crossing_intent_events']}\n")
    L.append(f"- Corridor source: {risk['corridor_source']}\n")
    L.append(f"- Ego risk: mean {risk['ego_risk']['mean']:.2f}, high on "
             f"{risk['ego_risk']['frames_high']}/{risk['ego_risk']['n_keyframes']} keyframes\n")
    L.append("  (M8b lowered these from mean 0.29 / 13 high frames: the old "
             "score charged every in-lane object 0.30 merely for existing, so "
             "warnings are now rarer and correspond to genuinely short TTC. "
             "Bucket thresholds were left at 0.30/0.60 — display calibration "
             "was deliberately NOT tuned against the braking labels.)\n")
    L.append("\nTop audited moments (all reasons machine-generated):\n")
    for m in risk["top_moments"][:5]:
        L.append(f"- [{m['score']:.2f}] {m['scene']} #{m['track']} "
                 f"{m['category']}: " + "; ".join(m["reasons"]) + "\n")

    try:
        rv = load("risk_validation.json")
    except FileNotFoundError:
        rv = None
    if rv:
        L.append("\n## 7. Does the risk output predict human braking? (M8)\n")
        L.append("The risk layer has no ground truth — nuScenes labels objects, not "
                 "danger. Proxy signal: hard braking within a horizon, derived "
                 "offline from ego pose (acausal; the perception path never sees "
                 "it). Predictors are ranked by ROC-AUC with bootstrap CIs.\n")
        hard = [s for s in rv["sensitivity"]
                if s["config"]["accel_thresh_ms2"] == -3.0
                and s["config"]["horizon_s"] == 5.0]
        if hard:
            s = hard[0]
            L.append(f"\n**Hard braking (<= -3.0 m/s^2 within 5s), n="
                     f"{s['n_valid_keyframes']} keyframes, {s['n_positive']} "
                     f"positive (base rate {s['base_rate']:.1%}):**\n\n")
            L.append("| predictor | ROC-AUC | 95% CI | AP | vs base rate |\n"
                     "|---|---|---|---|---|\n")
            for name, m in sorted(s["predictors"].items(),
                                  key=lambda kv: -kv[1]["roc_auc"]):
                lo, hi = m["roc_auc_ci"]
                L.append(f"| {name} | {m['roc_auc']:.3f} | [{lo:.2f}, {hi:.2f}] "
                         f"| {m['avg_precision']:.3f} | "
                         f"{m['ap_lift_over_base_rate']:.2f}x |\n")
        ttc_auc = straw_auc = ours_auc = float("nan")
        if hard:
            p = hard[0]["predictors"]
            ttc_auc = p["inverse TTC only"]["roc_auc"]
            ours_auc = p["risk score (ours)"]["roc_auc"]
            straw_auc = p["box area (strawman)"]["roc_auc"]
        L.append("\n**Findings.**\n")
        L.append(f"1. *Metric grounding is validated.* Inverse TTC — which exists "
                 f"only because of the metric lift and Kalman velocity — is the "
                 f"strongest predictor of hard braking (AUC {ttc_auc:.3f}, "
                 f"{hard[0]['predictors']['inverse TTC only']['ap_lift_over_base_rate']:.1f}x "
                 f"base rate on AP) and the only predictor whose CI clears chance. "
                 f"It beats the box-area strawman the plan called worthless "
                 f"({straw_auc:.3f}, CI includes 0.5).\n")
        L.append("2. *Improving the lift improved exactly the predictors that "
                 "depend on it.* Re-running this harness after M9's fused lift "
                 "raised inverse-TTC AUC 0.709 -> 0.737 and its AP 1.91x -> 2.42x "
                 "base rate, while the box-area strawman — which uses no metric "
                 "information — stayed bit-identical at 0.548. That is the "
                 "causal link between metric grounding and risk quality, measured "
                 "rather than asserted.\n")
        L.append(f"3. *The aggregation defect found here has been fixed (M8b).* "
                 f"The original score ADDED a flat +0.30 for anything in the ego "
                 f"lane, piling 33% of frames onto one tied value and scoring "
                 f"0.655 — below its own TTC component. Zone now MULTIPLIES a "
                 f"continuous hazard (noisy-OR of TTC, a continuous RSS gap "
                 f"ratio, and crossing intent), which lifted the composite to "
                 f"{ours_auc:.3f} (AP "
                 f"{hard[0]['predictors']['risk score (ours)']['ap_lift_over_base_rate']:.1f}x "
                 f"base rate) — statistically indistinguishable from inverse TTC "
                 f"({ttc_auc:.3f}), with overlapping CIs. No constant was fitted "
                 "to these labels; only the structure changed. The composite now "
                 "matches TTC rather than beating it, so the lane/RSS/intent terms "
                 "add no measurable prediction on 25 events — but they no longer "
                 "subtract, and they carry the explanatory content a bare TTC "
                 "number cannot.\n")
        L.append("4. *At gentle braking thresholds (-1.5 m/s^2) nothing predicts "
                 "well and box area leads.* Expected: gentle urban deceleration is "
                 "driven by traffic density, queues and traffic lights — none of "
                 "which this system perceives. Box area is a decent traffic-density "
                 "proxy; it is not a danger measure.\n")
        L.append("\n**Caveats.** " + " ".join(rv["caveats"]) + " Sensitivity across "
                 "3 thresholds x 3 horizons is in `out/eval/risk_validation.json`; "
                 "the ordering above holds throughout the hard-braking regime and "
                 "reverses in the gentle regime.\n")

    try:
        pitch = load("pitch_sensitivity.json")
        loc = load("localization.json")
    except FileNotFoundError:
        pitch = loc = None
    if pitch and loc:
        L.append("\n## 8. Robustness (M10)\n")
        L.append("Both experiments replay a cached detection stream, "
                 "perturbing what the system BELIEVES (assumed extrinsics, "
                 "assumed ego pose) while ground truth stays fixed.\n")
        L.append("\n**Pitch sensitivity.** mini has no hill scenes, but a "
                 "pitch error is exactly what a crest, dip, banked curve or "
                 "brake-dive produces. Median range error:\n\n")
        L.append("| assumed-pitch error | IPM only | fused (current) |\n"
                 "|---|---|---|\n")
        ipm_rows = {r["pitch"]: r for r in pitch["variants"]["IPM only"]["rows"]}
        fus_rows = {r["pitch"]: r
                    for r in pitch["variants"]["fused (M9 default)"]["rows"]}
        for p in (0.0, 1.0, 2.0, 3.0, -2.0, -3.0):
            if p in ipm_rows:
                L.append(f"| {p:+.0f}° | {pct(ipm_rows[p]['overall'])} | "
                         f"{pct(fus_rows[p]['overall'])} |\n")
        L.append(f"\nIPM degrades "
                 f"{pitch['variants']['IPM only']['degradation_factor']:.1f}x at "
                 f"+/-2°; the fused lift degrades "
                 f"{pitch['variants']['fused (M9 default)']['degradation_factor']:.1f}x. "
                 "The plan's 'every IPM system lies on hills' is now measured "
                 "rather than asserted — and M9 hardened the system against it "
                 "as an unplanned side effect, because the width prior takes "
                 "range from apparent size and intrinsics, not from an assumed "
                 "ground plane.\n")
        L.append("\n**Localisation dependence.** The BEV filters run in the "
                 "global frame using nuScenes `ego_pose` — GT localisation a "
                 "camera-only system would not have. Degrading it:\n\n")
        L.append("| assumed pose quality | range err | velocity err |\n"
                 "|---|---|---|\n")
        for v in loc["variants"]:
            L.append(f"| {v['label']} | {pct(v['range_err'])} | "
                     f"{v['vel_err']:.2f} m/s |\n")
        base_v = loc["variants"][0]["vel_err"]
        mid = next(v for v in loc["variants"] if v["yaw_sigma_deg"] == 0.5)
        L.append(f"\nThe dependence is mild: consumer-grade quality "
                 f"(0.3m / 0.5°) costs {mid['vel_err'] / base_v - 1:.0%} of "
                 "velocity accuracy, so the headline numbers are not an "
                 "artefact of nuScenes' SLAM. Range is nearly untouched by pose "
                 "noise because it is measured in the ego frame before any "
                 "global transform. Tracking without localisation at all keeps "
                 f"range at {pct(loc['no_localization']['range_err'])} but makes "
                 "absolute velocity unrecoverable by construction — only "
                 "ego-relative motion survives, which is why M3 chose the "
                 "global frame and why TTC (the best braking predictor) needs "
                 "it.\n")

    L.append("\n## 9. Known limitations (measured, not hypothetical)\n")
    L.append("- **Night**: lane corridor mostly unavailable; detection/tracking "
             "recall drops (tables 2-3). Risk falls back to fixed corridor, "
             "flagged per record.\n")
    L.append("- **IPM occluded-bottom drift** (largely FIXED in M9): barrier-"
             "occluded boxes biased range high with phantom velocity. Depth did "
             "not fix it (M6); the width prior did — occluded-object error "
             "13.2% -> 5.5%. IPM is still used below 12m, where the effect is "
             "small.\n")
    L.append("- **Depth near-field bias**: DA-V2-metric overestimates <10m "
             "(58% median) with a nonlinear, non-calibratable profile "
             "(gt/depth 0.63->1.06).\n")
    L.append("- **Cyclist detection** is weak (AP 0.13): COCO bicycle-vs-rider "
             "box mismatch; needs fine-tuning on full trainval.\n")
    L.append("- **Flat-ground assumption**: no slope scenes in mini, but the "
             "failure mode is now bounded by the pitch sweep above — the "
             "current lift degrades only 1.1x at +/-2° (IPM alone: 4.9x).\n")

    L.append("- **Risk composite adds no measurable prediction over TTC alone** "
             "(M8b): after restructuring it matches inverse TTC (0.732 vs "
             "0.737) rather than beating it. Whether lane/RSS/intent add real "
             "signal is unresolvable on 25 braking events; needs trainval.\n")

    L.append("\n## 9. Artifacts\n")
    L.append("- Demo videos: `out/demo/scene-0103_demo.mp4` (day), "
             "`out/demo/scene-1100_demo.mp4` (night); per-milestone demos "
             "alongside.\n")
    L.append("- Metrics JSONs: `out/eval/*.json` | Harnesses: `scripts/eval_*.py` "
             "| Tests: `tests/` (91 passing)\n")

    path = os.path.join(ROOT, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
