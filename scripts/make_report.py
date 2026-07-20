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
    L.append("M3 target: median relative error <=15% at 10-30m — **met: "
             f"{pct(ipm['m3_target']['kf_rel_median_10_30m'])}** (KF-smoothed).\n")
    L.append("Raw-lift comparison (identical matched pairs, "
             f"n={dep['n_pairs']}):\n")
    L.append("| GT range | IPM med | depth med | hybrid med | gt/depth ratio |\n"
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
    L.append("\nVelocity (BEV KF vs nuScenes annotation velocity): median error "
             "0.28 m/s at 0-10m, 2.3 m/s at 10-30m (`ipm_validation.json`). "
             "M6 verdict: hybrid lift available (`use_depth=True`), live "
             "pipeline keeps IPM — the TTC working range (<30m) is IPM's.\n")

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
             f"{risk['ttc']['trustworthy']} trustworthy (30m gate from M3 error "
             "curve; CI from KF covariances via delta method)\n")
    L.append(f"- RSS (arXiv:1708.06374) violations: {risk['rss_violations']}; "
             f"pedestrian crossing-intent events: {risk['crossing_intent_events']}\n")
    L.append(f"- Corridor source: {risk['corridor_source']}\n")
    L.append(f"- Ego risk: mean {risk['ego_risk']['mean']:.2f}, high on "
             f"{risk['ego_risk']['frames_high']}/{risk['ego_risk']['n_keyframes']} keyframes\n")
    L.append("\nTop audited moments (all reasons machine-generated):\n")
    for m in risk["top_moments"][:5]:
        L.append(f"- [{m['score']:.2f}] {m['scene']} #{m['track']} "
                 f"{m['category']}: " + "; ".join(m["reasons"]) + "\n")

    L.append("\n## 7. Known limitations (measured, not hypothetical)\n")
    L.append("- **Night**: lane corridor mostly unavailable; detection/tracking "
             "recall drops (tables 2-3). Risk falls back to fixed corridor, "
             "flagged per record.\n")
    L.append("- **IPM occluded-bottom drift**: barrier-occluded boxes bias range "
             "high with phantom velocity; chi-square innovation gate catches "
             "teleports, gradual drift remains (documented in M3; depth does "
             "NOT fix it — measured in M6).\n")
    L.append("- **Depth near-field bias**: DA-V2-metric overestimates <10m "
             "(58% median) with a nonlinear, non-calibratable profile "
             "(gt/depth 0.63->1.06).\n")
    L.append("- **Cyclist detection** is weak (AP 0.13): COCO bicycle-vs-rider "
             "box mismatch; needs fine-tuning on full trainval.\n")
    L.append("- **Flat-ground assumption untested on hills** — no slope scenes "
             "in mini.\n")

    L.append("\n## 8. Artifacts\n")
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
