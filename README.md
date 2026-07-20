# ADP — Autonomous Driving Perception

A camera-only perception system built on nuScenes: 2D detection → multi-object
tracking → **metric BEV lift** → lanes → an explainable risk layer (TTC, lane
assignment, pedestrian intent, RSS). The thesis, from [PLAN.md](PLAN.md):
detection is a solved API call — the value is *metric grounding*, turning
boxes into meters, m/s, and defensible risk quantities, each validated against
3D ground truth.

```
frame → yolo11m@960 (2D det) → ByteTrack (12Hz) → IPM lift → BEV Kalman (global frame)
     → YOLOP lanes → BEV corridor → risk: TTC±CI | lane zone | ped intent | RSS
```

**End-to-end: 61ms p50 / 117ms p95** on an RTX 4060 Laptop (≈1.8m / 3.5m of
staleness at 30 m/s). All numbers below: nuScenes v1.0-mini, 10 scenes,
404 keyframes, CAM_FRONT. Full audit trail in [REPORT.md](REPORT.md).

---

## Findings by stage

### Detection (M1)
- **yolo11m @ 960px** won model selection: mAP@0.5 **0.419** at ~18ms — beats
  RT-DETR-L (0.399) at half its latency. Resolution matters on 1600px frames:
  640→960 gained +4 mAP, mostly on small/distant objects.
- ONNX export verified to **0.06px max deviation** from PyTorch — but ONNX
  Runtime CPU runs ~260ms vs 18ms GPU, so the live pipeline stays on PyTorch
  (ONNX kept as the TensorRT on-ramp).
- Fine-tuning deferred deliberately: 404 mini keyframes are too few. Known
  cost: **cyclist AP 0.13** (COCO "bicycle" box excludes the rider; nuScenes
  includes it).

### Tracking (M2)
- Own ~150-line ByteTrack implementation (two-pass association; low-confidence
  recovery pass keeps IDs alive through partial occlusion) so the Kalman state
  stays ours. **Association runs at 12Hz on camera sweeps** — 2Hz keyframes
  are too coarse for IoU matching — and is scored at keyframes: MOTA 0.27,
  IDF1 0.51, 80 ID switches.
- Recall falls exactly with range (62% <30m → 31% 50m+): losses are
  small-object detection misses, not tracker failures.
- A looser-threshold variant won +3.6 recall but cost 46% more ID switches —
  **rejected**: identity continuity feeds velocity, and velocity feeds TTC.

### Metric lift — IPM (M3), the load-bearing result
- Box bottom-center → ground plane (ego z=0) → **BEV Kalman in the global
  frame** (ego-frame constant-velocity would make parked cars "accelerate"
  when ego turns). Measurement noise follows the analytic r²/(f·h) law.
- **Target met: 11.9% median range error at 10–30m** (≤15% required); 4.2% at
  0–10m. Velocity vs nuScenes annotation velocities: **0.28 m/s** median at
  0–10m, 2.3 m/s at 10–30m.
- Occlusion teleports (a barrier-occluded car "jumping" from 45m to 95m) blew
  up far-range velocity until a **chi-square innovation gate** (reject >99.9%
  Mahalanobis outliers, re-seed after 6 persistent) cut 50m+ error 61%→45%.
  The *gradual* occluded-bottom drift remains — it is ungateable by design.

### Lanes (M4)
- YOLOP segmentation (plain PyTorch — CLRNet/LaneATT need CUDA ops that don't
  build on Windows) → points lifted through the same IPM → dependency-free
  slice-and-link clustering → quadratic fits **in BEV space** → ego corridor
  with EMA smoothing, jump gating, and an explicit no-corridor state.
- **Lane-width median 3.43m** across valid corridors — a free end-to-end
  geometry sanity check (real urban lanes ≈3.5m).
- Marked roads: 44–88% corridor validity. Parking lots / intersection idling /
  dark night: 0% — **correctly**: there are no markings there; the system
  reports "no corridor" instead of hallucinating.
- Bug found via demo review: the clusterer couldn't bridge ~6m dash gaps, so
  dashed center lines produced zero fits. Gap bridging lifted validity 29→35%.

### Risk layer (M5)
- **TTC** = bumper gap / closing speed with a **1σ interval from the delta
  method** over the Kalman covariances, trust-gated at 30m (the M3-measured
  error knee): 1,446 computed, 531 trustworthy.
- **RSS** longitudinal safe distance (Shalev-Shwartz et al., arXiv:1708.06374)
  — 29 violations flagged. Lateral RSS is a documented scope cut.
- **Lane zones** against the curved M4 corridor, falling back to a fixed
  ±1.85m corridor when lanes are absent — with the source recorded per record.
- **69 pedestrian crossing-intent events** (lateral BEV velocity vs corridor).
- Every record carries machine-generated reasons ("ego lane; TTC 1.8±0.4s,
  closing 12 m/s") — never a bare colored box. Review caught a "pedestrian
  closing at 35.8 m/s" ghost (corrupted night-scene filter velocity scoring
  0.99): fixed with a **category velocity-plausibility gate** that suppresses
  TTC/RSS from impossible speeds and says so in the reasons.

### Depth experiment (M6) — a designed decision, honestly resolved
Head-to-head on 1,480 identical matched pairs, Depth Anything V2 metric
(39ms fp16) vs raw IPM:

| GT range | IPM | depth | hybrid (IPM<30m, depth beyond) |
|---|---|---|---|
| 0–10m  | **4.1%** | 58.1% | 4.1% |
| 10–30m | **11.0%** | 17.7% | **10.6%** |
| 30–50m | 22.8% | **12.7%** | 14.6% |
| 50m+   | 39.3% | **15.4%** | 15.4% |

- Two hypotheses died: depth does **not** fix the occluded-bottom failure
  (IPM still edges it on occluded objects <50m), and the near-field bias is
  **not calibratable** (gt/depth ratio 0.63→1.06 — nonlinear, not a scale).
- **Verdict:** hybrid implemented (`PerceptionPipeline(use_depth=True)`), but
  the live risk pipeline keeps IPM — TTC/RSS are gated to <30m, exactly where
  IPM is both better and free.

### Per-condition (M7)
| | day (7 scenes) | night (3 scenes) |
|---|---|---|
| detection mAP@0.5 | 0.489 | 0.338 |
| tracking recall <30m | 62.7% | 57.9% |
| lane corridor available | 35% | 35%* |

*Night lane numbers survive on the two lit scenes; the dark one finds nothing.
Night risk leans on the fallback corridor — flagged per record, not hidden.

## Known limitations (measured, not hypothetical)
- Night degrades everything above; per-condition tables in REPORT.md.
- IPM's gradual occluded-bottom drift → range bias + phantom velocity (depth
  doesn't fix it; measured).
- Depth near-field bias is nonlinear and non-calibratable.
- Cyclist detection weak (COCO box-definition mismatch; needs trainval
  fine-tuning).
- Flat-ground assumption untested on hills — mini has no slope scenes.
- Absolute AP/MOTA read low vs hand-labeled benchmarks: GT rectangles are
  projections of 3D boxes (systematically loose). Consistent for comparisons.

## Repository

```
adp/
  calib/      camera model, SE3, ray-plane intersection (tested to 1e-9)
  data/       nuScenes access layer (keyframes + 12Hz sweeps)
  detect/     yolo11m/RT-DETR wrapper, COCO→ADP class map
  track/      ByteTrack + the constant-velocity Kalman (ONLY source of velocity)
  lift/       IPM, Depth Anything V2, hybrid switch, BEV state + innovation gate
  lanes/      YOLOP wrapper, BEV clustering/fitting, corridor smoothing
  risk/       TTC±CI, RSS, lane zones, ped intent, transparent scoring
  viz/        BEV canvas + composite frame renderer
  pipeline.py the full spine as one object
scripts/      eval harnesses (one per milestone) + run_demo.py + make_report.py
tests/        91 tests, all passing
out/eval/     metrics JSONs   |   out/demo/  annotated videos
```

## Running it

```bash
pip install -r requirements.txt   # plus torch with CUDA
# place nuScenes v1.0-mini at ./v1.0-mini (tables in v1.0-mini/v1.0-mini/)

python -m pytest tests/ -q                      # 91 tests
python scripts/check_data.py                    # M0: GT projection sanity renders
python scripts/eval_detection.py                # M1: model benchmark
python scripts/eval_tracking.py                 # M2: MOTA/IDF1
python scripts/eval_ipm.py                      # M3: range-error validation
python scripts/eval_lanes.py                    # M4: corridor stats
python scripts/eval_risk.py                     # M5: full-pipeline risk stats
python scripts/eval_depth.py                    # M6: depth-vs-IPM decision
python scripts/run_demo.py --scenes scene-0103  # annotated demo video
python scripts/make_report.py                   # regenerate REPORT.md
```

## Documents
- [PLAN.md](PLAN.md) — the staged plan with per-milestone exit criteria and
  status.
- [REPORT.md](REPORT.md) — the full evaluation report (per-condition tables,
  latency accounting, top audited risk moments).
