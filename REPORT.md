# ADP Evaluation Report
All numbers measured on nuScenes v1.0-mini (10 scenes, 404 keyframes, CAM_FRONT), RTX 4060 Laptop GPU. Every table below is reproducible from the referenced harness script and its JSON in `out/eval/`.
**Honest-numbers preamble:** 2D GT boxes are projections of 3D boxes (systematically loose → absolute AP/MOTA read low); mini has no rain or hill scenes (slope failure mode untested); detector and lane model are pretrained only, no fine-tuning (404 frames is too few). Conditions: 7 day scenes, 3 night scenes.
## 1. Pipeline & latency
```
frame -> yolo11m@960 (2D det) -> ByteTrack (12Hz) -> IPM lift -> BEV Kalman (global frame)
      -> YOLOP lanes -> BEV corridor   -> risk: TTC+CI | lane zone | ped intent | RSS
```
| module | p50 ms | note |
|---|---|---|
| detection (yolo11m@960, PyTorch GPU) | ~18 | ONNX parity verified (max 0.06px) |
| tracking + IPM lift + BEV KF | ~1 | |
| lanes (YOLOP + BEV fit) | ~27 | can run at half rate |
| depth (optional hybrid, off by default) | ~39 | DA-V2-metric fp16 |
| **end-to-end** | **61** (p95 117) | staleness at 30 m/s: ~1.8m (p95 3.5m) |

## 2. Detection (AP@0.5 vs projected 3D GT)
Model selection table: `out/eval/detection_baseline.json` (yolo11m@960 selected: mAP 0.419 @ ~18ms).
| condition | mAP@0.5 (GT-weighted) | car | pedestrian |
|---|---|---|---|
| day (7 scenes) | 0.489 | 0.687 | 0.394 |
| night (3 scenes) | 0.338 | 0.633 | 0.311 |

## 3. Tracking (12Hz sweeps, scored at 2Hz keyframes)
| condition | MOTA | IDF1 | IDsw | recall <30m | 30-50m | 50m+ |
|---|---|---|---|---|---|---|
| day | 0.271 | 0.528 | 64 | 62.7% | 50.5% | 32.9% |
| night | 0.252 | 0.467 | 16 | 57.9% | 31.6% | 25.9% |

## 4. Metric lift (range error vs 3D GT)
M3 target: median relative error <=15% at 10-30m — **met: 11.9%** (KF-smoothed).
Raw-lift comparison (identical matched pairs, n=1480):
| GT range | IPM med | depth med | hybrid med | gt/depth ratio |
|---|---|---|---|---|
| 0-10m | 4.1% | 58.1% | 4.1% | 0.63 |
| 10-30m | 11.0% | 17.7% | 10.6% | 0.87 |
| 30-50m | 22.8% | 12.7% | 14.6% | 0.93 |
| 50m+ | 39.3% | 15.4% | 15.4% | 1.06 |

Per-condition raw-IPM median error (<50m), from per-pair records:
| condition | n pairs | IPM med | depth med |
|---|---|---|---|
| day | 957 | 13.4% | 17.5% |
| night | 243 | 10.5% | 17.4% |

*Note: night IPM error reading better than day is a survivorship artifact — night detection only yields matched pairs on close, well-lit objects, so the night sample is skewed toward easy ranges (see the night recall drop in table 3).*

Velocity (BEV KF vs nuScenes annotation velocity): median error 0.28 m/s at 0-10m, 2.3 m/s at 10-30m (`ipm_validation.json`). M6 verdict: hybrid lift available (`use_depth=True`), live pipeline keeps IPM — the TTC working range (<30m) is IPM's.

## 5. Lanes (no vector GT in mini -> degradation stats)
| condition | corridor valid (smoothed) | width median |
|---|---|---|
| day | 35% | 3.49m |
| night | 35% | 3.27m |

Zero-validity scenes are parking lots / intersection idling / dark night — 'no corridor' is the correct output there; the risk layer falls back to a fixed ±1.85m corridor and records the fallback per object.

## 6. Risk layer
- Records at keyframes: 2092 (ego 319, adjacent 601, off-path 1171)
- TTC: 1446 computed, 531 trustworthy (30m gate from M3 error curve; CI from KF covariances via delta method)
- RSS (arXiv:1708.06374) violations: 29; pedestrian crossing-intent events: 69
- Corridor source: {'detected': 735, 'fallback': 1357}
- Ego risk: mean 0.29, high on 13/404 keyframes

Top audited moments (all reasons machine-generated):
- [0.71] scene-1100 #8 car: ego lane (fallback corridor); TTC 0.5±0.1s, closing 9.8 m/s
- [0.70] scene-1100 #20 car: ego lane (fallback corridor); TTC 0.5±0.1s, closing 9.5 m/s
- [0.70] scene-1100 #2 car: ego lane (fallback corridor); TTC 0.5±0.1s, closing 8.6 m/s
- [0.69] scene-0916 #121 car: ego lane (fallback corridor); TTC 0.7±0.1s, closing 9.8 m/s
- [0.66] scene-1100 #2 car: ego lane (fallback corridor); TTC 1.0±0.1s, closing 9.0 m/s

## 7. Known limitations (measured, not hypothetical)
- **Night**: lane corridor mostly unavailable; detection/tracking recall drops (tables 2-3). Risk falls back to fixed corridor, flagged per record.
- **IPM occluded-bottom drift**: barrier-occluded boxes bias range high with phantom velocity; chi-square innovation gate catches teleports, gradual drift remains (documented in M3; depth does NOT fix it — measured in M6).
- **Depth near-field bias**: DA-V2-metric overestimates <10m (58% median) with a nonlinear, non-calibratable profile (gt/depth 0.63->1.06).
- **Cyclist detection** is weak (AP 0.13): COCO bicycle-vs-rider box mismatch; needs fine-tuning on full trainval.
- **Flat-ground assumption untested on hills** — no slope scenes in mini.

## 8. Artifacts
- Demo videos: `out/demo/scene-0103_demo.mp4` (day), `out/demo/scene-1100_demo.mp4` (night); per-milestone demos alongside.
- Metrics JSONs: `out/eval/*.json` | Harnesses: `scripts/eval_*.py` | Tests: `tests/` (91 passing)
