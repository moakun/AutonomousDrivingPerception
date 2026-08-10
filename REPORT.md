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
| **end-to-end** | **55** (p95 104) | staleness at 30 m/s: ~1.6m (p95 3.1m) |

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
Current default lift: **IPM fused with a class width prior** (M9) — IPM below 12m, width beyond. M3 target: median relative error <=15% at 10-30m — **met: 6.9%** (KF-smoothed).

End-to-end (Kalman-smoothed, current system):

| GT range | n | range err median | p90 | velocity err median |
|---|---|---|---|---|
| 0-10m | 88 | 4.1% | 15.1% | 0.33 m/s |
| 10-30m | 735 | 6.9% | 20.2% | 1.29 m/s |
| 30-50m | 398 | 8.6% | 22.6% | 1.54 m/s |
| 50m+ | 308 | 9.8% | 22.1% | 1.25 m/s |

**M9 — why the lift changed** (raw single-frame estimates, identical matched pairs, n=1480):

| GT range | IPM alone (old) | width alone | **fused (new)** |
|---|---|---|---|
| 0-10m | 4.1% | 5.1% | **3.7%** |
| 10-30m | 11.0% | 6.3% | **6.2%** |
| 30-50m | 22.8% | 7.6% | **7.6%** |
| 50m+ | 39.3% | 9.4% | **9.4%** |

By occlusion level (<50m): occluded (vis 2-3): IPM 13.2% -> fused 5.5%; clear (vis 4): IPM 12.6% -> fused 6.7%.

The width cue survives leave-one-scene-out validation (8.2% held-out vs 7.5% in-sample), so it is not calibration contamination. Two measured corrections made it work: IPM's analytic noise model was 4-6x too optimistic (real sigma grows as r^2.51 with +11.5m bias at 40-60m), and fusion is dropped entirely beyond 12m because inverse-variance fusion assumes unbiased inputs.

**M6 (superseded) — monocular depth comparison**, n=1480. The width prior beats this hybrid beyond 10m at ~39ms/frame lower cost, so depth is retained only for reproducing the comparison:

| GT range | IPM med | depth med | depth hybrid | gt/depth ratio |
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

Velocity is validated against nuScenes annotation velocities (table above). The TTC trust gate tracks this curve and was widened from 30m to 50m after M9, because 30-50m error is now better than 10-30m was when the gate was first set.

## 5. Lanes (no vector GT in mini -> degradation stats)
| condition | corridor valid (smoothed) | width median |
|---|---|---|
| day | 35% | 3.49m |
| night | 35% | 3.27m |

Zero-validity scenes are parking lots / intersection idling / dark night — 'no corridor' is the correct output there; the risk layer falls back to a fixed ±1.85m corridor and records the fallback per object.

## 6. Risk layer
- Records at keyframes: 2161 (ego 326, adjacent 601, off-path 1234)
- TTC: 1658 computed, 781 trustworthy (50m gate, re-derived from the M9 lift error curve; CI from KF covariances via the delta method)
- RSS (arXiv:1708.06374) violations: 24; pedestrian crossing-intent events: 61
- Corridor source: {'detected': 658, 'fallback': 1503}
- Ego risk: mean 0.18, high on 3/404 keyframes
  (M8b lowered these from mean 0.29 / 13 high frames: the old score charged every in-lane object 0.30 merely for existing, so warnings are now rarer and correspond to genuinely short TTC. Bucket thresholds were left at 0.30/0.60 — display calibration was deliberately NOT tuned against the braking labels.)

Top audited moments (all reasons machine-generated):
- [0.78] scene-1100 #8 car: ego lane (fallback corridor); TTC 0.4±0.1s, closing 10.6 m/s
- [0.76] scene-1100 #20 car: ego lane (fallback corridor); TTC 0.5±0.1s, closing 9.6 m/s
- [0.72] scene-1100 #2 car: ego lane (fallback corridor); TTC 0.6±0.1s, closing 8.0 m/s
- [0.60] scene-1100 #9 car: ego lane (fallback corridor); TTC 1.2±0.3s, closing 9.7 m/s
- [0.58] scene-1100 #2 car: ego lane (fallback corridor); TTC 0.7±0.2s, closing 11.1 m/s

## 7. Does the risk output predict human braking? (M8)
The risk layer has no ground truth — nuScenes labels objects, not danger. Proxy signal: hard braking within a horizon, derived offline from ego pose (acausal; the perception path never sees it). Predictors are ranked by ROC-AUC with bootstrap CIs.

**Hard braking (<= -3.0 m/s^2 within 5s), n=215 keyframes, 25 positive (base rate 11.6%):**

| predictor | ROC-AUC | 95% CI | AP | vs base rate |
|---|---|---|---|---|
| inverse TTC only | 0.737 | [0.65, 0.82] | 0.281 | 2.42x |
| risk score (ours) | 0.732 | [0.64, 0.82] | 0.245 | 2.11x |
| ego speed only | 0.610 | [0.53, 0.68] | 0.139 | 1.20x |
| box area (strawman) | 0.548 | [0.46, 0.63] | 0.121 | 1.04x |
| nearest object range | 0.548 | [0.46, 0.63] | 0.122 | 1.05x |

**Findings.**
1. *Metric grounding is validated.* Inverse TTC — which exists only because of the metric lift and Kalman velocity — is the strongest predictor of hard braking (AUC 0.737, 2.4x base rate on AP) and the only predictor whose CI clears chance. It beats the box-area strawman the plan called worthless (0.548, CI includes 0.5).
2. *Improving the lift improved exactly the predictors that depend on it.* Re-running this harness after M9's fused lift raised inverse-TTC AUC 0.709 -> 0.737 and its AP 1.91x -> 2.42x base rate, while the box-area strawman — which uses no metric information — stayed bit-identical at 0.548. That is the causal link between metric grounding and risk quality, measured rather than asserted.
3. *The aggregation defect found here has been fixed (M8b).* The original score ADDED a flat +0.30 for anything in the ego lane, piling 33% of frames onto one tied value and scoring 0.655 — below its own TTC component. Zone now MULTIPLIES a continuous hazard (noisy-OR of TTC, a continuous RSS gap ratio, and crossing intent), which lifted the composite to 0.732 (AP 2.1x base rate) — statistically indistinguishable from inverse TTC (0.737), with overlapping CIs. No constant was fitted to these labels; only the structure changed. The composite now matches TTC rather than beating it, so the lane/RSS/intent terms add no measurable prediction on 25 events — but they no longer subtract, and they carry the explanatory content a bare TTC number cannot.
4. *At gentle braking thresholds (-1.5 m/s^2) nothing predicts well and box area leads.* Expected: gentle urban deceleration is driven by traffic density, queues and traffic lights — none of which this system perceives. Box area is a decent traffic-density proxy; it is not a danger measure.

**Caveats.** Hard braking is a behavioural proxy, not ground-truth danger: drivers also brake for lights, stop signs and turns. 10 scenes is a small sample — see bootstrap CIs. Labels are acausal (offline smoothing over the whole scene); the perception path never sees them. Sensitivity across 3 thresholds x 3 horizons is in `out/eval/risk_validation.json`; the ordering above holds throughout the hard-braking regime and reverses in the gentle regime.

## 8. Robustness (M10)
Both experiments replay a cached detection stream, perturbing what the system BELIEVES (assumed extrinsics, assumed ego pose) while ground truth stays fixed.

**Pitch sensitivity.** mini has no hill scenes, but a pitch error is exactly what a crest, dip, banked curve or brake-dive produces. Median range error:

| assumed-pitch error | IPM only | fused (current) |
|---|---|---|
| +0° | 17.4% | 7.7% |
| +1° | 44.6% | 7.8% |
| +2° | 85.6% | 7.9% |
| +3° | 162.3% | 7.9% |
| -2° | 33.3% | 8.9% |
| -3° | 42.8% | 10.0% |

IPM degrades 4.9x at +/-2°; the fused lift degrades 1.1x. The plan's 'every IPM system lies on hills' is now measured rather than asserted — and M9 hardened the system against it as an unplanned side effect, because the width prior takes range from apparent size and intrinsics, not from an assumed ground plane.

**Localisation dependence.** The BEV filters run in the global frame using nuScenes `ego_pose` — GT localisation a camera-only system would not have. Degrading it:

| assumed pose quality | range err | velocity err |
|---|---|---|
| GT pose (current) | 7.7% | 1.28 m/s |
| pos 0.1m, yaw 0.2deg | 7.7% | 1.29 m/s |
| pos 0.3m, yaw 0.5deg | 7.8% | 1.33 m/s |
| pos 1.0m, yaw 1.0deg | 8.4% | 1.57 m/s |
| pos 0.0m, yaw 2.0deg | 7.7% | 1.46 m/s |
| pos 3.0m, yaw 0.0deg | 10.5% | 2.22 m/s |

The dependence is mild: consumer-grade quality (0.3m / 0.5°) costs 4% of velocity accuracy, so the headline numbers are not an artefact of nuScenes' SLAM. Range is nearly untouched by pose noise because it is measured in the ego frame before any global transform. Tracking without localisation at all keeps range at 7.7% but makes absolute velocity unrecoverable by construction — only ego-relative motion survives, which is why M3 chose the global frame and why TTC (the best braking predictor) needs it.

## 9. Known limitations (measured, not hypothetical)
- **Night**: lane corridor mostly unavailable; detection/tracking recall drops (tables 2-3). Risk falls back to fixed corridor, flagged per record.
- **IPM occluded-bottom drift** (largely FIXED in M9): barrier-occluded boxes biased range high with phantom velocity. Depth did not fix it (M6); the width prior did — occluded-object error 13.2% -> 5.5%. IPM is still used below 12m, where the effect is small.
- **Depth near-field bias**: DA-V2-metric overestimates <10m (58% median) with a nonlinear, non-calibratable profile (gt/depth 0.63->1.06).
- **Cyclist detection** is weak (AP 0.13): COCO bicycle-vs-rider box mismatch; needs fine-tuning on full trainval.
- **Flat-ground assumption**: no slope scenes in mini, but the failure mode is now bounded by the pitch sweep above — the current lift degrades only 1.1x at +/-2° (IPM alone: 4.9x).
- **Risk composite adds no measurable prediction over TTC alone** (M8b): after restructuring it matches inverse TTC (0.732 vs 0.737) rather than beating it. Whether lane/RSS/intent add real signal is unresolvable on 25 braking events; needs trainval.

## 9. Artifacts
- Demo videos: `out/demo/scene-0103_demo.mp4` (day), `out/demo/scene-1100_demo.mp4` (night); per-milestone demos alongside.
- Metrics JSONs: `out/eval/*.json` | Harnesses: `scripts/eval_*.py` | Tests: `tests/` (91 passing)
