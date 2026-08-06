# ADP — Autonomous Driving Perception: Project Plan

**Thesis:** Detection is a solved API call. The value of this project is *metric grounding* — turning
2D boxes into positions in meters, velocities, and defensible risk quantities (TTC, lane assignment,
RSS envelopes). Everything upstream of the BEV lift is off-the-shelf; the lift and the risk layer are ours.

**Spine:**

```
frame → 2D detection → tracking (IDs over time) → 3D/BEV lift → risk scoring
                          ↘ lanes (separate branch, fused in BEV) ↗
```

**Scoping guardrail:** if time runs short, cut lanes or cut risk breadth — never cut metric
grounding. Vehicles + pedestrians + accurate BEV distances + TTC is a real perception system.
All four done superficially (risk from box area) is a demo.

---

## Milestone overview

| # | Milestone | Depends on | Core deliverable | Status |
|---|-----------|-----------|------------------|--------|
| M0 | Data & calibration | — | nuScenes pipeline + camera model utilities | ✅ done — geometry round-trips at machine precision; GT boxes render correctly on all 10 mini scenes |
| M1 | Detection | M0 | Fine-tuned RT-DETR/YOLO, ONNX export, eval harness | ✅ done (fine-tuning deferred) — yolo11m@960 selected (mAP@0.5 0.419, ~18ms GPU p50); ONNX parity verified (max 0.06px). ORT-CPU is ~260ms → live pipeline stays on PyTorch GPU; TensorRT later. Fine-tuning deferred: mini's 404 frames too few — revisit with full trainval if cyclist AP (0.13) blocks downstream. |
| M2 | Tracking | M1 | ByteTrack/OC-SORT with per-track Kalman velocity | ✅ done — own ByteTrack impl @12Hz sweeps, eval @2Hz keyframes: MOTA 0.27, IDF1 0.51, 80 IDsw; recall by range 0.62/<30m, 0.46/30–50m, 0.31/50m+. Looser-threshold variant rejected (+recall but IDsw 80→117). Tracker step ~1ms. |
| M3 | Metric lift v1 (IPM) | M0, M2 | BEV positions from ground-plane homography, validated vs lidar GT | ✅ done — **target MET**: median rel range err 4.2% @0–10m, 11.9% @10–30m (≤15% target); 24% @30–50m, 45% @50m+. Velocity err 0.28 m/s @0–10m, 2.3 @10–30m. Global-frame BEV Kalman + chi-square innovation gate (occlusion teleports rejected). Known failure mode documented: barrier-occluded box bottoms bias range high + phantom velocity (gradual drift, ungateable) → M6 depth candidate. No slope scenes in mini; deferred to full dataset. |
| M4 | Lanes in BEV | M0 | Lane detection + polynomial fit in BEV space | ✅ done — YOLOP seg (27ms) + BEV slice-link clustering (dash-gap bridging) + quadratic fits + EMA-smoothed ego corridor. Marked roads: 44–88% corridor validity, width median 3.43m (sane). Unmarked scenes (parking lot, intersection idle, dark night) correctly report no-corridor instead of hallucinating. No vector lane GT in mini → qualitative + degradation stats per exit criteria. |
| M5 | Risk layer | M2, M3 (M4 optional) | TTC + lane assignment + RSS distance, per-object risk report | ✅ done — TTC w/ delta-method CI from KF covariances + 30m trust gate; RSS longitudinal (arXiv:1708.06374); lane zones vs curved M4 corridor w/ recorded fallback; ped crossing intent; category velocity-plausibility gate (kills ghost 35 m/s pedestrians). Transparent score + reasons on every record. Full pipeline in adp/pipeline.py: p50 61ms / p95 117ms. Stats: 531 trustworthy TTCs, 29 RSS violations, 69 crossing-intent events, top moments all "car in ego lane closing fast" (sane). Night scenes lean on fallback corridor — flagged per record. |
| M6 | Metric lift v2 (depth) | M3 | Depth Anything V2 / Metric3D lift; keep only if it beats IPM | ✅ done — verdict: **hybrid, but live risk pipeline keeps IPM**. DA-V2-metric (39ms fp16): far field wins big (30–50m: 12.7% vs IPM 22.8%; 50m+: 15.4% vs 39.3%) but severe nonlinear near-field bias (0–10m: 58%!, gt/depth ratio 0.63→1.06 — not scale-calibratable). Occlusion hypothesis DISPROVEN: IPM still edges depth on occluded objects <50m. HybridLift (IPM<30m, depth beyond): 4.1/10.6/14.6/15.4% — best of both; opt-in via PerceptionPipeline(use_depth=True). Default off: TTC/RSS working range is IPM's domain anyway. |
| M7 | Demo & evaluation report | all | Side-by-side camera + BEV visualization, per-condition metrics | ✅ done — REPORT.md: per-condition (day/night) tables for detection/tracking/lift/lanes, latency-as-staleness accounting (61ms p50 → ~1.8m @30m/s), measured-limitations section, artifact index. Composite renderer (adp/viz/compose.py): risk-colored boxes + TTC labels + why-text + BEV w/ corridor, velocity vectors, uncertainty rings, RSS arc. Demo videos: scene-0103 (day), scene-1100 (night) via scripts/run_demo.py. |

M4 and M6 are the designated cut candidates. M6 is explicitly conditional: it ships only if it
measurably beats IPM on the M3 validation set.

---

## M0 — Data & calibration (don't skip)

**Goal:** a dataset with 3D ground truth wired up end-to-end, and camera geometry code that is
tested before anything downstream consumes it.

Tasks:
1. Download **nuScenes** (start with `v1.0-mini` for pipeline development, full `trainval` later).
   KITTI is the fallback if disk/bandwidth is a problem. BDD100K only if 2D-scale diversity is
   needed later — it cannot validate distances (2D-only).
2. Build a data-access layer over `nuscenes-devkit`: iterate samples → (image, intrinsics K,
   extrinsics T_cam→ego, 3D box annotations, lidar points).
3. Camera model utilities with unit tests: project 3D→2D, back-project pixel + ground plane → 3D,
   undistort. Test by projecting nuScenes 3D GT boxes into images and checking overlap with 2D boxes.
4. Dashcam path (later, optional): checkerboard calibration script, or focal-length estimate from
   vanishing points. Parked until M7.

**Exit criteria:**
- Can render any nuScenes sample with 3D GT boxes projected into the camera image correctly.
- Geometry utils pass round-trip tests (project → back-project error < 1px on synthetic points).

## M1 — Detection

**Goal:** reliable 2D boxes for {car, truck, bus, pedestrian, cyclist} at real-time rates.

Tasks:
1. Baseline with pretrained RT-DETR and YOLOv8/v11 (no training) on nuScenes camera frames;
   measure per-class AP against 2D projections of GT. Pick the better latency/accuracy point.
2. Fine-tune the chosen model on the 5 target classes. Do **not** train from scratch.
3. Export to ONNX immediately; run under ONNX Runtime (TensorRT later if GPU allows). Latency is a
   design constraint, not a retrofit — record ms/frame from day one.
4. Eval harness: per-class AP@0.5, and latency percentiles (p50/p95), logged per run.

**Exit criteria:**
- ≥ baseline pretrained AP on all 5 classes after fine-tuning (no regressions).
- ONNX inference numerically matches PyTorch (within tolerance) and runs ≥ 30 FPS on target hardware,
  or the actual budget is measured and documented.

## M2 — Tracking

**Goal:** stable IDs over time and smoothed velocity — this is what turns detection into perception.

Tasks:
1. ByteTrack first (association only, trivial to integrate); OC-SORT as the comparison.
2. Per-track Kalman filter state includes velocity — this is the *only* place velocity is ever
   computed. Never difference raw per-frame positions downstream.
3. Track lifecycle tuning: birth/death thresholds, occlusion tolerance (nuScenes has heavy
   occlusion at intersections).
4. Metrics: MOTA/IDF1 against nuScenes GT tracks (project 3D GT tracks to 2D for comparison).

**Exit criteria:**
- IDs persist through short occlusions in qualitative review of ≥ 5 scenes.
- IDF1 measured and logged; velocity estimates visibly smooth (no sign-flipping frame to frame).

## M3 — Metric lift v1: ground-plane homography (IPM)

**Goal:** BEV position per tracked object, with honest error bars. This is the first "ours" module.

Method: assume flat road; use camera height + intrinsics; project each box's bottom-center ray to
the road plane → (x, y) in ego frame. Cheap, fast; known failure modes: hills, banked curves,
occluded box bottoms.

Tasks:
1. Implement IPM lift on box bottom-centers → ego-frame BEV coordinates.
2. Feed BEV positions into the M2 Kalman filters (track in BEV space, not image space) so velocity
   comes out in m/s.
3. **Validation harness (the load-bearing task):** for every tracked object matched to a nuScenes
   GT box, compare estimated range vs lidar/GT range. Report error as a function of distance
   (bins: 0–10m, 10–30m, 30–50m, 50m+), and separately for flat vs sloped scenes.
4. BEV canvas visualization: ego at origin, tracked objects with velocity vectors, GT ghosts
   overlaid for debugging.

**Exit criteria:**
- Range error curve published in the eval report. Target: **≤ 15% median error at 30m** on flat
  scenes. If met, M6 (depth) must justify its extra latency against this number.
- Failure modes demonstrated, not hidden: at least one hill/slope scene documented with its error.

## M4 — Lanes (cut candidate #1)

**Goal:** lane geometry in BEV so risk can say "in my path", not just "a car exists".

Tasks:
1. Pretrained CLRNet or LaneATT for image-space lane points (no training initially).
2. Project lane points to BEV via the M3 homography; fit polynomials/splines **in BEV space**
   (image-space curve fits are perspective-distorted and useless for geometry).
3. Ego-lane identification + adjacent-lane bands (simple lateral-offset model is fine).
4. Temporal smoothing of lane fits (lanes don't teleport).

**Exit criteria:**
- Ego lane correct in qualitative review across highway + urban scenes; degrades gracefully
  (reports "no lane" rather than hallucinating) under occlusion.

## M5 — Risk layer

**Goal:** defensible, computable risk — every score traceable to published quantities, never
pixel-area heuristics.

Components:
1. **TTC** = longitudinal gap / relative closing speed, from BEV Kalman state (M3 + M2).
   - Report with a confidence interval propagated from the Kalman covariance.
   - Gate by range: TTC flagged low-confidence beyond ~30m (distance error compounds into the
     ratio; the velocity derivative is garbage at 50m). Trustworthiness threshold is measured
     in M3, not assumed.
2. **Lane assignment** (needs M4; fallback: straight-ahead ego corridor of fixed width): ego lane /
   adjacent / off-path per object.
3. **Pedestrian intent**, cheap version: BEV position relative to road edge + lateral velocity —
   is the velocity vector pointed into the ego corridor within N seconds?
4. **RSS minimum safe distance** (Mobileye's Responsibility-Sensitive Safety): compute the formal
   longitudinal + lateral safe-distance envelope per object; flag violations. Cite the paper —
   this replaces invented thresholds with a published model.
5. Output contract: per-object structured risk record — `{id, class, range, closing_speed,
   ttc ± ci, lane, rss_violation, score}` — plus one scalar ego risk. The overlay must render the
   *why* ("TTC 1.8s, same lane, closing 12 m/s"), never just a color-coded box.

**Exit criteria:**
- Risk output is fully explainable from the record fields; no magic numbers without a cited source.
- Sanity suite: hand-picked scenes (lead car braking, pedestrian crossing, empty road) produce the
  obviously-correct ranking.

## M6 — Metric lift v2: monocular depth (conditional; cut candidate #2)

**Goal:** beat IPM where it breaks (slopes, occluded box bottoms) — *measured*, not assumed.

Tasks:
1. Depth Anything V2 (metric variant) or Metric3D; sample depth in the lower box region,
   median-filter to a range estimate.
2. Run the exact M3 validation harness on the same scenes. Compare error curves and latency.
3. Decision gate: adopt depth (or a hybrid: IPM near/flat, depth on slopes) only if the accuracy
   gain justifies the added inference time (~100–200ms class of cost). Otherwise document the
   negative result — that's a finding, not a failure.

Monocular 3D detection (FCOS3D / MonoFlex / BEVFormer-style) is noted as the stretch beyond this,
out of scope unless M0–M7 land early.

## M7 — Demo & evaluation report

**Goal:** the artifact that proves this is a perception system, not a YOLO tutorial.

Tasks:
1. Side-by-side visualization: camera overlay (boxes, IDs, lane fits, risk annotations with
   reasons) + BEV canvas (ego, objects, velocity vectors, lane geometry, RSS envelopes).
2. End-to-end runner: nuScenes scene → annotated video. Optional: calibrated dashcam footage.
3. **Per-condition evaluation report:** all metrics broken down by day/night/rain and
   flat/sloped. Clean-daytime-only numbers are dishonest; per-condition breakdowns are more
   honest and more interesting.
4. Latency accounting: per-module ms + end-to-end staleness in meters at 30 m/s (200ms = 6m).
   Either run modules at different rates or document the staleness explicitly.

**Exit criteria:** a demo video + a metrics report someone skeptical could audit.

---

## Cross-cutting engineering rules

- **Velocity comes from the Kalman filter only.** Never difference raw depths/positions.
- **Every metric estimate carries uncertainty** (range error %, TTC CI) and it propagates downstream.
- **Latency budget is tracked per module from day one**; serial stacking of detect + depth + track +
  lanes can hit 200ms without noticing.
- **Validate against ground truth before trusting any module** — a distance estimate that was never
  checked against lidar is fiction.
- **Failure modes get documented, not hidden**: hills for IPM, night/rain/glare for everything.

## Stack

Python · PyTorch · ONNX Runtime (TensorRT if available) · OpenCV for geometry ·
`nuscenes-devkit` for data · NumPy/SciPy for filters and fits.

## Proposed repository layout

```
adp/
  data/        # nuScenes access layer, sample iteration, GT matching
  calib/       # camera model, projection/back-projection, homography
  detect/      # model wrappers, ONNX export, inference
  track/       # ByteTrack/OC-SORT integration, BEV Kalman filters
  lanes/       # lane detection wrapper, BEV curve fitting
  lift/        # IPM lift, depth lift, common lift interface
  risk/        # TTC, lane assignment, pedestrian intent, RSS
  viz/         # camera overlay, BEV canvas, side-by-side composer
configs/       # per-experiment YAML (model, thresholds, latency budgets)
scripts/       # download data, run pipeline, export models, make demo video
eval/          # metric harnesses (AP, IDF1, range-error curves, per-condition)
tests/         # geometry round-trips, filter sanity, risk sanity scenes
```

## Sequencing note

M0 → M1 → M2 → M3 is a strict chain and is the critical path — at the end of M3 the project is
already "real" (metric BEV perception, validated). M4 can proceed in parallel after M0. M5 needs
M3; it gains lane assignment when M4 lands. M6 only runs if M3's numbers leave room to improve.

---

# Phase 2 — validation & hardening (M8–M11)

M0–M7 built and measured the system. Phase 2 addresses what those measurements exposed. Nine
improvements, grouped into four milestones; improvement numbers (#1–#9) are referenced in commits
and issue text.

| # | Milestone | Improvements | Status |
|---|-----------|--------------|--------|
| M8 | Risk-layer quantitative validation | #1 | ✅ done — mixed result, see below |
| M9 | Width-prior range fusion | #2 | ✅ done — adopted as default lift |
| M10 | Robustness experiments | #3, #4 | planned |
| M11 | Engineering hardening | #5–#9 | planned |

## M8 — Risk-layer quantitative validation (#1)

**The gap:** every other stage has a validated error curve; M5 has sanity tests and an eyeball
audit of top moments. The scoring weights in `adp/risk/scoring.py` (0.30 ego base, 0.45 TTC ramp,
0.25 RSS, 0.20 intent) are defensible but unvalidated — the risk layer is the part that makes this
a perception *system* rather than a distance estimator, and it currently rests on hand-picked
constants.

**The proxy label, already in the downloaded data:** differentiate `ego_pose` over each scene
(offline, acausal smoothing — appropriate for label generation) to get ego longitudinal
acceleration, then label each keyframe by whether hard braking begins within a horizon.

Tasks:
1. `adp/eval/braking.py` — ego kinematics from pose trajectory (Savitzky-Golay smoothed speed and
   acceleration), hard-braking event labelling with configurable threshold/horizon, and exclusion
   of frames where the ego is already stopped (braking is undefined there).
2. `adp/eval/ranking.py` — ROC-AUC and average precision, implemented directly (no new dependency,
   independently testable).
3. `scripts/eval_risk_validation.py` — run the pipeline once, cache per-keyframe features + labels,
   then compute metrics from cache (`--from-cache`) so threshold sweeps are free.
4. **Baseline comparison — the project's central claim, made testable:** does the metric-grounded
   risk score predict human braking better than
   (a) max box-area fraction (the plan's explicit strawman: "risk derived from box area"),
   (b) nearest-object range alone (metric, but no TTC/lanes/RSS),
   (c) inverse-TTC alone (isolates what lane assignment + RSS + intent add), and
   (d) ego speed alone (trivial predictor control)?
5. Threshold/horizon sensitivity: report at ≥2 braking thresholds so the result isn't an artifact
   of one cut.

**Exit criteria:** ROC-AUC and average precision for the risk score and all four baselines, with
base rate and event counts stated; the risk score beats the box-area strawman, or the loss is
reported honestly. Caveats documented (drivers brake for lights/turns, not only for risk objects;
10 scenes is a small sample; this is correlation with human behaviour, not ground-truth danger).

### M8 result (hard braking ≤ −3.0 m/s² within 5s; n=215, 25 positive, base rate 11.6%)

| predictor | ROC-AUC | 95% CI | AP vs base rate |
|---|---|---|---|
| **inverse TTC only** | **0.709** | [0.62, 0.80] | **1.91×** |
| risk score (composite) | 0.642 | [0.54, 0.75] | 1.38× |
| ego speed only | 0.610 | [0.53, 0.68] | 1.20× |
| box area (strawman) | 0.548 | [0.46, 0.63] | 1.04× |
| nearest object range | 0.543 | [0.45, 0.62] | 1.04× |

1. **Metric grounding validated.** Inverse TTC — a quantity that exists only because of the M3
   lift and M2 Kalman velocity — is the strongest predictor and the only one whose CI clears
   chance. It beats the box-area strawman decisively (0.71 vs 0.55, CI overlapping 0.5).
2. **The composite score is worse than its own best component** (0.64 vs 0.71). The flat +0.30
   ego-lane base swamps the TTC ramp and parks 33% of frames on one tied value. Confirmed
   mechanically: excluding tied frames doesn't help (AUC 0.479), so the aggregation weighting is
   wrong — the inputs are fine.
3. **At gentle thresholds (−1.5 m/s²) nothing predicts well and box area leads.** Expected, and
   worth stating: gentle urban deceleration is traffic-density and traffic-light driven, and this
   system perceives neither. Box area proxies traffic density; it is not a danger measure.

**Follow-up (M8b, open):** restructure scoring so dynamics dominate static lane membership — zone
as a *multiplier* on dynamic terms rather than an additive base, and continuous RSS-gap ratio
instead of a binary violation flag. Deliberately NOT fitting the existing constants to this
sample: 25 positive events across 10 scenes would overfit, and the sensitivity sweep shows the
label's own character changes with threshold. The structural change should be validated on full
trainval, where the event count supports it.

## M9 — Width-prior range fusion (#2)

**The gap:** M3 documented gradual occluded-box-bottom drift (range biased high, phantom velocity);
M6 *disproved* depth as the fix. The remedy has to be geometric.

Tasks: box pixel-width + class width priors (car ≈1.8m, truck ≈2.5m) give a range estimate
independent of the bottom edge; fuse as a second Kalman measurement with its own noise model, and
use IPM-vs-width disagreement as an occlusion detector. Re-run the M3 harness for the verdict.

**Exit criteria:** range-error curve vs IPM baseline, broken down by visibility level (the
occluded-object bins are the ones that must improve); if it doesn't help, document like M6.
Secondary win if it lets the velocity-plausibility gate in `scoring.py` become a backstop rather
than a load-bearing patch.

### M9 result — adopted as the default lift (`use_width=True`)

Median relative range error, n=1480 matched pairs:

| GT range | IPM (old default) | width alone | **fused (new default)** | M6 depth hybrid |
|---|---|---|---|---|
| 0–10m | 4.1% | 5.1% | **3.7%** | 4.1% |
| 10–30m | 11.0% | 6.3% | **6.2%** | 10.6% |
| 30–50m | 22.8% | 7.6% | **7.6%** | 14.6% |
| 50m+ | 39.3% | 9.4% | **9.4%** | 15.4% |

By visibility (<50m): occluded (vis 2–3) **13.2% → 5.5% (−59%)**; clear (vis 4) 12.6% → 6.7%
(−47%). The hypothesis was that occluded objects would improve; they improved most, but clear
objects improved nearly as much.

Five findings, each forced by the previous result:

1. **The width cue is far stronger than expected** and survives leave-one-scene-out validation
   (8.2% held-out vs 7.5% in-sample — only +10% optimism), so it is not calibration contamination.
2. **Aspect-ratio conditioning is a trap if taken at face value.** It cuts spread 32.6% → 11.7%
   for cars, but substituting W = a + b·(w/h) into z = f·W/w_px gives z = f·a/w_px + f·b/h_px —
   72–82% of the estimate is *height*-derived, the very dimension occlusion truncates. Two
   estimates are therefore maintained: conditioned (accurate) and unconditioned (truncation-immune).
3. **Single-frame occlusion detection is impossible** at 33% spread; the gate accumulates a
   per-track EMA of normalised disagreement instead, exploiting the fact that barriers persist
   while noise does not.
4. **IPM's analytic noise model was fantasy** — measured σ is 4–6× larger and grows as r^2.51
   (`scripts/calibrate_lift_noise.py`, n=1480), with large positive *bias* at range (+11.5m at
   40–60m). Now the default in `GroundPlaneLift.range_meas_std`; the analytic form is retained
   behind `measured=False`.
5. **Fusion loses to width alone beyond ~12m** because inverse-variance fusion assumes unbiased
   inputs and IPM is biased — any non-zero weight drags the estimate long. Hence the switch:
   fuse below 12m, drop IPM above.

Also measured and rejected: substituting the truncation-immune estimate when the gate fires. It
costs 11.7% → 33% spread on ~30% of samples, outweighing the ~20% truncation bias it avoids
(10–30m bin degraded 6.3% → 10.4%). The immune estimate stays a detector only.

**Consequence for M6:** the width prior supersedes the depth hybrid — better beyond 10m and
~39ms/frame cheaper. `use_depth` is retained only to reproduce the M6 comparison.

## M10 — Robustness experiments (#3, #4)

**#3 — Pitch sensitivity without hill data.** mini has no slope scenes, but perturbing camera pitch
±1–3° in the extrinsics and re-running `eval_ipm.py` produces an error-vs-pitch curve. Converts
"every IPM system lies on hills" from caveat to measured sensitivity, and covers ego pitch under
braking/acceleration, which happens on flat roads too.

**#4 — Localization dependence.** BEV filters live in the global frame using nuScenes `ego_pose` —
ground-truth localization a camera-only system would not have. Re-run M3 with ego motion from a
cheap visual-odometry / constant-turn-rate estimate and report the delta.

**Exit criteria:** both curves in REPORT.md; #4's result reframes the headline accuracy numbers
honestly (either "barely matters" or a stated dependency).

## M11 — Engineering hardening (#5–#9)

- **#5 Config consolidation.** The `configs/` directory in the layout above was never built;
  thresholds are scattered across eight modules (ByteTrack defaults, `GATE_MD2`/`MAX_REJECTS`,
  RSS params, `MAX_TRUST_RANGE`, scoring weights, `SWITCH_RANGE_M`, lane constants). One
  dataclass/YAML — prerequisite for M8 weight tuning and any sweep.
- **#6 Provenance in eval outputs.** Every `out/eval/*.json` should carry git SHA, timestamp, and
  the config dump. Makes REPORT.md genuinely reproducible rather than nominally so.
- **#7 End-to-end pipeline test.** All current tests are module-level; nothing instantiates
  `PerceptionPipeline`. A ~10-frame smoke test with loose assertions (no NaNs in BEV state,
  well-formed risk records, metrics above floors) catches integration breakage unit tests
  structurally cannot.
- **#8 Lane-stride hold-timer bug.** `pipeline.py` calls the corridor smoother with `dt=0.0` on
  lane-skip frames, so the 1s hold advances only on update frames — with `lane_stride=2` the
  effective hold is ~2s wall-clock. Charge real time or make the intent explicit.
- **#9 Risk bucket hysteresis.** Nothing debounces low/caution/high, so an object hovering at a
  threshold oscillates frame to frame. Enter at 0.6 / exit at 0.5 plus 2–3 frame confirmation.

**Exit criteria:** #5–#7 landed with tests; #8 fixed with a regression test; #9 verified on the
demo videos (no visible flicker).

## Larger, later (not part of the nine)

- **Fine-tune on full trainval** — the known fix for cyclist AP 0.13 and the night recall drop;
  biggest number-mover across every downstream stage, costs a ~300GB download and GPU hours.
- **Tiled/sliced inference on the far field** — recall is 31% beyond 50m largely because objects
  are tiny at 960px; slicing the upper-centre region may recover some without retraining. Gate on
  the latency cost.
- **Multi-camera** — nuScenes has six; surround coverage would catch adjacent-lane cut-ins earlier.
  Substantial work, and the risk layer is forward-focused, so this ranks last.
