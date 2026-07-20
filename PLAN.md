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
| M5 | Risk layer | M2, M3 (M4 optional) | TTC + lane assignment + RSS distance, per-object risk report |
| M6 | Metric lift v2 (depth) | M3 | Depth Anything V2 / Metric3D lift; keep only if it beats IPM |
| M7 | Demo & evaluation report | all | Side-by-side camera + BEV visualization, per-condition metrics |

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
