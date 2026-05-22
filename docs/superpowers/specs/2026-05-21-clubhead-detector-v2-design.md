# Clubhead Detector v2 — Design

- **Date:** 2026-05-21
- **Status:** Approved design, ready for implementation planning
- **Branch context:** `mvp/club-path-replay`
- **Supersedes the detection foundation behind:** the slice-diagnosis work (`SwingPathDiagnoser`), which depends on a clean clubhead path.

## 1. Context & problem

SwingSensei's phase-1 feature is a slice fixer, built on an on-device clubhead
detector whose per-frame detections are cleaned, bridged, and stitched into a
club path. The current detector is a CreateML object detector (single
`clubhead` class, "Full Network" architecture, grid size 19). Its latest
training run (`swing-tracker-ml 6`, 1000 iterations) scores:

| Metric | Score |
|---|---|
| mAP @ IoU 50% | 0.63 |
| mAP @ varied IoU (strict localization) | 0.18 |
| Training loss (converged) | ~0.36 |

Running the app on real swings shows **all four** failure modes:

1. **Wobbly / jittery path** — the clubhead is found but boxes are loose, so the
   traced arc is noisy frame-to-frame. The 0.63 → 0.18 gap between IoU50 and
   varied-IoU mAP is this problem stated numerically: roughly-right boxes, not
   tight ones.
2. **Missed frames / dropouts** — the model loses the clubhead on parts of the
   swing, especially the fast, motion-blurred downswing.
3. **Fails on new clips** — works on the developer's own clips, breaks on
   different phones, lighting, angles, and backgrounds.
4. **Wrong / phantom detections** — locks onto hands, ball, or background
   clutter instead of the clubhead.

**Root causes.** The training data is ~2,200 frames that are almost entirely
the developer's own swings (1,360 app captures plus a handful of `IMG_46xx`
phone clips and range/impact clips) — one or two phones, one or two settings.
That homogeneity explains symptom 3 directly. CreateML compounds it: it is a
convenience tool with the controls hidden — no augmentation control, a coarse
YOLOv2-era architecture, an opaque (almost certainly random) train/val split
that leaks near-identical adjacent frames and inflates reported metrics, and no
per-epoch metrics, no test set, no reproducibility.

This is not a tweak. It is a rebuild of the detection foundation, and it is a
prerequisite for continuing the slice-analysis work.

## 2. Goals & success criteria

**Deliverable:** "Clubhead Detector v2" — a YOLO-based on-device detector that
replaces the CreateML model, plus the data, training, and evaluation
infrastructure to keep improving it.

**Success criteria** — measured by a committed eval harness on a held-out test
set of full swings from sources never used in training:

- **Baseline first.** The harness's first run scores the *current* CreateML
  model (and the developer's in-progress model) to establish exact starting
  numbers before final targets are locked.
- Per-frame detection rate inside the trackable swing window — a large jump
  over baseline, targeting ~95%+.
- Median center-point error ≤ ~1.5% of frame width; p90 ≤ ~3%.
- Phantom detections on no-clubhead frames near zero.
- **Generalization is the real bar:** test sources are phones / players /
  settings absent from training.

Target numbers are provisional and will be confirmed against the measured
baseline; the harness, not intuition, decides when the model is good enough.

## 3. Out of scope

- Keypoint / pose model (predicting the clubhead as a point) — a possible
  phase 2 *if* measured wobble persists after boxes are tight.
- Temporal / multi-frame model — deferred until the per-frame model is good.
- The slice analysis itself (`SwingPathDiagnoser` and downstream) — resumes
  after the detector is rebuilt.
- Ball detection.

## 4. Decision: YOLO over CreateML

The model path is **Ultralytics YOLO11n, trained in Python, exported to Core
ML.** Rationale, mapped to the four symptoms:

- **Wobbly path** — YOLO11 is anchor-free with a modern feature-pyramid
  backbone built for small objects, and input resolution is a controlled knob
  (start 960). Finer architecture + higher resolution is the direct fix for
  box tightness.
- **Missed frames** — YOLO allows explicit motion-blur augmentation so the
  model trains on blurred clubheads on purpose.
- **Fails on new clips** — the train/val/test split is explicit (folder
  arrangement), so it can be done **by source video** and the val/test numbers
  finally reflect real generalization instead of frame-adjacency leakage.
- **Phantom detections** — background/negative images (empty label files) are
  first-class hard negatives.

It also gives per-epoch metric curves, PR curves, a confusion matrix, and
per-image prediction visualizations; reproducibility (training is a committed,
diffable script + config); and a clean Core ML export (`nms=True`) that Vision
still consumes as a standard object detector — so the Swift integration barely
changes. YOLO11n is ~2.6M parameters, sized for the Neural Engine, and trains
locally on Apple Silicon (MPS) in tens of minutes — no cloud.

It is a true drop-in for the current architecture: still box → center, so
existing box labels are reused as-is and `ClubHeadDetector.swift` barely
changes. This keeps integration risk low while unlocking the controls that
fix the symptoms.

## 5. The data engine

The labeling bottleneck — hand-drawing every box — is the project's real
constraint. The fix is to stop *drawing* boxes and start *correcting* them, and
to never label a frame a machine could have labeled. Section 6's dataset work
is therefore not "label everything, then train once"; it is an **iterative
loop**: label a batch → train → pre-label + track-propagate the next batch →
correct → auto-accept the easy frames → retrain → repeat. Each turn the model
improves *and* the labeling gets faster.

Four throughput multipliers, all in scope:

1. **Model-assisted pre-labeling loop.** Each iteration's trained model
   pre-fills boxes on the next batch of unlabeled frames; the human nudges and
   accepts rather than drawing from scratch. `build_label_studio_tasks.py`
   already does this with `loose_seed` predictions — it is rewired to consume
   the current trained model's output.
2. **Tracker-based interpolation between keyframes** — the biggest single
   multiplier. The clubhead moves on a smooth arc; a human labels sparse
   keyframes and an OpenCV CSRT tracker propagates boxes across the swing, with
   the human fixing only where the tracker drifts. Slow phases (address,
   top-of-backswing) propagate near-perfectly; only the fast downswing needs
   dense attention. Realistically 5–10× throughput.
3. **Smart frame sampling.** Consecutive near-identical frames teach the model
   almost nothing but cost full effort each. A sampling policy queues frames
   densely only where appearance changes fast (downswing/impact) and sparsely
   elsewhere — diversity across swings beats density within a swing.
4. **Confidence-based auto-accept.** Once the model is decent, predictions that
   are both high-confidence and part of a smooth track are auto-accepted as
   labels with light spot-checking; human clicks go only to low-confidence or
   disagreement frames. Kicks in from the second iteration onward.

Raw source footage comes primarily from YouTube — down-the-line and face-on
swings, instruction channels, slow-mo compilations — each new video a new
"source" for generalization.

## 6. Dataset

**Sources.** Consolidate the existing ~2,200 own-swing labeled frames and the
developer's in-progress labels with the Label Studio pool (~2,382 YouTube
frames plus range/impact). The in-progress labeling work folds straight in.

**Labeling spec** (a new doc written in Phase 0). Defines precisely what
"clubhead" means, partial-occlusion handling, and the motion-blur rule.
**Decided convention:** box the full visible clubhead *including* its
motion-blur streak — it is consistent and matches what is on screen. Flagged
for revisit in phase 2 if it proves to drive wobble.

**Negative frames.** Include frames with no clubhead (address, far
follow-through, non-swing) as empty label files — first-class hard negatives
against phantom detections.

**By-source split.** Each source video belongs to exactly one of train / val /
test. No frame from a test swing ever appears in train. A committed
`split_manifest.json` records the source → split assignment. This is the single
most important fix for "fails on new clips."

**Format.** Export to YOLO format — images plus `.txt` label files plus a
`data.yaml`. Raw images stay under `data/` (gitignored); the split manifest and
configs are committed.

**Target.** Label the full diverse pool, prioritizing distinct
swings / phones / angles over raw frame count. Rough end state: ~4–6k labeled
frames across 40+ distinct sources, grown iteratively via the data engine.

## 7. Training pipeline

All model code lives in a new `ml/clubhead/` directory:

```
ml/
  .venv/                  # gitignored; ultralytics + torch
  clubhead/
    requirements.txt
    README.md
    config/
      data.yaml           # YOLO dataset config (paths + class)
      hyp.yaml            # hyperparameters + augmentation knobs
    train.py              # Ultralytics training entry point
    export.py             # Core ML export
    eval/
      evaluate.py         # eval harness (see Section 8)
      reports/            # generated; gitignored
    label_tools/
      sample_frames.py    # smart frame sampling
      track_propagate.py  # CSRT keyframe -> track propagation
      build_tasks.py      # pre-label Label Studio task builder
      auto_accept.py      # confidence-based auto-accept
    dataset/
      split_manifest.json # source -> train/val/test assignment
```

Raw images and Label Studio task JSON stay in `data/` (already gitignored).
Existing labeling scripts under `data/labeling/clubhead/` migrate into
`ml/clubhead/label_tools/` opportunistically as they are touched.

**Environment.** Dedicated venv at `ml/.venv`; `requirements.txt` pins
`ultralytics` and `torch`. Training runs locally on MPS.

**Model.** YOLO11n (nano). If `n` underperforms after a couple of data-engine
iterations, `11s` is a one-line change.

**Config — all version-controlled knobs.** Input resolution starts at 960.
Augmentation includes explicit motion blur (via Albumentations), HSV jitter,
scale, translation, flip, and mosaic. `train.py` reads `data.yaml` + `hyp.yaml`
so every run is reproducible and diffable.

**Export.** `export.py` runs `model.export(format='coreml', nms=True)`,
producing a `.mlpackage` that Vision consumes as a standard object detector.

## 8. Eval harness

`ml/clubhead/eval/evaluate.py` runs any model — old CreateML or new YOLO — over
the held-out test swings and reports:

- **Per-frame detection rate** within the trackable window.
- **Center-point error** — median + p90, as a percentage of frame width.
- **Phantom rate** — false positives on negative frames.
- **Path jitter** — a curvature-noise metric on the resulting center sequence.
- **Visual overlays** — predicted path vs. ground truth drawn on the video.

The harness's first run scores the current CreateML model and the in-progress
model to fix the baseline. It is the scoreboard for every data-engine
iteration and the authority on when the success criteria are met.

## 9. iOS integration

- Replace `ios-native/SwingSenseiNative/ClubHeadModel.mlmodel` with the new
  `.mlpackage`.
- Because the export is Vision-compatible, `ClubHeadDetector.swift` changes are
  limited to the model reference — `VNCoreMLModel`,
  `VNRecognizedObjectObservation`, and the existing `bestDetection` logic
  (including the Vision bottom-left → top-left flip) stay.
- Re-tune the confidence threshold against the eval harness.
- `ClubHeadDetectionService.cleanedAndBridged` and `ClubPathStabilizer`
  post-processing stay untouched — this work fixes the detector, not adds
  smoothing (consistent with the no-path-smoothing decision).
- Verify per-frame inference latency on a real device.

## 10. Sequencing

**Phase 0 — Foundations.** Stand up `ml/clubhead/`, the venv, and Ultralytics.
Build the eval harness and assemble the held-out test set. Run the **baseline**
on the current CreateML model and the in-progress model. Write the labeling
spec.

**Phase 1 — First YOLO model.** Consolidate existing + in-progress labels into
YOLO format with a by-source split. Build the data-engine tools (smart frame
sampler, CSRT tracker-propagation, pre-label task builder). Train YOLO11n v1;
eval vs. baseline.

**Phase 2…N — Data-engine iterations.** Each turn: sample frames → pre-label
(model predictions + tracker propagation) → human-correct in Label Studio →
confidence auto-accept the easy frames → add to the dataset → retrain → eval.
Repeat until the success criteria are met or the metric curve flattens.

**Phase Final — Ship.** Export Core ML, integrate into the app, re-tune the
threshold, verify on a real device.

**Scoping note.** This is one coherent spec, but the Phase 2…N loop is
inherently open-ended. The implementation plan will specify Phases 0–1 and the
*first* data-engine iteration concretely, then treat the loop as a documented,
repeatable process rather than a fixed step count.

## 11. Risks & open questions

- **Manual labeling labor.** The plan builds the tooling and process; a human
  still does the correction clicks. The data engine minimizes this but cannot
  eliminate it.
- **CSRT drift on the fast downswing.** Tracker propagation is weakest exactly
  where the clubhead moves fastest; that region will still need dense human
  attention. Acceptable — the tracker's win is the slow phases.
- **YOLO11n capacity.** If nano cannot hit the localization target,
  the fallback is `11s`, at some on-device cost — to be judged against measured
  latency.
- **Core ML export fidelity.** The Vision-compatible export path
  (`nms=True`) must be verified to produce `VNRecognizedObjectObservation`
  results; if not, `ClubHeadDetector.swift` needs raw-output decoding instead.
- **Test-set size.** The held-out set must be large and diverse enough for its
  metrics to be stable; too small and the scoreboard is noisy.
