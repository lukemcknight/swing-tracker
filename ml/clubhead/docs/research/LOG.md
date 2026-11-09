# Clubhead detector research log

One verified, new-to-this-log finding per entry. Nothing here should repeat
an earlier entry. Entries are appended in run order (newest at bottom).

---

## 2026-11-09

**Note on sources:** this run's brief pointed at `ml/clubhead/eval/results/v1-failure-analysis.md`
and `ml/clubhead/docs/data-engine.md` for ground truth. Neither file exists at the
current `HEAD` (detached at `mvp/club-path-replay`) — the repo's clubhead work is
at an earlier CreateML-baseline stage (`ml/clubhead/eval/baseline/createml-v6.md`:
30.4% detection, 80% phantom rate on the 3-clip held-out set), not the
YOLO11n/CoreML/82%-detection state the brief describes. `labeling-spec.md` does
exist and does confirm the "box the full motion-blur streak" convention the brief
cites. I could not re-verify the specific 82%/85.9%/54%-Roboflow/elongation-1.60
figures against repo files, so this entry evaluates the finding on its own merits
rather than against those numbers.

**What it is:** `SoccerSynth-Detection` (Qin, Yeung, Umemoto, Fujii — Nagoya
University / Fujii Lab, 2024), a synthetic data generator and companion dataset
for soccer-player detection that explicitly simulates camera motion blur, random
lighting and textures. Trained/fine-tuned with YOLOv8n and benchmarked against
real datasets (SoccerNet-Tracking, SportsMOT): the paper reports the synthetic
set "matched the performance of real datasets and significantly outperformed
them in images with motion blur," and improved results further when used as a
pre-training set ahead of fine-tuning on real data.

- URL: https://github.com/open-starlab/SoccerSynth-Detection
- Licence (verified by reading `LICENSE.md` in the repo directly): Apache
  License, Version 2.0. Apache-2.0 is permissive and explicitly allows
  commercial use, modification and redistribution, subject to attribution and
  marking changed files (Section 4) and the standard "AS IS" warranty
  disclaimer (Section 7). **Commercial use: permitted.**
- The repo links a Windows data generator and the generated dataset via
  Google Drive; I could not confirm those specific Drive links are still live
  (did not attempt the download), only that the repo, its README claims and
  its licence file are real and current.

**Which failure mode it addresses:** motion blur (primarily), with secondary
relevance to camouflage/low-contrast via its "broad range of random lighting
and textures."

**Why it helps this model specifically:** This is not a golf dataset — it's
evidence that a *synthetic motion-blur generation pipeline* measurably closes
the blur gap for a small, fast-moving sports object detected with a YOLO
model, which is close to our exact situation (YOLO11n, clubhead vs. soccer
player — different object, same blur/lighting problem class). Our own blur
gap is a labeling/sourcing problem, not a modeling one: real footage with
genuinely blurred, labeled clubheads is scarce (median box elongation ~1.6
per the brief), and acquiring more real blurred footage is slow. A generator
that can render a 3D clubhead (or even a simple elongated proxy mesh) swinging
through a scene with simulated camera/object motion blur, varied lighting,
and auto-generated tight bounding boxes would let us manufacture exactly the
class of examples we're short on, at whatever volume and blur-severity
distribution we choose, with ground-truth boxes that don't depend on a human
annotator's eye for "where does the streak end."

**Effort vs. payoff:** High effort, potentially high payoff. The tool itself
doesn't hand us golf data — it's a precedent and possibly an adaptable
pipeline (engine/blur method aren't documented on the repo page, so the
generator internals would need to be inspected directly, and a golf club head
+ swing motion would need to be modeled from scratch; soccer players are a
very different asset to author than a thin elongated metal/graphite object
whose blur streak geometry is the entire point). Realistic path: treat this
as a reference implementation to study (how they generate the blur kernel,
how they randomize lighting/texture, how they auto-label) rather than a
drop-in tool, then build a minimal custom synthetic-swing renderer scoped
narrowly to clubhead + blur streak, which is a much smaller asset-authoring
problem than full synthetic soccer scenes. Not a quick win, but it's the
first concrete, licence-clean technique found so far that directly targets
manufacturing the motion-blur examples the training set is short on, rather
than just hoping for more real footage.
