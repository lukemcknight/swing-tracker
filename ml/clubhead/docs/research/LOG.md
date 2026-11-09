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

---

## 2026-11-09 (second run today)

**Environment note:** this session's network egress is restricted to package
registries (PyPI, npm, etc.) and MCP hosts only — `github.com`,
`raw.githubusercontent.com`, `arxiv.org`, `kaggle.com`, `huggingface.co`,
`roboflow.com`, `paperswithcode.com` and `pubmed.ncbi.nlm.nih.gov` all refused
the CONNECT at the proxy (403/404/timeout) when tested directly. This matches
the "egress allowlist" limitation this log has flagged many times before, not
a new regression, so it isn't re-escalated here. Everything below was
confirmed through the WebSearch tool's own fetch (which is not subject to this
session's egress block) rather than a direct page fetch from this session —
same workaround prior entries in this log have used when hitting the same
wall.

**Duplicate check performed:** before searching, `git log --all --oneline --
ml/clubhead/docs/research/LOG.md` was read (229 unique commits touching this
file across all branches) and used to rule out re-logging. Roughly a dozen
candidate leads were checked this run and rejected as duplicates before
landing on the one below: MoSA-Det (PubMed 41933033, unverifiable — PubMed
blocked and no independent corroboration or code found, so dropped rather
than logged on thin evidence), GreenVCOD (arXiv:2501.10914, exists, camouflage
area, no code found — plausible candidate but set aside in favor of a
stronger lead below), DRF-YOLO (World Electric Vehicle Journal, Oct 2025,
YOLOv11-based, ExDark low-light benchmark, no code — set aside as
substantively redundant with the already-logged TinyDark-YOLO entry, same
benchmark/base-architecture/no-code profile), AICaddy, PiTrac,
golf-club-tracking/SwingTrace, LeanCOD, EMIP, TinyDark-YOLO, CAMotion,
Vcamba/Mamba-VCOD, ARKit `sceneDepth`/`ReferenceObject`, and Zero-DCE — all
already in this log under those names.

**What it is:** "Lightweight Multi-Frame Integration for Robust YOLO Object
Detection in Videos" — Yitong Quan, Benjamin Kiefer, Martin Messmer, Andreas
Zell (Chair of Cognitive Systems, University of Tübingen — the same group
behind the already-logged BlurBall/WASB work). Published at ECMR 2025 (European
Conference on Mobile Robots), arXiv:2506.20550, also on IEEE Xplore
(document 11162972). The method: stack `k` consecutive video frames as extra
input channels into an otherwise-unmodified YOLO detector, train normally,
but compute the loss only against the single target frame's labels (the
other stacked frames are context, not separately-supervised outputs). The
paper's own stated motivation is that single-frame detectors struggle with
"transient challenges including motion blur, occlusions, and abrupt
appearance changes," and its qualitative Grad-CAM++ analysis shows the
multi-frame model attending more tightly to the object specifically in
motion-blur, occlusion and glare frames.

- URL: https://arxiv.org/abs/2506.20550 (mirror: https://ieeexplore.ieee.org/document/11162972/)
- Licence: the arXiv page carries the standard arXiv.org perpetual
  non-exclusive licence, which covers the paper text only, not an
  implementation. No GitHub or other code repository was found in multiple
  searches — **existence-only, no code to license-check.** This doesn't block
  using the technique, though: the mechanism itself (stack frames as input
  channels, supervise one frame) is not an artifact anyone can copyright, and
  implementing it is a dataloader/input-shape change to our own existing
  Ultralytics YOLO11n training pipeline, not a dependency on this paper's
  (nonexistent) code release.
- Reported results (BOAT360, a small fisheye maritime dataset the authors
  built for this paper, **509 boxes total** per a third-party critical
  review): 3-frame stacking improved YOLOv7-tiny from 41.1 to 53.9 mAP@0.5
  (+12.8 pts) and gave a smaller +8.1-pt gain for full YOLOv7, over a
  single-frame baseline. The same third-party review flags real weaknesses:
  no error bars or multi-seed runs, a frame-count choice (7 vs. 3 "best")
  that is inconsistent between the paper's own sections, gains that are
  **not monotonic** (too many stacked frames hurts in dynamically-changing
  scenes), and a headline number resting on a very small eval set.

**Which failure mode it addresses:** Motion blur, primarily and explicitly —
it's one of the three named "transient challenges" the paper targets, and the
qualitative examples shown are motion blur, occlusion and glare, not
camouflage. Weak/indirect relevance to camouflage only insofar as both
failure modes share the underlying idea "single-frame appearance is
insufficient, give the model temporal context" — but this paper's own
evidence doesn't touch camouflage-style low-contrast failures at all, so it
should not be read as a camouflage fix.

**Why it helps this model specifically:** Every other temporal/multi-frame
technique already in this log carries real adoption friction — EMIP needs a
frozen optical-flow foundation model and ships no licence; Vcamba needs
Mamba/CUDA kernels and is licence-blocked; YOLOV/YOLOV++ needs a different
detector family (YOLOX) end to end; the SAM2-family entries (EdgeTAM,
EfficientTAM, CamSAM2, WOFTSAM) need a foundation-model dependency this
on-device CoreML pipeline doesn't otherwise have. This technique has none of
that: it is a pure input/training-loop change to the exact detector family
we already use (YOLO11n), requires no new foundation model, no new runtime
dependency, and inherits no third-party licence at all because we'd write it
ourselves. For a clubhead that goes from "clean box" to "blur streak" over 2-3
frames, giving the detector the preceding frame(s) as extra input channels is
a direct, mechanistic answer to exactly the ambiguity a blurred single frame
creates — "where was the head a moment ago" is additional signal a single
blurred frame doesn't carry on its own.

**Effort vs. payoff:** By far the cheapest thing in this entire log to
actually go test on our own data — no licence negotiation, no foundation
model to integrate, no new architecture family, just a dataloader change
(stack N frames) and a training run on data we already have. That is the
entire appeal. The payoff side is genuinely uncertain, not just hedged for
politeness: the one paper measuring this used a 509-box dataset with no error
bars and an admitted frame-count inconsistency, so the headline +12.8 mAP
number is not something to trust at face value — it is a signal that the
idea is worth a same-day internal A/B test (train YOLO11n with 1 vs. 3 stacked
frames on our own held-out set and measure directly), not evidence that it
will work for us. Recommended as a cheap experiment to actually run, not as a
technique to adopt on the paper's word.
