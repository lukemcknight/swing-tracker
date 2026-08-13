# Clubhead detector — research log

Append-only log of verified, external findings that could improve the
clubhead detector. Each entry is added by an automated research run. Entries
already logged are off-limits for future runs — do not repeat them.

Note (2026-08-12, first run): the run brief referenced
`ml/clubhead/eval/results/v1-failure-analysis.md` and `ml/clubhead/docs/data-engine.md`
as read-first context. Neither exists in this repo yet — the repo is
currently at the Phase 0 baseline stage (CreateML v6 baseline captured,
30.4% detection rate; see `ml/clubhead/eval/baseline/createml-v6.md` and
`docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`), not the
later YOLO11n v1 state the brief describes. This run proceeded using the
failure-mode framing given directly in the brief (camouflage vs. motion
blur), since that framing is independently well-supported by the labeling
spec's motion-blur convention and by general small-object-detection
literature, but the specific stats quoted in the brief (82%/77%/85.9%,
elongation percentiles, etc.) could not be cross-checked against repo files.

---

## 2026-08-12 — Frame-averaging / learned motion-blur synthesis (Brooks & Barron, CVPR 2019)

**What it is.** "Learning to Synthesize Motion Blur" (Brooks & Barron, CVPR
2019) is a neural method for synthesizing physically realistic motion blur
from a burst of sharp frames (e.g. high-frame-rate/slow-motion video). Its
core finding, useful even without their trained model: naive temporal
averaging of consecutive high-fps frames is a physically grounded
approximation of real camera-shutter motion blur (a real exposure literally
integrates light over the time the shutter is open), and their learned model
is a refinement that fixes the boundary/ghosting artifacts plain averaging
produces. Reference code (training + eval harness, no confirmed pretrained
weights in what I could inspect) is at
`google-research/google-research/tree/master/motion_blur`.

**URL.** https://github.com/google-research/google-research/tree/master/motion_blur
(paper: Brooks & Barron, "Learning to Synthesize Motion Blur", CVPR 2019,
arXiv:1811.11745 — arxiv.org itself was unreachable from this sandbox's
egress proxy, so the paper's claims here are sourced from the GitHub
README and secondary summaries, not the PDF directly; this should be treated
as one notch below full verification).

**Licence (verbatim, from the `google-research/google-research` repo's
top-level `LICENSE` file, fetched directly).** Apache License, Version 2.0.
Key clause: "perpetual, worldwide, non-exclusive, no-charge, royalty-free,
irrevocable copyright license to reproduce, prepare Derivative Works of,
publicly display, publicly perform, sublicense, and distribute the Work."
**Commercial use: permitted**, subject to standard Apache-2.0 conditions
(retain notices, state changes). No dataset is bundled with this code, so no
separate dataset licence applies.

**Which failure mode.** Motion blur (primary). Not camouflage.

**Why it helps this model specifically.** The stated blur problem isn't lack
of blurry-looking footage — it's lack of *labelled* examples where the
elongated motion-blur streak is captured accurately in the box, because the
existing label pool skews near-square (per the brief's elongation stats).
This technique sidesteps the "go find more blurred footage" problem
entirely: if the app or data-engine pipeline can capture (or already has)
high-frame-rate/slow-motion swings of its own — which needs no new licence,
since it's the project's own footage — consecutive raw frames can be
averaged to synthesize a genuinely blurred frame from a burst of sharp ones,
with the true motion path known from the intermediate frames. That known
path is exactly what the existing CSRT tracker-propagation tooling
(`docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`, data
engine item 2) already produces, so the elongated bounding box for the
synthesized blurred frame can be computed automatically as the union/hull of
the clubhead's tracked position across the averaged window — no new manual
labelling. This directly targets the "training set is short of genuinely
blurred, correctly-boxed clubheads" gap without waiting on indoor/low-light
footage that doesn't exist yet.

**Effort vs. payoff.** Medium effort, likely good payoff. Effort: this is
not a drop-in augmentation — no confirmed pretrained model was found in the
inspected code, so realistically only the *idea* (frame-averaging + tracked
union-box) is being reused, not the repo's code directly. Implementing it
means: (a) capturing/locating slow-motion source footage of swings (phone
120/240fps modes are plausible if the app or dev already shoots them), (b)
writing an averaging-window augmentation script, (c) deriving the blur-box
as the hull of tracked sub-frame positions. All three are buildable with
tools the data engine already has (CSRT propagation) or with a few lines of
NumPy — no new external dependency or licence risk. Payoff: this attacks the
motion-blur gap at its actual root (labelled elongated boxes, not just
"blurrier-looking pixels"), which naive Albumentations `MotionBlur` (already
in the v2 design's augmentation list) does not — that applies a uniform
kernel to the whole frame and does not touch the label geometry at all, so
it cannot fix under-representation of elongated boxes. Caveat: the technique
itself is unverified beyond the GitHub README + LICENSE (arXiv was
unreachable), so treat the *paper's specific quantitative claims* as
unverified even though the *code, its licence, and its stated purpose* are
confirmed.

---

## 2026-08-13 — TrackNetV4 motion-attention fusion for small/fast sports-object tracking

**What it is.** TrackNetV4 ("Enhancing Fast Sports Object Tracking with
Motion Attention Maps," arXiv:2409.14543) extends the TrackNet family
(originally built for tracking small, blurry, sometimes-invisible tennis
balls in broadcast video) with a "Motion-Aware Fusion" mechanism: a
frame-differencing map computed across a sliding window of 3 consecutive
frames is passed through a learnable "motion prompt layer" to produce a
motion-attention map, which is then fused (element-wise) with the network's
normal visual features before the detection/heatmap head. The point of the
mechanism is to give the model a *motion*-derived signal, independent of
appearance, for exactly the situation where appearance alone is
insufficient — which is a near-exact description of the camouflage failure
mode in the failure-analysis brief (dark clubhead against dark clothing or
foliage: zero candidate detections from the appearance-only detector even at
confidence 0.05). Reference implementation (TensorFlow, training/eval/predict
scripts, pretrained checkpoints, RESULT.md with numbers) is at
`github.com/TrackNetV4/TrackNetV4`.

**URL.** https://github.com/TrackNetV4/TrackNetV4 (paper:
arXiv:2409.14543 — arxiv.org itself is unreachable from this sandbox's
egress proxy, same as last run's finding, so the paper's claims here are
sourced from the repo's README and `docs/RESULT.md`, fetched directly, not
the PDF).

**Licence (verbatim, from the repo's `LICENSE` file, fetched directly).**
MIT License. "Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software... subject to the following
conditions: The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software." **Commercial
use: permitted.** No dataset is bundled in a way that would impose a
separate licence on model weights trained from scratch on this project's own
data; the repo's own tennis/badminton training sets are for reference only
and are not proposed for reuse here.

**Verified results (from `docs/RESULT.md`, on the repo's "new tennis
dataset," TrackNetV4-TypeB vs. baseline TrackNetV2).** Accuracy 79.91% vs.
77.46% (+2.45pp), Precision 93.88% vs. 91.20% (+2.68pp), Recall 83.04% vs.
82.32% (+0.72pp), F1 88.13% vs. 86.53% (+1.60pp). These are ball-tracking
numbers on broadcast tennis, not golf — cited only to confirm the motion-
attention mechanism measurably beats an appearance-only baseline in a
directly analogous small/fast/low-contrast-object setting, not as an
expected uplift for this model.

**Which failure mode.** Camouflage (primary) — this is the first log entry
to address it. Secondarily useful for motion blur too, since a moving-but-
blurred clubhead still produces a frame-difference signal even when its
static appearance is ambiguous.

**Why it helps this model specifically.** The failure-analysis brief's
camouflage cases are described as visually sharp frames with zero
detections — i.e., the appearance-only YOLO11n detector has nothing to
latch onto because the clubhead genuinely resembles its background in a
single frame. Motion is the one signal a single static frame cannot carry
but a golf swing has in abundance (the clubhead is one of the fastest-moving
things in the frame by design). A frame-differencing / motion-attention
input is therefore structurally suited to exactly this gap, unlike more
data or better single-frame augmentation, which cannot fix a detector that
has no appearance cue to exploit in the first place.

**Important caveat — this is not a drop-in.** TrackNetV4 is architecturally
a different animal from the current YOLO11n pipeline: it's a heatmap-
regression network over a fixed 3-frame window, not a single-shot bounding-
box detector, and it isn't validated for CoreML export or on-device inference
budgets (the README doesn't discuss mobile deployment). Two more caveats
specific to this app: (1) TrackNet-family models assume a largely static or
broadcast-stabilized camera — frame differencing on a handheld phone video
(the actual capture condition here) will pick up camera motion as well as
club motion unless the frames are first stabilized/registered, which the
current pipeline doesn't yet do; (2) the output is a point heatmap, not a
box, so it would need adaptation (or use purely as a candidate-region
proposal / confidence-boost signal ahead of the existing YOLO box regressor)
to fit the current label/eval format.

**Effort vs. payoff.** High effort, uncertain-but-real payoff. Effort: not
reusable as a drop-in replacement — realistically only the *architectural
idea* (fuse a frame-difference-derived motion-attention map into the
detector's features) transfers, and building it means either (a) adding a
motion-attention branch/input channel to the existing YOLO11n architecture
and retraining, which is a nontrivial model-surgery task and would need
re-validation of the CoreML export path, or (b) running a lightweight
frame-differencing pre-filter as a separate signal that boosts confidence
on low-confidence YOLO candidates in the camouflage regime, which is far
cheaper but a much weaker version of the idea and still needs camera-motion
compensation to be reliable on handheld footage. Payoff: this is the first
verified idea in the log that attacks camouflage specifically, and it's the
right *kind* of fix (motion-derived, not appearance-derived) for a failure
mode that is by definition an appearance failure — but the camera-motion
caveat above is real and unaddressed by the source work, so this should be
scoped as a research spike (does simple frame-differencing survive typical
phone-swing camera shake?) before any model-surgery investment.

---

## 2026-08-13 — GolfDB checked and ruled out (dataset area, negative result)

**What it is.** GolfDB ("A Video Database for Golf Swing Sequencing,"
McNally & Vats, CVPR Workshops 2019) is the best-known public golf-swing
video dataset and the obvious first hit for anyone searching "golf swing
dataset." It compiles 580 YouTube videos (1,400 trimmed clips, ~390k
frames) of PGA/LPGA/Champions Tour professionals. This run checked it
specifically to see whether it (or its curation approach) could plug either
the indoor/low-light gap or the motion-blur gap. It cannot, for three
independent, verified reasons below — logging this so no future run
re-spends effort chasing it.

**URL.** https://github.com/wmcnally/golfdb (paper via search-indexed
excerpt only — both arxiv.org/pdf/1903.06528 and
openaccess.thecvf.com are blocked by this sandbox's egress proxy, same
restriction noted in prior runs).

**Licence (verbatim, fetched directly from the repo README).** "The code in
this repository is licensed under a Creative Commons Attribution-
NonCommercial 4.0 International License." **Commercial use: NOT permitted.**
No separate, more-permissive licence is stated for the video data/labels
themselves (they are distributed as YouTube URLs + annotation files under
the same repo).

**Which failure mode.** Neither, in practice — checked against both.

**Why it doesn't help this model, specifically (three independent
dealbreakers, not just the licence).**
1. **Licence.** CC BY-NC-4.0 forecloses commercial use outright, matching
   this project's hard licence requirement.
2. **Annotation format doesn't match what's needed.** Verified directly from
   the README: GolfDB's labels are *temporal event frames* (8 swing-phase
   markers — address, top, impact, etc.) plus per-video metadata, not
   spatial clubhead bounding boxes. Even under a permissive licence, it
   would require fully re-labelling every frame from scratch to be usable
   for this detector — it contributes zero ready-made box labels.
3. **It was explicitly curated to exclude motion blur.** A direct quote
   surfaced via search indexing of the paper text: "to alleviate
   obscurities caused by motion blur, only high quality videos were
   considered." (Sourced from a search-engine excerpt of the paper, not a
   direct PDF fetch — treat this one specific quote as one notch below full
   verification, consistent with how arXiv-sourced claims have been flagged
   in prior entries.) This is the opposite of what this project needs: the
   gap is a shortage of genuinely blurred, correctly-boxed examples, and
   GolfDB's own selection criteria filtered blur *out*. It is broadcast
   footage of tour pros in good light, not indoor/simulator/overcast/older-
   phone footage either — so it doesn't touch the camouflage-adjacent
   indoor gap the brief flags as unmeasured.

**Effort vs. payoff.** Low effort (one search-and-verify pass), zero
payoff — that is the finding. Recorded as a negative result per the run
brief's instruction to log "nothing new" honestly rather than pad the log;
in this case there *was* something to check (a specific, verifiable, named
dataset), it just came up empty on inspection. Future runs searching the
"golf datasets" area should treat GolfDB as checked and move to less
obvious sources — e.g. searching for simulator-vendor or swing-app training
data licensing, or golf-adjacent (not golf-specific) blurred-fast-object
sport datasets, rather than re-discovering GolfDB.