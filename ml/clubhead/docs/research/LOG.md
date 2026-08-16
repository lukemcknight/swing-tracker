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

---

## 2026-08-13 — GolfPose (ICPR 2024): golf-specific club-keypoint model, checked and blocked on licensing (golf pose/tracking area, negative-with-caveats result)

**What it is.** GolfPose (Lee et al., "GolfPose: From Regular Posture to
Golf Swing Posture," ICPR 2024) is, as far as this run could find, the only
public, code-released project that treats the golf **club** itself as a
pose-estimation target with explicit club keypoints, rather than just
inferring club position from body pose. The official repo
(`github.com/MingHanLee/GolfPose`) ships MMDetection/MMPose configs and
pretrained checkpoints for several model variants, including club-only
(`GolfPose-2D(C)`) and combined golfer+club (`GolfPose-2D(GC)`,
`GolfPose-3D(GC)`) pose estimators, fine-tuned from HRNet/ViTPose/DEKR/MixSTE
backbones on a purpose-built "GolfSwing" dataset (17 golfer keypoints + up to
5 club keypoints, with 3D ground truth from a motion-capture rig
synchronized to normal RGB cameras). Reported club-model accuracy in the
README: HRNet-w48 0.857 AP, ViTPose-H 0.870 AP, DEKR 0.858 AP (AP, not MPJPE
— no per-joint pixel-error number is given for the club variants).

I also checked two derivative/independent repos that cite this same
line of work — `mamoonik/golf-swing` (claims a from-scratch reimplementation
achieving "34% improvement on club keypoints" via a club-detection refinement
step, but is really citing the earlier Jiang et al. 2022 ICMEW "GolfPose"
paper, a different work with the same name) and `ryanboscobanze/GolfPosePro`
(MIT-licensed, but explicitly does not track the club head — wrist-only).
Neither is a substitute for the ICPR 2024 GolfPose code/dataset.

**URL.** https://github.com/MingHanLee/GolfPose (paper:
ICPR 2024, DOI 10.1007/978-3-031-78305-0_25, paywalled at
link.springer.com/dl.acm.org — not fetchable from this sandbox's egress
proxy, so claims here are sourced from the repo's README, fetched directly,
not the PDF).

**Licence — VERIFIED ABSENT. Commercial use: NOT confirmed permitted; treat
as forbidden by default.** Checked directly:
`raw.githubusercontent.com/MingHanLee/GolfPose/main/LICENSE` and
`.../master/LICENSE` both return HTTP 404 — there is no LICENSE file in the
repo, and the README contains no licence badge or licence text anywhere.
Under GitHub's own terms of service, code with no explicit licence is "all
rights reserved" by default: it is visible for reading, but nobody else
(including for commercial use) has any grant to use, copy, modify, or
distribute it. The dataset is separately gated — the README states "Please
email mhlee.cs09@nycu.edu.tw to authorize the dataset download" — with no
stated terms of use surfaced anywhere in the README once authorized; gated
academic-contact datasets of this kind are conventionally research/
non-commercial, but that is an inference, not a confirmed fact, since no
terms text was found to quote. **Both the code and the dataset are therefore
logged as blocked pending an explicit licence grant from the authors** — do
not use either without first emailing the authors and getting written terms
that permit commercial use.

**Which failure mode.** Neither directly — this is a model-architecture/
dataset-methodology finding, not a camouflage or motion-blur fix per se.
Closest fit is camouflage-adjacent: a model trained to regress explicit club
keypoints (not just detect a box) has to learn a location prior and
part-relationship structure for the club that a box-only detector like the
current YOLO11n never gets, which is one documented way small/occluded/
low-contrast object recall improves in pose literature (the model has more
to condition on than raw appearance-in-a-crop). It does not touch motion
blur — the GolfSwing dataset's capture setup (mocap-synced RGB) is not
described as including blurred frames.

**Why it (would) help this model specifically, if unblocked.** This is the
first thing found in three runs of this log that is golf-specific and
club-specific rather than a generic small-object or motion technique
adapted from another sport — it is directly on-target for "does anyone
already solve golf club localization." If the licence question resolved
favorably, the natural use would not be swapping in a pose model wholesale
(architecture mismatch with the on-device YOLO11n/CoreML pipeline is as real
here as it was for TrackNetV4, logged 2026-08-13) but using the released
club-keypoint checkpoints as an **auto-labelling assist**: run GolfPose-2D(C)
over the project's own unlabelled phone footage to propose club keypoints,
convert to a box via a small dilation, and route through human review before
accepting into the training set. That would directly attack the stated
54%-Roboflow/29%-own-footage data-mix imbalance by cutting the cost of
turning more of the app's own footage into labelled training data — but only
if the authors' terms permit that kind of derivative use, which is unverified.

**Effort vs. payoff.** Low effort to find, currently near-zero payoff
because of the licence block — this is the finding. If a future run (or the
project owner directly) gets written commercial-use terms from
mhlee.cs09@nycu.edu.tw, effort would jump to medium (stand up MMDetection/
MMPose, run inference on a footage sample, build the box-derivation +
review-queue step) with genuinely good payoff (cheaper labelling of the
underrepresented own-footage slice). Until then this should be treated as
"licence-blocked, contact the authors before spending any implementation
time," not as an actionable technique. Recorded so future runs don't
re-discover the same repo and re-do this same licence check.

---

## 2026-08-13 — Channel-stacked multi-frame YOLO (Quan et al., ECMR 2025): same-family architecture fix for blur *and* camouflage

**What it is.** "Lightweight Multi-Frame Integration for Robust YOLO Object
Detection in Videos" (Quan, Kiefer, Messmer, Zell — ECMR 2025, also on arXiv
as 2506.20550) modifies a standard YOLO detector to take `n` consecutive
video frames stacked along the channel axis as input (Early Fusion: the
first conv layer processes `3n` channels jointly, its weights initialized by
repeating the pretrained single-frame weights `n` times), while supervising
only the label for the single latest frame — i.e. no new label format and no
added temporal module (no RNN/optical-flow/attention block). Reference
implementation (built on YOLOv7/YOLOv7-tiny, training + test code, multiple
pretrained checkpoints for n=3..9) is at
`github.com/yitong-quan/yolov7-multi-frame`, fetched and inspected directly
(root file listing, README, and LICENSE.md content all confirmed live, not
just search-indexed).

**URL.** https://github.com/yitong-quan/yolov7-multi-frame (paper: ECMR
2025 / arXiv:2506.20550 / IEEE Xplore document 11162972 — all three of
arxiv.org, ieeexplore.ieee.org, and the papers.cool arXiv mirror are blocked
by this sandbox's egress proxy, same restriction noted in every prior run of
this log, so the paper's quantitative results below are sourced from
search-engine-indexed abstract text, not a direct PDF/HTML fetch; treat
those specific numbers as one notch below full verification, per this log's
established convention. The repo README and LICENSE.md **were** fetched
directly and are fully verified.)

**Licence (verbatim, from `LICENSE.md`, fetched directly).** GNU General
Public License, Version 3, 29 June 2007 (full GPL-3.0 text, Free Software
Foundation). **Commercial use of the code is NOT straightforwardly
permitted** — GPL-3.0 is copyleft: distributing a binary that incorporates
GPL-3.0 code (or code derived from it) obligates you to make the complete
corresponding source available under GPL-3.0 too. This matters concretely
here: this repo builds on the official YOLOv7 codebase, which is itself
GPL-3.0, and the project's current detector is YOLO11n (Ultralytics), whose
own licence is AGPL-3.0 unless the project holds a paid Ultralytics
commercial licence — that is a separate, pre-existing question this entry
does not resolve, but it means the project should already know its own
answer to "can we ship a YOLO-family model commercially." **Do not import or
adapt this repo's code into the shipping pipeline without legal review.**
What is safely reusable regardless of licence: the *technique* itself
(stack n frames on the channel axis, widen/re-init the first conv layer) is
a well-known, generic architectural idea, not a copyrightable expression —
it can be reimplemented from the paper's description directly against
YOLO11n's own first-conv layer without touching this repo's GPL code at all.

**Which failure mode.** Both, explicitly. The README states the goal in
those terms: "improve robustness against blur and occlusion" via injecting
"temporal cues at the pixel level." The (search-indexed) abstract text adds
that motion blur, occlusions, and abrupt appearance changes "severely
degrade single-frame detection performance" and that stacking frames
"improves detection robustness, especially for lightweight models,
effectively narrowing the gap between compact and heavy detection
networks" on MOT20Det and BOAT360 (unverified numbers — see caveat above).

**Why it helps this model specifically.** This is the most architecturally
compatible temporal-method finding logged so far. TrackNetV4 (logged earlier
today) is a heatmap-regression network over a fixed window — a different
model family from the current single-shot YOLO11n/CoreML pipeline, with an
unclear export/on-device story. This technique keeps the exact YOLO
single-shot-detector shape; the only structural change is the first
convolution's input channel count and its weight initialization. That means
it plausibly retains CoreML exportability and on-device inference budgets in
a way TrackNetV4 does not — a claim this entry cannot fully verify (no CoreML
export was attempted here) but is a much smaller leap than a heatmap
architecture swap. It targets camouflage the same way TrackNetV4 was logged
as targeting it — motion is a signal a static frame lacks but the moving
clubhead has — while also directly targeting motion blur, since a
channel-stacked window gives the network genuine multi-frame temporal
context around a blurred frame instead of asking a single blurred,
near-square-boxed training example to teach the shape of a blur streak on
its own, which is the root gap this log's first entry (frame-averaging,
2026-08-12) also independently identified.

**Effort vs. payoff.** Medium effort, good-if-verified payoff. Effort:
reimplementing against YOLO11n means (a) writing a dataloader that yields
n-frame stacks with a single target label (straightforward — Ultralytics'
YOLO11 dataloader is built to be subclassed), (b) widening the first conv
layer and re-initializing by repeating pretrained weights (a few lines, same
trick this repo already demonstrates for YOLOv7), (c) re-running the CoreML
export and checking the exported model still meets the app's inference
latency budget on-device, which is the one real unknown and the item most
worth spiking first. No GPL code needs to ship — only the idea. Payoff:
if it works, this is a same-architecture, same-export-path improvement
(unlike TrackNetV4), which meaningfully de-risks adoption versus every other
temporal-method finding logged so far — but the actual accuracy delta is
unverified (blocked from the paper itself), so the honest recommendation is
a small experimental spike (train an n=3 channel-stacked YOLO11n variant on
the existing training set, compare against the existing baseline on the
held-out test set) before treating the payoff as real.

---

## 2026-08-14 — PSF-based blur synthesis with box expansion (Sayed & Brostow, CVPR 2021): a second, cheaper route to correctly-elongated blur labels

**What it is.** "Improved Handling of Motion Blur in Online Object Detection"
(Mohamed Sayed & Gabriel Brostow, UCL — CVPR 2021, pp. 1706–1716,
arXiv:2011.14448) studies five remedy classes for the sharp/blurred
performance gap in object detectors. The one directly relevant here is their
label-generation remedy: they convolve sharp training crops with generated
motion-blur point-spread-function (PSF) kernels (parameterised by blur type
and exposure length via `--param_index`/`--high_exposure`/`--low_exposure`),
and — critically — **expand the ground-truth bounding box to match the
kernel's blur extent** (`--expand_target_boxes` in their training/eval code)
instead of leaving the original sharp-object box in place. Per the search-
indexed abstract/README text, this "custom label generation aimed at
resolving spatial ambiguity" is what "markedly improves object detection" in
their experiments, more than the blur-conditioning/ensemble remedies they
also tried. Reference implementation (PyTorch, built on torchvision's
detection reference code, includes `generate_PSFs.py`, `motion_blur/`, and
the full train/eval CLI with the flags above) is at
`github.com/mohammed-amr/detectInBlur`, fetched and inspected directly (root
file listing and full README fetched via raw.githubusercontent.com and
github.com — both live and confirmed, not just search-indexed).

**URL.** https://github.com/mohammed-amr/detectInBlur (paper: CVPR 2021 /
arXiv:2011.14448 / project page visual.cs.ucl.ac.uk/pubs/handlingMotionBlur —
arxiv.org, the UCL project page, and deepai.org were all blocked by this
sandbox's egress proxy, consistent with every prior run of this log, so the
paper's specific quantitative deltas were not directly readable; only a
search-engine-indexed summary confirming the label-generation remedy is the
strongest of the five and the authors' names/venue/page range could be
cross-checked. Treat the *existence and mechanism* of the PSF+box-expansion
technique as verified — it's visible directly in the repo's README and CLI
flags, not just claimed — but treat any specific accuracy number as
unverified, since none could be fetched.).

**Licence (verified directly).** No LICENSE, LICENSE.md, LICENSE.txt, or
COPYING file exists anywhere in the repository root (checked the live file
listing at github.com/mohammed-amr/detectInBlur/tree/master, and confirmed
`raw.githubusercontent.com/.../LICENSE` returns 404 on both `master` and
`main`). The README states "most of this repo is based on the detection
reference code from torchvision" (torchvision's own reference scripts are
BSD-3-Clause, which does permit commercial use), but the author's own
additions — `generate_PSFs.py`, the blur-application code, the box-expansion
logic, the blur-estimator classifier — carry no licence grant of their own.
Under GitHub's default-licence rule, unlicensed code is "all rights
reserved": **commercial use of this repo's own code is NOT confirmed
permitted; treat as forbidden by default** until the author (Mohamed Sayed)
grants explicit terms. Same posture as the GolfPose entry logged 2026-08-13.
What *is* safely reusable regardless: PSF-based motion-blur synthesis
(convolving a sharp crop with a directional blur kernel) is a decades-old,
generic image-processing technique with no original-expression claim
attached to it (OpenCV and countless textbooks describe the identical
operation) — reimplementing "convolve with a motion PSF, then grow the
label box along the kernel's direction/length" from the paper's description
requires none of this repo's actual code.

**Which failure mode.** Motion blur, specifically and only. Does not touch
camouflage.

**Why it helps this model specifically.** This is a second, independently-
sourced route to the exact same root gap the 2026-08-12 frame-averaging
entry identified: labelled boxes in this project's training set are
near-square (median elongation 1.60) because blurred training examples are
scarce, not because annotators mislabel blur when it's present (the labeling
spec already instructs boxing the full streak). Frame-averaging (08-12)
fixes this by *capturing new footage* (a burst of sharp frames, averaged,
with the box derived from tracked sub-frame positions) — it needs
slow-motion/high-fps source video the project may or may not already have.
This technique instead fixes it by *synthesizing blur from footage already
in hand*: take an existing sharp, correctly-boxed clubhead crop from the
training set, convolve it with a directional PSF kernel of chosen length/
angle, and analytically grow the box by that same kernel's extent — no new
capture, no tracker, no burst-mode footage required, just the labels and
images the project already has. The two techniques are complementary, not
competing: frame-averaging produces genuinely more realistic blur (true
optical integration) but depends on footage the project may not have yet;
PSF synthesis is available immediately from the existing dataset but the
blur is synthetic/approximate rather than physically captured. Running both
is plausible and cheap since neither requires new licensed assets.

**Effort vs. payoff.** Low-to-medium effort, plausible payoff, capped by an
unverified accuracy claim. Effort: PSF generation and convolution is a small
amount of NumPy/PIL code (no need for this repo's training loop or model
code, just the blur-kernel-generation idea), and the box-expansion rule is
pure geometry (grow the axis-aligned box by the kernel's projected
length along its angle) — this is the cheapest-to-implement idea logged in
this file so far, since it needs no new footage, no tracker, no architecture
change, and no GPU-hours beyond normal training. Payoff: directly produces
more elongated, correctly-boxed training examples, which is exactly the
labelled-data gap the brief describes — but the paper's actual reported
accuracy delta for this remedy could not be fetched in this run (egress
blocked on every hosting domain tried), so "markedly improves" is a
paraphrase of a search-indexed summary, not a number this entry can stand
behind. Recommended as a cheap experimental addition alongside the
frame-averaging idea, not as a substitute for eventually getting real
low-light/indoor footage (still unmeasured, per the brief) — synthetic blur
on well-lit source images does not create the exposure/noise/dynamic-range
characteristics that real indoor motion blur will actually have.

---

## 2026-08-14 — DTUM: direction-coded temporal module for dim/low-contrast moving targets (IEEE TNNLS 2023), from the infrared-surveillance literature rather than sports tracking

**Area covered.** This run first spent effort re-checking the "commercial
golf video/image dataset, especially low-light/indoor/simulator/older-phone"
area (rotation bullet 1), since it's the least-covered area in the log and
the brief flags indoor performance as never measured. That search came up
empty: a fresh Roboflow Universe search surfaced only outdoor/broadcast golf
datasets already in the same family as the ones the project already draws
54% of its training data from, and a separate GitHub/academic search for
indoor-simulator or low-light golf swing footage found nothing beyond
GolfDB (already checked and ruled out on 2026-08-13). `universe.roboflow.com`,
`roboflow.com`, `app.roboflow.com`, `api.roboflow.com`, `kaggle.com`,
`huggingface.co`, `arxiv.org`, `zenodo.org`, `figshare.com`, and
`opendatalab.com` are all unreachable from this sandbox's egress proxy
(`curl` to each returns "CONNECT tunnel failed, response 403"; only
`github.com` and `raw.githubusercontent.com` are reachable), which caps how
far a dataset search can go in this environment — noted here so a future run
doesn't re-spend a whole cycle re-discovering the same block. Given that
dead end, this entry instead logs a genuinely new finding from rotation
bullet 3 (small/low-contrast/camouflaged object detection, temporal
methods), specifically chosen from a research field the log hasn't touched
yet.

**What it is.** DTUM ("Direction-Coded Temporal U-Shape Module for
Multiframe Infrared Small Target Detection," Li et al., IEEE Transactions on
Neural Networks and Learning Systems 2023) is from the infrared/thermal
surveillance small-target-detection literature — a different research
community from the sports-broadcast-tracking lineage TrackNetV4 (logged
2026-08-13) comes from, working on a structurally similar problem: detecting
a small, dim, low-contrast target that is nearly invisible in a single
static frame because it barely differs in appearance from cluttered
background. DTUM's mechanism is a direction-coded convolution block (DCCB)
that encodes each target's motion *direction* across a short multi-frame
window into learned features, feeding a U-shaped temporal module that
enhances the moving target's signal while suppressing background false
alarms. Unlike TrackNetV4 (a full heatmap-regression network built around
frame-differencing + a learned attention map), the repo's README describes
DTUM explicitly as a plug-in module that "can be equipped with most
single-frame networks to leverage spatial-temporal information" — i.e. it
is pitched as an add-on to an existing single-shot detector's backbone
rather than a wholesale architecture replacement, which is structurally
closer to what modifying YOLO11n would require. The authors also built and
released NUDT-MIRSDT, a dataset of dim, small, moving infrared targets
against cluttered backgrounds with both mask- and point-level annotations,
built specifically for this problem shape.

**URL.** https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM
(paper: IEEE TNNLS 2023, IEEE Xplore document 10321723, also indexed at
PubMed 37976190 — IEEE Xplore, PubMed, and arXiv were all unreachable from
this sandbox for direct verification, same egress restriction as every
prior run's finding; the paper's title, venue, year, and mechanism
description are corroborated across three independently-indexed sources
(IEEE Xplore, PubMed, ResearchGate) converging on the same description, and
the repo's own README — fetched directly and live — independently confirms
the DCCB/temporal-module mechanism and the plug-in-compatibility claim in
the authors' own words, so the mechanism is treated as verified even though
the PDF itself could not be read for exact reported numbers).

**Licence (verified directly).** No LICENSE, LICENSE.md, LICENSE.txt, or
COPYING file exists in the repository — checked directly via
`raw.githubusercontent.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM/main/<file>`
and the equivalent `master/` path for all four filenames; every one returned
HTTP 404 (confirmed via `curl`, not just a page render), while
`README.md` returns HTTP 200 on both branches, confirming the repo and both
branch names are live and the absence of a licence file is real, not a
fetch failure. Under GitHub's default-licence rule, this means **commercial
use of this repo's code is NOT confirmed permitted; treat as forbidden by
default** until the authors grant explicit terms — same posture as the
GolfPose (2026-08-13) and detectInBlur (2026-08-14, PSF-synthesis) entries
already in this log. The NUDT-MIRSDT dataset is not bundled in the repo at
all — it's distributed externally via BaiduYun and Google Drive links with
no separate licence text stated anywhere in the README, so it inherits the
same "not confirmed permitted" status and is additionally ungated only by a
generic file-sharing link (no request-access process, unlike GolfPose's
dataset), which is worth noting but does not change the licence conclusion.
What *is* safely reusable regardless of the code's licence status: the
described technique — encoding a moving object's frame-to-frame direction
into a convolutional feature via a direction-coded kernel, then fusing that
motion-direction signal with single-frame appearance features ahead of a
detection head — is an algorithmic idea describable and reimplementable
from the paper's own mechanism description (as summarized in the repo
README and corroborating abstracts) without copying any of this repo's
actual code.

**Which failure mode.** Camouflage, primarily and specifically — DTUM's own
problem framing ("dim small target" against "cluttered background," typical
signal-to-clutter ratios low enough that single-frame appearance is
ambiguous) is close to a direct restatement of this project's camouflage
failure mode (dark clubhead against dark clothing/foliage, zero detections
even at confidence 0.05 in visually sharp frames). Not applicable to motion
blur — DTUM's target model is a dim-but-sharp point/small-blob target, not
an elongated blur streak.

**Why it helps this model specifically.** This is the second independent
verified source (after TrackNetV4) converging on the same conclusion from a
completely different research field: when a single frame's appearance is
genuinely insufficient — not just noisy or low-resolution, but ambiguous
because the target and background actually look alike — the standard
remedy in the literature is to inject a motion-derived signal computed
across frames, not to keep improving the single-frame appearance model.
Two independent literatures (sports broadcast tracking and infrared
surveillance) landing on structurally the same fix for structurally the
same problem is stronger evidence that this is the right general direction
for camouflage than either paper alone would be. DTUM adds something
TrackNetV4 didn't: an explicit plug-in framing compatible with existing
single-frame detector backbones (closer to "add a module to YOLO11n" than
"replace YOLO11n with a heatmap network"), and a purpose-built dim-small-
moving-target dataset (NUDT-MIRSDT) that, licence permitting, could serve as
a pretraining or architecture-validation source for a motion-direction
module even before any golf-specific data is touched — though the domain
gap (infrared thermal imagery vs. visible-light phone video) means it could
only validate the *architecture*, not directly transfer as training data.

**Important caveat — same camera-motion problem as TrackNetV4, independently
confirmed here.** Infrared surveillance/tracking systems in this literature
are typically mounted on tripods, gimbals, or slow-panning platforms —
DTUM's direction-coded motion signal assumes the dominant frame-to-frame
motion is the target's, not the camera's. A handheld phone recording a golf
swing has substantial camera motion of its own (especially during the
downswing, when the phone is often also moving to track the club), which
would corrupt a naive direction-coded motion signal exactly as it would
corrupt TrackNetV4's frame-differencing input. Any prototype would need
frame registration/stabilization ahead of the motion-encoding step. DTUM is
also pure PyTorch research code with no CoreML export path or mobile
inference budget discussed anywhere in the repo — same deployment gap as
every architecture-change idea logged so far.

**Effort vs. payoff.** High effort, moderate-confidence payoff, and mostly
confirmatory rather than novel in its conclusion. Effort: comparable to or
slightly cheaper than the TrackNetV4 idea — the DCCB is a small, well-
specified convolutional block (not a full network) that could plausibly be
prototyped as an added input branch to YOLO11n's backbone without adopting
DTUM's full U-shaped multiframe architecture, but it still requires
architecture surgery, retraining, camera-motion compensation research, and
re-validation of the CoreML export path — none of which is small. Payoff:
this entry's main value is *confirmatory*, not novel — it strengthens
confidence that motion-direction encoding is the right general shape of fix
for camouflage (two unrelated literatures agree) rather than introducing a
new mechanism the project hadn't already identified via TrackNetV4. Given
that, and given the unresolved licence and camera-motion problems shared
with TrackNetV4, this should not be treated as a second, independent thing
to build — it's evidence to weight the existing TrackNetV4-style research
spike (does a motion-direction/frame-difference signal survive handheld
camera shake?) higher, not a separate line of work.

---

## 2026-08-14 — dj_masters checked and ruled out (golf pose/club-tracking area, negative result: unimplemented accuracy claim + broken licence claim + tainted training pipeline)

**What it is.** `matiarj/dj_masters` ("DJ Masters," GitHub) is a personal
golf-swing-analysis project combining MediaPipe/YOLOv8 body-pose estimation
with a "custom-trained YOLO model" for golf club detection, plus a
"Motion-consistent Tracking" component described as using Kalman filtering
to smooth the club's trajectory across frames. It surfaced in a search for
new golf-specific club-tracking work (this log's rotation area 4) as a
distinct project from the ones already logged here (GolfPose, 2026-08-13;
`mamoonik/golf-swing` and `ryanboscobanze/GolfPosePro`, both referenced as
asides in that same entry). On inspection it does not hold up, for three
independent, verified reasons below.

**URL.** https://github.com/matiarj/dj_masters (repo root, README.md,
`GOLFDB_SETUP_GUIDE.md`, and `CLUB_DETECTION_IMPROVEMENTS.md` all fetched
directly and confirmed live).

**Licence — claimed MIT, but no LICENSE file exists. Commercial use: NOT
confirmed permitted; treat as forbidden by default.** The README states
"MIT License - see LICENSE file for details," but
`raw.githubusercontent.com/matiarj/dj_masters/main/LICENSE` returns HTTP
404, and no `LICENSE`/`LICENSE.md`/`LICENSE.txt` appears anywhere in the
repo's root file listing (confirmed via direct fetch of the GitHub file
tree, not just search-indexing). Under GitHub's default-licence rule, a
README's licence claim with no accompanying LICENSE file grants nothing —
the code remains "all rights reserved" by default. Same posture this log
has already applied to GolfPose (2026-08-13), detectInBlur (2026-08-14),
and DTUM (2026-08-14): a stated intent to be permissively licensed is not a
licence.

**Second, independent dealbreaker: the repo's own `GOLFDB_SETUP_GUIDE.md`
documents GolfDB (CC BY-NC-4.0, non-commercial only — verified and logged
here on 2026-08-13) as the data source its setup process is built around.**
Even if the code licence were fixed, any club-detector weights trained by
following this project's own documented pipeline would inherit a
non-commercial data provenance problem, making the resulting weights
unusable for this app regardless of what licence the code itself carries.

**Third, independent dealbreaker: the headline "80%+ detection accuracy"
claim is not a measured result.** `CLUB_DETECTION_IMPROVEMENTS.md` lists
"Detection rate: 21% → 60-80%" under a section literally titled "Expected
Improvements," with no test-set description, no metric definition
(precision/recall/F1/IoU threshold — none stated), and no reported
methodology anywhere in the repo's own documentation. The Kalman-filter
"Motion-consistent Tracking" that the README's badges present as a current
feature is, per `CLUB_DETECTION_IMPROVEMENTS.md`'s own "Recommended Next
Steps" section, unimplemented — only a skeleton `cv2.KalmanFilter(4, 2)`
stub is present, not a working tracker. The badge-level marketing in the
README does not match the project's own internal status documentation.

**Which failure mode.** Neither, in practice — nothing here is validated
enough to attribute an effect to. If the described (but unimplemented)
Kalman-filtered trajectory-smoothing idea were ever built, it would be
camouflage-adjacent (bridging a frame where the detector returns zero
candidates by carrying forward a predicted position from prior confident
detections) — the same idea this log's TrackNetV4 and DTUM entries reach
by a more rigorous route. But that idea is generic, well-established
control-theory (Kalman filtering for object tracking predates this
project by decades) and this repo contributes no working implementation,
no benchmark, and no golf-specific insight beyond stating the idea as a
TODO — so there is nothing here worth adopting over just implementing a
standard Kalman/constant-velocity tracker directly from first principles
if that direction is pursued.

**Effort vs. payoff.** Low effort (one search-and-verify pass), zero
payoff — that is the finding. Recorded per the brief's instruction to log
"nothing new" honestly: this is a case where a plausible-sounding search
hit (golf club detection, 80%+ accuracy, Kalman tracking) evaporates on
direct inspection of the actual repo contents rather than its README
badges. Worth noting as a general caution for future runs in this log:
GitHub project READMEs for small/personal repos can state accuracy figures
and feature lists that are aspirational rather than achieved, and licence
badges that are aspirational rather than backed by an actual LICENSE file
— both should be checked against the repo's own file listing and internal
docs, not taken at README-badge value, exactly as this run did.

---

## 2026-08-14 — Copy-Paste augmentation (Ghiasi et al., CVPR 2021): a data-synthesis route to camouflage, not an architecture change

**What it is.** "Simple Copy-Paste Is a Strong Data Augmentation Method for
Instance Segmentation" (Ghiasi et al., CVPR 2021, arXiv:2012.07177) pastes a
real labelled object cutout onto a different background image, with the new
bounding box (and mask) computed directly from the pasted region — no
generative model, no learned harmonization network. This run checked an
independent, unofficial implementation, `conradry/copy-paste-aug`, and
fetched both the README and the actual augmentation code
(`copy_paste.py`) directly, not just a description. Verified mechanism, read
from the code itself: bounding boxes are *outputs*, recomputed by extracting
min/max nonzero coordinates from the pasted mask
(`extract_bboxes()`) after merging masks via
`np.logical_and(mask, np.logical_xor(mask, alpha))` to correctly punch out
occluded original content; blending is Gaussian edge-smoothing of the paste
mask (`gaussian(alpha, sigma=sigma)`, default `sigma=3`), i.e.
`img = paste_img * alpha + img * (1 - alpha)` — a soft feathered edge, not
Poisson/gradient-domain blending. **Important limitation confirmed from the
code:** it requires segmentation masks as input, not bare boxes — it cannot
run on box-only annotations without first deriving a mask.

**URL.** https://github.com/conradry/copy-paste-aug (paper: CVPR 2021 /
arXiv:2012.07177 — arxiv.org itself was unreachable from this sandbox's
egress proxy, consistent with every prior run's restriction, so the paper's
own reported accuracy numbers were not fetched; the augmentation mechanism
above is verified directly from the reimplementation's own code, not the
paper).

**Licence (verified directly, not just README-claimed — see the dj_masters
entry above for why that distinction matters).**
`raw.githubusercontent.com/conradry/copy-paste-aug/main/LICENSE` was fetched
directly and returns the actual MIT License text (Ryan Conrad, 2020):
"Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files... to deal in the
Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software." **Commercial use: permitted.**

**Which failure mode.** Camouflage, primarily. This is the first entry in
the log to attack camouflage from the *data* side rather than the
*architecture* side (TrackNetV4 and DTUM, both logged above, both require
adding a temporal/motion module to the model and re-validating the CoreML
export). Secondarily useful for motion blur: nothing stops pasting an
already-blur-augmented clubhead crop (from either the frame-averaging or
PSF-synthesis entries logged above) onto a new background, combining both
fixes in one augmentation pass.

**Why it helps this model specifically.** The camouflage failure as
described is a single-frame appearance problem: a dark clubhead against
dark clothing or cluttered foliage produces zero candidates because the
detector has never seen enough examples of *that specific hard contrast
combination* — not because the model architecture is incapable of a box
detection task. Copy-paste directly manufactures more of exactly that
combination: take real, already-correctly-boxed clubhead crops from the
existing training set and composite them onto real hard backgrounds pulled
from this project's own footage. The labeling spec already keeps
unannotated negative frames as first-class data (follow-through, non-swing
footage) — those are a ready, zero-new-capture, zero-licence-risk source of
exactly the dark-clothing/foliage backgrounds needed, since they're already
this project's own footage. This needs no new external dataset, no new
capture, and — unlike TrackNetV4 or DTUM — no architecture change or
re-validation of the on-device CoreML export path, since the model and
input format are untouched; only the training set grows.

**Effort vs. payoff.** Low-to-medium effort, plausible payoff, and the
cheapest camouflage-directed idea in this log so far specifically *because*
it doesn't touch the model. Effort: the mask requirement is the one real
obstacle — this project's labels are boxes only (per the labeling spec), so
either (a) a cheap approximate mask can be derived per crop (e.g. an
ellipse or the box itself, feathered at the edge, without a real
segmentation model, which likely loses little for a small, roughly-blob-
shaped clubhead) or (b) a lightweight off-the-shelf background-removal
step generates a real mask once per crop, which is a one-time preprocessing
cost, not a per-training-run cost. Composition (sampling a background from
the existing negative frames and pasting a blur-augmented clubhead crop
with feathered edges) is a small, self-contained script. Payoff: directly
produces more of the specific hard training examples the camouflage failure
mode is short on, with no architecture risk — but composited realism is
inherently imperfect (lighting/shadow/scale mismatch between the crop's
original scene and the new background), so this should be treated as
*more hard examples*, not a faithful simulation of real camouflage
conditions, and evaluated on the held-out test set before trusting it to
move the needle.

---

## 2026-08-15 — RT-Focuser (ICTA 2025): lightweight edge/CoreML deblurring as an inference-time preprocessing step, not a training-side fix

**What it is.** Every motion-blur entry logged so far (frame-averaging
2026-08-12, PSF-synthesis 2026-08-14, channel-stacked multi-frame YOLO
2026-08-13) attacks the training side: make blurred, correctly-boxed
examples so the detector *learns to recognize* a blur streak. RT-Focuser
("A Real-Time Lightweight Model for Edge-side Image Deblurring," Wu et al.,
IEEE ICTA 2025, arXiv:2512.21975) is a different remedy class entirely: a
small U-shaped deblurring network (a "Lightweight Deblurring Block" with
edge-aware feature extraction, a "Multi-Level Integrated Aggregation"
encoder-fusion stage, and a "Cross-source Fusion Block" decoder) meant to
run **at inference time**, sharpening each blurred frame *before* it reaches
a downstream detector — no retraining of the detector required at all.
Reference implementation (PyTorch + ONNX, training and inference scripts,
pretrained weights already included in the repo, CoreML export path
documented) is at `github.com/ReaganWu/RT-Focuser`.

**URL.** https://github.com/ReaganWu/RT-Focuser (paper: IEEE ICTA 2025,
arXiv:2512.21975 — arxiv.org and ieeexplore.ieee.org are both unreachable
from this sandbox's egress proxy, same restriction as every prior run of
this log, so the paper's own PSNR/SSIM claims are sourced from the repo's
own README table, not the PDF. The repo itself — README, `LICENSE`,
`requirements.txt`, `Inference_Image_Torch.py`, `Inference_Video_ONNX.py`,
and a specific weight file,
`Pretrained_Weights/rt_focuser_wint8_afp16.onnx` — was fetched directly via
`raw.githubusercontent.com` and confirmed live (HTTP 200 on every path
checked), not just search-indexed, so the existence of working code and
included pretrained weights is fully verified, unlike several prior entries
in this log where only a paper claim, not code, could be checked.)

**Licence (verbatim, from the repo's `LICENSE` file, fetched directly).**
MIT License, Copyright (c) 2025 ReaganWu. "Permission is hereby granted,
free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software... subject to the following conditions: The above copyright notice
and this permission notice shall be included in all copies or substantial
portions of the Software." **Commercial use: permitted**, including of the
included pretrained weights (the MIT grant in this repo is not scoped to
code-only the way some prior entries' licences were silent on weights/data —
the weights ship inside the same repo under the same LICENSE file, with no
separate licence file or restriction stated for `Pretrained_Weights/`).

**Which failure mode.** Motion blur, specifically and primarily. Not
camouflage — a deblurring pass sharpens edges and texture but does not
manufacture an appearance cue where the object's color genuinely matches its
background; it would not turn a dark clubhead invisible against dark
clothing into something detectable.

**Why it helps this model specifically.** This is the first entry in the log
that requires **zero changes to the existing model, training set, or
labels** — the current YOLO11n CoreML detector stays exactly as it is. The
proposal is purely additive: run RT-Focuser over each captured frame before
handing it to the existing detector. Three things make it a specifically
good fit for this app, verified directly from the README, not inferred: (1)
**it is already deployed and benchmarked on iPhone via CoreML** — 146.72 FPS
on an iPhone 15 (A16 Bionic), which is the exact target runtime class this
project ships to, not a generic "should be exportable" claim like most prior
architecture entries in this log; (2) it is tiny (5.85M params, 15.76
GMACs) and fast (6ms/frame on an RTX 3090, sub-frame-time even on CPU per
the README's OpenVINO/ONNX Runtime numbers), so it plausibly fits inside the
existing per-frame latency budget alongside YOLO11n rather than replacing
it; (3) unlike the training-side blur fixes already logged, this can be
prototyped and measured **immediately against the existing eval harness**
with no training run at all — pull the pretrained ONNX weights, run them as
a preprocessing pass over the held-out test set's frames (especially the
fast-downswing frames where blur is worst), and re-score with the unchanged
`chdet.evaluate` harness to see whether detection rate or center error moves
on the existing CreateML/future-YOLO baseline. That is a same-day
experiment, not a data-engine or architecture-surgery commitment.

**Important caveats.** (1) Trained/benchmarked on the GoPro deblurring
benchmark (Nah et al., general camera-shake and object-motion blur), not
golf-specific data — domain transfer to a golf clubhead's specific
elongated-streak blur shape is unverified and is exactly what the same-day
harness experiment above would test. (2) Deblurring restores *plausible*
sharpness from a lossy blur, not the true pre-blur image — for a clubhead
moving fast enough that its shape is smeared across many pixels, RT-Focuser
may sharpen edges without actually recovering enough real structure for the
existing appearance-only YOLO11n detector to latch onto; the PSNR/SSIM
numbers in the README measure image-quality fidelity, not downstream
detection-rate lift, and the paper's own reported numbers could not be
fetched to check whether they evaluate a detection task at all (arXiv
blocked). (3) Adds a second model to the on-device pipeline, which is a real
latency and binary-size cost even if individually each model is fast —
total added time and combined CoreML memory/compute budget on-device is
unmeasured here. (4) This is complementary to, not a substitute for, the
training-side blur entries already logged (frame-averaging, PSF-synthesis,
channel-stacked YOLO) — those fix the *label* gap so the detector learns
what a real blur streak looks like; this fixes the *input* by removing blur
before detection. Running both is plausible.

**Effort vs. payoff.** Low effort for a first measurement, genuinely unknown
payoff pending that measurement — and that's exactly what makes this a good
next step rather than a full commitment. Effort: no training required at
all — download the included ONNX weights, run `Inference_Video_ONNX.py` (or
adapt it) over a handful of held-out test-set clips, feed the deblurred
frames through the existing CoreML detector, and diff the eval harness
report against the unmodified-frame baseline. This is the cheapest
experiment logged in this file to date, in wall-clock terms, because the
model is pretrained and the harness already exists. Payoff: if detection
rate improves meaningfully on the fast-downswing frames, this is a
zero-retraining win deployable as a small preprocessing addition to the
existing iOS pipeline; if it doesn't (per caveat 2), the experiment still
answers a real open question cheaply and rules out an entire remedy class
without having spent architecture-surgery or data-engine effort first. It
should be tried before, not after, any of the higher-effort architecture
changes already logged (TrackNetV4, DTUM, channel-stacked YOLO) precisely
because it's so much cheaper to falsify or confirm.

---

## 2026-08-15 — SLT-Net (CVPR 2022): a third, differently-mechanized field converging on motion for camouflage — plus a dead-end check on its MoCA-Mask benchmark's licence

**What it is.** "Implicit Motion Handling for Video Camouflaged Object
Detection" (Cheng et al., CVPR 2022) is the founding work of a distinct
research field this log hadn't touched yet: video camouflaged object
detection (VCOD) — objects that are hard to see in a single frame
specifically *because* their appearance matches the background, made
detectable by motion across frames. That is a closer literal match to this
project's camouflage failure mode than either of the two camouflage entries
already logged (TrackNetV4, from sports-broadcast tracking; DTUM, from
infrared surveillance) — this field's entire premise is "camouflage in
ordinary RGB video, fixed with motion," not a related-but-different problem
adapted to camouflage. Architecturally it's also a different mechanism from
both prior entries: instead of raw frame-differencing (TrackNetV4) or a
hand-designed direction-coded convolution (DTUM), SLT-Net builds a **dense
correlation volume** between consecutive frames (the same kind of
representation RAFT-style optical-flow networks use) to *implicitly* capture
inter-frame motion, jointly trained end-to-end with the segmentation loss —
so the motion representation itself is learned, not a fixed differencing
operator — then a **short-term detection module** predicts a mask per frame
pair and a **long-term refinement module** (a spatio-temporal transformer
over T short-term predictions) enforces consistency across the sequence.
The paper also released **MoCA-Mask**, a VCOD benchmark of 87 video
sequences / 22,939 frames (67 animal species, natural environments,
pixel-level masks on every 5th frame), built by adding dense masks to the
earlier MoCA ("Moving Camouflaged Animals") dataset. Repository (PyTorch,
training/eval code for both modules, pretrained checkpoints linked via
Google Drive) fetched and inspected directly at
`github.com/XuelianCheng/SLT-Net` — README, root file listing, and both
`LICENSE` paths confirmed live via direct fetch, not just search-indexing.

**URL.** https://github.com/XuelianCheng/SLT-Net (paper: CVPR 2022,
openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Implicit_Motion_Handling_for_Video_Camouflaged_Object_Detection_CVPR_2022_paper.pdf
— thecvf.com, arxiv.org, and paperswithcode.com's own arXiv mirror were all
unreachable from this sandbox's egress proxy, same restriction noted in
every prior run of this log, so the mechanism description above is sourced
from the repo's own README plus a search-engine-indexed excerpt of the
paper's abstract, not the PDF directly; treat the mechanism's *existence*
as verified (it's stated in the authors' own repo) but the paper's specific
reported accuracy numbers as unfetched and therefore not cited here.)

**Licence (verified directly, not README-claimed).**
`raw.githubusercontent.com/XuelianCheng/SLT-Net/main/LICENSE` and the
equivalent `master/LICENSE` path both return HTTP 404 — no LICENSE,
LICENSE.md, or COPYING file exists anywhere in the repo. **Commercial use
of the code is NOT confirmed permitted; treat as forbidden by default**
until the authors grant explicit terms — same posture already applied in
this log to GolfPose, detectInBlur, DTUM, and dj_masters (2026-08-13/14
entries). What's safely reusable regardless: the *architectural idea*
(correlation-volume-based implicit motion estimation, jointly trained with
the detection/segmentation head, rather than a fixed differencing operator)
is describable and reimplementable from the paper's own mechanism
description without copying this repo's code.

**A second check this run made and could not resolve: MoCA/MoCA-Mask's own
licence.** A search-engine summary (not the primary source) claimed the
underlying MoCA dataset is "available to download for commercial/research
purposes under a Creative Commons Attribution 4.0 International License."
This could **not** be verified — the dataset's home page,
`robots.ox.ac.uk/~vgg/data/MoCA/`, is blocked by this sandbox's egress
proxy at both the raw-fetch and tool level (confirmed via two independent
fetch attempts, including a Wayback Machine fallback, both blocked), so
this claim is explicitly **not trusted** and is not being logged as a
usable dataset. One piece of corroborating-the-other-way evidence: this run
also checked CAMotion (`github.com/Garyson1204/CAMotion`), a newer
(2026) VCOD benchmark from the same research community that extends the
same MoCA lineage — its README states in the project's own words: **"The
CAMotion dataset is released for academic research only. Commercial use is
strictly prohibited without permission from the authors."** That doesn't
prove MoCA itself is non-commercial, but it means the field's more recent,
directly-comparable benchmark explicitly forecloses commercial use, which
is reason enough to not act on the unverified CC-BY claim for MoCA without
independently confirming it from the primary source first (e.g. from a
machine with unrestricted egress). Recorded so a future run doesn't
re-attempt the same blocked fetch or mistake the unverified claim for a
confirmed one.

**Which failure mode.** Camouflage, specifically and directly — this is the
closest-matching research field found in three runs' worth of camouflage
searching (TrackNetV4 2026-08-13, DTUM 2026-08-14, this entry), since VCOD's
entire premise is single-frame appearance failing due to genuine
background-similarity, fixed by a motion signal. Not applicable to motion
blur.

**Why it helps this model specifically.** Two things make this a real
addition to the log rather than a third redundant restatement of "use
motion for camouflage": (1) it's evidence from a *third independent
research community* (sports broadcast, infrared surveillance, and now
dedicated video-camouflage research) converging on the same general
direction, which is stronger triangulation than two; (2) it introduces a
specific, different candidate mechanism for the camera-motion problem both
prior camouflage entries flagged as real and unresolved — raw
frame-differencing (TrackNetV4) or direction-coded convolution (DTUM) both
assume the dominant inter-frame motion is the target's, which breaks on a
handheld phone with its own camera shake. A *learned*, jointly-optimized
correlation volume is not obviously immune to this (it's still built from
raw frame content), but it is a materially different bet than a fixed
differencing operator, since the network could in principle learn to
partially discount coherent whole-frame motion (camera shake) in favor of
motion that's spatially localized and inconsistent with the rest of the
frame (the swinging club) — this is a hypothesis, not a claim the paper
makes or that this run could verify, but it's a concretely different thing
to test in the camera-motion research spike that TrackNetV4's and DTUM's
entries already called for, not a reason to treat this as a fourth
unrelated idea.

**Effort vs. payoff.** High effort, mostly confirmatory payoff, same
posture as DTUM. Effort: not a drop-in — the correlation-volume + short-
term/long-term-transformer architecture is a substantial departure from
single-shot YOLO11n, with the same unresolved CoreML-export and
on-device-budget questions as TrackNetV4 and DTUM, and the same licence
block on the reference code. Payoff: this entry's primary value is
strengthening the case (now three fields, not two) that motion-based
camouflage handling is the right general direction, plus one concrete,
testable implementation detail (try a learned correlation-volume motion
representation, not just raw differencing, when that research spike
happens) — it should not be treated as a fourth separate architecture to
build. The MoCA-licence dead end is itself worth the entry: it closes off a
specific, plausible-sounding "maybe there's a usable general-purpose
camouflage-video dataset" lead cheaply, rather than leaving a future run to
re-discover and re-chase the same unverifiable claim.

---

## 2026-08-15 — AICaddy (`oswinkil-git/AICaddy-A-Golf-Club-Tracer`) checked and ruled out (golf pose/club-tracking area, negative result: genuinely permissive licence, but no shippable artifact)

**What it is.** This is this log's third run today (after RT-Focuser and
SLT-Net, both logged above), rotated into the golf-specific pose/club-
tracking area since the prior two entries covered motion blur and
camouflage back-to-back. AICaddy is a small GitHub project — "a Python
program and Yolov8 model that aims to allow anyone to use machine learning
to trace their golf swing for better analysis" — whose README claims the
model is "already trained on 6000+ images of golf club heads (just
drivers)." It surfaced as the most specific, GitHub-hosted (i.e.
fetchable, unlike Roboflow/Kaggle/HuggingFace, all still blocked by this
sandbox's egress proxy per every prior run's finding) hit for "golf club
head detection YOLO github pretrained model."

**URL.** https://github.com/oswinkil-git/AICaddy-A-Golf-Club-Tracer
(README.md, LICENSE, and main.py all fetched directly via
raw.githubusercontent.com and confirmed live — HTTP 200 on all three, not
search-indexed).

**Licence (verbatim, from the repo's `LICENSE` file, fetched directly).**
BSD 3-Clause License, Copyright (c) 2023, oswinkil. "Redistribution and use
in source and binary forms, with or without modification, are permitted
provided that [attribution/no-endorsement conditions are met]." **Commercial
use: permitted** — this is a genuine, verified permissive licence, unlike
several prior golf-tracking entries in this log (GolfPose, dj_masters) that
turned out to have no real licence grant at all.

**Which failure mode.** Would be camouflage-adjacent if it worked (a
club-specific detector checkpoint could plug the same "does anyone already
solve golf club localization" gap GolfPose was logged for on 2026-08-13) —
moot, per the dealbreaker below. Not motion blur.

**Why it doesn't help this model, despite the clean licence (verified,
not just claimed).** Fetched `main.py` directly: it loads
`model = YOLO('model.pt')` — a local weights file — but the repo's file
listing (fetched directly from the GitHub tree view, not just the README)
contains exactly three files: `LICENSE`, `README.md`, `main.py`. **No
`model.pt`, no dataset, and no download link for either is anywhere in the
repo.** The "trained on 6000+ images" claim in the README is exactly the
kind of unimplemented/aspirational claim this log flagged as a pattern in
the dj_masters entry (2026-08-14) — stated in prose, not backed by an
artifact in the repo itself. Practically: this project ships a licence with
nothing to license. There is no weights file to run, no dataset to retrain
from, and no way to reproduce the claimed model from what's actually in the
repo. Also worth noting: even the claimed training scope (drivers only,
6000+ images, unstated capture conditions) would not have touched the
indoor/low-light gap this project actually needs — the README says nothing
about lighting or capture conditions at all.

**Effort vs. payoff.** Low effort (one search-and-verify pass), zero
payoff — that is the finding. Recorded so a future run doesn't re-discover
this same repo and re-spend a cycle checking it: the licence is real and
clean, but there is nothing shippable behind it. General pattern now
confirmed across three separate golf-tracking-repo checks in this log
(GolfPose's licence-blocked-but-real dataset, dj_masters's fake licence
badge, and now AICaddy's real licence but phantom weights file): small
personal golf-CV GitHub repos in this space consistently oversell what they
actually deliver in the repo itself. Future runs in this area should verify
the artifact (weights/dataset actually present and fetchable), not just the
README's claims or the LICENSE file's presence, before logging anything as
usable.

---

## 2026-08-16 — BlurBall: blur-aware labeling convention + joint position/blur-attribute detector for table tennis (Gossard et al., CVPR 2026 Workshops)

**Area covered.** Rotated back into motion blur (bullet 2), specifically the
"datasets of fast-moving blurred small objects in sport" sub-area, which
this log had not touched directly before (prior blur entries — frame-
averaging, PSF box-expansion, RT-Focuser — are all synthesis/preprocessing
techniques, not an existing real-world blurred-small-object sports
dataset). The commercial-dataset area (bullet 1) was not re-attempted this
run since two prior runs (2026-08-13, 2026-08-14) already exhausted it and
hit the same egress wall (Roboflow/Kaggle/HuggingFace/Dryad/NextCloud all
return `CONNECT tunnel failed, response 403` from this sandbox — confirmed
again this run against `datadryad.org` before pivoting away from it).

**What it is.** BlurBall ("Joint Ball and Motion Blur Estimation for Table
Tennis Ball Tracking," Gossard, Radovic, Ziegler, Zell — University of
Tübingen, accepted CVPR 2026 Workshops/CVSports) is a detector for a
structurally identical problem to this project's blur gap: a small, fast
sports object (table tennis ball) that is frequently a motion-blur streak,
not a point, in broadcast-style footage. Two things are directly relevant,
independent of whether the dataset itself turns out to be usable:

1. **A blur-aware labeling convention that is a direct refinement of this
   project's own rule.** `docs/labeling-spec.md`'s motion-blur rule already
   says "box the full visible clubhead including its motion-blur streak" —
   but a plain axis-aligned box only encodes *extent*, not *direction* or
   *center*. BlurBall's convention instead places the annotated point at the
   **center of the blur streak** (not the ball's leading or trailing edge,
   which is what naive point-labeling defaults to and what the project's
   own elongation-percentile finding suggests may be happening implicitly
   when annotators under-draw blur boxes) and explicitly records blur
   **length and orientation** as separate fields alongside position. Their
   stated motivation for this convention is exactly this project's own
   observed problem: naive point/box labeling "introduces asymmetry and
   ambiguity to the detection task" for blurred objects.
2. **A concrete, working mechanism for turning that richer label into a
   training target**: from a (center, length, orientation) blur annotation,
   generate a heatmap that is an elongated Gaussian smeared along the blur
   axis, rather than a circular point-Gaussian — "blur-aware heatmaps...
   guide the network to capture both the ball center and its motion
   extent." This is a heatmap/keypoint-detector technique (HRNet backbone
   with Squeeze-and-Excitation attention), not a YOLO box-regression
   technique, so it does not port directly to YOLO11n's architecture — but
   the underlying idea (encode blur length + orientation as explicit
   auxiliary supervision, not just a bigger box) is architecture-agnostic
   and could inform a future auxiliary regression head or loss term on top
   of YOLO11n's existing box output.

**URL.** Code: https://github.com/cogsys-tuebingen/blurball (`README.md`
and `LICENSE.md` fetched directly via `raw.githubusercontent.com`, HTTP 200,
confirmed live). Project page:
https://cogsys-tuebingen.github.io/blurball/ (unreachable directly —
`github.io` is blocked by this sandbox's egress proxy like every other
non-`github.com`/`raw.githubusercontent.com` host — but the same content is
published verbatim on the repo's `gh-pages` branch, fetched directly via
`raw.githubusercontent.com/cogsys-tuebingen/blurball/gh-pages/index.html`,
HTTP 200, and used as the primary source for the abstract, dataset
description, and labeling-convention claims above). Paper: arXiv:2509.18387
(arxiv.org itself unreachable from this sandbox; title, venue, authors, and
abstract corroborated identically across the GitHub README, the gh-pages
project page, and independently-indexed ResearchGate/arXiv-HTML search
snippets, so treated as verified short of reading the PDF itself). Dataset:
hosted on University of Tübingen NextCloud
(`cloud.cs.uni-tuebingen.de`) — this host is also blocked by the sandbox's
egress proxy, so the dataset's actual contents could not be inspected.

**Licence.** **Code: MIT** (verbatim from the repo's `LICENSE.md`, fetched
directly — "Permission is hereby granted, free of charge, to any person
obtaining a copy of this software... to deal in the Software without
restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software"). **Commercial use of the code: permitted.** **Dataset: licence
NOT confirmed.** Neither the README nor the gh-pages project page states a
licence for the released table-tennis dataset itself (no CC-BY/CC0/ODC
notice found anywhere in either fetched source), and the dataset is
distributed via a bare NextCloud share link with no accompanying terms file
that could be located. Per this log's established convention (same posture
as DTUM's NUDT-MIRSDT dataset, 2026-08-14): **treat the dataset as NOT
confirmed permitted for commercial use** until the authors state explicit
terms — only the training/model code is a verified permissive licence here.
This matters doubly for this project, since the underlying footage is
described as coming from "diverse table tennis scenes" (i.e. likely
broadcast/online recordings, the same copyright-provenance problem GolfDB
was ruled out for on 2026-08-13), which is exactly the kind of source
material that tends to carry restrictive redistribution terms even when a
paper's own code is openly licensed.

**Which failure mode.** Motion blur, specifically and only. Not camouflage
— BlurBall's problem is a fast, blurred-but-visible ball against a
relatively uncluttered table-tennis-table background, not a low-contrast
object indistinguishable from clutter.

**Why it helps this model specifically.** This is the first source in the
log to attack the *labeling* side of the motion-blur gap rather than the
*data-volume* side (frame-averaging, PSF synthesis) or the *inference-time*
side (RT-Focuser deblurring). The brief's own evidence — median labelled
elongation of only 1.60 despite a spec that instructs full-streak boxing —
is consistent with annotators defaulting to a tight box around the most
visually salient part of the streak rather than the true full extent,
exactly the labeling ambiguity BlurBall's paper calls out by name. Adopting
a center-of-blur + explicit length/orientation convention (even while
keeping YOLO-style box output for training, by deriving the box
deterministically from center+length+orientation instead of hand-drawing
it) would make blur-labeling **mechanical and consistent** rather than
subjective — directly targeting the "box the full streak" rule's actual
failure mode (inconsistent human judgment of what "full" means) rather than
requiring new footage or a new architecture to fix.

**Effort vs. payoff.** Low-to-medium effort for the labeling-convention
idea, unclear payoff since it is unbenchmarked for this exact use; higher
effort and lower confidence for the heatmap-mechanism idea. Effort (part
1): changing the labeling spec to require center + length + orientation
instead of a free-hand box, and adding a small geometry step that derives
the axis-aligned box deterministically from those three numbers, is a
schema and tooling change (labeling-spec.md, the annotation tool, and the
box-derivation code) — no new footage, no architecture change, comparable
effort to the already-logged PSF box-expansion idea. Effort (part 2): the
blur-aware-heatmap idea requires either bolting an auxiliary
length/orientation regression head onto YOLO11n (architecture surgery, CoreML-
export risk, same category as DTUM/TrackNetV4/SLT-Net) or abandoning
YOLO11n's box-regression paradigm for a heatmap-based one entirely (a much
larger change with no CoreML-export precedent shown here) — treat as a
research direction, not a near-term change. Payoff: plausible for part 1
(directly targets a labeling-consistency problem the brief already
diagnosed with hard numbers), unverified for part 2 (no reported accuracy
delta from this run — the PDF was not reachable — and the mechanism is
demonstrated only on point-object ball tracking, not elongated golf-club-
head geometry, which is a bigger domain gap than it looks). The dataset
itself is not usable regardless of payoff, per the licence finding above.