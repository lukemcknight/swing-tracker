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

---

## 2026-08-16 (second run) — RealBlur: a real, low-light/indoor camera-shake blur dataset with a verified commercial-use licence (Rim et al., ECCV 2020)

**Area covered.** Bullet 1 (commercial datasets, especially low-light/
indoor footage) — this run deliberately avoided re-querying
Roboflow/Kaggle/HuggingFace/Zenodo/Dryad/NextCloud, all confirmed blocked
by this sandbox's egress proxy in three prior runs (2026-08-14, 2026-08-15,
and again earlier today), and instead searched for GitHub-hosted work in
adjacent (non-golf) low-light/blur research, since `github.com` and
`raw.githubusercontent.com` are the only hosts this sandbox can reach
directly. Rotated away from motion-blur *technique* papers (this log's
2026-08-16 first run today, BlurBall, already covered that) toward the
dataset bullet specifically, which — unlike the technique bullet — has
produced nothing usable in three prior attempts.

**What it is.** RealBlur ("Real-World Blur Dataset for Learning and
Benchmarking Deblurring Algorithms," Rim, Lee, Won, Cho — POSTECH, ECCV
2020) pairs blurred and sharp images of the *same* scene, captured
simultaneously through a beam-splitter dual-camera rig (one camera at a
long exposure, one at a short exposure, geometrically aligned) — i.e. the
blur is physically real camera output, not synthesized or approximated.
4,738 image pairs across 232 scenes, released as two subsets: RealBlur-R
(camera RAW) and RealBlur-J (JPEG, camera ISP-processed). Per a
search-indexed paper excerpt (ECVA/ecva.net PDF link surfaced in search
results, but ecva.net itself was not fetched directly — same egress
restriction as arXiv in every prior run of this log, so treat the *scene
count and composition* below as one notch below full verification, per
this log's established convention): the blur is described as caused by
**camera shake**, captured in **low-light environments including streets at
night and indoor rooms**, specifically "to cover the most common scenarios
for motion blur." This is the direct, verified-to-exist answer to what
every prior dataset search in this log has failed to find: a low-light/
indoor blur dataset, real (not synthetic).

**URL.** Code + README: https://github.com/rimchang/RealBlur (`README.md`
fetched directly via `raw.githubusercontent.com`, HTTP 200, confirmed
live). Paper: ECCV 2020, `ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf`
(not fetched directly — ecva.net is not `github.com`/`raw.githubusercontent.com`
and was not tested against this sandbox's proxy, so the scene-composition
claim above rests on a search-engine-indexed excerpt, not the PDF).
**Dataset host is NOT reachable from this sandbox**: the primary download
link (`cgdata.postech.ac.kr/sharing/YhKdbtvD0`) returned a connection
failure (`curl` exit with no response, HTTP code `000`), the project page
(`cg.postech.ac.kr/research/RealBlur/`) returned HTTP 403, and the
alternate Google Drive link (`drive.google.com/drive/folders/...`) is
blocked outright by this sandbox's egress proxy (`EGRESS_BLOCKED`,
confirmed via the fetch tool's explicit error, not a timeout). **This run
could not actually download or inspect a single image from the dataset —
only the repo's own README (which states the licence) and secondary
search-indexed summaries (which describe the scene composition) were
verified.** This is a real gap: per the task brief's own instruction to
"check whether the data or code is genuinely downloadable today," the
honest answer here is *not from this sandbox* — a machine with unrestricted
egress would need to confirm the link is still live before this is treated
as actionable.

**Licence (verbatim, from the repo's `README.md`, fetched directly — no
separate `LICENSE` file exists in the repo; the grant is stated inline).**
"The RealBlur dataset is released under CC BY 4.0 license." **Commercial
use: permitted**, subject to CC BY 4.0's attribution requirement. This is a
meaningfully stronger licence finding than most dataset-adjacent entries in
this log to date (GolfPose, DTUM's NUDT-MIRSDT, BlurBall's table-tennis
set, SLT-Net's MoCA-Mask all logged as "not confirmed" or blocked) — the
grant is explicit, in the authors' own words, in a directly-fetched file,
not inferred or aspirational. The code itself (SRN-Deblur, DeblurGAN-v2
training/eval scripts, included as git submodules) is not covered by this
statement and was not separately checked — irrelevant here regardless,
since only the *dataset* is of interest, not these specific (older,
non-CoreML) deblurring architectures.

**Which failure mode.** Motion blur — but a different blur regime than
every technique previously logged, which matters for how it's used. All
four prior blur entries (frame-averaging 2026-08-12, PSF-synthesis and
channel-stacked YOLO 2026-08-13/14, RT-Focuser 2026-08-15, BlurBall
2026-08-16) model or synthesize *subject*-motion blur: a small, fast object
smearing across an otherwise sharper frame, which is what a clubhead does.
RealBlur's blur is **camera-shake** blur from a long exposure of a largely
static scene — closer to "the whole frame smears somewhat uniformly"
than "one object streaks while the background stays sharp." It does not
directly simulate a blurred clubhead. What it *does* provide, and what no
other blur source logged so far provides, is the **low-light/indoor
lighting, noise, and dynamic-range characteristics** the brief flags as
completely unmeasured for this model.

**Why it helps this model specifically.** RT-Focuser (logged 2026-08-15)
is a promising zero-retraining deblur-preprocessing candidate, but its own
caveat (already logged) is that it was trained/benchmarked on GoPro — an
outdoor-daylight, general camera-and-object-motion blur benchmark — with
domain transfer to golf's specific blur "unverified." RealBlur does not fix
that domain gap for the *clubhead-streak* part (wrong blur regime, as
above), but it is the first source found in four runs of dataset searching
that could plausibly fix the *lighting-domain* part of that same gap: if
RT-Focuser (or any deblur-preprocessing net) is fine-tuned or validated
against RealBlur's low-light/indoor image pairs before being trusted on
this app's own indoor/simulator-bay footage, that at least tests whether
the deblurring step degrades gracefully under the noise/exposure
conditions of dim indoor phone video — a cheap, real check the brief's own
"indoor performance has never been measured" gap calls for, done with an
existing off-the-shelf resource instead of waiting on new golf-specific
indoor capture. It is explicitly a partial, adjacent fix — not a
golf-specific or subject-blur-specific one — and should be scoped as such.
(Aside, not separately verified: the same authors released a 2022 ECCV
follow-up, RSBlur, at `github.com/rimchang/RSBlur`, describing a
"realistic blur synthesis pipeline" that models camera ISP effects — noise,
CRF, saturation — on top of synthetic blur; its README states no licence
and no `LICENSE` file exists in the repo, checked directly, so it is NOT
logged as usable here, only noted so a future run doesn't need to
re-discover it from scratch if the licence question is ever resolved.)

**Effort vs. payoff.** Low effort to identify, genuinely unresolved payoff
because the dataset itself could not be inspected or downloaded from this
sandbox. Effort: the licence question is fully answered (permitted), so if
a future run or the project owner can reach `cgdata.postech.ac.kr` (or the
Google Drive mirror) from an unrestricted network, downloading a sample and
visually confirming the indoor/low-light scene claim is a same-day check.
Payoff: capped and indirect — this is not a source of golf-specific
training data, not a source of subject-motion-blur examples, and not
independently confirmed to be downloadable; its only concrete use is as a
lighting/noise-domain sanity check for a deblur-preprocessing candidate
already logged. Recommended as a cheap follow-up validation step for the
RT-Focuser idea (2026-08-15), not as a standalone action item — and the
"dataset host unreachable from this sandbox" finding itself is worth
recording so a future run doesn't re-attempt the same blocked fetch paths.

---

## 2026-08-16 (third run) — SAHI: slice-based fine-tuning as a resolution fix, distinct from the tiled-inference half of the technique

**What it is.** "Slicing Aided Hyper Inference and Fine-Tuning for Small
Object Detection" (Akyon, Onur Altinuc & Temizel, ICIP 2022) and its
official implementation, the `sahi` library, describe **two** separable
techniques, not one: (1) slice/tile an image into overlapping crops, run
detection on each crop plus the full image, and merge results at inference
time; (2) **train** on sliced crops (small objects occupy far more of a
crop's pixels than of the full frame) so the network learns higher-fidelity
small-object features even when later run at normal, unsliced resolution.
Every prior camouflage entry in this log (TrackNetV4, DTUM, SLT-Net,
channel-stacked YOLO, Copy-Paste) treats camouflage as an appearance/motion
problem. This is the first entry to treat it as a **resolution** problem:
YOLO11n downsamples the full frame to a fixed square input, and a small,
distant clubhead can be reduced to only a handful of pixels before the
network ever sees it — a failure mode indistinguishable, in the zero-
detections-even-at-conf-0.05 symptom described in the brief, from true
visual camouflage, but with a different fix.

**URL.** Code: https://github.com/obss/sahi (repo confirmed live, `README.md`
and `LICENSE` fetched directly). Paper: arXiv:2202.06934 (arxiv.org itself
unreachable from this sandbox, per this log's standing egress limitation —
paper claims here are sourced from the abstract as indexed by search and
from the repo's own description of the method, not the PDF).

**Licence (verbatim, from `LICENSE` in `obss/sahi`, fetched directly via
`raw.githubusercontent.com`).** MIT License, "Copyright (c) 2020 obss."
"Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software..." **Commercial use: permitted.** No
dataset ships with this repo — it's a library, not a data source — so no
separate dataset licence question applies.

**Which failure mode.** Camouflage — but, per the framing above, only the
subset of "camouflage" frames that are actually a too-small-for-input-
resolution problem rather than a genuine color/texture-similarity problem.
Not motion blur (slicing a blurred streak into tiles doesn't restore lost
detail; if anything it's neutral-to-harmful there since a streak can span
tile boundaries).

**Why it helps this model specifically.** Two distinct, honestly-scoped
uses:
1. **Diagnostic, cheap, do this first.** Before investing further in
   camouflage-specific architecture or data work, check whether the
   existing camouflage-failure frames in the eval set are actually low-
   resolution-of-subject failures: crop each failing frame tightly around
   the (human-known) clubhead location and re-run the current model on the
   crop alone. If detection recovers on the crop, the "camouflage" failures
   are at least partly a resolution artifact, not an appearance problem —
   which redirects effort toward training-time slicing (or simply training
   at higher input resolution) instead of the temporal/appearance-focused
   fixes already logged. This costs a single evaluation script, no
   retraining, no new data.
2. **If step 1 confirms the resolution hypothesis**, slice-based
   fine-tuning (train on tight crops around labeled clubheads, mixed with
   full-frame negatives, at the same 640-ish input size) is a training-data
   preprocessing change, not an architecture or runtime change — it costs
   nothing at inference time, unlike tiled inference (below), and is
   directly compatible with the existing YOLO11n/CoreML export pipeline.

**Important caveat — the inference-time half of SAHI is likely a dead end
for this app and should not be pursued.** Tiled inference multiplies
inference calls per frame by the number of tiles (the repo's README does
not state a specific overhead figure, and this could not be independently
benchmarked from this sandbox, but the mechanism itself — N crops plus one
full-frame pass, each a separate forward pass — is inherently several times
more compute per frame). This app runs live, on-device, per-frame video
detection; that cost is very unlikely to fit a real-time mobile budget on
top of an already-lightweight YOLO11n. The README's own emphasis (large,
high-resolution static images and satellite imagery) is a different
use case from real-time phone video, and this log should not conflate the
two: only the fine-tuning half of the SAHI paper is being recommended here.

**Effort vs. payoff.** Diagnostic step: very low effort (one script, uses
data and a model already in hand), and it's a fork in the road — a cheap
way to find out whether several already-logged camouflage entries
(TrackNetV4, DTUM, SLT-Net) are solving the right problem before spending
real effort on any of them. Fine-tuning step (if warranted): low-to-medium
effort (a crop-generation preprocessing script plus a retrain, no new
architecture), payoff unverified until the diagnostic is run. Tiled
inference: not recommended, effort would be moderate but payoff is
presumptively negative for this app's real-time on-device constraint.

---

## 2026-08-16 (fourth run) — SINet-V2: single-frame, appearance-based camouflage detection — a fourth, structurally different mechanism, and licence-blocked despite an Apache-2.0 file

**Area covered.** This is the fourth run today. The first three (BlurBall,
RealBlur, SAHI) covered motion blur, datasets, and a resolution-framed take
on camouflage, respectively. This run re-checked the golf-specific
club-tracking area first (bullet 4, not touched today) with a fresh search
for anything released since the last check (2026-08-15, AICaddy) — one new
repo surfaced, `onkar-99/Golf-Ball-Tracking` (misleadingly named; it actually
tracks the *clubhead*, using YOLOv5 plus optical-flow/Dlib-tracker fallbacks
for blurred frames, and its own README states the fallbacks "were not as
promising as were expected to be" — but it has no LICENSE file and no
weights included in the repo, only a Google Drive link, so it's the fourth
instance of the exact "no shippable artifact" pattern already documented in
this log for GolfPose, dj_masters, and AICaddy; not worth a separate full
entry, noted here only so a future run doesn't re-discover it). Given that
dead end, this run pivoted to a genuinely unexplored sub-area within bullet
3: every camouflage entry logged so far (TrackNetV4, DTUM, SLT-Net) is
**motion**-based — it takes multiple frames as input and separates the
clubhead from clutter by how it moves. None of them address the case where
only a *single* frame is available or motion happens to be ambiguous. The
dedicated **camouflaged object detection (COD)** literature — a separate
research field from video camouflaged object detection (VCOD, already
covered via SLT-Net) — solves single-frame camouflage directly, using
appearance cues (fine boundary/texture discrimination) rather than motion.
That's the new thing checked this run.

**What it is.** SINet-V2 ("Concealed Object Detection," Fan, Ji, Cheng, Shao
et al., IEEE TPAMI 2022 — the journal extension of the original SINet, CVPR
2020) is a well-established, actively-cited reference implementation in the
COD field. Its "Search and Identification Network" architecture has two
stages: a coarse localization stage (find candidate regions that might
contain a concealed object) followed by a refinement stage using a Neighbor
Connection Decoder (NCD) and Group-Reversal Attention (GRA) modules that
sharpen the object's boundary against a visually-similar background — i.e.,
the mechanism is explicitly about exploiting subtle edge/texture
discontinuities between object and background *within one frame*, not
motion across frames. It's a segmentation network (outputs a per-pixel
mask), not a box detector. Repo (PyTorch, training/eval scripts, an
`AWESOME_COD_LIST.md` survey of the wider field) fetched and inspected
directly at `github.com/GewelsJI/SINet-V2` — README, LICENSE, and root file
listing all confirmed live via direct fetch, not search-indexed.

**URL.** https://github.com/GewelsJI/SINet-V2 (paper: IEEE TPAMI 2022,
DOI 10.1109/TPAMI.2021.3085766 — IEEE Xplore unreachable from this
sandbox's egress proxy, same restriction as every prior run of this log, so
claims here are sourced from the repo's own README and LICENSE files,
fetched directly, not the PDF).

**Licence — a real conflict between the LICENSE file and the README's own
words, resolved conservatively.** `raw.githubusercontent.com/GewelsJI/
SINet-V2/main/LICENSE` returns the standard Apache License 2.0 full text
verbatim (grants "perpetual, worldwide, non-exclusive, no-charge,
royalty-free, irrevocable copyright license to reproduce, prepare
Derivative Works of, publicly display, publicly perform, sublicense, and
distribute the Work" — ordinarily commercial-use-permitting). **But the
README, fetched directly and separately, states in the authors' own words:
"The source code is free for research and education use only. Any
commercial usage should get formal permission first."** This is a direct
contradiction between the machine-readable LICENSE grant and the authors'
stated intent in prose. Per this log's established practice of trusting
verified primary-source text over inference, and given the explicit,
unambiguous README restriction: **commercial use is NOT confirmed
permitted — treat as forbidden by default until the authors (contact via
the repo) grant explicit written permission**, exactly as the README itself
instructs. This is a new licence-verification lesson worth flagging for
future runs: an Apache-2.0/MIT LICENSE file's presence is necessary but not
sufficient — the README can (and here does) impose an additional
restriction the LICENSE file's text does not itself contain, so both must
be checked, not just whichever is fetched first.

**Which failure mode.** Camouflage, specifically and directly — arguably a
closer literal match to the failure-mode description than any of the three
motion-based entries already logged. The brief's own wording is that
camouflage failures occur "even in visually sharp frames" with "ZERO
candidate detections... at confidence 0.05" — that is a description of a
single-frame appearance failure, which is COD's exact problem statement,
not VCOD's (which assumes multi-frame context is available and useful).
Not applicable to motion blur — COD's target objects are typically static
or slow-moving (frogs, insects, snakes in natural habitat), so the field
has no blur-handling mechanism to borrow.

**Why it helps this model specifically — and an important honest tension
with the three motion-based entries already logged.** If the camouflage
failure really is "the clubhead's colour/texture genuinely matches its
background in this one frame," a motion-based fix (TrackNetV4, DTUM,
SLT-Net) only helps if the frames around the failing frame carry a usable
motion signal — exactly the camera-shake caveat all three of those entries
already flag as unresolved for handheld phone video. An appearance-based
fix doesn't have that dependency: it would work (or not) on the failing
frame in isolation. That makes single-frame COD a genuinely different bet,
not a restatement of the same idea — worth testing independently of
whether the camera-motion problem for the motion-based entries ever gets
resolved. The honest complication, not papered over: COD's target domain
(large-ish, often-static camouflaged animals filling a meaningful fraction
of the frame) is a substantial geometry mismatch with a small, often
elongated (per the brief's own elongation-percentile finding), fast-moving
clubhead — the field's boundary/texture-refinement mechanism is tuned for
a very different object scale and motion regime, and nothing in the
repo or README claims applicability to tiny fast objects. This should be
read as "a structurally interesting, unverified hypothesis," not as
evidence the technique will transfer.

**Effort vs. payoff.** High effort, low-confidence payoff, and blocked by
licence regardless. Effort: SINet-V2 is architecturally a segmentation
network, not a box detector — even setting the licence question aside,
adopting the *idea* (a boundary/texture-refinement mechanism inserted into
the existing single-frame YOLO11n pipeline, or run as a separate
segmentation pass whose mask is converted to a box for use as a candidate-
region proposal ahead of the existing detector) is a nontrivial engineering
project with no CoreML-export precedent shown anywhere in the repo, on top
of the same architecture-surgery/retraining costs already flagged for
TrackNetV4/DTUM/SLT-Net. Payoff: genuinely uncertain given the object-scale
domain gap above, and moot for direct code/weight reuse until the licence
question is resolved by contacting the authors. The one thing worth
carrying forward cheaply: if a future camera-motion research spike (already
called for by the TrackNetV4/DTUM/SLT-Net entries) finds that motion-based
signals are unreliable on handheld footage, single-frame appearance-based
COD is the field to come back to as the alternative bet — logged here so
that pivot doesn't require rediscovering the field from scratch.

---

## 2026-08-17 — TrackNetV3's InpaintNet: post-hoc trajectory rectification for zero-detection frames (ACM MMAsia 2023) — a fifth, different-in-kind mechanism: recovery, not appearance or architecture

**Area covered.** Rotated to bullet 4 (golf-specific pose/club tracking) first, since it had not been touched since 2026-08-15 (AICaddy) and the brief single-line note logged in the fourth 2026-08-16 run. A fresh search found only one new hit,
`rlarcher/GolfTracker` — a 2017 student class project (CMU 16-423) doing ball-flight trajectory drawing via OpenCV contour detection, `master` branch confirmed, no LICENSE file, no dataset, no trained weights, description text only, YouTube demo from 2017. This is the fifth instance of the "no shippable artifact" pattern already documented in this log (GolfPose, dj_masters, AICaddy, `onkar-99/Golf-Ball-Tracking`) — not worth a separate entry, noted here only so a future run doesn't re-spend a cycle on it. That dead end prompted a pivot to a specific, previously-unexamined sub-thread of bullet 3/2: checking whether the TrackNet sports-tracking lineage (already partially covered here via TrackNetV4, 2026-08-13) has a member that addresses the *recovery* side of detection failure — what happens after a frame produces no usable candidate — rather than the *prevention* side (giving the model a better signal to detect on in the first place), which is what every camouflage/blur entry logged so far attempts.

**What it is.** TrackNetV3 ("Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification," Chen, Chen, Wang, Yang, Chen, Ik — ACM International Conference on Multimedia in Asia 2023, DOI 10.1145/3595916.3626370) is the direct predecessor to TrackNetV4 within the same TrackNet lineage this log has already partially covered, but it targets a structurally different problem: not "how do we give the detector a motion cue instead of an appearance cue" (TrackNetV4's contribution), but "what do we do when the detector's heatmap confidence for a frame is near-zero" — i.e., a real, working answer to the exact symptom described in this project's own camouflage failure mode (zero candidate detections even at low confidence). It has two components, verified directly from the repo's code layout and README, not just prose claims: (1) a prediction module that takes an estimated background (per-clip background subtraction) as an auxiliary input channel alongside the raw frame, trained with mixup augmentation; (2) an **InpaintNet module** — a separate, smaller U-Net, confirmed present as an actual trainable model in `model.py` and invocable via `python train.py --model_name InpaintNet --seq_len 16` (not just described in prose) — that takes the sequence of predicted (x, y) positions across a clip, builds a mask flagging the frames where heatmap confidence collapsed to near-zero, and inpaints plausible positions for exactly those masked frames from the surrounding trajectory context. This is a **post-hoc trajectory-repair pass that runs after per-frame detection, on the sequence of detected positions, independent of why any individual frame failed** — occlusion, motion blur, or (by direct analogy) camouflage all manifest identically as "no usable per-frame detection," and the rectification module doesn't need to know which cause produced the gap.

**URL.** https://github.com/qaz812345/TrackNetV3 (`master` branch; README, `LICENSE`, and root file tree — including `train.py`, `test.py`, `predict.py`, `model.py`, confirming `InpaintNet` is a real, trainable component, not just a paper claim — all fetched/confirmed directly via `raw.githubusercontent.com` and the GitHub tree view, live). Paper: ACM MMAsia 2023, DOI 10.1145/3595916.3626370 (`dl.acm.org` is blocked by this sandbox's egress proxy, same restriction as every prior run of this log for non-GitHub hosts, so the specific accuracy figures below are sourced from search-engine-indexed excerpts of the paper's abstract/body, not the PDF or HTML directly — treat those specific numbers as one notch below full verification, per this log's established convention; the *mechanism description* — background-subtraction auxiliary input, mixup, inpainting-mask-driven trajectory correction via U-Net — is corroborated identically across the ACM abstract excerpt and the repo's own README, so the mechanism itself is treated as verified).

**Licence (verbatim, from the repo's `LICENSE` file, fetched directly).** MIT License, Copyright (c) 2024 qaz812345. "Permission is hereby granted, free of charge, to any person obtaining a copy of this software... to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software... subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies." **Commercial use of the code: permitted.** The pretrained checkpoints (both TrackNet and InpaintNet) are distributed via an external Google Drive link, not bundled in the repo, and were not independently fetched or inspected in this run (Google Drive was not tested against this sandbox's egress proxy) — irrelevant regardless, since these checkpoints are trained on badminton, not golf, and the recommendation below is to retrain `InpaintNet` from scratch on this project's own detection sequences, not reuse the badminton weights. **The underlying Shuttlecock Trajectory Dataset (badminton match footage, distributed via a HackMD-linked SharePoint download) has no confirmed licence** — `hackmd.io` is blocked by this sandbox's egress proxy, so its terms could not be checked directly, and as professional/amateur match broadcast footage it is presumptively copyright-restricted, matching the same provenance problem GolfDB (2026-08-13) was ruled out for. This is not a blocker for the finding: only the MIT-licensed *code/architecture* is being proposed for reuse, retrained on this project's own already-labelled detection sequences, which needs no external dataset at all — `InpaintNet`'s training input is (x, y, visibility) trajectory triples derivable from any already-labelled clip, golf or otherwise.

**Which failure mode.** Both, but in a different sense from how "both" was used for channel-stacked YOLO (2026-08-13) or Copy-Paste (2026-08-14). This does not fix *why* a frame produces zero detections — it does not touch appearance (camouflage's root cause) or blur-streak geometry (motion blur's root cause) at all. It fixes what happens *after*: given a swing clip's sequence of per-frame YOLO11n outputs, some of which are gaps (camouflage: zero candidates even at conf 0.05; motion blur: a detection with low confidence or a badly-fit box on a heavily blurred frame), it fills the gaps with a trajectory-consistent estimate. It is a recovery layer, not a detection-quality fix, and is honestly scoped as such — it should not be read as reducing the *true* per-frame detection rate reported in the brief (82%/77%), only as a candidate fix for the **downstream selection ceiling** (measured at 85.9%) and the whole-swing 77% number, both of which are about producing a usable club-position sequence across a clip, not single-frame accuracy in isolation.

**Why it helps this model specifically.** This is the first entry in five runs of this log that is architecturally *compatible with the existing pipeline by construction*, not despite it — it requires zero changes to YOLO11n, zero retraining of the detector, and zero CoreML re-export of the main model, because it operates entirely on the detector's *output* (a sequence of positions/confidences), not its input or internals. Concretely: run the existing on-device YOLO11n exactly as it is today, log the per-frame candidate position and confidence across a swing clip (data the app's downstream "software selection" logic — the thing scored at 85.9% — must already be consuming in some form), flag the low/zero-confidence frames, and pass the sequence through a small, separately-trained rectification model (or, as a v0 with literally zero new models: constrained spline/cubic interpolation between the nearest high-confidence detections, which is the classical special case InpaintNet's U-Net is a learned generalization of). Every other camouflage/blur entry logged in this file (TrackNetV4, DTUM, SLT-Net, SINet-V2, channel-stacked YOLO, RT-Focuser) requires either retraining the detector on new inputs/architecture or adding a second full-frame model to the per-frame inference path; this is the cheapest structural fit found so far because it runs on trajectory coordinates, not images — a handful of floats per frame, not a second CNN forward pass. It directly targets the "software selection ceiling of 85.9%" number named in the brief, which is explicitly a whole-swing, sequence-level metric — precisely InpaintNet's operating level, not a per-frame one.

**Important honest caveats.** (1) The reported accuracy jump (a secondary-source-sourced 87.72% → 97.51%, unverified beyond the search excerpt, on a shuttlecock task, not golf) should not be read as a golf-transferable number — badminton rallies are pure ballistic/aerodynamic trajectories with far more mid-air path predictability than a golf swing, where the clubhead's velocity, direction, and even visibility (behind the body during backswing/downswing) change in more complex, less purely-parabolic ways; the technique's *applicability* is verified, its *effect size* on this specific problem is not. (2) Camouflage gaps and blur gaps are not symmetric inputs to this technique: a genuinely blurred-but-still-partially-visible clubhead usually still yields a low-confidence detection with real (if imprecise) position information for InpaintNet to lean on, whereas a true zero-candidate camouflage frame gives InpaintNet nothing but the surrounding frames' trajectory to extrapolate from — the technique is a strictly weaker fix for camouflage than for blur, since it degrades to open-loop extrapolation exactly when the appearance signal is worst. (3) This is a mitigation for the symptom the brief measures downstream (whole-swing/selection accuracy), not a fix for the per-frame detection-rate numbers (82%/77%) themselves, which is a meaningfully different claim than most other entries in this log make — it should be pursued as a complement to, not a substitute for, the appearance/architecture-side fixes already logged, and its value depends entirely on whether the app's actual UX (drawing a continuous club path) cares more about sequence plausibility than raw per-frame recall, which this run could not confirm from the docs available.

**Effort vs. payoff.** Low effort for a first measurement, plausible payoff scoped narrowly and honestly. Effort: no changes to the shipping detector or its export pipeline at all. A v0 (constrained spline interpolation across existing eval-harness output, no new model) is an afternoon's script reusing data the harness likely already produces per clip. A v1 (retraining `InpaintNet`'s small U-Net architecture — reimplemented from the paper's description and this repo's `model.py` as a reference, not by importing the repo's GPL-free MIT code directly if avoiding even the reimplementation risk is preferred) on this project's own labelled sequences is a self-contained, low-dependency training job, unrelated to and no riskier than the main detector's training pipeline. Payoff: directly targets the two whole-swing metrics the brief names (85.9% selection ceiling, 77% through-swing) rather than the per-frame numbers every other entry targets, which is a genuinely different lever — but per caveat (3) above, its real-world value hinges on an unconfirmed assumption about what the app's downstream logic actually optimizes for, so the honest recommendation is to try the zero-new-model v0 (spline interpolation over existing eval output) first, as a cheap way to test whether sequence-level rectification measurably helps at all, before investing in a trained InpaintNet-style model.

---

## 2026-08-17 (second run) — Degradation Estimation Network (DEN): zero-shot synthetic low-light/noise augmentation for object detection (McGE'25 workshop)

**Area covered.** Rotated to bullet 5 (ways to synthesize/augment training data), specifically targeting a gap none of the 17 prior entries close: every dataset entry to date (RealBlur, GolfDB, ExDark-class low-light sets — see below) either fails on licence or fails to actually be golf/indoor footage the project can use, and the brief's own caveat stands unaddressed: indoor/low-light performance has never been measured, and nothing in the log so far lets the team *synthesize* that lighting domain from footage already in hand, the way the PSF-synthesis (2026-08-14) and Copy-Paste (2026-08-14) entries already do for blur and camouflage respectively. This run first re-checked whether a commercial-use low-light dataset had appeared since the last dataset-area attempt (2026-08-16, RealBlur): `cs-chan/Exclusively-Dark-Image-Dataset` (ExDark, 7,363 images, 12 classes, low-light-to-twilight) surfaced as the best-known low-light object-detection dataset, but its own README states "can be used only for non-commercial research purpose" — a fourth confirmed non-commercial dead end in this area (after GolfDB, MoCA/CAMotion, NUDT-MIRSDT), not worth a full separate entry, noted here so a future run doesn't re-discover it. That dead end is what prompted the pivot to a synthesis-side technique instead.

**What it is.** "Towards a General-Purpose Zero-Shot Synthetic Low-Light Image and Video Pipeline" (Lin et al., ACM MMAsia McGE'25 workshop, DOI 10.1145/3746278.3759376) trains a small Degradation Estimation Network (DEN) that estimates the parameters of a physics-informed sensor-noise model directly from an input frame, then uses those estimated parameters to synthesize a realistic low-light/noisy version of *any* ordinary sRGB image or video clip — no RAW sensor data and no per-camera calibration metadata required (verified directly from the repo: setup instructions build the pipeline from the YouTube-VOS dataset, which is ordinary processed video, not RAW). This is a fundamentally different kind of low-light fix than everything else in this log: RealBlur (2026-08-16) and RT-Focuser (2026-08-15) both address blur under low light; nothing so far addresses the plain fact that the training set's ~1,112 non-test frames are, per the brief, overwhelmingly outdoor/well-lit footage, with the noise, dynamic-range compression, and contrast loss of real indoor/dim capture entirely absent from the label pool. This is a way to manufacture that domain variation from the project's *own already-labelled* outdoor frames, with the labels needing zero adjustment (unlike the blur-box-expansion techniques already logged, degrading an image's exposure/noise doesn't move the clubhead, so the existing box stays exactly correct).

**URL.** Code: https://github.com/JoanneLin168/degradation-estimation-network (`README.md` and `LICENSE` both fetched directly via `raw.githubusercontent.com`, HTTP 200, confirmed live). Paper: arXiv:2504.12169 / ACM DOI 10.1145/3746278.3759376 (arxiv.org and doi.org are both unreachable from this sandbox's egress proxy, the same standing restriction every prior run of this log has hit, so the specific reported detection-accuracy deltas — a search-indexed claim of "up to... 62% AP50-95" improvement — are sourced from a search-engine excerpt of the abstract, not the PDF, and are **not** corroborated anywhere in the repo's own README, which reports no benchmark numbers at all. Treat that 62% figure as unverified and do not repeat it as a confirmed result.).

**Licence (verbatim, from the repo's `LICENSE` file, fetched directly).** Apache License, Version 2.0, January 2004. "Grant of Copyright License... perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work." **Commercial use: permitted**, subject to standard Apache-2.0 conditions (retain notices, state changes). Pretrained weights (`best.pt`) are distributed via a GitHub Release attached to the same repo, under the same LICENSE — not a separate, unstated grant the way several prior entries' bundled weights have been (e.g. RT-Focuser's weights were explicitly covered by the repo's MIT file; DTUM's and BlurBall's external dataset links were explicitly not).

**Which failure mode.** Neither cleanly, and that distinction matters — logging it honestly rather than force-fitting it into the brief's two-bucket framing. This does not synthesize a motion-blur streak (it's a noise/exposure model, not a motion-kernel model) and does not, by itself, make a clubhead's colour match its background (true colour-camouflage is unaffected by sensor noise). What it does is degrade contrast and add noise the way real dim capture does — which is camouflage-*adjacent*, since lower contrast and added noise make an already-marginal appearance cue (dark clubhead against dark clothing/foliage) harder still to discriminate, and is a plausible, cheap way to stress-test whether the current model's camouflage failures get meaningfully worse under exactly the lighting conditions the brief flags as unmeasured — but this is a hypothesis this run could not verify (no experiment was run), not a confirmed mechanism.

**Why it helps this model specifically.** The single most useful thing this offers is not "improves accuracy" (unverified — see licence section) but "makes an unmeasured gap measurable without new capture." The brief is explicit that indoor performance has never been measured because there is no usable indoor test set (`indoor_test` is quarantined). Applying DEN to the existing 3-clip held-out outdoor test set produces a synthetic-but-principled proxy for "how does this exact model perform once real-world noise and low light are present" — a same-day sanity check, not a training change, and one that needs no new licensed data, no new capture, and no labelling (the boxes don't move). If that experiment shows detection rate collapsing under synthetic low light, it's a concrete, load-bearing argument for prioritizing real indoor capture over further outdoor-data work; if it doesn't collapse, that's useful evidence the opposite way. Only after that diagnostic would training-set augmentation (applying DEN to the ~1,112 training frames to manufacture indoor-like variants) be worth the follow-on effort.

**Important caveat verified directly from the repo, not inferred.** The README states plainly: **"Inference: Currently not implemented yet."** Training code (`train.py`) and released weights exist and are fetchable, but there is no working script in the repo today that takes an arbitrary input image/video and emits its synthetic low-light counterpart — the exact "no shippable artifact" pattern this log has repeatedly flagged for golf-specific repos (GolfPose, dj_masters, AICaddy, `onkar-99/Golf-Ball-Tracking`, `rlarcher/GolfTracker`), here surfacing in a non-golf, otherwise well-verified, Apache-2.0 repo instead. Unlike those, this one at least ships a trained checkpoint and a documented model architecture, so writing the missing inference wrapper (load `best.pt`, run the DEN forward pass, apply the estimated noise/degradation parameters to a target frame) is a bounded, buildable task from what's already public — but it is not a pip-install-and-go tool today, and that gap should not be glossed over.

**Effort vs. payoff.** Low-to-medium effort for the diagnostic use, higher and less certain for the training-augmentation use. Effort: the diagnostic (run DEN, once someone writes the missing inference call, over the 196-frame held-out test set and re-score with the existing `chdet.evaluate` harness) is a small, bounded engineering task — one inference wrapper plus a re-run of an eval harness that already exists — not a retraining commitment. Training-set augmentation is a bigger step (regenerate degraded variants of ~1,112 training frames, decide a realistic degradation-strength distribution, retrain) and its payoff is unverified beyond a search-indexed, non-golf, non-repo-confirmed accuracy claim. Payoff: capped honestly by two things — the missing inference code (real but bounded engineering cost) and the fact that this is a noise/exposure model only, not a fix for either failure mode's actual mechanism (colour-similarity for camouflage, motion-kernel shape for blur) — its value is in turning "indoor is unmeasured" into "indoor is measurable today," which is a smaller but immediately actionable claim than the training-augmentation upside, and should be scoped as such rather than oversold.

---

## 2026-08-17 (third run) — GolfPose licence status changed: a LICENSE file now exists and permits commercial use of derived models (update to the 2026-08-13 entry, not a new finding)

**Area covered.** Golf-specific pose/club-tracking. This is a re-check of an already-logged item, justified because it directly contradicts a prior verified finding rather than repeating it: the 2026-08-13 entry above logged `github.com/MingHanLee/GolfPose` as licence-blocked because `raw.githubusercontent.com/MingHanLee/GolfPose/main/LICENSE` and `.../master/LICENSE` both returned HTTP 404 on that date. This run re-checked the same two URLs directly and both now return HTTP 200 with real EULA content — the repo added a LICENSE file sometime between 2026-08-13 and today. That is new, verifiable information, not a repeat of the earlier entry.

**URL.** https://github.com/MingHanLee/GolfPose. Verified today via `raw.githubusercontent.com/MingHanLee/GolfPose/main/LICENSE` (HTTP 200) and `raw.githubusercontent.com/MingHanLee/GolfPose/master/LICENSE` (also HTTP 200, same content) — checked both branch names specifically because that is exactly where the 2026-08-13 run's 404s were reported, to make sure this run wasn't just hitting a different, wrong path. Repo currently shows 43 stars, a `LICENSE` file at root alongside `README.md`, `environment.yml`, `golfpose_3d.py`, `mmdet_test.py`, `mmpose_test.py`.

**Licence (verbatim clauses, fetched directly from the raw file, not summarized from the README).** "Research & Development": internal research and ML model training — permitted. "Commercialization of Derived Models": integrating trained models into commercial products — permitted. "No Redistribution": may not publish, share, distribute, or make the raw Dataset (images, videos, annotations) available to any third party. "No Resale": may not resell or otherwise monetize the raw data. "No Re-identification": may not use for facial recognition or identifying individuals. "No Endorsement": may not claim authors'/institutions' endorsement. Attribution required in academic or commercial product documentation. Provided "AS IS," no warranties; licence terminates automatically on violation, requiring deletion of all copies. **Commercial use: permitted for training and shipping derived models; raw data itself may not be redistributed or resold.** This resolves the 2026-08-13 entry's blocker — that entry's stated condition for moving from "near-zero payoff" to "medium effort, good payoff" (get written commercial-use terms from the authors) has now been met by the repo itself, no email needed to establish the licence question (an email is still required to obtain the actual download link, per the README: "Please email mhlee.cs09@nycu.edu.tw to authorize the dataset download" — that gate is about *access*, not *terms*, and is unchanged).

**Which failure mode.** Neither directly, unchanged from 2026-08-13's assessment — this is a data-volume/auto-labelling finding, not a camouflage or blur fix. Repeating the 2026-08-13 caveat because it still applies and matters more now that the licence question is resolved: the GolfSwing dataset's capture setup is mocap-synced RGB with physical markers on the club (17 golfer keypoints + up to 5 club keypoints), not casual phone footage, so it is presumptively well-lit and clean-background — it does not diversify the *visual conditions* (camouflage, blur) this project is short on, only the sheer volume of correctly-licensed club-location ground truth.

**Why it helps this model specifically.** Unchanged from 2026-08-13's proposal, now unblocked: use the released club-keypoint checkpoints (`GolfPose-2D(C)`, HRNet-w48/ViTPose-H/DEKR variants, reported 0.857–0.870 AP in the repo's own README) as an auto-labelling assist over this project's own unlabelled phone footage — run inference, convert the 5 club keypoints to a tight box via small dilation, route through human review before accepting into the training set. That directly attacks the 54%-Roboflow/29%-own-footage imbalance named in the brief by cutting the cost of labelling more of the app's own footage, which is the footage that actually contains the camouflage and (per the brief's caveat) likely blur cases this project cares about — GolfPose itself doesn't supply blur/camouflage examples, but it could cheapen the labelling of examples the project already has or can capture.

**Effort vs. payoff.** Now medium effort, plausible payoff — upgraded from the 2026-08-13 entry's "low effort to find, near-zero payoff, licence-blocked." Effort: email `mhlee.cs09@nycu.edu.tw` for the dataset link (only needed if using the checkpoints' training data directly rather than just running the released pretrained checkpoints, which the README implies are usable standalone), stand up MMDetection/MMPose per `environment.yml`, run `mmdet_test.py`/`mmpose_test.py`-style inference over a sample of the project's own footage, build the keypoints-to-box conversion plus a human-review queue. This is real integration work, not a config change, and inherits the same architecture-mismatch caveat the 2026-08-13 entry raised (MMPose/HRNet is not the on-device YOLO11n/CoreML pipeline; this is a labelling-time tool, not something that ships in the app). Payoff: genuine but indirect — cheaper labelling of the underrepresented own-footage slice, not a direct accuracy fix for either named failure mode. Recommend: worth a spike (get the checkpoints running, label-assist a small batch, measure reviewer time saved vs. hand-labelling) before committing to full pipeline integration; do not expect this alone to move the camouflage or motion-blur numbers.

---

## 2026-08-17 (fourth run) — YOLO26: the direct successor to YOLO11 in the same Ultralytics repo, with a small-object-specific training change (STAL)

**Area covered.** Rotated to bullet 3 (small/low-contrast object detection techniques), avoiding the three areas already used today (recovery/bullet-4-adjacent for TrackNetV3's InpaintNet, synthesis/bullet 5 for DEN, golf-specific/bullet 4 for the GolfPose licence re-check). Two candidate leads were checked and discarded before this one: "One-Shot Badminton Shuttle Detection for Mobile Robots" (arXiv:2603.06691v2), whose abstract description of a 20,510-frame indoor/outdoor egocentric shuttlecock dataset with deliberately-included motion-blur frames looked promising, but arxiv.org is blocked by this sandbox's egress proxy (the standing restriction every prior run has hit) and no mirror, GitHub repo, or dataset host for it could be found by search — it is **not logged as a finding** because it could not be verified per this log's own rule, only noted here so a future run doesn't re-spend a cycle re-discovering the same dead end. A search for "YOLO-Ball" (tennis ball detection under blur/occlusion, SAGE journal) also came up with no accessible code or dataset repository — same disposition, not logged.

**What it is.** YOLO26 is Ultralytics' newest model generation (2026), released and maintained in the same `ultralytics/ultralytics` GitHub repo that already produces the `YOLO11n` checkpoints this project's own brief says the current detector is built from — it is not a fork, a paper-only architecture, or a different maintainer, and its `n` (nano) variant (`yolo26n.pt`) is a parameter-budget match for what this project already runs on-device. Two training-side changes are new relative to YOLO11, verified directly from the model's own docs page (fetched via GitHub, not the arxiv/blog mirrors): **Progressive Loss**, which shifts training-loss emphasis toward matching the model's inference-time head rather than an auxiliary training-only head; and **STAL (Small-Target-Aware Label Assignment)**, which specifically improves how many small objects get assigned a positive training label during label assignment (the step that decides which anchors/points count as "this is the object" during training) — small objects are disproportionately likely to fall through standard label-assignment thresholds, which is a direct, if generic, description of what a golf clubhead is in most frames of this dataset.

**URL.** https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo26.md (fetched directly via the GitHub blob view; docs.ultralytics.com itself is blocked by this sandbox's egress proxy, so the GitHub-hosted copy of the same doc was used instead — content matches what search-engine excerpts of the blocked page show, so this is treated as fully verified, not a downgrade). Repo: https://github.com/ultralytics/ultralytics (and a slimmer quickstart mirror at https://github.com/ultralytics/yolo26).

**Licence (verbatim, from `raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE`, fetched directly).** GNU Affero General Public License v3 (AGPL-3.0), "a free, copyleft license for software and other kinds of works, specifically designed to ensure cooperation with the community in the case of network server software." **Commercial use is not simply "permitted": it is permitted only under AGPL-3.0's copyleft terms (which, applied to a network-facing/on-device product, generally means the complete corresponding source of anything built on it must be made available), or by purchasing a separate paid Ultralytics Enterprise licence for closed-source commercial use.** This is not a new problem this entry introduces — the 2026-08-13 channel-stacked-YOLO entry already logged that the project's current `YOLO11n` dependency carries the identical AGPL-3.0/Enterprise choice, since YOLO26 ships from the exact same repo under the exact same top-level `LICENSE` file. Logged here only to confirm explicitly: adopting YOLO26 changes nothing about that already-open licensing question, for better or worse — it is exactly as commercially-usable (or not, absent an Enterprise licence) as the `YOLO11n` the project already depends on.

**Which failure mode.** Neither mechanism-specifically, and that should not be oversold. STAL improves *label assignment during training* for small objects generally — it is not an appearance-similarity fix (does nothing for a clubhead that is genuinely camouflaged in colour/texture against its background) and not a motion-blur-geometry fix (does nothing about the training set's shortage of correctly-elongated blur boxes; STAL operates on label *assignment*, not label *shape*). Its plausible benefit is a small, generic uplift to small-object recall/precision that could marginally help *both* regimes indirectly, since a golf clubhead is a small object in essentially every frame regardless of which failure mode is in play — but no accuracy comparison against YOLO11n on any small-object benchmark was found or verified in this run (only a CPU inference-speed claim — "up to 43% faster CPU ONNX inference... on an Intel Xeon CPU" — which is unrelated to accuracy).

**Why it helps this model specifically.** This is the cheapest possible integration path of anything logged in this file to date: no new architecture, no new input modality, no new dataset, no CoreML-export uncertainty (CoreML is a documented first-class export target for YOLO26, same as YOLO11), and no change to the training pipeline beyond swapping which pretrained checkpoint the existing Ultralytics trainer starts from (`yolo11n.pt` → `yolo26n.pt`). Every other entry in this log requires either new data, a new architecture graft, or a new dependency with its own CoreML-export risk; this requires editing one string in an existing training config. That cheapness is also its ceiling: because STAL is a generic small-object training improvement with no documented mechanism aimed at colour-similarity or blur-streak geometry, it should be treated as a free, low-priority "try it and see" checkpoint swap once the project's training pipeline actually exists (it currently does not — the repo is still at the Phase 0 CreateML-baseline stage, per this file's 2026-08-12 opening note, not yet at the YOLO11n stage the brief describes), not as a substitute for any of the camouflage- or blur-specific entries already logged.

**Effort vs. payoff.** Very low effort, low-to-modest and largely unverified payoff. Effort: once the project reaches the point of actually training a YOLO model (it has not yet, per repo state), this is a one-line checkpoint change plus a re-run of the existing `chdet.evaluate` harness — no new code, no new licence risk beyond the AGPL-3.0/Enterprise question the project already has open for YOLO11n. Payoff: genuinely uncertain — no small-object accuracy delta vs. YOLO11n was found or verified (only an unrelated CPU-speed number), and the mechanism (training-time label assignment) does not target either failure mode's actual root cause the way, e.g., the PSF-based blur-box synthesis (2026-08-14) or Copy-Paste camouflage augmentation (2026-08-14) entries do. Recommend: worth defaulting to `yolo26n.pt` instead of `yolo11n.pt` when the project's real training run happens, purely because it is a zero-cost substitution with no plausible downside found — but do not expect it to move either the camouflage or motion-blur numbers on its own, and do not delay on higher-payoff, already-logged entries waiting to "try YOLO26 first."

---

## 2026-08-18 — Motion-Informed Enhancement (Bjerge, Frigaard & Karstoft, Sensors 2023): a sixth camouflage mechanism, and the first that needs zero architecture change — channel-encoded motion inside an ordinary 3-channel image

**Area covered.** Bullet 3 (small/low-contrast/camouflaged object detection, temporal methods). Chosen deliberately over bullet 1 (datasets): a fresh attempt this run to reach `universe.roboflow.com` (golf-clubhead search) confirmed the same standing block every prior run has hit (`EGRESS_BLOCKED`), and a web search for indoor/simulator-bay golf datasets surfaced nothing beyond GolfDB (already ruled out 2026-08-13) and general 2026 indoor-golf-industry news, not a dataset — so this run pivoted, as several before it have, to a technique area where GitHub/search access is actually productive. Five camouflage mechanisms are already logged (TrackNetV4 frame-differencing+attention, DTUM direction-coded convolution, SLT-Net learned correlation volume, SAHI resolution/tiling, SINet-V2 single-frame boundary refinement) — this is a sixth, checked specifically because a search for "motion informed small object detection camouflage" (deliberately outside sports/surveillance, the two fields already mined) surfaced an unrelated domain — ecological insect monitoring — with a mechanism none of the five already-logged entries use.

**What it is.** "Motion Informed Object Detection of Small Insects in Time-lapse Camera Recordings" (Bjerge, Frigaard, Karstoft — Sensors 2023, DOI 10.3390/s23167242, also arXiv:2212.00423) tackles a close structural analogue of this project's camouflage problem: a small, low-contrast object (an insect) against cluttered, visually similar background (foliage/flowers) in a single frame, detected with a standard single-frame CNN detector (YOLOv5 / Faster R-CNN). Their fix, Motion-Informed Enhancement (MIE), is not an architecture change at all — it is a preprocessing step that recolors an ordinary 3-channel image using a 2-frame history, then feeds that recolored image to an *unmodified* detector. Verified directly from the repository's own code (`yolov5/utils/datasetsMotionRGB.py`, `LoadImages.motion_image()`, fetched via `raw.githubusercontent.com` and quoted here verbatim, not paraphrased from the README):

```python
def motion_image(self, im):
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=0)
    if self.imgPrevGray is None:
        self.imgPrevGray = img_gray.copy()
    imgDiff = cv2.absdiff(self.imgPrevGray, img_gray)
    if self.imgPrevDiff is None:
        self.imgPrevDiff = imgDiff.copy()
    imgSumDiff = cv2.add(self.imgPrevDiff, imgDiff)
    if self.imgPrev is None:
        self.imgPrev = im.copy()
    imgMotion = self.imgPrev.copy()
    imgMotion[:,:,0] = self.imgPrev[:,:,0]/2 + self.imgPrev[:,:,2]/2
    imgMotion[:,:,2] = imgSumDiff
    ...
    return imgMotion
```

`im` is OpenCV BGR, so channel 0 is blue and channel 2 is red. The result: the **red channel is replaced** with a 2-frame accumulated grayscale motion-difference signal (`imgSumDiff`); the **blue channel is replaced** with a blend of the previous frame's own blue and red channels (compressed appearance, freeing up the true blue channel for the motion payload); the **green channel is left untouched** (the previous frame's real green, an intact appearance cue). The output is still an ordinary H×W×3 image — no widened first conv layer, no extra input tensor, no new model input format. Only two frames of history are kept (current + previous), not a longer window.

**URL.** Code: https://github.com/kimbjerge/insectsFlowers (`README.md`, `LICENSE`, and `yolov5/utils/datasetsMotionRGB.py` all fetched directly via `raw.githubusercontent.com`, HTTP 200, confirmed live — the `motion_image()` function above is quoted from that direct fetch, not inferred). Paper: Sensors 2023, DOI 10.3390/s23167242 / arXiv:2212.00423 (`mdpi.com`, `pmc.ncbi.nlm.nih.gov`, and `arxiv.org` are all blocked by this sandbox's egress proxy — the same standing restriction every prior run of this log has hit for non-GitHub hosts — so the paper's own quantitative results below are sourced from search-engine-indexed excerpts, not the PDF/HTML directly; treat those specific numbers as one notch below full verification, per this log's established convention. The *mechanism* — the exact code quoted above — is fully verified from the primary source, unlike the numbers.)

**Licence (verbatim, from the repo's `LICENSE` file, fetched directly).** GNU General Public License, Version 3, 29 June 2007. **Commercial use of the repo's code is NOT straightforwardly permitted** — GPL-3.0 is copyleft, same posture already applied in this log to the channel-stacked-YOLO entry (2026-08-13) and to `yolov7-multi-frame`'s own GPL-3.0 licence: distributing a product that incorporates GPL-3.0 code (or code derived from it) obligates releasing the complete corresponding source under GPL-3.0. **Do not import this repo's code into the shipping pipeline without legal review.** What is safely reusable regardless of licence: the technique itself, as shown above, is roughly six lines of generic OpenCV image arithmetic (grayscale conversion, Gaussian blur, `absdiff`, `add`, channel assignment) — a well-known class of operation with no original-expression claim attachable to it, fully reimplementable from the description in this entry without copying any of this repo's actual file. The companion insect dataset (107,387 time-lapse images, per the paper) is not proposed for reuse here — wrong domain, and its licence was not checked in this run since it isn't relevant to a code-only reuse.

**Which failure mode.** Camouflage, specifically and only. Not motion blur — MIE's motion signal is a *difference* map used to reveal a barely-visible-but-present object, not a mechanism for representing or boxing an elongated blur streak; a genuinely blur-smeared clubhead would still just look diffusely blurred in the red channel, no more usefully so than in the original frame.

**Why it helps this model specifically.** Every camouflage mechanism logged so far (TrackNetV4, DTUM, SLT-Net, SINet-V2) requires architecture surgery: a new input branch, a widened first layer, an attention/correlation module, or a switch to a segmentation network — all of them carry the same open questions this log has repeated five times now (CoreML exportability, on-device latency budget, handheld camera-motion robustness) before a single experiment can even run. MIE is structurally different: **the model is untouched.** YOLO11n already accepts a 3-channel image; MIE only changes what's *in* those three channels before the frame reaches the network. That means the entire idea — recompute two of three channels from a 2-frame history, keep the network and its CoreML export pipeline byte-for-byte identical — can be prototyped as a standalone preprocessing script run once over the existing held-out test set's frames, scored with the unmodified `chdet.evaluate` harness, with zero retraining and zero export risk to test whether it helps at all. This is the same "cheap to falsify" property that made RT-Focuser (2026-08-15) attractive, but aimed at camouflage instead of blur, and cheaper still: RT-Focuser needs a second on-device model in the inference path; MIE needs no model at all, just per-frame OpenCV arithmetic the existing capture pipeline can run trivially within a phone's frame budget. It also sidesteps the camera-motion caveat differently than TrackNetV4/DTUM/SLT-Net do: because the motion channel is only ever a same-image-format *hint* fed into a still fundamentally appearance-based detector (the intact green channel and the blue-channel appearance blend still carry real texture/colour), a camera-shake-corrupted motion channel degrades toward "less useful," not toward "detector receives garbage it was never trained to handle" the way a naive frame-difference *input branch* to a purpose-built temporal architecture might.

**Important caveats.** (1) The reported F1 improvement (YOLO micro-F1 0.49→0.71, Faster R-CNN 0.32→0.56, per search-indexed excerpts of the paper, not independently verified from the PDF) is on honeybees against natural flower/foliage backgrounds in static-mounted time-lapse camera traps, not handheld phone video of a golf swing — the domain gap (object shape, motion speed, camera stability, image cadence — time-lapse implies seconds between frames, not 30/60fps video) is real and untested here. Whether a clubhead's much faster inter-frame displacement (the brief's own 54.8% impact-speed-frame statistic) still produces a clean, useful `absdiff` signal rather than a motion channel that's saturated or already itself blurred is an open, cheaply-testable question, not a confirmed transfer. (2) The current YOLO11n was (presumably) trained on ordinary RGB frames — feeding it MIE-recolored frames at inference time without also retraining on MIE-recolored frames is very likely to hurt, not help, accuracy, since the network has never seen that channel semantics; the honest v0 experiment is therefore "train (or fine-tune) on MIE-recolored frames, then evaluate," not "run the existing trained model on recolored frames unchanged" — a retraining step, not truly a zero-training change, despite the architecture staying fixed. (3) The 2-frame-only history (no window/multi-frame accumulation) is simpler than TrackNetV4/DTUM's designs and may be correspondingly weaker at separating true object motion from noise; the paper does not compare against a longer-window variant.

**Effort vs. payoff.** Low effort for a first measurement, real-if-unverified payoff, and the cheapest camouflage architecture-adjacent idea logged in this file to date because it needs no architecture change at all. Effort: the entire technique is reimplementable in well under an hour from the six lines quoted above — no GPL code needs to ship, no dependency added. A first experiment (recolor the held-out test set's frames with a from-scratch reimplementation of `motion_image()`, fine-tune or lightly retrain the existing detector on a small MIE-recolored slice of the training set, re-score with `chdet.evaluate`) is a same-week spike, not a data-engine or multi-week architecture commitment, and is markedly cheaper than every one of the five camouflage entries already logged. Payoff: genuinely unverified for this project's specific object/camera/motion regime (caveat 1) and does require some retraining despite the architecture staying fixed (caveat 2) — so this should be scoped as "cheap enough to try before any of the heavier camouflage architecture entries," not as a confirmed win. Recommended as the next camouflage experiment to actually run, ahead of TrackNetV4/DTUM/SLT-Net/SINet-V2, purely on cost: it is the only camouflage-directed idea in this log that requires no new model, no widened input, and no CoreML re-validation.

---

## 2026-08-18 (second run) — BlenderProc: procedural 3D-CG rendering with a built-in physically-based motion-blur control, as a route to precisely-labeled synthetic data for *both* failure modes at once

**Area covered.** Bullet 5 (synthesizing/augmenting training data). Deliberately distinct from every synthesis entry already logged: PSF-based blur synthesis (2026-08-14) and the Brooks & Barron frame-averaging approach (2026-08-12) both start from a *real, already-captured* sharp image and convolve/blend in a blur kernel; Copy-Paste (2026-08-14) composites a *real* cut-out clubhead onto a *real* background image; DEN (2026-08-17) perturbs a *real* frame's noise/exposure. None of them can put a clubhead in a pose, background, or lighting condition that was never actually filmed. This run checked whether a genuinely different mechanism — full procedural 3D rendering, where the clubhead, background, camera path, and blur are all synthetic and controllable — is a real, usable, verifiable option, rather than assuming it (domain-randomized synthetic CG training data is a well-established idea in robotics/pose-estimation literature generally, but this log's own rule is not to log a technique without checking whether *this specific* implementation is live, downloadable, and licensed today).

**What it is.** BlenderProc (`DLR-RM/BlenderProc`, developed by the German Aerospace Center's Robotics and Mechatronics group) is a Python pipeline that drives Blender to procedurally generate photorealistic synthetic training images: import a 3D model (`.obj`/`.fbx`/`.blend`/`.ply`), place it and the camera under domain randomization (random pose, lighting, background/HDRI, textures, distractor objects), and render RGB plus ready-made **COCO or BOP-format bounding-box/segmentation annotations** — the labels come from the renderer, with zero human labeling cost, unlike every other entry in this log. Critically for this project, it has a documented, first-class, physically-modeled motion-blur feature, not a bolt-on: `bproc.renderer.enable_motion_blur(motion_blur_length=0.5, rolling_shutter_type=..., rolling_shutter_length=...)`, verified directly from the repo's own official example (`examples/advanced/motion_blur_rolling_shutter/README.md`, fetched live via `raw.githubusercontent.com`): the shutter opens `motion_blur_length` fraction of a frame-interval before the object's keyframe pose and closes the same amount after, and a separate rolling-shutter model can be layered on top to mimic a phone's actual (non-global) shutter readout. This is a categorically different, and more physically grounded, blur mechanism than PSF-based post-hoc convolution (2026-08-14), because it comes from Blender's actual per-sub-frame object/camera motion during rendering, not an approximated kernel applied after the fact.

**URL.** Code: https://github.com/DLR-RM/BlenderProc (root `README.md` and `LICENSE`, and the motion-blur example's `README.md`, all fetched directly via `raw.githubusercontent.com`, HTTP 200, confirmed live — the `enable_motion_blur` signature and behavior above is quoted/paraphrased from that direct fetch of the example doc, not inferred from a blog post). Docs site `dlr-rm.github.io` is blocked by this sandbox's egress proxy (the same standing restriction this log has hit repeatedly for non-GitHub hosts), so the function's full docstring was not read; the example README's own description of the parameter was enough to confirm the feature is real and documented, not enough to reproduce the full API surface.

**Licence (verbatim, from the repo's `LICENSE`, fetched directly).** GNU General Public License, Version 3, 29 June 2007. Same copyleft posture already applied elsewhere in this log to GPL-3.0 code (channel-stacked-YOLO 2026-08-13, Motion-Informed Enhancement above): **do not vendor BlenderProc's source into the shipping app or training pipeline's distributed artifacts without legal review.** The important nuance specific to *this* entry, though: BlenderProc is proposed here only as an **offline data-generation tool run at data-prep time**, never linked into or shipped with the app or the exported CoreML model. The FSF's own long-standing GPL FAQ position is that output produced by running a GPL tool is not itself a derivative work of the tool merely because the tool is GPL-licensed (the standard analogy is a GCC-compiled binary not being GPL just because GCC is GPL) — on that reading, rendered images and the model trained on them would not inherit BlenderProc's GPL terms, only BlenderProc's own source code would. This is a legal interpretation, not something this log can certify; **flag for actual legal review before relying on it**, but it is a materially different, and probably-fine, licence posture than importing GPL code into the shipped pipeline the way the Motion-Informed Enhancement entry above warns against.

**Which failure mode.** Both, uniquely among this log's synthesis entries. Motion blur: `enable_motion_blur` gives exact, controllable, physically-modeled blur length/direction tied to real 3D motion, directly targeting the training set's actual gap (median labelled-box elongation 1.60, most labels near-square) by letting the data engine manufacture arbitrarily many correctly-elongated, correctly-labeled blurred clubheads without waiting on real capture. Camouflage: the same pipeline's domain randomization (random backgrounds/textures/lighting/distractors, confirmed as a documented, first-class feature of the tool in an earlier search this run, not just the motion-blur example) can place a rendered clubhead against dark-clothing-like or foliage-like textures at will, targeting the zero-candidate-detection camouflage failure the same way Copy-Paste (2026-08-14) does, but with full control over blur and pose simultaneously — something Copy-Paste's real-image-compositing approach cannot do, because it can only recombine crops that were actually captured.

**Why it helps this model specifically.** This project's stated blocker on the indoor/low-light blur gap is that **no real indoor footage has ever been captured or measured** (`indoor_test` is quarantined, per the brief), and separately, camouflage failures were found on frames that were visually sharp and well-lit — i.e., real capture of the *combination* the model actually fails on (dark-clothing or foliage background, genuine blur streak, correct elongated label) barely exists yet in either direction. BlenderProc is the only synthesis technique logged in this file that does not require *any* real photograph of the failure condition to manufacture a labeled example of it: a single reasonably-detailed 3D clubhead model, rendered against randomized backgrounds/lighting with `enable_motion_blur` sweeping a range of lengths and directions along a plausible swing arc, produces boxes that are ground-truth-correct by construction (no labeling-spec ambiguity, no annotator judgment call) for exactly the elongated-streak-against-cluttered-background examples the training set is short on. That is a strictly larger claim than PSF-synthesis or Copy-Paste, each of which is still bottlenecked by needing a real photo of *something* (a sharp real clubhead to blur, or a real background to composite onto) in roughly the right condition.

**Important caveats.** (1) **No verified free-to-use golf club 3D model was actually confirmed this run** — a search surfaced claims that Meshy's golf-tagged asset library is CC0, but no specific model's licence page was individually fetched and checked, so this is a plausible starting point, not a confirmed one; each candidate asset needs its own licence check before use, same as every dataset entry in this log. (2) Sim-to-real domain gap is real and unmeasured: even excellent domain randomization does not guarantee a model trained partly on rendered clubheads generalizes to real phone video — this is a well-known, general risk of synthetic-CG training data, not something specific to BlenderProc, and the only way to know is to run the experiment and score it on the existing real held-out test set. (3) This is by far the highest up-front-effort synthesis idea logged to date: it requires sourcing or modeling a reasonably accurate 3D clubhead (several club types, ideally), building a BlenderProc scene script (camera path along a swing arc, background/lighting randomization, motion-blur sweep), and a render farm or patient local rendering — materially more engineering than any post-hoc blur or compositing technique already logged, none of which need 3D assets or a renderer at all. (4) COCO/BOP is the confirmed native annotation output, not YOLO `.txt` — a (trivial, well-trodden) format-conversion step is needed before the renders slot into this project's existing YOLO-format dataset pipeline.

**Effort vs. payoff.** High effort, potentially high and uniquely-shaped payoff, and the first synthesis entry in this log that could address the camouflage-plus-blur *combination* directly rather than one failure mode at a time. Effort: real and substantial — 3D asset sourcing/creation, scene-script engineering, a rendering budget, and a format-conversion step, before a single synthetic image reaches the training set; this is a multi-week undertaking, not a same-day spike like Motion-Informed Enhancement or a config-flag change like YOLO26. Payoff: uncapped in principle (arbitrarily many perfectly-labeled examples of the exact failure combination this model has almost none of) but entirely unverified for this specific model until an experiment is actually run, and gated on two real open questions this run could not resolve — a genuinely free-for-commercial-use club model, and the sim-to-real gap. Recommended only as a **later-phase bet**, after the cheaper, already-logged single-failure-mode fixes (Motion-Informed Enhancement for camouflage, PSF-synthesis for blur) have been tried and scored — not as a first move, given the up-front cost.

---

## 2026-08-18 (third run) — CaddieSet (Jung et al., CVPR 2025 Workshops) checked and ruled out: launch-monitor golf dataset, permissive licence, but no imagery released (golf dataset area, negative result)

**What it is.** This log's third run today, rotated back into the golf-
specific dataset area since the first two runs today (Motion-Informed
Enhancement, BlenderProc) were both camouflage/blur-mechanism entries, and
that area hasn't had a hit since AICaddy was ruled out on 2026-08-15.
CaddieSet is a golf swing dataset published at CVPR 2025 Workshops
(CVSPORTS): "swing videos and ball flight estimates of 8 individuals with
diverse golf skills were collected using a **camera-based launch
monitor**," comprising 1,757 shots (924 face-on, 833 down-the-line views).
The "camera-based launch monitor" capture method is notable because that
class of device (TrackMan, GCQuad-style units) is exactly the kind of
hitting-bay/indoor-simulator setup this project has never captured or
measured — the only golf-specific candidate so far in this log whose
capture conditions plausibly overlap the unmeasured indoor gap at all.

**URL.** https://github.com/damilab/CaddieSet (paper:
https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Jung_CaddieSet_A_Golf_Swing_Dataset_with_Human_Joint_Features_and_CVPRW_2025_paper.html
— blocked by this sandbox's egress proxy, so not read directly; the GitHub
repo's README.md and LICENSE were fetched directly via
raw.githubusercontent.com and confirmed live).

**Licence (verbatim, from the repo's `LICENSE` file, fetched directly).**
MIT License, copyright damilab 2024. "Permission is hereby granted, free of
charge, to any person obtaining a copy of this software... to deal in the
Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software." **Commercial use: permitted** — a real,
unambiguous permissive licence, same tier as AICaddy's BSD-3-Clause.

**Which failure mode.** Would be motion-blur-adjacent if usable (launch-
monitor capture in a hitting bay is a real source of the indoor/lower-light
conditions this project has zero measured data on) — moot, per the
dealbreaker below.

**Why it doesn't help this model, despite the clean licence.** Fetched the
repo's actual README directly (not the paper abstract): the released
dataset is **extracted numeric features only** — ball-flight metrics
(carry, speed, spin, direction) and 21-22 derived biomechanical measurements
(joint angles, hip rotation, weight shift) per shot, output by the launch
monitor's own pose pipeline. There is no raw video, no frame images, and no
bounding-box or keypoint annotation released anywhere in the repo — the
source videos referenced in the paper were used to *derive* these numbers
but were never published. This is the same dealbreaker pattern as GolfDB
(links to YouTube, not distributable frames) and dj_masters (claims not
backed by a shippable artifact), just with a genuinely clean licence this
time, same as AICaddy. A dataset with no images cannot produce a single
training frame for an object detector, regardless of how well its capture
conditions would otherwise match the indoor gap this project needs filled.

**Effort vs. payoff.** Low effort (one search-and-verify pass: fetched
README + LICENSE directly, no speculation), zero payoff. Recorded so a
future run doesn't re-spend a cycle on this repo. Fourth golf-dataset-area
entry in this log (after GolfDB, dj_masters, AICaddy) to confirm the same
pattern: small-to-mid golf-CV data releases in this space either don't ship
imagery at all, or don't ship a real licence — never both a real licence
and real images together yet. Worth another look only if damilab later
releases the source videos (no indication in the repo that they plan to);
until then this is a dead end, not a lead.

---

## 2026-08-18 (fourth run) — TinyDark-YOLO: a YOLO11-based low-light head, found but not independently verifiable in this sandbox (negative-with-caveats result)

**Area covered.** Bullet 3 (small/low-contrast/camouflaged object detection),
approached from a fresh angle — illumination adaptation rather than motion —
after checking this log's own dead-end notes (2026-08-17 second run: ExDark
is non-commercial; 2026-08-16 second run: RealBlur is the one confirmed
commercial low-light asset so far) to avoid re-treading them.

**What I found and could not clear this log's own bar for.** A paper titled
"TinyDark-YOLO for adaptive and lightweight object detection in low-light
conditions" (Zheng, Chen & Zhang), published in *Scientific Reports*
(`https://www.nature.com/articles/s41598-026-58443-9`, June 2026),
describing a YOLO11-based detector with three additions: an Adaptive Gamma
Enhancement (AGE) module, an Attention-based Intra-scale Feature Interaction
(AIFI) module, and a "Lite Efficient Head" (LEHead) aimed at cutting
compute while improving weak-feature detection in dark scenes, evaluated on
ExDark. Being built on the exact YOLO11 family this project already ships
would make it an unusually direct fit if real.

**Why this is not being logged as a usable finding.** `nature.com` hit this
sandbox's standing egress block (the same one every prior run has recorded
for `arxiv.org`, `universe.roboflow.com`, and others) on every fetch
attempt, including via an archive.org mirror, which is itself blocked here.
That means:
- I could not read the paper itself — everything above is search-engine
  synthesis of snippets, not something I read and verified firsthand. This
  log's own rule ("do not log anything you have not confirmed exists") is
  met only for the paper's *existence* (title/authors/journal/DOI-shaped ID
  are consistent across several independent search results) — not for any
  claim about its actual method, numbers, or contribution beyond what a
  search summary asserts.
- No GitHub repository, code release, or weights were found by search. If
  none exists, this is idea-only — a from-scratch reimplementation based on
  a secondhand summary, which is a materially weaker starting point than
  every other architecture entry in this log (BlurBall, DTUM, SLT-Net,
  Channel-stacked multi-frame YOLO, TrackNetV4 all had a real repo I could
  fetch and read).
- The licence is unconfirmed. *Scientific Reports* lets authors choose
  between CC BY 4.0 and CC BY-NC-ND per article (confirmed as the journal's
  general policy, not this article specifically) — search snippets leaned
  toward NC-ND for this piece, but I could not fetch the article's own
  rights statement to confirm which applies here. Note for whoever revisits
  this: even a confirmed NC-ND tag on the article/PDF would restrict
  redistributing the paper itself, not a clean-room reimplementation of the
  disclosed architecture — that distinction would need the actual method
  section to act on responsibly, which I don't have.

**Which failure mode.** Camouflage-adjacent (low ambient light compounding
low contrast) — same bucket as DEN (2026-08-17 second run), not motion
blur.

**Effort vs. payoff.** Not assessable yet, and that is the finding. This
entry exists so a future run with working `nature.com` access (or that
finds a code mirror/GitHub port this search missed) can pick it up at
"confirm the licence and check for code" instead of re-discovering the
paper from zero. Logged as a lead, explicitly not a recommendation.

---

## 2026-08-19 — SloMoDeblur/SloMoBlur: a real (not synthetic) 42k-pair smartphone motion-blur dataset built on the exact frame-averaging method this log's first entry proposed — existence confirmed, licence NOT confirmed (sandbox egress block)

**Area covered.** Bullet 2 (motion blur — specifically "datasets of
fast-moving blurred small objects in sport," extended to the closest
adjacent thing actually findable: a general-purpose real smartphone-blur
dataset, since three prior runs of the sport-specific sub-area have only
surfaced BlurBall, already logged 2026-08-16). Checked two other leads
first and discarded them before this one, so a future run doesn't
re-spend a cycle: (1) "SHOP" (Cooper & Isaacs, arXiv:2203.15228,
pose-guided deblur→detect→ROI-filter pipeline for handheld objects in
blurry video, IEEE-published) — no GitHub repo could be found by search,
so its code/licence status is unverifiable from here, same disposition as
this log's other no-repo leads; not logged as a full entry. (2)
`LOUEY233/Deblur-YOLO` (joint GAN-deblur + YOLO detection, IJCNN 2021) —
repo is real and MIT-licensed (verified directly via
`raw.githubusercontent.com/LOUEY233/Deblur-YOLO/master/LICENSE`), but its
own README states the repo is missing training/testing datasets,
pretrained weights, and video-testing capability — the same "no shippable
artifact" pattern this log has flagged repeatedly (GolfPose, dj_masters,
AICaddy, `onkar-99/Golf-Ball-Tracking`, `rlarcher/GolfTracker`, DEN); not
worth a full entry over that.

**What it is.** "Deblurring in the Wild: A Real-World Image Deblurring
Dataset from Smartphone High-Speed Videos" (Mahmud, Noki, Majumder, Al
Radi, Sukanto, Lubaina, Khan — University of Dhaka, arXiv:2506.19445)
builds a dataset, released as **SloMoBlur** (Hugging Face:
`masterda/SloMoBlur`), using exactly the method this log's very first
entry (2026-08-12, Brooks & Barron frame-averaging) proposed the project
build for itself: shoot high-frame-rate slow-motion video on an ordinary
modern phone (an iPhone 15 Pro specifically), then average temporally
contiguous frames to synthesize a physically-real long-exposure blur
image, with the temporally-centered sharp frame kept as ground truth. Per
search-indexed excerpts of the paper (30 frames averaged from a 240fps
capture, simulating a ~1/8s effective exposure): **42,000+ high-resolution
(1920×1080) blur-sharp pairs, "8 times the amount of different scenes"
versus prior deblurring benchmarks, "including indoor and outdoor
environments, with varying object and camera motions,"** and a finding
that state-of-the-art deblurring models benchmarked against it show
"significant performance degradation" versus their reported numbers on
older benchmarks (i.e., this is harder/more realistic blur than what
existing deblur models, including RT-Focuser's GoPro-trained baseline
already logged 2026-08-15, were tuned against).

**URL.** Paper: https://arxiv.org/abs/2506.19445 (also indexed at
`huggingface.co/papers/2506.19445`). Dataset:
https://huggingface.co/datasets/masterda/SloMoBlur. **Every one of these
hosts — `arxiv.org`, `huggingface.co`, plus `researchgate.net`,
`researchsquare.com`, and `themoonlight.io` (secondary sources checked
as fallbacks) — returned `EGRESS_BLOCKED` from this sandbox on direct
fetch, consistent with every prior run's standing restriction.** No
GitHub repository for this project could be found by search. Everything
above (scene count, indoor/outdoor claim, frame-count/exposure-simulation
parameters, benchmark-degradation claim) is therefore sourced from
search-engine-indexed excerpts only, not a primary-source read — one
notch below full verification, per this log's established convention.
**Existence of the dataset and paper is confirmed** (consistent title,
authors, arXiv ID, and Hugging Face dataset path across multiple
independent search results); **its licence is not** — no licence tag,
CC/MIT/other text, or terms-of-use snippet surfaced in any search
performed this run, and the one page that would authoritatively state it
(the Hugging Face dataset card's metadata) could not be fetched.

**Which failure mode.** Motion blur, specifically — and unlike RealBlur
(2026-08-16, camera-shake blur from a static beam-splitter rig), this is
architecturally the **same blur regime** the golf clubhead produces:
object/scene motion averaged over a real exposure window on a handheld
phone, not whole-frame camera shake. Not camouflage.

**Why it helps this model specifically.** Two distinct, honestly-scoped
uses, neither blocked by the unresolved licence: (1) **Validates the
method, not just the dataset.** This log's first-ever entry (2026-08-12)
proposed frame-averaging high-fps footage as a way to manufacture
genuinely blurred, correctly-boxed training examples from the project's
own footage, but flagged it as an untested idea with "no confirmed
pretrained model" backing it. SloMoDeblur is independent, published
confirmation that the identical technique — averaging slow-mo frames from
a modern iPhone, the same device class this app targets — produces a
large, real, benchmark-grade blur dataset, not a hypothetical. That
derisks recommending the same approach for this project's own data
engine: it's now a demonstrated, published method, not just a plausible
idea. (2) **A same-device-class deblur validation set, if the licence
permits.** RT-Focuser (2026-08-15, already logged as the leading
zero-retraining deblur-preprocessing candidate) was flagged with an
open, unverified caveat: it was trained/benchmarked on GoPro footage, an
outdoor-daylight, general-motion blur benchmark, with "domain transfer to
golf's specific blur unverified." SloMoDeblur is real iPhone footage
including indoor scenes — if licence-cleared, running RT-Focuser's
pretrained weights against a SloMoDeblur sample (rather than only the
project's own limited outdoor footage) would be a cheap, real test of
whether RT-Focuser's deblurring generalizes to the actual device class
and indoor/outdoor mix this app needs, closing exactly the domain-gap
caveat that entry left open — without needing any new golf-specific
capture.

**Effort vs. payoff.** Low effort to identify, payoff currently capped at
zero by an unresolved licence, same honest posture as the RealBlur entry
(2026-08-16). Effort: if a future run or the project owner can reach
`huggingface.co/datasets/masterda/SloMoBlur` from an unrestricted network,
confirming the licence tag and downloading a sample is a same-day check;
this sandbox's own repeated egress failures across five different hosts
this run (arXiv, HuggingFace, ResearchGate, Research Square, themoonlight)
make clear this is a sandbox limitation, not a sign the resource doesn't
exist or isn't worth chasing. Payoff, once/if the licence is confirmed
permissive: capped and indirect for the same reason RealBlur's was — this
is not golf-specific and not a clubhead-shaped object, so its direct value
is as (a) a derisking data point for the already-recommended frame-
averaging technique and (b) a same-device-class validation set for the
already-logged RT-Focuser deblur candidate, not as training data for the
detector itself. Recommended next step: check the licence from an
unrestricted network before doing anything else with it; do not download
or use the dataset based on this entry alone, since "commercial use
permitted" has not been established.

---

## 2026-08-19 (second run) — "Explainable Graph-Based Golf Swing Analysis" (Applied Sciences, April 2026) checked and ruled out: club-keypoint consumer, not a detector, private unreleased capture (golf pose/tracking area, negative result)

**Area covered.** Rotated to bullet 4 (golf-specific pose/club tracking
papers, benchmarks, or open-source implementations), the area least touched
in the last several runs (last full entry was AICaddy, 2026-08-15; the only
thing since then was the 2026-08-17 GolfPose licence-status update, not a
new find). Avoided bullet 1 (dataset) and bullet 2 (motion blur), both used
in the immediately preceding runs (CaddieSet/TinyDark-YOLO on 08-18,
SloMoDeblur on 08-19 first run).

**What it is.** "Explainable Graph-Based Golf Swing Analysis Integrating
Club and Body Keypoints for Ball Flight Outcome Prediction" (MDPI *Applied
Sciences* 16(8):3813, published April 2026). It trains graph neural networks
(ST-GCN, STGAT) over a unified spatio-temporal graph of body joints *plus*
golf club keypoints to predict three ball-flight outcomes (spin axis, launch
direction, ball speed), with Integrated Gradients used for phase-specific
interpretability. Checked specifically because it is one of the only 2026
papers found that treats the club, not just the body, as a first-class
tracked entity — the same angle that made GolfPose (2026-08-13) worth
logging.

**URL.** https://www.mdpi.com/2076-3417/16/8/3813 — blocked by this
sandbox's egress proxy on direct fetch (`EGRESS_BLOCKED`, same restriction
as every MDPI/arXiv/HuggingFace host hit by prior runs of this log); the
description here is sourced from search-engine-indexed abstract text only,
not a primary-source read. No GitHub repository, dataset link, or
supplementary-code reference for this specific paper turned up in three
different targeted searches (including one aimed directly at data-
availability/supplementary-material mentions).

**Why it doesn't help this model (the finding, not a licence question this
time).** Two independent, verified-enough-to-act-on reasons this is not
usable, regardless of licence:
1. **It is not a club detector — it assumes club keypoints as a given
   input.** The graph model consumes body+club keypoints; it does not
   describe or release a model that locates the club in a raw video frame.
   Even under an ideal licence, this project would still need its own
   clubhead detector to feed it — it solves a different, downstream problem
   (predicting ball flight from an already-tracked swing), not the
   detection problem this log exists to fix.
2. **The capture is private and unreleased.** Per the indexed abstract, the
   321 driver-swing sequences were collected "from six amateur golfers in a
   controlled studio setting" synchronized to TrackMan ball-flight data — a
   small, bespoke, non-public capture, with no dataset release, GitHub repo,
   or supplementary-code link found anywhere in three targeted searches.
   There is nothing here to download, licensed or not.

**Which failure mode.** Neither — ruled out before the camouflage/blur
question was even reachable, since there's no detector or dataset to
evaluate against either failure mode.

**Effort vs. payoff.** Low effort (one search-and-verify pass, consistent
with this log's other negative results), zero payoff — recorded so a future
run doesn't re-discover this same paper while searching the golf-pose area
and re-spend a cycle confirming it's a dead end. The golf-specific
pose/tracking area of this rotation is increasingly thin: of five things
checked so far (GolfDB, GolfPose, dj_masters, AICaddy, and now this paper),
only GolfPose has ever cleared to something implementable, and only after
its licence changed between runs. A future run in this area should consider
widening the search to golf *swing-analysis SaaS/app vendors* with public
technical blog posts or patents describing their detection approach, rather
than continuing to search for more academic papers in the same thin vein.

---

## 2026-08-19 (third run) — Deblur-YOLO checked and ruled out: joint detection+deblur GAN architecture, real MIT licence, but no code ever shipped (motion blur area, negative-with-caveats result)

**Area covered.** Rotated to bullet 2 (motion blur — blur-robust detection
architectures specifically), avoiding bullet 1 (dataset area — SloMoDeblur,
this run's immediately preceding run) and bullet 4 (golf pose/tracking —
the second run today). Also distinct from the architecture-side entries
already logged for camouflage (TrackNetV4, DTUM, SINet-V2, channel-stacked
multi-frame YOLO, Motion-Informed Enhancement) and for blur specifically
(RT-Focuser, DEN, PSF-based synthesis, BlurBall) — none of those is a
joint detection+deblurring network, which is the angle this entry checks.

**What it is.** "Deblur-YOLO: Real-Time Object Detection with Efficient
Blind Motion Deblurring" (Zheng, Wu, Jiang, Lu & Gupta, IJCNN 2021). A
YOLO-based detector fused with a GAN-based blind-deblurring front end: a
dilated feature-pyramid generator restores a sharp image from a blurred
input, trained against a pair of multi-scale spectral-norm discriminators
plus a *detection* discriminator (i.e. the deblurring loss is shaped by
what helps detection, not just pixel fidelity) — structurally the closest
thing found so far to "take a YOLO model and make it blur-robust" rather
than a wholesale architecture swap or a temporal/multi-frame trick.

**URL.** https://github.com/LOUEY233/Deblur-YOLO (paper:
https://ieeexplore.ieee.org/document/9534352, IJCNN 2021, paywalled, not
fetched directly — this entry is sourced from the repo's own abstract
reproduction, which is verified primary-source text, not a search summary).

**Verification (repo cloned directly into this sandbox, GitHub egress is
not blocked here).** `git clone` succeeded. Contents: a `README.md`
(abstract, architecture description, citation), a `LICENSE` file, and two
`Arch/`/`Vis/` folders containing PDFs and PNGs of the architecture diagrams
and paper figures only — **no source code, no training or testing dataset,
no pretrained weights**. The README's own TODO list confirms this directly:
"Upload Training Dataset", "Upload Testing Dataset", "Update Code", and
"Upload Pretrained Weight" are all unchecked. `git log` shows the repo's
last commit is from 2021-11-29 — almost five years stale, effectively
abandoned mid-TODO.

**Licence, verbatim (`LICENSE` file, read directly, not inferred).**
```
MIT License

Copyright (c) 2021 ShenZheng2000

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```
Commercial use: **permitted** by the licence text — but there is no
"Software" to license in any usable sense; the grant covers architecture
diagrams and a README.

**Which failure mode.** Motion blur, directly — but only as a design
reference, not as adoptable code. The detection-driven deblurring-GAN idea
(shape the deblur loss by what the downstream detector needs, not by pixel
PSNR alone) is a genuinely relevant pattern for this model's specific gap:
a from-scratch reimplementation ahead of a frozen YOLO11n export is a much
bigger lift than any preprocessing option already logged (RT-Focuser, DEN),
and CoreML/on-device GAN inference at video frame rate is a real feasibility
question this entry does not resolve.

**Effort vs. payoff.** Low-to-medium effort (repo clone + direct file read,
no network egress issues this time since GitHub itself is reachable) for a
confirmed-negative-with-caveats result: real permissive licence, directly
on-topic architecture, but zero shippable artifact — same shape as the
TinyDark-YOLO and AICaddy entries, and worth recording so a future run
doesn't re-find this repo and re-spend a cycle discovering the same empty
TODO list. Unlike TinyDark-YOLO (blocked domain, never actually reached),
this one was fully inspected — the "nothing to use" verdict here is
solid, not a placeholder for a retry.

---

## 2026-08-19 (fourth run) — SAM-PM (CVPR 2024 Workshops): a seventh camouflage mechanism, verified permissive, but the one that rules out foundation-model VCOD as a category for this deployment target

**What it is.** "SAM-PM: Enhancing Video Camouflaged Object Detection using
Spatio-Temporal Attention" (Meeran, Adethya T & Mantha, CVPR 2024 Workshops,
pp. 1857-1866) adapts Meta's Segment Anything Model (SAM) to video
camouflaged object detection by keeping SAM's own encoder/decoder frozen and
adding a trainable "SAM Propagation Module" that enforces temporal
consistency via spatio-temporal cross-attention between consecutive frames'
SAM features — i.e. the camouflage-relevant motion signal is injected as a
lightweight add-on module rather than by retraining the whole backbone.
Reference implementation (PyTorch, full train/eval pipeline, bundles a copy
of Meta's `segment_anything` code, evaluated via MATLAB scripts against the
MoCA and CAD benchmarks) is at `github.com/SpiderNitt/SAM-PM`, fetched and
inspected directly — root file listing, README, and LICENSE all confirmed
live via direct GitHub fetch, not search-indexing. The README itself is
thin: no reported benchmark numbers, no stated SAM backbone variant
(ViT-B/L/H), and no explicit pretrained-checkpoint download link despite a
`ckpt/` directory existing — so this entry cites mechanism and licence only,
not any accuracy claim.

**URL.** https://github.com/SpiderNitt/SAM-PM (paper: CVPR 2024 Workshops /
arXiv:2406.05802 — arxiv.org is blocked by this sandbox's egress proxy, same
restriction as every prior run of this log, so the abstract and any
benchmark numbers could not be fetched directly; only the repo's own
README/LICENSE, which is where the claims below actually come from).

**Licence (verbatim, from the repo's `LICENSE` file at the `master` branch,
fetched directly).** Apache License, Version 2.0. **Commercial use:
permitted**, standard Apache-2.0 conditions (retain notices, state changes).
No separate, more restrictive licence applies to the propagation-module code
itself. Note this says nothing about the MoCA/MoCA-Mask or CAD benchmark
datasets used for the paper's own evaluation — this project would train on
its own data, not those benchmarks, so their licence status (already flagged
unresolved for MoCA-Mask in the 2026-08-15 SLT-Net entry) doesn't block
reuse of the code/technique here.

**Which failure mode.** Camouflage, directly — same problem framing as the
six camouflage entries already logged (TrackNetV4, DTUM, SLT-Net, SINet-V2,
Motion-Informed Enhancement, InpaintNet). Not applicable to motion blur.

**Why it helps this model specifically — and why it mostly doesn't.** This
is the seventh independently-sourced camouflage mechanism in this log, and
by itself it isn't new information about *whether* motion/temporal signals
help camouflage — six prior entries from four unrelated research fields
already established that. What this entry actually adds is a boundary case
on *how expensive is too expensive*: SAM's frozen encoder is a foundation-
model backbone (SAM ViT-B alone is ~91M parameters; ViT-L and ViT-H are
larger still), run once per frame at inference regardless of which variant,
against YOLO11n's on-device footprint of roughly 2.6M total parameters.
Every prior camouflage entry flagged an unresolved CoreML-export/on-device-
budget question as a caveat; this is the first one where the answer isn't
"unresolved," it's "already known to be no" — a SAM-sized encoder cannot run
per-frame on an iPhone at video-capture rate alongside everything else the
SwingSensei app already does, independent of how good the propagation
module's temporal-consistency signal turns out to be.

**Effort vs. payoff.** Low effort to check, essentially zero payoff as a
model to adopt, but real payoff as a category-closing result. Effort: this
was a fast verify (README + LICENSE fetch, no code run). Payoff: this
closes off SAM/SAM2-based video-camouflaged-object-detection as a practical
line of investigation for this specific on-device deployment target — a
category that, per this run's search, includes multiple other 2024-2025
SAM-based VCOD papers (e.g. "ST-SAM," "Phantom-Insight," "TokenMotion")
that a future run might otherwise be tempted to check one at a time. All of
them share the same disqualifying trait (a foundation-model encoder at
inference time), so a future run should not re-spend a cycle verifying each
one individually unless the project's deployment target changes (e.g. a
server-side preprocessing step becomes acceptable instead of fully
on-device inference). The camouflage rotation area is now well-covered by
this log (seven mechanisms, four research fields, consistent conclusion:
motion helps, architecture-matching to YOLO11n/CoreML is the real
bottleneck) — a future run in this area would get more value from actually
scoping the camera-motion research spike every entry since TrackNetV4 has
called for, or from testing the one architecturally-compatible candidate
already logged (channel-stacked multi-frame YOLO, 2026-08-13), than from
finding an eighth mechanism.

---

## 2026-08-20 — DeFMO / the "FMO" (Fast Moving Objects) deblurring literature: an offline data-engine/labeling-QA tool for the motion-blur failure mode, not an on-device fix

**Note on this run's environment.** This sandbox's egress proxy blocked
every non-GitHub host tried this run — `arxiv.org`, `nature.com`,
`universe.roboflow.com`, `huggingface.co`, `semanticscholar.org`,
`paperswithcode.com`, `kaggle.com`, and `r.jina.ai` all returned
`EGRESS_BLOCKED`, a broader block than prior runs report (several earlier
entries could at least reach `arxiv.org` or a Roboflow project page). Only
`github.com` was reachable, which is why this entry's search deliberately
converged on a GitHub-hosted result: it is the one class of source this run
could actually verify first-hand rather than from a search snippet. A
promising non-GitHub lead was found and dropped for this reason: "MoSA-Det:
motion state adaptive object detection for sports videos" (Scientific
Reports, April 2026, `nature.com/articles/s41598-026-43231-2`) directly
targets both this project's failure modes (its abstract, per search
snippets only, names "motion blur-induced feature degradation" and
"temporal aggregation failure" from "excessive inter-frame displacement" as
its two target problems) but its full text, licence, and code/data
availability could not be loaded — logging it as unverified would repeat
the exact pattern this log already flags as low-value (see the 2026-08-19
first-run SloMoDeblur entry). Also checked and rejected as a repeat: MoSA-Det
and any further camouflage-mechanism search would have been a second
consecutive run on the same rotation area the 2026-08-19 fourth-run entry
just closed out — see that entry's own note not to spend a cycle finding an
eighth mechanism.

**What it is.** DeFMO ("Deblurring and Shape Recovery of Fast Moving
Objects," Rozumnyi et al., CVPR 2021) and its companion repos
(`rozumden/MotionFromBlur`, CVPR 2022, and `rozumden/fmo-deblurring-benchmark`)
belong to a named research subfield — "FMO" (Fast Moving Objects) — built
specifically around the case this project's own labeling spec describes:
a single small object, one per frame, that appears as a motion-blur streak
rather than a sharp shape. DeFMO takes one blurred frame and outputs a
temporal super-resolution: an ordered sequence of sharp sub-appearances of
the object across the exposure window, plus a recovered 2D/3D shape mask —
i.e. it doesn't just sharpen the streak, it decomposes it into where the
object actually was at each instant within that single frame's exposure.
Verified by direct GitHub fetch (not search-indexed): `rozumden/DeFMO`'s
README confirms an MIT `LICENSE` file, a pretrained-model download link
(Polybox-hosted `.zip`, load into `./saved_models`), and working
inference/training scripts. The training dataset (`ShapeBlur`, synthetic:
ShapeNet objects + DTD textures + VOT backgrounds, rendered in Blender
2.79b) is **not** redistributed in the repo — the README states plainly:
"Due to this and also the ShapeNet licence, we cannot make the pre-generated
dataset public" — only the generation recipe is. The companion
`fmo-deblurring-benchmark` repo (also MIT, per its GitHub license badge)
ships three synthetic/lab evaluation sets (TbD: uniformly-colored spheres;
TbD-3D: textured spheres with 3D motion; Falling Objects: arbitrary shapes)
via a `download_datasets.sh` script — none of it golf imagery, and this run
could not confirm the remote hosts behind that script are still live.

**URL.** https://github.com/rozumden/DeFMO (paper: CVPR 2021,
arXiv:2012.00595 — unreachable this run, same egress block noted above) and
https://github.com/rozumden/fmo-deblurring-benchmark.

**Licence (from the GitHub-displayed license badge/file on both repos,
fetched directly, not search-indexed).** MIT License on both `DeFMO` and
`fmo-deblurring-benchmark`. **Commercial use of the code: permitted.** The
pretrained DeFMO weights are distributed from the same repo under the
README's plain instruction to download and use them, with no separate,
more restrictive licence stated for the weights themselves — but note the
weights were trained on ShapeNet-derived synthetic shapes (generic
household-object-style geometry, not clubs), so their *licence* is clear
while their *applicability* to a golf clubhead without retraining is not
(see caveats). The `ShapeBlur` training set itself is explicitly **not**
redistributable (ShapeNet's own non-commercial-research licence blocks
that), so reusing DeFMO's training pipeline as-is, rather than just its
pretrained weights, would require regenerating shapes from a
commercially-licensed source instead of ShapeNet.

**Which failure mode.** Motion blur, specifically — not camouflage, for the
same reason the 2026-08-15 RT-Focuser entry already gives (recovering
sharper structure from a blur streak doesn't fix a case where the object's
colour already matches the background; there's no blur to undo there).

**Why it helps this model specifically.** Every motion-blur entry logged so
far is either a training-side fix (make correctly-boxed blurred examples:
frame-averaging, PSF-synthesis, channel-stacked multi-frame YOLO,
BlenderProc) or an inference-side fix (sharpen before detecting:
RT-Focuser). This is a third, distinct use: an **offline labeling-QA and
data-engine tool**, not a training-time or inference-time model change to
the shipped pipeline at all. The project's own stated blur gap is that
labelled boxes are suspiciously close to square (median elongation 1.60)
even though the spec instructs annotators to box the full streak — i.e.
there's an open question of whether annotators are under-boxing genuinely
blurred frames, or whether the training set genuinely lacks blurred frames
to begin with. DeFMO gives a way to check mechanically, on captured footage
the app already owns: run it over a captured swing clip, and its recovered
sub-frame trajectory directly implies what the *correct*, fully-elongated
streak box should have been for that frame — a way to audit existing labels
or auto-suggest corrected boxes for new footage, instead of relying only on
annotator judgement. This is a complementary, different-in-kind role to
every prior blur entry: it doesn't generate new training images (BlenderProc,
PSF-synthesis) and it doesn't run on-device (RT-Focuser) — it's a one-time
or periodic offline pass over the data-engine's own footage.

**Important caveats.** (1) The pretrained weights' generalization to a golf
clubhead's actual shape (thin blade/hosel geometry, metallic reflectance) is
unverified — ShapeNet's synthetic training shapes are generic
household/toy-style objects, and the benchmark's own eval sets are spheres
and "arbitrary shapes," not anything club-like; DeFMO may need retraining on
club-shaped geometry to be trustworthy as a labeling aid, which reopens the
Blender-2.79b/ShapeNet-licence problem in caveat (2) below rather than being
a drop-in tool. (2) Retraining or extending the pipeline requires Blender
2.79b (released 2019, obsolete) and a source of 3D shapes with a licence
that permits this project's commercial use — ShapeNet's own licence does
not, so a golf-club-shaped retrain would need a different asset source
(e.g. a purchased or self-modeled clubhead mesh), which is extra work not
included in anything verified this run. (3) This produces label *guidance*,
not ground truth — DeFMO's shape recovery is itself a model output with its
own error, so any auto-suggested box would need human review, not blind
acceptance, same as any other auto-labeling aid. (4) Neither repo was run in
this sandbox (no GPU/compute available here) — verification this run is
limited to confirming the code, licence, and pretrained-weight links are
real and live, not to confirming DeFMO's actual deblurring quality on
anything resembling golf footage.

**Effort vs. payoff.** Moderate effort, uncertain but directly-on-point
payoff. Effort: downloading the pretrained weights and running inference on
a handful of already-captured, already-blurry own-swing clips is a bounded,
single-day experiment (no Blender, no retraining, no ShapeNet needed for
this first check) — only *extending* it to golf-shaped retraining would hit
the Blender/licence wall in caveats (1)-(2). Payoff: if DeFMO's recovered
sub-frame trajectory on real swing footage looks plausible even
off-the-shelf, it directly answers the log's open question about whether
the near-square median box elongation reflects genuinely sharp training
frames or under-boxed blurred ones — the single most actionable unresolved
question the motion-blur failure mode currently has, more actionable than
adding an eighth architecture candidate. If the off-the-shelf weights
produce garbage on a club-shaped, metallic, fast-moving object (plausible,
per caveat 1), that's still a cheap, useful negative result before
committing to a full retrain.

---

## 2026-08-20 (second run) — CamDiff: diffusion-inpainting camouflage-scene augmentation (Luo, Wang, Wu, Sakaridis, Cheng, Fan, Van Gool — CAAI AIR 2023), an eighth camouflage mechanism and the first generative-synthesis one

**Area covered.** Before starting fresh research, this run first read the
full log (~2,300 lines, 27 prior entries back to 2026-08-12) to avoid
repeating ground. Note for future runs: the branch `research/clubhead-YYYY-
MM-DD` for **today's date already existed on `origin`** with prior commits
before this run started (this routine appears to fire multiple times per
day, not once) — always `git fetch origin research/clubhead-<today>` and
rebase onto it before starting local work, rather than branching fresh from
`main`/whatever HEAD happens to be checked out, or you will duplicate
already-logged findings and hit a rejected push. This run's first research
pass (Albumentations `MotionBlur` + Sayed & Brostow CVPR 2021 box-dilation)
turned out to exactly duplicate the entry already logged 2026-08-14 (line
434) — caught only because the log was read in full before committing.
Discarded that draft and rotated into area 5 (data-synthesis techniques),
specifically a generative/diffusion angle none of the seven prior
camouflage entries (TrackNetV4, DTUM, SLT-Net, channel-stacked YOLO,
Motion-Informed Enhancement, SAM-PM, SINet-V2) or the two prior
data-synthesis entries (Copy-Paste, BlenderProc) had tried.

**What it is.** CamDiff ("Camouflage Image Augmentation via Diffusion
Model," Luo, Wang, Wu, Sakaridis, Cheng, Fan, Van Gool — CAAI Artificial
Intelligence Research 2023, also arXiv:2304.05469) uses a latent diffusion
inpainting model to synthesize new **salient (non-camouflaged) objects
into existing camouflaged-scene images**, with CLIP-based zero-shot
classification used as a filter to reject failed generations that don't
match the intended object prompt or that accidentally blend into the
background instead of standing out. The stated purpose is the opposite
direction from what this project needs (they add visible clutter to
camouflage scenes to diversify hard-negative backgrounds for a COD model),
but the underlying mechanism — diffusion-inpaint a target object into an
existing background image, filter failures with CLIP, keep the original
scene's other labels intact — is directly invertible for this project's
actual need: diffusion-inpaint a **clubhead** into backgrounds chosen
specifically for low contrast (dark clothing, dense foliage crops pulled
from the project's own footage or public sources), which is exactly the
manufactured-negative-turned-positive the camouflage failure mode is short
of. This is the first generative/diffusion-based data-synthesis technique
in this log — mechanically distinct from Copy-Paste's simple cut-and-paste
compositing (2026-08-14) and BlenderProc's physically-based 3D rendering
(2026-08-18), since a diffusion model can blend lighting/shadow/texture at
the boundary in a way flat compositing cannot, which matters specifically
for a camouflage augmentation whose entire point is subtle blending into
the background rather than looking obviously pasted-in.

**URL.** Code + released 5GB augmented dataset:
https://github.com/drlxj/CamDiff (`README.md` fetched directly via
`raw.githubusercontent.com`, HTTP 200, confirmed live; repo contains
`inpainting_diff.py`, `clip_classification.py`, and generated-image folders
`new`/`new1+1`/`new3`, per direct repo-listing fetch). Paper: CAAI AIR 2023
/ arXiv:2304.05469 — arxiv.org and sciopen.com (the CAAI AIR journal host)
are both unreachable from this sandbox's egress proxy, the same standing
restriction noted in every prior run of this log, so the mechanism
description above is corroborated across the GitHub README (fetched
directly) plus independently-indexed ResearchGate/ADS/ETH Zürich
(people.ee.ethz.ch) search snippets that converge on the same description
— treated as verified in mechanism, not in reported accuracy numbers, which
could not be read from the PDF.

**Licence — VERIFIED ABSENT. Commercial use: NOT confirmed permitted;
treat as forbidden by default.** Checked directly via
`raw.githubusercontent.com/drlxj/CamDiff/<branch>/<file>` for `LICENSE`,
`LICENSE.md`, `LICENSE.txt`, and `COPYING` on both `main` and `master` —
all eight return HTTP 404, while `README.md` returns HTTP 200 on the same
branch, confirming the repo is live and the absence is real, not a fetch
failure. Same posture as GolfPose-before-its-license-update, detectInBlur,
DTUM, dj_masters, and ZoomNeXt/EASE (the latter two checked and discarded
this run in favor of CamDiff — also no LICENSE file, but a less useful
mechanism: both are supervised COD detectors, an architecture category this
log already has five license-blocked examples of). **Do not use this
repo's code or its released 5GB dataset without the authors' explicit
written terms.** What is reusable regardless of the code's licence status:
diffusion-inpainting an object into a chosen background with a CLIP-based
accept/reject filter is a generic, describable pipeline buildable from
scratch against any commercially-licensed diffusion inpainting model (e.g.
Stability AI's SDXL inpainting checkpoints under their own commercial
terms, not evaluated here) and the MIT-licensed open CLIP implementation —
none of CamDiff's own code needs to ship.

**Which failure mode.** Camouflage, specifically and only. Not motion blur
— nothing about diffusion inpainting addresses blur-streak geometry.

**Why it helps this model specifically.** Every camouflage fix logged so
far is either an architecture change (TrackNetV4, DTUM, SLT-Net,
channel-stacked YOLO, SAM-PM, SINet-V2 — all requiring model surgery and
unverified CoreML export) or a labeling/resolution change (SAHI). This is
the first one that stays entirely on the data side: if it works, it plugs
into the existing YOLO11n training pipeline with zero architecture change,
the same appeal BlenderProc and Copy-Paste have for their respective
problems. The specific value-add over Copy-Paste (already logged) is
blend quality: Copy-Paste's flat compositing leaves a visible seam that a
network can learn to key on as a shortcut cue, which is a bad shortcut to
teach a detector whose entire job is finding objects that do *not* stand
out from their background — a diffusion inpainter blending lighting,
shadow, and local texture at the object boundary is a mechanistically
better match for synthesizing genuinely low-contrast training examples.

**Effort vs. payoff.** Medium-high effort, uncertain payoff, and the
weakest-verified entry in the log's camouflage-mechanism family so far.
Effort: this is not a drop-in — it requires standing up a diffusion
inpainting model (with its own separate licence to check), writing prompt/
mask-selection logic to target dark-clothing and foliage regions specif-
ically (CamDiff's own prompts target generic "salient object" diversity,
not this project's specific low-contrast-background requirement, so the
prompt/region-selection work is net-new, not reusable), and building the
CLIP-filter step to reject inpaints that look obviously synthetic — a
non-trivial data-engineering project, not a config change. Payoff is
genuinely unverified: no experiment here (or, as far as could be read, in
the source paper) tests diffusion-inpainted objects for the *reverse* task
(making an object blend in) rather than the paper's actual task (making an
object stand out), so treat "this would work for manufacturing camouflaged
positives" as a plausible, mechanism-grounded hypothesis, not a demonstrated
result. Recommended as a research spike (generate a small batch, spot-check
whether a human can tell they're synthetic, before any pipeline investment)
rather than a committed data-engine change.

---

## 2026-08-20 (fourth run) — GoPro & REDS: the canonical *object-motion* blur/sharp paired datasets, built by frame-averaging, with raw sub-frames shipped — probable CC BY 4.0 (verbatim read blocked)

**What it is.** The two standard real motion-blur/sharp *paired* deblurring
benchmarks, both from Seungjun Nah et al.:
- **GoPro** (a.k.a. GOPRO_Large; Nah, Kim & Lee, "Deep Multi-Scale
  Convolutional Neural Network for Dynamic Scene Deblurring", CVPR 2017):
  3,214 blur/sharp pairs at 1280×720 (2,103 train / 1,111 test). Blur is
  synthesised by **averaging consecutive short-exposure frames from GoPro
  Hero4 240 fps video** — i.e. the exact frame-averaging construction this
  log's first entry (2026-08-12, Brooks & Barron) proposed. Crucially, the
  companion **GOPRO_Large_all** ships *all the raw sharp sub-frames* used to
  build the blur, so you can regenerate blur yourself by averaging an
  arbitrary number of frames (i.e. pick your own synthetic shutter length).
- **REDS** (Nah et al., NTIRE 2019 Challenge on Video Deblurring &
  Super-Resolution): 300 sequences × 100 frames at 720×1280 (240 train / 30
  val / 30 test), blur built by high-fps frame interpolation + averaging with
  a measured camera response function (more physically faithful than plain
  averaging).

**URL.**
- GoPro: https://seungjunnah.github.io/Datasets/gopro (author page) and the
  author's own mirror https://huggingface.co/datasets/snah/GOPRO_Large
- REDS: https://seungjunnah.github.io/Datasets/reds
- (Both author pages `seungjunnah.github.io` AND the HuggingFace mirror were
  egress-blocked in this sandbox, as was arxiv.org — same block wall every
  prior run hit.)

**Licence — PROBABLE CC BY 4.0, but NOT read verbatim (honest gap).** Three
independent secondary sources agree both datasets are **CC BY 4.0**
(commercial use permitted *with attribution*): (1) the GS-Blur paper
(arXiv:2410.23658) tabulates GoPro and REDS as "CC BY 4.0"; (2) the original
author's own HuggingFace mirror `snah/GOPRO_Large` carries a machine-readable
`license: cc-by-4.0` metadata tag; (3) curated deblurring-dataset lists
repeat the same. That the *original author* tagged his own mirror cc-by-4.0
is strong provenance. **BUT** I could not open the authoritative licence text
on either the github.io author page or the HF card (both blocked), so this is
convergent-secondary evidence, **not** the verbatim primary read the log's
standard demands. Treat commercial-use as *probable* and verify the licence
line on the HF card / author page from an unblocked network before relying on
it. (Note the standard caveat for scraped video datasets: a CC BY tag on the
compilation does not by itself launder any third-party footage inside it —
though GoPro/REDS were self-shot by the authors, which mitigates this.)

**Which failure mode.** Motion blur. Not camouflage.

**Why it helps THIS model specifically — and why it is NOT the already-logged
RealBlur or SloMoDeblur.** The blur here is **dynamic-scene / object-motion**
blur (things moving in the frame), which is a closer statistical match to a
fast clubhead streak than the already-logged **RealBlur** (2026-08-16), whose
blur is *camera-shake* (global, handheld) in low light — a different kernel
family. Against the 2026-08-19 **SloMoDeblur** entry (a smartphone
frame-averaging blur set whose licence was totally unconfirmed), GoPro/REDS
is the *licence-cleaner* counterpart (author-tagged CC BY 4.0 vs. nothing) and
adds one thing neither prior entry has: **GOPRO_Large_all's raw sub-frames**.
That raw-frame corpus is a ready, off-the-shelf way to build and unit-test the
frame-averaging augmentation pipeline (log entry #1) end-to-end — averaging N
sub-frames, deriving the elongated box — *before* the project has captured any
golf-specific high-fps footage. It is a **development/validation asset for the
augmentation code and (optionally) pre-training a deblur preprocessor**, not
detector training data: it contains zero clubheads and zero golf domain, so
it cannot close the golf-blur *labelled-example* gap on its own.

**Effort vs. payoff — modest, and honestly incremental.** Low effort to
obtain (single download, standard format). Payoff is indirect and capped:
(a) as a **prototyping corpus** for the frame-averaging pipeline it is
genuinely handy and de-risks that idea cheaply; (b) as **deblur-preprocessor
pre-training** data it feeds the inference-time-deblur route the RT-Focuser
entry (2026-08-15) already flagged as questionable for on-device iOS, so
inherits that skepticism; (c) it does **nothing** for the core "need real
blurred *clubheads*" gap. Recommendation: use GOPRO_Large_all's raw frames to
build/validate the frame-averaging augmentation now (cheap, no licence risk to
the *code*), but treat the datasets themselves as scaffolding, not training
data — and confirm the CC BY 4.0 line verbatim from an unblocked network
before shipping anything derived from the pixels. Net: a useful enabling
asset, not a model-accuracy fix in itself.

---

## 2026-08-21 — CADDIE (CVPR 2026 Workshops): a golf-club-specific, real-time-compact pose paper exists, but this sandbox cannot verify anything past its title (existence-only result, golf pose/tracking area)

**What it is.** "CADDIE: Compact Adaptive Detection-Driven Inference for
Real-Time Golf Club Pose Estimation" — Jung, Changsoo; Yang, Fan; Blanchard,
Nathaniel; Wong, HonYung (Fujitsu Research of America and Colorado State
University), published in the Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR) Workshops, June 2026,
CVsports workshop, pp. 9978–9987. The title alone is a near-exact match for
this project's deployment constraint (real-time, "compact," detection-driven
club pose, i.e. specifically not a heavy offline model) rather than a
generic sports-pose paper adapted after the fact, which is why this is
logged despite the thin verification below rather than discarded.

**URL.**
https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Jung_CADDIE_Compact_Adaptive_Detection-Driven_Inference_for_Real-Time_Golf_Club_Pose_CVPRW_2026_paper.pdf
(listed on https://openaccess.thecvf.com/CVPR2026_workshops/CVsports). The
paper's existence, exact title, full author list, institutions, venue, and
page range are corroborated identically across multiple independent search
results (the CVF open-access listing plus separate indexed citations), which
is good evidence the paper is real — but every domain that could serve its
actual content is blocked from this sandbox: `openaccess.thecvf.com` (the
PDF and the workshop listing itself), `www.semanticscholar.org`,
`www.google.com`, and IEEE Xplore (no reachable document link was ever
surfaced for the "final published version" the CVF listing references) were
all tried and all refused. `arxiv.org` — also tried, per this log's standing
restriction — has no listing under this title, so there is no preprint
mirror either. No GitHub repository, project page, or author lab page
publishing this work was found despite targeted searches (including
co-author Nathaniel Blanchard's CSU/CU Boulder pages).

**Licence — UNKNOWN, not merely unconfirmed.** Zero licence information was
found for either code or data, because no code or data location was found at
all. **Do not treat as available for any use, commercial or otherwise, until
a primary source is actually read.**

**Which failure mode.** Unknown — cannot be determined without the abstract
or method section. The title's "detection-driven" phrasing suggests a
two-stage pipeline (club detection feeding a pose/keypoint stage), which
would be architecturally relevant to a camouflage fix in the same way the
already-logged GolfPose entry (2026-08-13/08-17) is, but this is a guess
from the title, not a verified claim, and is flagged as such.

**Why this is being logged anyway.** This log's standing rule is not to log
things that can't be confirmed to exist — this paper's existence and exact
identifying metadata are confirmed via multiple converging sources, which is
different from a single unverifiable claim. What's unverified is everything
about its actual content and any code/data terms. Given three prior "golf
pose/tracking" entries in this log (GolfPose, dj_masters, AICaddy, graph-
based swing analysis) turned up nothing usable, and this is the first
2026-vintage, purpose-built, real-time-golf-club paper this log has found,
it is worth a named pointer for the project owner — who may have institutional
access (an IEEE Xplore or CVF member login, or a direct email to the
authors) this sandbox does not — rather than silently discarding it.

**Effort vs. payoff.** Low effort spent (a single search-and-verify pass,
~6 queries, all content fetches blocked), payoff currently zero and
unknowable from here. This is *not* an actionable technique yet — it is a
lead. Recommended next step for whoever has real network/institutional
access: pull the actual PDF from IEEE Xplore or CVF, read the method and any
code/data availability statement, and only then decide whether it's a
architecture reference (most likely, given "detection-driven inference") or
a labelled-data source. Do not spend implementation effort on this entry
until someone has actually read the paper.

---

## 2026-08-21 (second run) — Golf dataset sweep for the day: one ruled out, one flagged unreadable (golf dataset area, negative result)

**What was checked.** This run rotated to the "golf swing video/image
datasets, especially indoor/low-light/simulator footage" area, which this
log has hit least often (only GolfDB and CaddieSet so far, both ruled out).
Two new leads surfaced and both dead-end the same way as prior entries in
this area.

**1. GolfPosePro (`github.com/ryanboscobanze/GolfPosePro`) — ruled out.**
Fetched directly. It is an MIT-licensed *tool* (MediaPipe-based swing
analyzer that uses `yt-dlp` to pull reference clips from YouTube Shorts at
run time), not a dataset — the repo's `input videos`/`output generated.mp4`
are worked examples, not a shipped corpus. No indoor/low-light footage is
bundled or mentioned. Same shape as the already-logged AICaddy/dj_masters
checks: permissively licensed code, nothing to train on.

**2. "On the Utility of Pose Estimation Models for Golf Swing Understanding"
(SCIRP, published December 2025) — flagged, not usable.** Multiple
independent search snippets (not the primary source — see below) describe
this paper as comparing YOLO Pose vs. MediaPipe Pose on golf swings using
"a custom dataset consisting of golf swing recordings across diverse
players, backgrounds, and lighting conditions" — the exact "diverse
lighting" property this project's test coverage lacks. That is the only
reason this is worth a named pointer rather than silent discard.

**URL.** https://www.scirp.org/journal/paperinformation?paperid=148105 —
**this sandbox's egress proxy blocks `www.scirp.org` outright** (same
failure mode as `arxiv.org`, `ar5iv.labs.arxiv.org`, `peerj.com`, and
`universe.roboflow.com`, all also tried and blocked this run). Every detail
above is reconstructed from third-party search-result snippets, not the
paper itself, so treat it as one notch below even the CADDIE entry's
verification level (CADDIE's *existence* was corroborated by independently
matching metadata across sources; here even that corroboration is thin —
only one source paraphrase was found, repeated verbatim-ish across search
results, which is more consistent with one search engine's summary
propagating than with independent confirmation).

**Licence — unknown.** No data-availability statement, licence, or download
link for the dataset was found in any snippet despite several targeted
searches. SCIRP articles are typically open-access (CC BY), which would
likely cover the *paper text*, but that says nothing about a redistribution
licence for the underlying video, which almost always needs separate
human-subject/institutional clearance that authors do not grant by default.

**Which failure mode.** Motion blur / camouflage (both, potentially) —
"diverse lighting conditions" is the closest any golf-specific source has
come this log to naming indoor/low-light footage directly. But note: the
paper's task is body-pose keypoints, not clubhead detection, so even if the
raw video were released, there is no reason to expect clubhead bounding
boxes in it — at best it would be unlabelled source footage for this
project's own labeling pipeline, not a drop-in training set.

**Why this is being logged anyway.** Following the same standing rule as
the CADDIE entry: don't silently drop a lead whose existence is plausible
just because this sandbox can't reach it. Unlike CADDIE, this one has a
real, weaker corroboration problem (single-source snippet, not
cross-verified) and a real, additional relevance problem (pose keypoints,
not clubhead boxes) even in the best case — so it should be weighted lower
than CADDIE was, not equally.

**Effort vs. payoff.** Low effort (a handful of search queries plus one
direct GitHub fetch), payoff most likely zero. Recommended next step for
whoever has real network access: read the SCIRP paper's data-availability
section first, before anything else — if it says "dataset available on
request" or similar, it is very unlikely to also grant a commercial
redistribution licence, in which case this lead should be closed out
without further effort. Do not prioritize this over the still-open CADDIE
lead.

## 2026-08-21 (third run) — WASB-SBDT: a verified, MIT-licensed, working multi-sport ball-tracking codebase (small/fast/blurred objects) — code confirmed real, paper still unreadable

**What it is.** `github.com/nttcom/WASB-SBDT` — code release for "Widely
Applicable Strong Baseline for Sports Ball Detection and Tracking" (BMVC
2023). It detects/tracks the ball in soccer, tennis, badminton, volleyball,
and basketball footage — small, fast-moving, frequently motion-blurred
objects, i.e. the same shape of problem as clubhead detection, just for a
different object. This rotates into the "multi-frame/temporal methods for
small, low-contrast objects" area, which this log has covered mostly via
papers (TrackNetV4, DTUM, Motion-Informed Enhancement, channel-stacked
multi-frame YOLO, BlurBall). WASB is different in kind: it is an actual
maintained repo with runnable code and pretrained weights per sport, not a
paper description.

**Verification performed.** Fetched the GitHub repo directly (not search
snippets). Confirmed real: `src/` (with `configs/`, `dataloaders/`,
`datasets/`, `detectors/`, `losses/`, `models/`, `optimizers/`, `runners/`,
`setup_scripts/`, `trackers/`, `utils/`), `Dockerfile`, `GET_STARTED.md`,
`MODEL_ZOO.md`, `README.md`. `GET_STARTED.md` names five real dataset setup
paths (soccer via a setup script, tennis via a SharePoint zip, badminton via
the TrackNetV2 zip, volleyball via two Google Drive files, basketball via a
setup script) and `MODEL_ZOO.md` lists pretrained weights per sport,
compared against DeepBall, DeepBall-Large, BallSeg, TrackNetV2,
ResTrackNetV2, and MonoTrack. This is a substantially more real artifact
than most "existence-only" entries in this log — it's not just a title.

**Licence — verified, permits commercial use.** Fetched
`raw.githubusercontent.com/nttcom/WASB-SBDT/main/LICENSE.md` directly and
read the full text:

> MIT License
>
> Copyright (c) 2023 NTT Communications Corporation
>
> Permission is hereby granted, free of charge, to any person obtaining a
> copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit
> persons to whom the Software is furnished to do so, subject to the
> following conditions: [...]

This covers the **code** (architecture, training/eval scripts, pretrained
weights) for commercial use. It does **not** cover the five underlying
sports datasets, which are separate third-party downloads (SharePoint,
Google Drive) each carrying their own original licence — those were not
checked here and must not be assumed permissive just because the WASB code
wrapping them is MIT.

**What could NOT be verified.** The actual architecture — whether it stacks
consecutive frames as multi-channel input (à la TrackNetV2's 3-frame
9-channel scheme), how many frames, backbone, and heatmap-regression
details — is in the paper, not the repo docs. `arxiv.org`,
`huggingface.co`, `papers.bmvc2023.org`, and `ui.adsabs.harvard.edu` were
all tried and all blocked by this sandbox's egress proxy, the same
recurring failure mode noted in the 2026-08-15, -19, -20, and -21 entries.
`MODEL_ZOO.md` and `GET_STARTED.md` were fetched but neither documents
input-frame count or architecture, only which pretrained weights exist per
sport. So: the code and licence are real and confirmed; the specific
mechanism that makes it work is not verified this run.

**Which failure mode.** Camouflage primarily (multi-frame temporal
detection of a small object against clutter — motion where appearance
alone fails), motion blur secondarily (balls in these sports are
frequently blurred at speed, same as clubheads). Same rationale as every
other multi-frame entry in this log.

**Why this doesn't move the needle much despite being real.** This is the
fourth or fifth verified multi-frame/temporal small-object mechanism logged
(TrackNetV4, channel-stacked multi-frame YOLO, DTUM, Motion-Informed
Enhancement, BlurBall) and doesn't introduce a new mechanism — it's a
working reference implementation of the same family of ideas already
covered, for a different sport. Its actual value here is narrow and
concrete: something a developer could clone today and read the `models/`
and `dataloaders/` source directly to see a real, shipped implementation of
this pattern, instead of reasoning from a paper abstract. That is genuinely
useful for *implementation* once the multi-frame direction is chosen, but
it is not new *evidence* that the direction is right, and it ships no golf
data.

**Effort vs. payoff.** Low-moderate effort (repo fetched directly, licence
file read verbatim, several architecture-detail fetches blocked). Payoff:
low as a new idea (redundant with prior entries), low-moderate as an
implementation reference (real, permissively licensed, runnable code
exists and was confirmed, unlike several earlier paper-only leads). Not
worth further sandbox time — the next useful step is someone with real
network access cloning the repo and reading `src/models/` directly rather
than continuing to hunt for the paper.

---

## 2026-08-22 — CAMotion: a benchmark that annotates motion blur and camouflage as co-occurring attributes on the same sequences (research-only, ruled out for training data)

**What it is.** `github.com/Garyson1204/CAMotion` — code and data release for
"CAMotion: A High-Quality Benchmark for Camouflaged Motion Object Detection
in the Wild" (arXiv 2604.08287, April 2026). It is a video camouflaged
object detection (VCOD) dataset of wild animals (batfish, octopus, geckos,
leopards, owls, insects, etc.) with pixel-level mask annotations, plus depth
maps, optical flow, eval scripts (`eval_video.py`, `eval_image.py`), and
predictions/checkpoints from 18 existing COD/VCOD methods run against it.

This rotates into the small/camouflaged-object-in-video area, but it is a
different kind of finding than the seven or eight VCOD *methods* already in
this log (TrackNetV4, DTUM, SLT-Net, SINet-V2, Motion-Informed Enhancement,
CamDiff, SAM-PM, InpaintNet): CAMotion is the first **benchmark** found here
that explicitly tags **motion blur** as a per-sequence challenge attribute
alongside camouflage, occlusion, and uncertain edges — i.e. it treats this
model's two "co-equal" failure modes as attributes that occur *together* on
the same footage, rather than as two separate research literatures. Every
prior blur-vs-camouflage entry in this log picked one mechanism or the
other; none pointed at a dataset built around both at once.

**Verification performed.** Fetched the GitHub repo and its README directly
(not search snippets). Confirmed real: `eval_video.py`/`eval_image.py`
present, working Google Drive and Baidu Netdisk links for the main dataset,
supplementary depth/optical-flow data, and an 18-model checkpoint/prediction
folder. The README explicitly lists "uncertain edge, occlusion, motion
blur, and shape complexity" as the per-sequence attribute set. `arxiv.org`
was blocked by this sandbox's egress proxy as usual (recurring failure mode
noted in the 2026-08-15, -19, -20, and -21 entries), so the paper's method
section (how attributes are used, exact split sizes) was not read — only
the repo/README content was verified.

**Licence — verified, commercial use is prohibited.** The README states
verbatim: *"The CAMotion dataset is released for academic research only.
Commercial use is strictly prohibited without permission from the
authors."* This is a hard research-only restriction with no ambiguity, so
this entry is a negative result for sourcing training data, same as
CaddieSet and MoCA-Mask earlier in this log. (A close relative, YUV20K —
`github.com/K1NSA/YUV20K`, arXiv 2604.09985, also April 2026, wildlife VCOD
with a similar motion-blur-plus-camouflage attribute framing — was checked
in the same search and is CC BY-NC 4.0, also non-commercial, and additionally
has no source code released yet, only the dataset. Not logged as a separate
entry since it duplicates CAMotion's restriction and offers less.)

**Which failure mode.** Both, genuinely — this is the first entry in this
log where that's true of the *data* rather than of a proposed technique.

**Why this doesn't move training forward, but could inform a decision
already implied by this log.** No clubhead frames are gained (wrong domain,
non-commercial licence). The real, narrow value: this log has now
accumulated ~8 distinct camouflage/temporal mechanisms and several
blur-synthesis routes without a shared way to compare them on footage that
has *both* problems at once, which is exactly this model's outdoor-vs-indoor
gap (the outdoor test set has camouflage without blur; indoor is expected to
have both). CAMotion's attribute tags mean someone could filter its test
split to sequences tagged both "motion blur" and generic camouflage, and use
it as a proxy benchmark to rank the previously-logged methods (e.g.
Motion-Informed Enhancement vs. DTUM vs. channel-stacked multi-frame YOLO)
before committing engineering time to implementing any one of them on real
golf footage. That is a research-planning use, not a training-data or
architecture fix.

**Effort vs. payoff.** Low effort (GitHub README fetched directly, licence
read verbatim, no downloads attempted). Payoff: low as training data (wrong
domain, non-commercial), low-moderate as a research-prioritization tool (it
could cheaply rank existing candidate methods on blur+camouflage footage
before implementation, which nothing else in this log offers) — but that
requires someone to actually download ~20GB+ of animal video and run the
eval harness, which is more setup cost than payoff unless the multi-frame
direction is already committed to. Not a strong finding; logged mainly
because it is the first evidence this log has found that blur and
camouflage are being treated as a joint problem anywhere in the literature,
which is directly relevant to this model's likely indoor failure mode.

---

## 2026-08-22 (second run) — Grounding DINO for open-vocabulary auto-labeling of raw phone footage (data-engine area, not a blur or camouflage technique)

**What it is.** `github.com/IDEA-Research/GroundingDINO` — the official
release of "Grounding DINO: Marrying DINO with Grounded Pre-Training for
Open-Set Object Detection." It takes an `(image, text)` pair and returns
boxes for whatever the text names (e.g. a prompt string like `"golf club
head ."`), with no fine-tuning and no fixed class list. This is a
data-engine finding, not a blur- or camouflage-technique finding: the
proposal is to run it once over raw, currently-unlabelled footage to
generate *candidate* boxes for a human to correct in Label Studio, rather
than drawing every box from scratch — i.e. attack the "only ~29% of training
data is the app's own phone footage" gap by making it cheaper to label more
of it, not by synthesizing new pixels or changing the detector.

**Verification performed.** Fetched the GitHub repo directly. Confirmed: the
README documents the `(image, text)` input/output contract with a worked
example (`"chair . person . dog ."` style prompts), `pip install -e .` is a
real, documented install path, and inference code (`demo/inference_on_a_image.py`
per the repo layout) ships in the repo — this is a working, downloadable
codebase, not a paper-only lead. Pretrained weights are linked from the same
README (Swin-T and Swin-B checkpoints). I did not run the model myself in
this sandbox (no GPU/weights download attempted), so the *specific* claim
that a bare "golf club head" prompt yields usable boxes on this project's
footage is unverified — see caveat below.

**Licence — verified, commercial use is permitted.** The repository's
license badge and footer read **Apache-2.0**, confirmed by fetching the repo
page directly (not a search snippet). Apache-2.0 permits commercial use,
modification, and redistribution subject to standard notice-retention
terms. No separate, more restrictive licence applies to the code; I did not
separately verify the licence terms of the individual Swin-T/Swin-B
checkpoint files, which is worth a follow-up if weights (not just code) end
up redistributed.

**Which failure mode.** Neither, directly — this is a data-engine/labelling
throughput tool, not a fix for camouflage or motion blur specifically. It is
logged because "ways to synthesise or augment training data" is one of the
rotation areas and every prior entry in that area has been about
synthesising *pixels* (blur kernels, diffusion inpainting, copy-paste,
BlenderProc renders) rather than about reducing the *labelling cost* of real
footage the project already has or could easily capture more of.

**Why it helps this model specifically — and an honest limit on that.** The
project's own stated gap is that only ~29% of training data is the app's own
phone footage, with the rest third-party Roboflow imagery; more own-footage
is repeatedly flagged elsewhere in this repo as the top data-engine
priority. Manual Label Studio annotation is the likely bottleneck to using
more of it. An open-vocabulary detector used as a pre-labelling pass (box
proposal, then human correction, not automatic acceptance) is a standard,
verifiable way to cut that cost — this exact workflow (open-vocabulary
annotation followed by pseudo-label review) is the subject of a paper found
in the same search, "DART: An Automated End-to-End Object Detection Pipeline
with Data Diversification, Open-Vocabulary Bounding Box Annotation,
Pseudo-Label Review, and Model Training" (arXiv 2407.09174) — arxiv.org was
blocked by this sandbox's egress proxy as in every prior entry that hit it,
so DART itself is logged as a corroborating title only, not independently
verified.

The honest limit: a general-purpose open-vocabulary model was pretrained on
web-scale image-text pairs, not on golf video, and "golf club head" is a
fine-grained, unusual phrase for it. It is plausible — and this log's own
two failure modes predict — that Grounding DINO does fine on easy, sharp,
well-lit frames (which are already the least useful ones to add more of)
and does *poorly* on exactly the camouflaged and motion-blurred frames this
model already fails on, since those are hard for any appearance-based
detector, foundation-model or not. So this tool likely raises labelling
*throughput* on the easy majority of frames, not the hard tail that
actually limits the 82%/77% numbers. It should be scoped as a volume/cost
tool for expanding own-footage coverage, not a fix for either named failure
mode, and its candidate boxes should never be auto-accepted without human
review given that mismatch risk.

**Effort vs. payoff.** Low-medium effort: `pip install`, download one
checkpoint, run inference over a batch of raw clips, spot-check candidate
boxes against a few already-labelled frames to gauge quality before trusting
it for volume — a half-day sanity check, not a training run. Payoff:
plausibly good but unproven here — if a human still has to review and fix
most boxes on the hard frames, the net time saved could be small; if it is
reliably close on easy/medium frames (most of any new raw clip), it could
meaningfully cut the labelling cost of finally using more own-footage,
including whatever indoor/simulator capture eventually replaces the
quarantined `indoor_test` set. Recommend a small pilot (one raw clip, count
how many of its candidate boxes need no correction) before committing to it
as a pipeline stage.

---

## 2026-08-22 (third run) — TOTNet: visibility-weighted loss + train-time occlusion augmentation for temporal ball tracking (Xu et al., *Computer Vision and Image Understanding*, 2026)

**What it is.** TOTNet ("Occlusion-aware temporal tracking for robust ball
detection in sports videos," Xu, Baniya, Wells, Bouadjenek, Dazeley, Aryal —
*Computer Vision and Image Understanding*, Elsevier, 2026, arXiv:2508.09650)
is a small-fast-ball tracker built on the same lineage this log has already
covered (it re-implements TrackNetV2, monoTrack and TTNet as baselines, and
uses WASB-SBDT — logged 2026-08-21 — as its strongest comparison point). Its
architecture is the now-familiar family already logged five times over
(TrackNetV4, DTUM, SLT-Net, channel-stacked YOLO, WASB-SBDT): 3D
convolutions over a stack of 5 consecutive frames, regressing a heatmap for
ball position instead of a single-frame bounding box. That part is not new
here and is not the reason for this entry.

What *is* new, and not yet logged in this file, is the specific train-time
technique the paper uses to make the model robust to frames where the ball
is barely or not visible at all — the closest published analogue to this
project's camouflage failure mode (zero candidate detections, not just a
weak one) found so far: (1) **occlusion augmentation** — during training,
the ball is synthetically hidden in a controllable fraction of frames
(`--occluded_prob`, used at 0.1 in the released config) so the model sees
many more "the target is not visibly present" examples than any real
dataset naturally contains; (2) **visibility-weighted loss** — each training
sample is labelled with a visibility level (0 = out of frame, 1 = clearly
visible, 2 = partially occluded, 3 = fully occluded) and the loss
(weighted binary cross-entropy) up-weights the harder, lower-visibility
levels (`--weighting_list 1 2 2 3`) instead of treating every frame
equally. The paper's own ablation (frame count 3/5/7/9, broken out by
visibility level) is a real, reported result, not just a mechanism claim.

**URL.** https://github.com/AugustRushG/TOTNet (README and LICENSE fetched
directly via `raw.githubusercontent.com` — both return HTTP 200 with
substantive content: a full hyperparameter table, ablation figures embedded
as `github.com/user-attachments/assets/...` image links — confirming a live,
populated repo, not a stub. GitHub's own web UI and API
(`github.com`, `api.github.com`, `codeload.github.com`) all returned
403/400 through this sandbox's egress proxy, same restriction this log has
hit on every non-raw GitHub host to date, so the exact file tree beyond
`README.md`/`LICENSE` could not be enumerated — but the README's own
content (hyperparameter tables, ablation description, a `torchrun` command
with named CLI flags matching the mechanism described above) is corroborated
independently via two separate fetches of the raw file and is treated as
verified to the same bar as this log's other GitHub-only entries.)

**Licence — code VERIFIED PERMISSIVE, data VERIFIED NON-COMMERCIAL. Read
these as two separate things.** `raw.githubusercontent.com/AugustRushG/TOTNet/main/LICENSE`
returns the standard MIT License text verbatim (`Copyright (c) 2024
August`, "Permission is hereby granted, free of charge... to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software..."), fetched and read directly, not summarized — **commercial use
of the code is permitted.** Separately, the paper's own dataset (TTA —
Table Tennis Australia, 9,159 samples with 1,996 occlusion cases, collected
with Paralympics Australia) is gated behind a signed access agreement and
the README states plainly: **"Commercial use and redistribution are
strictly prohibited"** for that dataset. This project has no use for the TTA
data itself (wrong sport, non-commercial anyway) — the value here is the
MIT-licensed *code and technique*, not the data.

**Which failure mode.** Primarily camouflage, in a narrower and more
directly-actionable sense than this log's five prior motion-based
camouflage entries. Those all propose new architecture (attention modules,
direction-coded features, foundation-model segmentation) to give the
network a better appearance-independent cue. TOTNet's contribution needs no
new architecture: it is a loss-weighting and augmentation change, directly
portable to the existing YOLO11n training pipeline without touching the
model graph or the CoreML export path at all. Concretely: during YOLO
training, synthetically blend/occlude the labelled clubhead region in some
fraction of frames (the analogue of `occluded_prob`), and if per-sample loss
weighting is available in the training config, up-weight frames the data
engine already knows are hard (low-contrast background, partial occlusion,
heavy blur) rather than weighting every frame equally. That is a real gap
in what this log has logged for the data-synthesis area (bullet 5) — every
prior synthesis entry (Copy-Paste, CamDiff, BlenderProc, PSF box-expansion)
manufactures more *positive*, correctly-labelled hard examples; none of them
changes *how much the loss cares* about the hard examples already in the
1,308-frame set once camouflaged/blurred frames exist. Secondarily motion
blur, only via the same generic "more temporal context helps" logic already
covered by the five prior multi-frame entries — the frame-count ablation
(accuracy peaks at 5 frames for racket sports, does not improve at 7 or 9,
attributed to fast direction changes) is a genuinely new, concrete data
point if this project ever tunes how many stacked frames the
channel-stacked-YOLO entry (2026-08-13) uses, but it is not itself a
blur-streak-geometry fix.

**Why it helps this model specifically.** The camouflage failure mode as
described in the brief is exactly "zero candidate detections... even at
confidence 0.05" — not a low-confidence miss, a total absence of signal.
Standard training (uniform loss weighting, no deliberate exposure to
target-invisible frames beyond whatever negative frames the labeling spec
already produces) has no mechanism that specifically teaches the network to
behave sanely as visibility craters, because every frame contributes
equally to the loss regardless of how hard it is. TOTNet's two changes are
a direct, cheap answer to that: force more exposure to near-invisible
targets during training, and make the loss function itself say those frames
matter more, not just add more of them as ordinary examples. Nothing else
already logged targets *loss weighting* specifically — Copy-Paste/CamDiff
add examples, PSF-synthesis fixes box shape, InpaintNet (2026-08-17)
patches gaps after detection — this is the first entry that touches how the
existing 1,308 frames are weighted during training itself.

**Effort vs. payoff.** Low effort, moderate-plausible payoff, and cheaper
than every architecture-swap entry already logged. Effort: this does not
require adopting TOTNet's heatmap-regression architecture (that would be a
large, unproven-for-CoreML rewrite, same caveat as every other
multi-frame-architecture entry in this log) — only two changes inside the
existing Ultralytics YOLO training config: an occlusion/camouflage-blend
augmentation applied at some tunable probability, and, if Ultralytics'
loss supports per-sample weighting (needs to be checked against the
installed version — not verified in this run), weighting hard frames
higher. Both are a training-config change, not a new dependency or a new
CoreML export risk. Payoff: genuinely unverified for this specific model —
the paper's reported gains are for ball-heatmap tracking in racket sports,
not bounding-box clubhead detection in golf, and the mechanism has not been
tested here. But it is the only entry in this log's camouflage bucket that
is a pure training-config change with no architecture or export risk at
all, which makes it worth a cheap try (train one YOLO run with occlusion
augmentation + hard-frame up-weighting, eval against the existing 3-clip
test set) before committing effort to any of the architecture-graft options
already logged.

---

## 2026-08-22 (fourth run) — iPhoneBlur: a real, current-generation iPhone motion-blur benchmark with a working, MIT-licensed frame-averaging synthesis script — the first working implementation of this log's own first idea

**What it is.** iPhoneBlur is a 2026 motion-deblurring benchmark: 7,400
blur-sharp image pairs (5,714 train / 1,686 test) extracted from 51 videos
shot on iPhone 17 Pro at 177-240fps, stratified into Easy/Medium/Hard
difficulty bands by PSNR (Easy >=30dB, Medium 24-30dB, Hard <24dB; the paper
reportedly shows a "7-9dB Easy-to-Hard performance drop hidden by aggregate
metrics" across six benchmarked deblurring architectures, NAFNet best at
31.2dB overall). The part that matters most for this project is not the
benchmark numbers but the synthesis code: `dataset/generate_iphoneblur.py`,
fetched and read directly from `raw.githubusercontent.com` (not just
described in the README), implements exactly the frame-averaging idea this
log's very first entry (2026-08-12, Brooks & Barron) proposed but could not
find a working implementation for. Concretely, the script: reads `.MOV`
files, converts frames to linear color space (`gamma_decode()`/
`gamma_encode()`, gamma=2.2), synthesizes blur as a weighted average over an
adaptively-sized window of neighboring frames (`min_window`/`max_window`,
default 3-21, via `synthesize_blur()`), quality-filters candidates against
PSNR/SSIM/LPIPS thresholds, deduplicates with perceptual hashing
(`imagehash.phash()`), and writes out blur/sharp JPEG pairs plus a
19-column metadata CSV (PSNR, SSIM, optical-flow motion magnitude, ISP
energy, difficulty label, etc.). Repo root (fetched directly, confirmed
live) contains `LICENSE`, `README.md`, `requirements.txt`, and `dataset/`,
`evaluation/`, `finetuning/`, `models/`, `Inferred_Notebooks/` directories —
this is a real, structured, working codebase, not a stub.

**URL.** https://github.com/C-loud-Nine/iPhoneBlur (dataset also listed on
Kaggle per the README, ~9.08GB JPG+CSV; paper apparently at
arXiv:2605.05990, "iPhoneBlur: A Difficulty-Stratified Benchmark for
Consumer Device Motion Deblurring" — both `kaggle.com` and `arxiv.org` are
blocked by this sandbox's egress proxy, same restriction as every prior run,
so the arXiv identifier and Kaggle listing are unverified beyond what the
GitHub README states; only `github.com`/`raw.githubusercontent.com` content
described below is independently fetched and confirmed.)

**Licence — code VERIFIED MIT, dataset licence claim NOT independently
verified. Read these as two separate things.**
`raw.githubusercontent.com/C-loud-Nine/iPhoneBlur/main/LICENSE` returns the
standard MIT License text (2026, attributed to Shafi Abdullah): "Permission
is hereby granted, free of charge... to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software," conditioned
only on retaining the notice. **Commercial use of the synthesis code and
any other repo code is permitted.** The README separately states the
*dataset* itself (the 7,400 pre-generated pairs hosted on Kaggle) is
"CC BY 4.0" and that both licences "permit academic and commercial use with
appropriate attribution" — but since `kaggle.com` is unreachable from this
sandbox, that specific claim was read only from the GitHub README, not
confirmed against Kaggle's own licence field the way this log's other
dataset entries (e.g. RealBlur, 2026-08-16) were verified. Treat the
dataset's CC BY 4.0 status as one notch below full verification; the code's
MIT status is fully verified. This project has more use for the code than
the pre-built dataset anyway (see below), which limits how much that
caveat matters.

**Which failure mode.** Motion blur, specifically and only. Does not touch
camouflage.

**Why it helps this model specifically.** The 2026-08-12 entry identified
the actual root gap correctly (labelled boxes are near-square because
genuinely blurred, correctly-boxed training examples are scarce — median
elongation 1.60) and proposed frame-averaging of high-fps footage as the
fix, but flagged that no confirmed pretrained model and no working code
were found — only the *idea*, sourced from a paper this sandbox couldn't
even read the PDF of. This entry closes that gap: `generate_iphoneblur.py`
is a working, inspectable, MIT-licensed implementation of the same idea,
already tuned specifically for iPhone footage (gamma-correct in linear
space before averaging, adaptive window sizing, PSNR/SSIM/LPIPS-based
quality gating to reject bad synthetic pairs, perceptual-hash dedup) — all
problems a from-scratch implementation would otherwise have to solve
itself. It is also more directly applicable than the GoPro/REDS entry
(2026-08-20, GoPro action-camera footage, licence only "probable" CC BY
4.0) or SloMoDeblur (2026-08-19, licence unconfirmed): this is phone-native,
current-generation capture (iPhone 17 Pro), matching this app's actual
capture device family, and the code's licence is fully confirmed rather
than probable. Two concrete uses: (1) adapt this script directly against
the project's own high-fps swing footage (if the app or a dev captures
120/240fps bursts) to generate blur/sharp pairs with the box then derived
as the tracked-position hull, exactly as 2026-08-12 proposed, but now with
working reference code instead of a from-scratch build; (2) even without
new capture, the PSNR/SSIM/optical-flow-based *quality metrics* in the
script are reusable standalone as a labeling-QA signal (does a candidate
"blurred" training frame actually have the motion-magnitude/PSNR profile of
real blur, versus a near-square mislabel) — a narrower, code-reusable
version of the same QA idea the DeFMO/FMO entry (2026-08-20) proposed only
as a general technique.

**Effort vs. payoff.** Low-to-medium effort, plausible payoff, still capped
by the same capture-availability caveat as 2026-08-12. Effort: the
synthesis script is small and dependency-light (OpenCV/NumPy/perceptual-
hashing-class tooling, no exotic ML framework), and directly portable — no
GPL/unlicensed-code risk like several architecture-idea entries in this log
carry. It does not, by itself, produce clubhead-specific training data: it
still needs (a) the project's own high-fps source footage, which this run
could not confirm exists yet, and (b) the box-derivation step (tracked-
position hull across the averaging window) that 2026-08-12 already scoped
using the existing CSRT tracker-propagation tooling. Payoff: this is the
strongest evidence yet that the frame-averaging approach is a solved,
reusable engineering problem rather than a research risk — the remaining
work is capture (does the app or a dev have slow-motion swing footage) and
integration (wiring the CSRT-derived box into this script's output format),
not algorithm design. Recommended as the concrete next step for the
2026-08-12 idea specifically: try adapting this script against any existing
high-fps footage before investing further in the PSF-synthesis (2026-08-14)
alternative, since real temporal-integration blur from actual footage is a
strictly better ground truth than a synthetic PSF kernel once real high-fps
source video exists.

---

## 2026-08-23 — MoSA-Det (Scientific Reports, April 2026): a motion-state-conditioned mechanism aimed directly at the excess-displacement failure this log already flagged as unresolved for TrackNetV4 and DTUM

**What it is.** "MoSA-Det: motion state adaptive object detection for sports
videos" (Yang, Sun, Ren — *Scientific Reports* vol. 16, article 15969, April
2026) is a sports-video detector built around the observation that fast
sports objects break single-frame *and* naive multi-frame detectors in two
distinct ways: (1) motion blur degrades single-frame appearance features
("indistinct boundaries, texture loss, significantly reduced feature
response intensity"), and (2) large inter-frame displacement breaks
temporal-fusion methods that assume small, alignable motion between frames
("existing temporal methods assume small inter-frame displacements... yet
fast-moving objects often exhibit excessive inter-frame displacement that
causes alignment failure"). It has two components: **MAAF**, a lightweight
motion-state estimator (inter-frame differencing + local correlation) that
drives a state-conditioned dynamic convolution with deformable spatial
sampling in the regions it estimates are moving fast; and **SGTA**, which
uses that same motion-state signal to *gate* temporal feature aggregation —
static regions fuse fully across frames for stability, fast-moving regions
aggregate less (or are down-weighted) specifically to avoid pulling in
misaligned-frame noise. Reported results: mAP@0.5 +1.7pp and mAP@0.75
+2.6pp over the strongest baseline on SoccerNet-Tracking, and +1.6pp /
+1.8pp on SportsMOT (soccer and multi-sport broadcast tracking, not golf).

**URL.** https://www.nature.com/articles/s41598-026-43231-2 (DOI
10.1038/s41598-026-43231-2). `nature.com`, `doi.org`, and
`api.semanticscholar.org` are all blocked by this sandbox's egress proxy —
consistent with every prior run's experience with academic-publisher
domains — so the article itself could not be fetched directly. Everything
above (mechanism description, component names, quoted phrases, and the
SoccerNet-Tracking/SportsMOT numbers) is sourced from web-search-indexed
excerpts of the article that were internally consistent across two
independently-worded search queries, not a direct read of the PDF/HTML.
Treat the mechanism description as reasonably corroborated (the same
distinctive phrasing — "excessive inter-frame displacement," MAAF, SGTA —
recurred verbatim across independent search hits) but one notch below full
verification, per this log's established convention for blocked domains. No
GitHub repository was found for this paper under either the paper's name or
any of the three authors' names — treat it as **paper-only, no released
code**, not merely "code not yet located."

**Licence — NOT permissive; do not treat this as reusable content.**
*Scientific Reports* lets authors choose between CC BY 4.0 (commercial use
permitted) and CC BY-NC-ND 4.0 (non-commercial, no derivatives) per
article — it is not a single default, so this had to be checked per-article,
not assumed. Two independent search queries both returned the same
boilerplate licence sentence for this specific article: "This article is
licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives
4.0 International License, which permits any non-commercial use, sharing,
distribution and reproduction in any medium or format... Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International License." **This
article is CC BY-NC-ND — commercial use of the article's content is NOT
permitted**, and since the primary source page could not be fetched
directly (egress-blocked), this specific licence determination should be
treated as corroborated-but-not-primary-verified, same caveat as the
mechanism description above. There is no separate code licence to evaluate
since no code was released. Per this log's established convention (applied
to DTUM, PSF-synthesis, GolfPose pre-2026-08-17, etc.): the underlying
*algorithmic idea* — motion-state-conditioned dynamic convolution gating
temporal aggregation strength — is a describable technique, not
copyrightable expression, and is reimplementable from the mechanism
description without reproducing any of the article's actual text or figures.
But note this is a materially weaker position than usual for this log: most
prior "idea-only, licence-blocked" entries at least had real code to point
to for implementation detail (DTUM, PSF-synthesis) even if the code itself
carried licence risk. Here there is no code at all — only a two-paragraph
mechanism description assembled from search snippets — so "reimplementable
from the description" is a much thinner starting point than usual.

**Which failure mode.** Both, explicitly, and — unusually for this log —
the paper's own framing of the *second* problem (temporal aggregation
failure from excessive inter-frame displacement) is a close structural
match for something already flagged as an open, unaddressed gap in two
earlier entries here, not just a generic "motion helps" claim.

**Why it helps this model specifically.** The camouflage-mechanism entries
already in this log (TrackNetV4, 2026-08-13; DTUM, 2026-08-14) both landed
on the same fix — inject a motion-derived signal since a single frame's
appearance is genuinely insufficient — but both entries independently
flagged the identical unresolved caveat: frame-differencing / motion-
direction encoding assumes small, alignable inter-frame motion, and a
golf clubhead at swing speed (or a handheld phone moving to track it) will
break that assumption exactly the way DTUM's own infrared-surveillance
literature doesn't have to worry about (tripod/gimbal-mounted cameras).
Neither entry offered a mechanism for what to do when that assumption
breaks — they just flagged it as a research-spike question. MoSA-Det's SGTA
component is, as far as this log has found, the first mechanism that
targets *that specific sub-problem* directly: instead of assuming
alignability everywhere, it uses the motion-state estimate itself to decide
*how much* to trust temporal fusion per-region, explicitly to avoid
"misaligned-frame noise" in fast-motion regions. For this model, that maps
onto the two failure modes at once in a way no earlier entry does as
cleanly: in the camouflage case (dark clubhead near swing speed against
dark background), MAAF's motion-state estimate is exactly the kind of
motion-vs-appearance-conflict signal DTUM and TrackNetV4 already argued for;
in the motion-blur case, MoSA-Det treats blur-induced feature degradation
and alignment failure as *the same underlying variable* (motion state)
rather than two separate problems needing two separate fixes — which is a
more parsimonious framing than this log's blur entries (frame-averaging,
PSF-synthesis, iPhoneBlur — all label/data-side fixes) and camouflage
entries (TrackNetV4, DTUM — appearance/architecture-side fixes) have
offered separately so far.

**Effort vs. payoff.** High effort, genuinely uncertain payoff, and the
weakest evidentiary footing of any architecture-idea entry logged so far.
Effort: same order of magnitude as TrackNetV4/DTUM — this is architecture
surgery on YOLO11n (a motion-state estimation branch, state-conditioned
dynamic/deformable convolution, and a gated temporal-fusion module), not a
drop-in change, and — as with every other architecture idea in this log —
CoreML exportability and on-device inference budget for the added modules
is completely unvalidated by the source material. Payoff: conceptually this
is the best-targeted idea logged yet for the *combination* of both failure
modes and specifically for the camera-motion caveat this log has flagged
twice before as unresolved, which is real value — but the practical
starting point is much thinner than DTUM or TrackNetV4 (no code, no
paper PDF read, mechanism reconstructed from search snippets, licence
corroborated but not primary-source-verified), so treat this as a reason to
prioritize the *camera-motion research spike* already recommended in the
DTUM entry (does a motion/frame-difference signal survive handheld swing
footage, and if not, does gating by estimated motion state fix it) over
committing to reimplementing MoSA-Det's specific architecture from a
secondhand description. This entry's main contribution is sharpening what
that spike should test, not a ready-to-build mechanism.

---

## 2026-08-23 (second run) — Zero-DCE: zero-reference low-light enhancement, original code non-commercial, but a verified Apache-2.0 independent reimplementation exists

**What it is.** "Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement" (Guo et al., CVPR 2020, usually called Zero-DCE, with a later
"Zero-DCE++" extension) is a tiny CNN (**DCE-Net**: 7 conv layers, 32
3x3 kernels each, ReLU, final Tanh layer emitting 24 per-pixel curve
parameters for 8 iterative enhancement steps) that learns to brighten a
low-light image by estimating a pixelwise tonal curve. The key property for
this project: it is **zero-reference** — trained with no paired
dark/bright ground truth at all, only four unsupervised losses (spatial
consistency, exposure control against a target well-exposedness level,
color constancy, and illumination smoothness). That means it can be trained
directly on raw, unlabeled low-light footage — exactly the kind of
indoor/simulator-bay/evening footage this project has never had a use for
until now, with no annotation step required.

**URL.** Original repo (non-commercial):
https://github.com/Li-Chongyi/Zero-DCE and
https://github.com/Li-Chongyi/Zero-DCE_extension (Zero-DCE++). Verified
independent, permissively-licensed reimplementation of the same
architecture, losses, and zero-reference training procedure: Keras's
official examples repository,
https://raw.githubusercontent.com/keras-team/keras-io/master/examples/vision/zero_dce.py
(page: `keras.io/examples/vision/zero_dce/`, blocked by this sandbox's
egress proxy, but the raw source file on `raw.githubusercontent.com`
fetched cleanly and was read directly, not summarized secondhand).

**Licence (verbatim, both fetched directly).** Original `Li-Chongyi/Zero-DCE`
README: "The code is made available for academic research purpose only...
Under Attribution-NonCommercial 4.0 International License." Same clause,
same wording, in `Li-Chongyi/Zero-DCE_extension`. **Commercial use of the
original authors' code: NOT permitted.** The `keras-team/keras-io` repo's
`LICENSE` file (fetched directly): **Apache License, Version 2.0** — "perpetual,
worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright
license to reproduce, prepare Derivative Works of... and distribute the
Work." **Commercial use: permitted**, standard Apache-2.0 conditions
(retain notices, mark changes). The Keras example is a clean-room
reimplementation of the published algorithm (same DCE-Net layer counts,
same four loss terms, same unpaired/zero-reference training loop), not a
fork of the NC-licensed original — using it as the starting point avoids
the original repo's licence entirely. One caveat: the Keras example itself
trains on the LoL Dataset (a third-party low-light benchmark) purely for
demonstration; that dataset's own licence was not checked here and is
irrelevant regardless, since the zero-reference property means this
project would retrain on its own footage, not reuse the demo weights.

**Which failure mode.** Camouflage — specifically the not-yet-tested
low-light subset of it. This is a preprocessing/enhancement technique, not
a detector architecture change or a labelled dataset.

**Why it helps this model specifically.** The brief's own caveat is that
the camouflage finding (82%/77%/85.9%, dark clubhead vs. dark
clothing/foliage) comes from bright, professionally-shot, fast-shutter
outdoor footage, and that indoor/low-light performance has never been
measured because the quarantined `indoor_test` set is unusable. Zero-DCE
targets exactly the gap between those two situations: it doesn't change
what the clubhead looks like (it isn't a detector), it changes how much
signal is available for the detector to work with when a frame is
genuinely underexposed — brightening a dark simulator-bay or evening frame
before it reaches YOLO11n restores contrast between a dark clubhead and a
dark background that a fixed exposure otherwise crushes into near-identical
pixel values, which is a different mechanism from every appearance- or
motion-based camouflage fix already in this log (TrackNetV4, DTUM, SLT-Net,
SINet-V2, Motion-Informed Enhancement, SAM-PM, MoSA-Det, Copy-Paste,
CamDiff — none of them touch exposure/illumination). Because training needs
no labels, it is also directly usable as a **data-engine tool**: run it
over any raw low-light phone footage the app already has or can cheaply
collect (no waiting on a licensed dataset) to produce brightened frames
that a human labeller can actually see well enough to box accurately, which
directly serves the labeling spec's existing rule that occluded/invisible
heads become negative frames — under-exposure currently forces some frames
into that bucket for lack-of-visibility reasons that have nothing to do
with true occlusion.

**Effort vs. payoff.** Low-to-medium effort, uncertain but cheap-to-test
payoff. Effort: DCE-Net is tiny (7 conv layers) — training it from scratch
on a small batch of the project's own dark footage is realistic in a day,
and a network this small should export to CoreML without the
budget/exportability doubts that hang over every architecture-surgery entry
in this log (MoSA-Det, DTUM, TrackNetV4). Two real open questions before
committing: (a) whether brightening actually helps YOLO11n's *learned*
features versus just making frames look better to a human — enhancement
networks are known to sometimes hurt downstream detectors that already
learned dark-domain features, so this needs an A/B eval on real held-out
dark frames, not an assumption; (b) whether to run it on-device at
inference time (extra latency + CoreML pipeline complexity for a real-time
app) versus offline-only as a labelling aid — the offline-only, labelling-aid
use is the safer, near-zero-risk first step and should be tried before any
on-device commitment. Given the total absence of any indoor/low-light
measurement to date, the honest framing is: this is a promising, cheap
experiment to run, not a validated fix.

---

## 2026-08-23 (third run) — FastViT: Apple's own hybrid conv-attention backbone, first-party CoreML-exported, but a generic capacity lever rather than a targeted fix for either failure mode

**Area covered.** Closest to bullet 3 (small/low-contrast object detection
techniques), but honestly it doesn't fit any of the five bullets cleanly —
it's not temporal, not camouflage-specific, not a labelled dataset, and not
a data-synthesis technique. It's logged anyway because it's a structurally
different lever from everything else in this file: every architecture entry
so far (TrackNetV4, DTUM, SLT-Net, SINet-V2, Motion-Informed Enhancement,
SAM-PM, MoSA-Det) adds a module or a motion signal to an existing
single-frame CNN detector. This is the first entry proposing to replace the
*backbone itself* with a hybrid conv+attention design, and the first
architecture entry in this log where the CoreML export path is demonstrated
by the model's own authors rather than assumed.

**What it is.** "FastViT: A Fast Hybrid Vision Transformer using Structural
Reparameterization" (Anasosalu Vasu, Gabriel, Zhu, Tuzel, Ranjan — Apple,
ICCV 2023). It's a mobile-efficient backbone family that uses
train-time-only multi-branch blocks (`MobileOneBlock`,
`ReparamLargeKernelConv`, both confirmed present in the repo's own source)
which collapse into single-branch convolutions at inference time
(structural reparameterization, the same trick MobileOne and RepVGG use),
combined with actual self-attention stages in the later network layers
(`fastvit_sa*` variants) rather than pure convolution throughout, unlike
YOLO11n's CSPDarknet-style backbone. Verified directly from source
(`raw.githubusercontent.com/apple/ml-fastvit/main/README.md` and
`models/fastvit.py`, both fetched directly, HTTP 200): the smallest variant,
FastViT-T8, is 76.2% ImageNet-1K top-1 at 0.8ms latency on an iPhone 12 Pro;
larger variants trade latency for accuracy up to 83.9%. The paper's own
COCO object-detection and ADE20K segmentation claims ("comparable
performance with 4.3x lower backbone latency" on Mask-RCNN detection,
"1.5x lower latency" on segmentation) come from a search-engine-indexed
excerpt of the abstract, not the paper itself — `arxiv.org` is blocked by
this sandbox's standing egress restriction, so per this log's convention
those two numbers are one notch below full verification; everything else
above is read directly from the repo.

**URL.** Code: https://github.com/apple/ml-fastvit (`README.md`,
`LICENSE`, `ACKNOWLEDGEMENTS`, and `models/fastvit.py` all fetched directly
via `raw.githubusercontent.com`, HTTP 200, confirmed live and real, not a
placeholder repo). Paper: ICCV 2023, arXiv:2303.14189 (not fetched
directly, see above). **Important verification gap:** the actual model
weights and the pre-exported CoreML packages are hosted on
`docs-assets.developer.apple.com`, which is blocked by this sandbox's
egress proxy (`EGRESS_BLOCKED`, confirmed via direct `curl` and via the
fetch tool, not a timeout) — the same class of gap this log hit with
RealBlur's dataset host. So: the *code and its licence* are fully verified
live; the *actual checkpoint and .mlpackage files* were not downloaded or
opened in this run, only linked from the directly-fetched README. A
follow-up from an unrestricted network should confirm those links still
resolve before this is treated as actionable.

**Licence (verbatim, from the repo's own `LICENSE` file, fetched
directly).** "Apple grants you a personal, non-exclusive license, under
Apple's copyrights in this original Apple software (the 'Apple Software'),
to use, reproduce, modify and redistribute the Apple Software, with or
without modifications, in source and/or binary forms; provided that if you
redistribute the Apple Software in its entirety and without modifications,
you must retain this notice..." No non-commercial clause anywhere in the
file. **Commercial use: permitted**, condition is just notice-retention on
verbatim redistribution (modified/derivative use, which is what training a
new backbone on this project's own data would be, carries no stated
condition at all). The `ACKNOWLEDGEMENTS` file (fetched directly) lists
three bundled subcomponents — RepVGG (MIT), RepLKNet (MIT), and the
MetaFormer/PoolFormer code (Apache-2.0) — all independently commercial-use
permissive, no NC or copyleft terms found anywhere in the dependency chain.
This is one of the cleanest licence findings in this log to date: a
first-party corporate author, no academic-only clause, no unverifiable
third-party licence.

**Which failure mode.** Weakly, both — which is itself the honest
weakness of this entry. The argument for camouflage: self-attention layers
in the later stages give the network a broader effective receptive field
than local convolution, which in principle helps weigh a low-contrast
clubhead against its surrounding pixels using more of the frame's context
rather than a small local window — the same broad-context intuition behind
DTUM's direction-coded convolution and SLT-Net's correlation volume, just
applied through attention instead of motion. The (weaker) argument for
blur: an elongated motion-blur streak spans more spatial extent than a
compact convolution kernel sees at once, and attention layers integrate
across that whole extent by construction. Neither claim is demonstrated by
Apple for this use case — FastViT was designed and benchmarked purely for
general ImageNet efficiency, not for low-contrast or blurred-object
detection, and no camouflage- or blur-specific benchmark for it exists
anywhere that was found in this run.

**Why it helps this model specifically.** Two things distinguish it from
every prior architecture entry, and one thing meaningfully undercuts it.
Distinguishing: (1) it is the only architecture-graft candidate in this
log where the authors themselves ship pre-exported CoreML packages
(`fastvit_t8_reparam.pth.mlpackage.zip` etc., linked directly from the
verified README) — every other architecture entry (MoSA-Det, DTUM,
TrackNetV4, SAM-PM) carries an open "will this even export to CoreML"
question that this one answers in the affirmative, for classification at
least, even though the actual files weren't downloadable from this
sandbox; (2) the reparameterization trick means the deployed model is a
plain single-branch CNN at inference time — no runtime attention-graph
complexity survives to the edge, which matters for an app already shipping
a CoreML pipeline. Undercutting it: the public checkpoints are ImageNet-1K
*classification* pretrained only (confirmed from the README's own model
zoo table) — YOLO11n's current backbone benefits from being pretrained
inside a COCO object-detection pipeline, a transfer-learning head start
this project would lose by swapping in a classification-only backbone
unless it separately does the mmdetection-based downstream detection
pretraining the repo's own source code is wired for (`from mmdet.models.builder
import BACKBONES as det_BACKBONES`, confirmed directly in `models/fastvit.py`
— the official downstream-task integration target is mmdetection/
mmsegmentation, not Ultralytics). No existing FastViT-into-Ultralytics-YOLO
integration was found in this search; wiring it into this project's actual
YOLO11n/Ultralytics training config would be original engineering, not a
one-line checkpoint swap like the YOLO26 entry (2026-08-17).

**Effort vs. payoff.** Medium-to-high effort, unvalidated and likely
overstated payoff. Effort: porting a non-Ultralytics-native backbone into
the Ultralytics YOLO11 config, re-wiring the detection head, and then
re-pretraining or fine-tuning without the COCO-detection transfer head
start YOLO11n currently has is a substantially bigger lift than the
STAL/YOLO26 checkpoint swap, and bigger than most single-module additions
already logged (Motion-Informed Enhancement, Zero-DCE). Payoff: entirely
speculative for this project's two failure modes — the camouflage and blur
arguments above are this entry's own inference from FastViT's general
design, not anything Apple measured or claimed. Given how many
more-targeted, already-verified mechanisms this log has for camouflage
(DTUM, SLT-Net, Motion-Informed Enhancement, SAM-PM, MoSA-Det, Zero-DCE)
and for blur (frame-averaging, PSF synthesis, channel-stacked YOLO,
RT-Focuser, BlenderProc), this should rank below all of them: a
lower-priority "if the targeted fixes stall" experiment, not a next step,
and the strongest thing it actually has going for it — the demonstrated
first-party CoreML export path — applies to classification, not yet to
detection.

---

## 2026-08-23 (fourth run) — Adding a P2/4 detection head to YOLO11n: a mechanical fix that targets the specific "zero candidate detections" symptom

**Area covered.** Bullet 3 (small/low-contrast object detection). Also the
first entry this log has logged that changes the *detection head resolution*
rather than adding a module, a motion signal, or a new backbone — every
architecture entry so far (TrackNetV4, DTUM, SLT-Net, SINet-V2,
Motion-Informed Enhancement, SAM-PM, MoSA-Det, FastViT) operates on top of
YOLO11n's existing P3/P4/P5 (stride 8/16/32) feature pyramid. This one
argues the pyramid itself is the wrong shape for this object.

**What it is.** Ultralytics' own repo ships `yolov8-p2.yaml`
(`ultralytics/cfg/models/v8/yolov8-p2.yaml`), a variant of the standard
detection config that adds a fourth, higher-resolution detection head at
the P2/4 feature map (stride 4, i.e. 1/4 input resolution) alongside the
usual P3/8, P4/16, P5/32 heads. Verified by fetching the raw file directly
(`raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8-p2.yaml`,
HTTP 200): structurally it inserts three extra layers (upsample P5→concat
P4, P4→concat P3, P3→concat P2) and changes the `Detect` module's input
list from `[15, 21, 24]` (3 scales) to `[18, 21, 24, 27]` (4 scales,
including the new P2 layer). Ultralytics does not ship an equivalent
`yolo11-p2.yaml` in the main repo (checked: not present in
`cfg/models/11/`), but the same edit applied to `yolo11.yaml` is a
well-documented community pattern — GitHub issue
`ultralytics/ultralytics#20267` and `ultralytics/ultralytics#20359` both
show users doing exactly this (`YOLO('yolo11n-p2.yaml')`), and a working
example config (`yolo11-p2.yaml`) is published on Hugging Face at
`davsolai/yolo11x-p2-coco`. This is a mechanical, same-repo, same-license
config edit, not a new codebase or a new dependency: since YOLO11n training
already goes through Ultralytics' `ultralytics` package (per this project's
own v1/v2 design docs), the P2 head carries no new licensing exposure
beyond what training with `ultralytics` already entails (AGPL-3.0 on the
training/repo code; Ultralytics' own position is that this does not extend
to the exported weights, which is the same position this project is
presumably already relying on for the existing YOLO11n baseline — not
independently re-verified here, flagged as a pre-existing assumption, not a
new one this entry introduces).

**Why it helps THIS model specifically.** The failure analysis's camouflage
symptom is not "low confidence, wrong box" — it's **zero candidate
detections even at confidence 0.05**. That symptom pattern (nothing fires
at any threshold) is the classic signature of an object whose extent, after
downsampling, no longer occupies a full cell at the coarsest strides the
network detects from. A clubhead is one of the physically smallest objects
in the frame class-for-class (a driver head is a few percent of frame
width in typical phone-distance swing footage), and YOLO11n's finest
detection stride today is P3 (1/8 resolution) — at 1080p that's already
downsampling the clubhead's extent by 8x before any detection head sees it,
and further for dark clubhead-on-dark-background cases where the true
signal is a handful of edge pixels, not a filled blob. The P2 head halves
that stride to 1/4, roughly quadrupling the spatial area each feature-map
cell represents at the finest scale, which is the standard, well-established
fix for exactly this "very small object, feature map has already lost it"
failure mode in the small-object-detection literature (this is *why* P2
variants exist at all — see `ultralytics/ultralytics` discussion #8227
cited in earlier searches). It is a weaker match for the blur failure mode
(P2 helps absolute pixel size, not elongation/aspect ratio directly), so
this should be read as a camouflage-primary, blur-secondary fix, not a
solution to both co-equally.

**Cost.** Ultralytics' own published GFLOPs figures show the P2 variant
roughly doubles compute for the nano scale (YOLOv8n: 8.9 GFLOPs → YOLOv8n-p2:
17.4 GFLOPs); the same rough 2x should be expected for YOLO11n-p2. Given
that plain YOLO11n→CoreML has been reported comfortably real-time on modern
iPhones (tens to 80+ FPS in third-party benchmarks, not this project's own
measurement), a 2x compute increase is very likely to still clear the
on-device real-time bar, but this project's own baseline latency numbers
were not available in this checkout to confirm the actual headroom — that
check belongs in the experiment, not this entry.

**Effort vs. payoff.** Low-to-moderate effort: no new dataset, no new
dependency, no new labelling convention — it's a YAML config edit plus a
retrain on the exact training set already in use, and a CoreML re-export
using the same pipeline already in place for the current model. This is
one of the cheapest experiments logged so far (comparable to the YOLO26
checkpoint swap from 2026-08-17), while targeting the single most concrete,
already-measured symptom in the failure analysis (zero detections at any
threshold) more directly than any camouflage entry logged before it — the
prior camouflage entries (SLT-Net, SINet-V2, SAM-PM, DTUM, Motion-Informed
Enhancement, CamDiff, Copy-Paste, Zero-DCE) all add appearance- or
motion-based signal on top of the existing pyramid; none of them change the
pyramid's finest resolution, which is the more direct lever if the root
cause is genuinely "the object is too small for the network to see," as
the confidence-0.05 zero-detection symptom suggests. Recommended as a
near-term experiment to run before, or alongside, the heavier camouflage
architecture changes already logged.

---

## 2026-08-24 — DFRCP: a YOLOv11-native blur-robustness module (Dynamic Fuzzy Feature Fusion), found but not independently verifiable in this sandbox (existence-only result)

**Area covered.** Motion blur specifically — blur-robust detection
architectures, this run's rotation target after the last several runs
covered architecture (P2/4 head), a generic backbone (FastViT), an
augmentation route (Zero-DCE), and a motion-blur architecture (MoSA-Det) on
2026-08-23. Checked this log's existing motion-blur architecture entries
(TrackNetV4, DTUM, MoSA-Det, channel-stacked multi-frame YOLO,
PSF-based blur synthesis) first to avoid re-treading them — DFRCP is a
distinct mechanism from all of them (see below).

**What I found.** A paper titled "Motion Blur Robust Wheat Pest Damage
Detection with Dynamic Fuzzy Feature Fusion" (arXiv:2601.03046, January
2026; a preprint version also indexed at Research Square,
`rs-8760445/v1`, under the title "Learning from Motion: A Dynamic Feature
Fusion Framework for Robust Agricultural Vision in Blurred Environments").
It proposes **DFRCP (Dynamic Fuzzy Robust Convolutional Pyramid)**,
described consistently across multiple independent search snippets as an
explicit **plug-in upgrade to the YOLOv11 feature pyramid** — the same
model family this project already trains and exports to CoreML. Mechanism,
per those snippets: it combines large- and medium-scale pyramid features
while preserving native representations, then synthesizes "fuzzy features"
by rotating and nonlinearly interpolating multiscale features and merging
them through a learned "transparency convolution" that trades off original
vs. fuzzy cues per-content ("Dynamic Robust Switch" units). This is a
structurally different mechanism from every blur entry already in this
log: MoSA-Det conditions dynamic convolution on an inter-frame-differencing
motion-state estimate; DTUM is a direction-coded temporal module; PSF-based
synthesis and channel-stacked multi-frame YOLO both operate on the input
side, not the feature pyramid. DFRCP is a single-frame, pyramid-internal
technique — no temporal/multi-frame input required, which would make it
cheaper to integrate than the multi-frame entries if it works as described.
Reported result (from the same snippets, not independently confirmed): ~10.4
percentage points of accuracy improvement over a YOLOv11 baseline on a
blurred test split, using a CUDA custom rotation/interpolation kernel
claimed to give "more than 400x speedup" over a naive implementation for
training-time practicality (inference-time cost not stated in any snippet
found).

**Why this cannot be logged as a usable finding yet.** Both `arxiv.org` and
`www.researchsquare.com` hit this sandbox's standing egress block (the same
recurring failure mode noted in nearly every prior entry for `arxiv.org`,
`nature.com`, `scirp.org`, `universe.roboflow.com`). I could not read the
paper itself — everything above is search-engine synthesis of snippets that
agree with each other closely enough (matching module names, matching
numeric claims) to support the paper's *existence* and *general shape*, but
not to independently confirm its method details or numbers firsthand. A
dedicated search for a GitHub repository, code release, or weights (`DFRCP
Dynamic Fuzzy Robust Convolutional Pyramid github`, and variants) returned
nothing — no code was found. That makes this idea-only, the same weaker
category as the TinyDark-YOLO entry (2026-08-18, fourth run): a from-scratch
reimplementation based on a secondhand summary, not something with a real
repo to fetch, read, and verify like BlurBall, DTUM, SLT-Net, or TrackNetV4
had.

**Licence — not applicable / not assessed.** No dataset transfer is at
stake here (the paper trains on a private ~3,500-image wheat pest dataset,
irrelevant to this project regardless of licence) — the only thing of
interest is the architecture technique itself, and with no code found there
is no licence to check yet. If code does surface later, it would need its
own licence check before use, same as every other architecture entry in
this log.

**Which failure mode.** Motion blur, and only motion blur — the paper's own
framing (camera-shake/jitter degrading edge-side detection) maps directly
onto this project's blur-elongation gap (median labelled box elongation
1.60), not camouflage.

**Effort vs. payoff.** Not assessable yet, and that is the finding, same
caveat as TinyDark-YOLO. If real and if the reported ~10.4pt number holds
outside the wheat-pest domain, this would be one of the cheaper blur fixes
in this log to try — single-frame, no new temporal input pipeline, and
explicitly YOLOv11-compatible — but none of that can be acted on without
either the paper's method section (to reimplement cleanly) or a real code
release. Logged as a lead only: a future run with working `arxiv.org` or
`researchsquare.com` access, or a later search that turns up a code mirror,
should pick this up at "read the method section and check for code" rather
than rediscovering it from zero.

---

## 2026-08-24 (second run) — ReynoldsFlow: a training-free, physics-inspired optical-flow method evaluated by its own authors on GolfDB, but with no code or license actually released (existence-only result)

**Area covered.** Bullet 3 (temporal methods for small/low-contrast/camouflaged
object detection), chosen because the last several runs (2026-08-23 through
today's first entry) had clustered on motion-blur architecture and generic
backbones. Checked this log's existing camouflage/temporal entries first
(TrackNetV4, DTUM, SLT-Net, SINet-V2, Motion-Informed Enhancement, CamDiff,
SAM-PM, TOTNet) — ReynoldsFlow is mechanistically distinct from all of them:
every one of those is a *learned* network (attention fusion, direction-coded
convolution, a trained correlation volume, a trained segmentation head, a
diffusion model, a trained visibility-weighted loss). ReynoldsFlow is not
learned at all.

**What I found.** A paper titled "ReynoldsFlow: Physics-Inspired Spatiotemporal
Flow Representation for Video Understanding" (arXiv:2503.04500; an earlier
version of the same arXiv ID was titled "ReynoldsFlow: Exquisite Flow
Estimation via Reynolds Transport Theorem"), by Chen, Lin, Huang & Wu
(University of Melbourne / National Yang Ming Chiao Tung University). Per
search-engine synthesis (the paper itself is unreadable — see below), it
derives a **training-free** flow representation from the Reynolds transport
theorem rather than learning one, positioned as an alternative to classical
Lucas-Kanade-style optical flow. The companion repo,
`github.com/wish44165/ReynoldsFlow`, is real and actively maintained (commits
dated March 2025, June 2025, February 2026, and August 6 2026 — five commits
total, the most recent only 18 days before this run). Directly fetched and
read its `README.md`: the paper's own evaluation datasets are listed there,
and **GolfDB is one of them**, alongside HMDB, UCF101, Anti-UAV, and a
drone-trajectory-reconstruction dataset — i.e. the authors themselves
benchmarked this exact method on golf-swing video and on two fast-moving
small/aerial-object tracking benchmarks (Anti-UAV, drone tracking), which is
close to as direct a match to "does motion separate a fast small object from
a cluttered background" as anything logged so far.

**Why this cannot be logged as a usable finding.** Fetched the repo root
directly: it contains **only `README.md`** — no source files, no requirements
file, no weights, despite 5 real commits and August 2026 activity. A direct
fetch of `LICENSE` at the repo root 404'd, and no license entry appears in
the GitHub sidebar — there is nothing to license-check because there is no
code to check. The README links a Zenodo DOI badge
(`10.5281/zenodo.21802410`) that might hold a data/code archive, but
`zenodo.org` and `doi.org` both hit this sandbox's standing egress block, so
that could not be inspected. `arxiv.org` and `ar5iv.labs.arxiv.org` are
likewise blocked, so the method itself (the actual flow-construction formula,
and any quantitative results on GolfDB or Anti-UAV) could not be read
firsthand — everything above beyond the directly-fetched README is
search-snippet synthesis, unconfirmed.

**Licence.** None found — no LICENSE file exists in the repo. Not usable
commercially or otherwise today, because there is no code to use.

**Which failure mode.** Camouflage, primarily — a training-free motion signal
is exactly the kind of complementary channel (alongside RGB) that could help
find a dark clubhead against dark clothing or foliage, where appearance
alone gives zero candidates. It is single-frame-pair (needs one prior frame,
not a learned temporal window), which would make it cheap to compute
on-device as an extra input channel if it works as described — but that is
speculative until the method can actually be read.

**Effort vs. payoff.** Not assessable, and that is the finding — same
category as this log's DFRCP and Deblur-YOLO entries: a real, actively
maintained repo and a real paper, directly relevant by the authors' own
choice of golf-swing and fast-small-object benchmarks, but zero usable
artifact today (no code, no license, no readable method). Worth a future
run picking this up specifically to check `zenodo.org` (the DOI badge) or a
later commit that might add the actual implementation — the repo's commit
cadence (four separate update dates over 17 months, most recently this
month) suggests it is not abandoned.

---

## 2026-08-24 (third run) — EMIP: an explicit optical-flow, two-stream video camouflaged-object-detection network (IEEE TIP 2025), real working code but no licence — the ninth distinct VCOD mechanism logged, and the direct "explicit" counterpart to this log's SLT-Net ("implicit") entry

**Area covered.** Bullet 3 (temporal/multi-frame methods for small,
low-contrast or camouflaged objects), rotating away from today's first two
runs (motion-blur architecture, then a training-free flow method) toward a
*learned* VCOD network, to check whether anything new has appeared in that
family since the last one logged (SAM-PM, 2026-08-19). Checked this log's
existing VCOD/camouflage entries first — TrackNetV4, DTUM, SLT-Net,
SINet-V2, Motion-Informed Enhancement, CamDiff, SAM-PM, TOTNet, CAMotion,
ReynoldsFlow — none use a frozen pretrained optical-flow model as an
explicit second stream, which is EMIP's distinguishing mechanism.

**What I found.** "Explicit Motion Handling and Interactive Prompting for
Video Camouflaged Object Detection" (Zhang, Xiao, Ji, Wu, Fu, Zhao),
published in *IEEE Transactions on Image Processing*, vol. 34, pp.
2853–2866, 2025 (arXiv:2403.01968, first posted March 2024). Its own
framing positions it directly against SLT-Net (already in this log,
2026-08-15): SLT-Net handles motion *implicitly* through a learned temporal
module, EMIP handles it *explicitly* — a two-stream architecture where one
stream runs camouflage segmentation and the other runs optical-flow
estimation via a frozen, pretrained flow backbone, fused through a
"camouflage feeder" and "motion collector". Code is at
`github.com/zhangxin06/EMIP` — fetched directly and confirmed real:
`train.py`, `train_long.py`, `test.py`, `test_long.py`, `test_of.py`, a
`configs/` directory, a `model/` directory, and loss-function code are all
present, which is more than DFRCP or ReynoldsFlow had (README only, or
README-only-with-broken-code-links respectively). No pretrained weights or
checkpoints are included in the repo. It evaluates on MoCA-Mask, CAD2016,
and COD10K — MoCA-Mask is the same benchmark SLT-Net and this log's
CAMotion entry already touched.

**Licence — confirmed absent, not usable.** No `LICENSE` file exists at
either `main` or `master` in the repo (`raw.githubusercontent.com` 404s on
both paths, checked directly), and the GitHub API confirms it:
`GET /repos/zhangxin06/EMIP` returns `"license": null`. No licence text
anywhere in the README either. Default copyright applies — the code is not
usable, commercially or otherwise, without contacting the authors. This is
a firmer negative than DFRCP (no code found at all) or ReynoldsFlow
(code found but empty) — here the code is real and complete, and the
blocker is purely the missing licence.

**Which failure mode.** Camouflage, explicitly and only — the paper is a
segmentation method (pixel masks on MoCA-Mask/CAD2016/COD10K), not a
bounding-box detector, so even with a permissive licence it would need
adaptation to this project's box-detection output, on top of everything
else.

**Effort vs. payoff — low, independent of the licence block.** Three
compounding problems, not just one: (1) no licence, full stop; (2) it is a
segmentation network needing box-conversion to fit this project's YOLO11n
output; (3) it depends on an unnamed "frozen pretrained optical flow
model" as a second stream — the README does not say which one, and this
session's egress block on `arxiv.org` and `ieeexplore.ieee.org` prevented
reading the method section to find out. A two-stream network carrying a
full optical-flow backbone alongside a segmentation backbone is a heavy,
almost certainly non-trivial-to-convert CoreML target for an on-device
iOS app, well past what DFRCP (single-frame, pyramid-internal) or
ReynoldsFlow (training-free, single-frame-pair) would have cost if either
had been usable. Logged mainly to close out the "explicit vs. implicit
motion handling" comparison this log's SLT-Net entry left open, and to mark
this specific paper as checked (real, working, unlicensed) so a future run
does not re-discover it from zero — not as an actionable lead.
