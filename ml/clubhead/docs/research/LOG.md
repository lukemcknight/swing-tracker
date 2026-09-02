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

---

## 2026-08-24 (fourth run) — "A high-quality sport ball dataset annotation based on videos" (Dryad, Apr 2026): a fast/small/blurred sports-object dataset, checked and ruled out for commercial use (motion blur area, negative result)

**Area covered.** Bullet 2 — datasets of fast-moving blurred small objects
in sport — rotating away from today's first three runs, which were all
camouflage/blur *architecture or mechanism* entries (DFRCP, ReynoldsFlow,
EMIP). The dataset side of this bullet hasn't had a hit since iPhoneBlur
(2026-08-22, fourth run) and GoPro/REDS (2026-08-20, fourth run), neither
of which is sport-specific. Checked this log's existing dataset entries
first (GolfDB, CaddieSet, the 2026-08-21 golf dataset sweep, RealBlur,
SloMoDeblur, GoPro/REDS, iPhoneBlur) — none of them cover ball-sports video
with explicit motion-blur-driven detection difficulty as their stated
premise, which is this dataset's whole reason for existing.

**What I found.** "A high-quality sport ball dataset annotation based on
videos" (Zou, Tianjian & Liu, Jun; Beijing University of Posts and
Telecommunications), published on Dryad 2026-04-30, DOI
`10.5061/dryad.3bk3j9m13`. Per two independently-issued WebSearch queries
that returned self-consistent detail (same institution, same publish date,
same DOI, same per-class object counts, same license split — direct fetch
blocked, see below): table tennis, tennis, and soccer videos, each with
over 10,000 annotated ball instances, in Ultralytics/YOLO-format bounding
boxes (x, y, w, h), built specifically to address small-size,
low-distinctive-feature, and motion-blur detection difficulty for fast
spherical sports objects — the same class of problem this project has
(small, fast, elongating-under-blur object), just for balls rather than
clubheads. Source footage is described as "open-source online videos or
competition videos," automatically de-identified (people blurred
frame-by-frame) using a YOLO26 model before release.

**Why this cannot be independently verified beyond snippets.** Both
`datadryad.org` (the annotation/landing page) and the companion video host
`zenodo.org` (DOI `10.5281/zenodo.19874312`, where the actual video files
are published separately) hit this sandbox's standing egress block — the
same recurring failure noted against `zenodo.org` and `doi.org` in this
log's ReynoldsFlow entry earlier today, and against `arxiv.org`,
`nature.com`, `scirp.org` and `universe.roboflow.com` in earlier entries.
The two WebSearch queries agreed closely enough (identical numbers, DOIs,
and license split, phrased in near-identical language each time) to
support the dataset's existence and general shape, but the license text
itself was not read firsthand from either the Dryad or Zenodo page.

**Licence — split, and the split rules it out.** Per the corroborating
snippets: the *annotation labels* are CC0 (public domain), but the *actual
video/image content* — the only part usable for training — is licensed
CC BY-NC-ND 4.0 on the separate Zenodo release. CC BY-NC-ND is one of the
most restrictive Creative Commons tiers: no commercial use, and no
derivative works (model training is arguably exactly that) even for
non-commercial use. **Commercial use: not permitted.** This is consistent
with the source description above — footage sourced from "competition
videos" (broadcast-adjacent) is exactly the kind of asset that typically
cannot be relicensed permissively by a downstream annotator.

**Which failure mode.** Motion blur, specifically — the dataset's own
framing (small, fast, blur-degraded spherical objects, annotated for
exactly that difficulty) is the closest dataset-level match in this log to
the clubhead's blur problem, closer than GoPro/REDS (generic camera-shake
objects, not sport) or SloMoDeblur (generic smartphone scenes, not sport).

**Effort vs. payoff.** Low effort, zero payoff for training data — the
NC-ND video licence forecloses use regardless of the domain mismatch
(balls, not clubheads) or resolution. The one non-dead-end angle: if a
future run can get past the Zenodo egress block and confirm the
annotation/de-identification methodology (reportedly built on YOLO26,
already flagged as a checkpoint-swap candidate in this log's 2026-08-17
entry), that could inform this project's own labeling-QA tooling, not
serve as a data source. Logged mainly to close off "ball-sport
motion-blur datasets" as a category and stop this specific one from being
rediscovered from zero — not an actionable lead.

---

## 2026-08-25 — RF-DETR: a DINOv2-backboned, Apache-2.0 detection transformer that shipped native CoreML export four weeks ago (architecture area, both failure modes, existence-and-code confirmed)

**Area covered.** Camouflage and motion blur, both — this is an
architecture-swap entry, the same category as this log's FastViT
(2026-08-23, third run) and P2/4-head (2026-08-23, fourth run) entries.
Checked those first: FastViT is a backbone-only, drop-in swap into the
existing YOLO11n pipeline; this is a full detector-architecture
replacement, a structurally different (and much higher-effort) proposal,
so it is not a repeat.

**What I found.** RF-DETR (`github.com/roboflow/rf-detr`, accepted ICLR
2026), a real-time, query-based ("DETR-style", no NMS) object detector
built on a **DINOv2** self-supervised vision-transformer backbone. Verified
directly from the repo's README and releases page (two independent
`WebFetch` reads, cross-checked against each other for consistency):

- **License, confirmed via the README's License section, quoted:** "The
  open-source `rfdetr` package and Apache-designated model weights are
  licensed under Apache License 2.0" — and the Nano/Small/Medium/Large
  detection variants (plus all Seg-N…Seg-2XL segmentation variants) are
  explicitly listed as Apache-2.0. Only the XL/2XL detection variants
  (`rfdetr_plus` extension) are PML 1.0 — irrelevant here since Nano/Small
  are the only sizes plausible for this project's on-device budget.
  **Commercial use is permitted** for the sizes that matter to us.
- **Native CoreML export is real and current**, not a roadmap promise:
  added in release `v1.9.0` (29 Jul 2026 — four weeks before this run),
  confirmed via the releases page, quoted verbatim: "Native CoreML export —
  `RFDETR.export(format="coreml")` produces a `.mlpackage` (mlprogram, iOS
  16+) directly from `torch.export`, no ONNX intermediary." The repo has
  shipped five further point releases since (through v1.9.4, 24 Aug 2026),
  so it is actively maintained, not abandoned mid-feature.
- **Sizes:** Nano is 30.5M parameters. That is roughly **12x the
  parameter count of the YOLO11n this project currently trains and ships**
  (YOLO11n ≈ 2.6M params). The 2.3ms latency figure quoted in RF-DETR's own
  benchmarks is **not an on-device number** — it reads as a GPU-class
  figure (the repo's benchmark table is not phone/ANE-specific) and I could
  not find any published iPhone or Apple Neural Engine latency for RF-DETR
  anywhere in what I read. This is an unverified, load-bearing gap: nothing
  here confirms Nano actually runs in real time on the phone hardware this
  app targets.

**Why this addresses both failure modes.** Two distinct, plausible-but-
unproven mechanisms:
- *Camouflage:* DINOv2 is self-supervised on ~1.7B images (per its own
  published pretraining set), so its features are trained for general
  visual discrimination rather than only on this project's ~54%
  Roboflow / ~29% first-party-phone label mix. The broader VCOD literature
  already surfaced in this log (SAM-PM, 2026-08-19 fourth run) is built on
  exactly this premise — that foundation-model features generalize better
  under low-contrast/appearance-ambiguous conditions than a detector
  trained end-to-end on a small labelled set. Unproven for clubhead-vs-
  foliage specifically; RF-DETR has never been benchmarked on this kind of
  target as far as I found.
- *Motion blur:* DETR-style detection predicts boxes as a direct
  regression from learned object queries, with no anchor grid and no
  anchor aspect-ratio priors the way YOLO11n's head has. In principle that
  removes one structural bias against predicting the kind of elongated
  box this project's own labeling spec calls for (the full blur streak,
  median elongation only 1.60 in current labels) — but this is a
  mechanism argument, not something I found demonstrated on blurred data
  anywhere in RF-DETR's own materials.

**Effort vs. payoff.** High effort, uncertain payoff — the most expensive
class of finding this log logs. Unlike FastViT (a backbone slotted into
the existing training/export pipeline), adopting RF-DETR means standing up
a new training pipeline end to end, then answering the one question that
actually decides whether this is viable: does RF-DETR-Nano hit real-time
on an iPhone via CoreML, or does its 12x parameter count over YOLO11n make
it a non-starter for a live swing-capture app before any accuracy question
even matters. That is a half-day-or-less spike (export a COCO-pretrained
Nano checkpoint, drop it on a device, measure latency) that would settle
the question cheaply, and it should happen before this project spends any
time on a real training run.

**Verification status.** README license section and releases-page CoreML
changelog entry: read directly, quoted verbatim above, consistent across
two independent fetches. DINOv2 pretraining-scale claim: from DINOv2's own
published material, not independently re-verified this run. On-device
latency: **not found, not verified — flagged explicitly as the open
question above**, not glossed over.

---

## 2026-08-25 (second run) — EdgeTAM: a CoreML-exportable, real-time on-device SAM2 variant that reopens a category this log previously closed

**What it is.** EdgeTAM (Zhou et al., CVPR 2025, Meta) is a distilled,
mobile-optimized variant of Segment Anything Model 2 (SAM2) for promptable
video segmentation and tracking: given a mask/box/point prompt on one
frame, it propagates a segmentation mask of that specific object through
subsequent frames using the same memory-attention temporal mechanism SAM2
uses, but with a much lighter encoder distilled for on-device speed. It is
not a detector — it does not find a class of object on its own — it tracks
whatever object it is told about across frames it has not seen labels for.
Reference implementation (PyTorch, official Meta Research repo, README and
LICENSE fetched directly from `github.com/facebookresearch/EdgeTAM`, not
search-indexing) includes exported-model instructions for iOS/macOS:
three separate CoreML models (Image Encoder ~9.6MB, Prompt Encoder ~2MB,
Mask Decoder ~8MB, ~20MB total) with reported on-device numbers of 16 FPS
on an iPhone 15 Pro Max for full video segmentation-and-tracking without
quantization (22x faster than SAM2), and 40.4 FPS for single-frame
Segment-Anything-style inference.

**URL.** https://github.com/facebookresearch/EdgeTAM (paper: CVPR 2025 /
arXiv:2501.07256 — arxiv.org remains blocked by this sandbox's egress
proxy, same restriction noted in every prior run of this log; claims below
come from the repo's own README and LICENSE file, fetched directly, plus
one independent web search corroborating the FPS/parameter figures against
the paper's own announcement thread, not from the PDF itself).

**Licence (verbatim, from the repo's `LICENSE` file, fetched directly).**
"Apache License, Version 2.0, January 2004." The README states explicitly:
"The EdgeTAM model checkpoints and code are licensed under Apache 2.0" —
i.e. this is one of the few entries in this log where the pretrained
*weights*, not just the code, are confirmed under a commercially-permissive
licence in the project's own words, not inferred from a generic repo
licence badge. **Commercial use: permitted**, standard Apache-2.0
conditions (retain notices, state changes).

**Which failure mode.** Primarily camouflage — this is a genuine, tested
route to answering the "camera-motion research spike" this log has flagged
as unresolved since the very first camouflage entries (TrackNetV4,
2026-08-13): instead of building a bespoke temporal-attention module into
YOLO11n's architecture, run YOLO11n as the primary per-frame detector and,
on a frame where it returns zero candidates even at low confidence, seed
EdgeTAM with the last confident prior detection's mask/box and let it
propagate a track through the camouflaged frames until YOLO reacquires.
Unlike the 2026-08-17 InpaintNet entry (pure trajectory interpolation, no
visual signal at all), this actually re-examines the pixels each frame
using real spatio-temporal attention, so it should degrade more gracefully
when the club changes direction under camouflage (e.g. top of backswing
against foliage) than a straight-line or spline interpolation would.
Secondary, weaker relevance to motion blur: propagating a mask through a
blurred frame doesn't need per-frame anchor-box regression to guess the
elongated streak shape the way YOLO11n's head does, but this is unverified
speculation, not something the sources below demonstrate.

**Why it helps this model specifically — and the real reason it's worth
logging.** The 2026-08-19 SAM-PM entry explicitly closed off "SAM/SAM2-
based video-camouflaged-object-detection" as a practical line of
investigation for this deployment target, on the specific grounds that a
SAM-sized encoder (ViT-B alone ~91M params) "cannot run per-frame on an
iPhone at video-capture rate" versus YOLO11n's ~2.6M total parameters —
and flagged that conclusion as applying to the whole category, including
future SAM2-based papers, "unless the project's deployment target changes."
EdgeTAM is the first candidate this log has found that actually reopens
that closure on its own terms: it's the same SAM2 mechanism, official
Meta code, with a genuinely small on-device footprint (~20MB across three
CoreML models, real reported FPS on an iPhone 15 Pro Max) rather than a
foundation-model encoder. That doesn't make it a fit by default — it means
the "already known to be no" from 08-19 needs to be downgraded back to
"actually worth a spike," which is itself the useful finding, independent
of whether EdgeTAM specifically turns out to work.

**Honest caveat that keeps this from being an easy win.** A companion
search this run surfaced "Benchmarking SAM2-based Trackers on FMOX"
(arXiv:2512.09633, a Fast-Moving-Objects tracking benchmark), whose
findings (read via search-result summary only — arxiv.org itself blocked,
so this is second-hand and explicitly flagged as such) state that SAM2-
family trackers "struggle with accurately tracking objects with very thin
or fine details especially when they are fast-moving," with reported
"error propagation and tracking instability... in challenging scenarios
involving fast-moving objects." A golf clubhead at impact speed is close
to the worst case this describes — thin, fast, exactly where this project
needs the recovery to work. EdgeTAM's own materials do not report FMO-style
benchmarks, so whether its distilled encoder is better or worse than full
SAM2 on this specific failure pattern is unknown either way, not just
untested by this project.

**Effort vs. payoff.** Medium effort, uncertain-but-real payoff. This is a
tracking/recovery layer bolted onto the existing detector, not a
replacement for it or a retraining exercise — the spike is: export the
three CoreML models from the official repo (documented, no custom
conversion work needed), feed it one of the existing camouflage-failure
clips seeded with a manually-drawn box on the last good frame, and see
whether the propagated mask stays on the clubhead through the failure
region or drifts. That's an afternoon, not a training run. The honest risk
is the FMOX caveat above: if EdgeTAM inherits SAM2's known weakness on
fast thin objects, the recovery may not survive exactly the frames where
camouflage most needs it (the same frames tend to be higher-speed swing
frames), and this would be discovered only by running the spike, not by
further literature search.

**Verification status.** README description, CoreML export section, and
FPS/parameter figures: read directly from the repo, quoted/paraphrased
above. LICENSE file: fetched and quoted directly. FMOX/SAM2-on-fast-objects
weakness claim: from a search-result summary of arXiv:2512.09633, not the
paper itself (blocked) — flagged as second-hand, not independently
confirmed.

---

## 2026-08-25 (third run) — 6-DOF camera-motion blur synthesis for object detection (Yang, Li & Zhang, ICPR 2024): the camera-shake half of blur this log's PSF/frame-averaging entries don't cover

**Area covered.** Rotated deliberately away from architecture/tracking
(bullets used the last two runs today: RF-DETR and EdgeTAM) back to bullet 2
(motion-blur synthesis specifically). Before logging this, checked whether
the CADDIE golf-pose lead (2026-08-21) had become reachable through any new
route — it has not: `openaccess.thecvf.com` still returns `EGRESS_BLOCKED`
this run, no GitHub repo or preprint mirror exists, and a targeted search for
indoor/simulator-bay golf datasets again surfaced nothing beyond the
industry-news noise every prior dataset-area run has already hit. Neither is
logged as a new finding — both are re-confirmed dead ends of entries already
in this file.

**What it is.** "6-DOF Motion Blur Synthesis and Performance Evaluation of
Object Detection" (Hanjin Yang, Feng Li, Lei Zhang — ICPR 2024, published as
a Springer LNCS chapter). Per consistent search-indexed summaries from two
independent indexes (see URL section), the method generates a random
six-degree-of-freedom **camera** motion trajectory (translation + rotation,
not a single object-local direction), maps that trajectory through the
scene geometry to a spatially-varying, per-pixel blur kernel field (not one
kernel for the whole frame), and — to make convolving with a different
kernel at every pixel computationally tractable — decomposes the kernel
field with non-negative matrix factorization (NMF) into a small basis of
kernels plus per-pixel mixing coefficients. The paper evaluates the
synthesized-blur training data specifically on **object detection**
performance (not just image-quality metrics), reporting results against
RealBlur, GoPro, and REDS — all three already logged in this file
(RealBlur: 2026-08-16; GoPro/REDS: 2026-08-20) as the standard blur
benchmarks, which is corroborating evidence this is a real, mainstream
motion-blur-for-detection paper and not a mis-indexed unrelated result.

**Why this is a genuinely different mechanism from what's already logged.**
Every blur-synthesis entry so far in this log operates **per-object**: PSF
convolution (Sayed & Brostow, 2026-08-14) applies one directional kernel to
a cropped clubhead and grows its box; frame-averaging (Brooks & Barron,
2026-08-12) and BlenderProc's `enable_motion_blur` (2026-08-18, second run)
both model the blurred object's own motion. None of them model **whole-frame
camera shake** — the blur a handheld phone itself contributes independent of
what the clubhead is doing — despite this log flagging "camera-motion
research spike" as an open, unresolved question in at least six prior
entries (TrackNetV4 2026-08-13, DTUM 2026-08-14, MoSA-Det 2026-08-23, and
others). This paper's 6-DOF *camera* trajectory model is the first blur-
synthesis technique found that targets exactly that missing half: apply it
to an existing sharp, correctly-boxed training frame and it degrades the
*entire image* with a physically-plausible handheld-camera blur field,
independent of and complementary to growing the clubhead's own box for its
motion — the two could be composed (object-local PSF/streak blur on top of
a globally camera-shaken frame) to approximate what real handheld swing
footage with a slow shutter actually looks like, which is closer to the
"genuinely blurred, correctly-boxed clubhead" gap the brief describes than
either technique alone.

**URL.** https://link.springer.com/chapter/10.1007/978-3-031-78444-6_1
(also indexed identically at https://dl.acm.org/doi/10.1007/978-3-031-78444-6_1,
same DOI, same title/author list/venue — independent-index corroboration of
existence, not just one source). `link.springer.com` is blocked by this
sandbox's egress proxy (same standing restriction as every prior Springer
link in this log, e.g. line 280); `dl.acm.org` and `scite.ai` (an AI-
summary/citation site that also indexes this paper) were also tried this run
and both blocked. **No primary-source text was read.** Everything above is
reconstructed from search-engine result summaries, which is one notch below
this log's stronger entries (e.g. EdgeTAM, WASB-SBDT) where the repo/README
itself was fetched directly — treat the mechanism description as plausible
and internally consistent (it matches standard 6-DOF-trajectory blur
literature and the RealBlur/GoPro/REDS benchmark choice is exactly what a
real blur-for-detection paper would use) but not independently confirmed.

**Licence — no code or repository found, so nothing to license.** Targeted
searches for a GitHub release under the authors' names or the paper title
turned up nothing (only unrelated repos, e.g. Wii Sports Resort mods).
Unlike the PSF entry (2026-08-14), where a real, inspectable-but-unlicensed
repo exists, here there is no artifact to grant or withhold rights over.
The *idea* — a randomized 6-DOF camera trajectory mapped to a spatially-
varying blur-kernel field, applied to already-owned training images — is
ordinary image-processing/data-augmentation technique with no original-
expression claim attached (the same posture this log already took on PSF
synthesis); the NMF-basis-decomposition trick is a standard, decades-old
linear-algebra method (not this paper's invention) used here purely for
compute efficiency, so reimplementing the described idea from scratch
carries no licensing exposure. **Commercial use: not a licensing question
for this project** since nothing of the authors' is being redistributed or
run verbatim — but also nothing shippable exists to save engineering time.

**Which failure mode.** Motion blur, specifically. Camera-shake blur is
orthogonal to camouflage (a still-frame appearance problem); it does not
help zero-detection-on-a-sharp-frame failures at all.

**Effort vs. payoff.** Medium effort, plausible but unverified payoff, and
strictly a reimplementation exercise (no code to adapt). Effort: the core
idea — sample a random 6-DOF camera trajectory, render or approximate the
resulting per-pixel blur-kernel field, convolve — is more engineering work
than the PSF entry's single directional kernel (spatially-varying
convolution, not one crop-level operation, is the whole reason the paper
needed the NMF trick to make it fast), but is still a data-augmentation
script against existing training images, not a retraining-architecture or
new-capture commitment. The NMF-efficiency trick specifically matters only
at full-frame resolution and real-time budgets; if this project only needs
an offline augmentation pass at training time (not on-device), a much
simpler brute-force spatially-varying convolution may suffice without
reimplementing the NMF step at all, lowering the real effort further.
Payoff: directly addresses a gap this log has repeatedly flagged as
unresolved (camera-shake blur, as opposed to object-motion blur) but every
specific number behind "improves object detection" comes from an unread
primary source, so treat as a promising, cheap-to-prototype idea worth a
literature-verification pass by someone with Springer/ACM access, not yet
as a validated result to build against.

**Verification status.** Paper existence, title, authors, venue (ICPR 2024),
and DOI: confirmed via two independent indexes (SpringerLink, ACM Digital
Library) returning identical metadata. Method description (6-DOF camera
trajectory, per-pixel blur field, NMF kernel-basis decomposition) and
evaluation datasets (RealBlur/GoPro/REDS): from search-result summaries
only, not the paper text — flagged as such, not independently confirmed.
No code/repository found by search.

---

## 2026-08-25 (fourth run) — D-FINE: an ICLR 2025 Spotlight real-time DETR with Apache-2.0 code, and a genuine gap it has that RF-DETR doesn't

**Area covered.** Architecture (this log's bullet touching small/low-contrast
detection capacity). Before picking this, two other leads were tried and
dropped: (1) `mamoonik/golf-swing`, a GitHub repo claiming 38-keypoint
body+club tracking built on the already-logged GolfPose architecture — its
README and code are real and it does produce club-keypoint output (LSD-based
hosel-keypoint correction, reported pixel-error improvement over an HRNet
baseline), but the repository ships **no LICENSE file at all**, which under
default copyright means no rights are granted for reuse, commercial or
otherwise — the same dead end this log already hit with `dj_masters`
(2026-08-14). Not logged as a separate entry since it adds no new pattern
beyond what that entry already established (unlicensed GolfPose-derivative
student repos). (2) A repeat check of `universe.roboflow.com` for the
"golf-club-tracking" dataset (6,750 images, surfaced by search) — still
`EGRESS_BLOCKED`, same standing wall as every prior run's Roboflow attempts;
not logged, since Roboflow-domain inaccessibility is already an established
fact in this file (e.g. 2026-08-18 second run, 2026-08-22 second run).

**What it is.** D-FINE ("Redefine Regression Task of DETRs as Fine-grained
Distribution Refinement," Peterande et al., **ICLR 2025 Spotlight**). Fetched
directly from `github.com/Peterande/D-FINE` (not just search summaries).
It reframes DETR-style bounding-box regression: instead of predicting a
single box offset, it predicts a probability distribution over fine-grained
bins for each box edge and refines that distribution iteratively across
decoder layers (Fine-grained Distribution Refinement, FDR), paired with a
self-distillation scheme (Global Optimal Localization Self-Distillation,
GO-LSD) that transfers the final, most-refined layer's localization
knowledge back to earlier layers at no extra inference cost. This is a
mechanistically distinct lever from every architecture entry already logged:
RF-DETR (2026-08-25, first run) swaps in a DINOv2 backbone for stronger
features; YOLO26 (2026-08-17) changes small-object label assignment
(STAL); the P2/4 head entry (2026-08-23) adds a higher-resolution detection
head; FastViT (2026-08-23) is a backbone swap. D-FINE instead changes *how
box coordinates themselves are regressed*, which is a plausible lever for
this project's problem shape: a clubhead's box is often ambiguous at its
edges even when the model fires at all (blurred streaks especially), and a
distributional-regression head is built exactly to handle boundary
ambiguity better than a single point-estimate offset.

**Reported numbers.** Five scale variants — Nano (4M params, D-FINE-N:
42.8% COCO AP), Small (10M, 48.5%), Medium (19M, 52.3%), Large (31M, 54.0%
at 124 FPS on a T4), X (62M, 55.8% at 78 FPS) — all read directly from the
repo's own README table, not a search summary. D-FINE-N at 4M params is in
the same rough capacity class as YOLO11n (this project's current model,
~2.6M params), making it a realistic like-for-like swap candidate rather
than a jump to a much heavier model. No small-object-specific AP breakdown
(e.g. COCO AP-small) is published in the README; this project's failure
mode is specifically small/low-contrast objects, and the headline AP
numbers do not tell us how D-FINE-N does on that slice specifically.

**Licence.** Apache License 2.0 — fetched `LICENSE` directly from
`raw.githubusercontent.com/Peterande/D-FINE/master/LICENSE` and confirmed
the file is the genuine Apache 2.0 text (matched header: "Apache License,
Version 2.0, January 2004... TERMS AND CONDITIONS FOR USE, REPRODUCTION,
AND DISTRIBUTION"). Commercial use permitted, same posture as RF-DETR.

**The gap RF-DETR doesn't have.** RF-DETR's entry (2026-08-25, first run)
noted it shipped a *native* CoreML export path. D-FINE's own README and
repo only document ONNX export (with simplification) and TensorRT
conversion — no CoreML or other mobile-format export is mentioned anywhere
in the repo, and a targeted search for a community CoreML conversion of
D-FINE (`"D-FINE" coreml export ios`) turned up nothing specific to this
model, only generic PyTorch→ONNX→coremltools guidance that applies to any
model. DETR-family architectures (deformable attention, custom ops) are
known in practice to be more failure-prone through coremltools than plain
convolutional YOLO-style nets, though this project has not attempted the
conversion itself to confirm or refute that for D-FINE specifically.

**Why this matters for this model specifically.** If D-FINE-N's
distributional box-regression genuinely produces tighter, more confident
edges on ambiguous (blurred or low-contrast) small objects, it is a direct
lever on this project's core measured problem — but that benefit is
unconfirmed for this exact shape (single small elongated object, not COCO's
general object mix) and comes bundled with a real, unquantified CoreML
export risk this project would have to absorb itself, unlike RF-DETR where
that risk is already retired upstream.

**Effort vs. payoff.** Medium-high effort, uncertain payoff, and strictly
higher-risk than the RF-DETR entry already logged today because of the
missing export path. Honest ordering: if this project spikes one DETR-family
architecture, RF-DETR (native CoreML export, DINOv2 features, already
logged) is the lower-risk first attempt; D-FINE is worth a second look only
if RF-DETR's CoreML export or accuracy disappoints, and even then the first
step would have to be a from-scratch ONNX→coremltools conversion spike
before any training investment, since that step alone could kill the whole
idea.

**Verification status.** Repo, method description, license text, and all
AP/parameter numbers: fetched and read directly from
`github.com/Peterande/D-FINE` and its raw `LICENSE` file, not from search
summaries. The "no CoreML export exists" claim is a search-derived
negative (absence of evidence), not a certainty — a conversion may be
technically possible but undocumented; that uncertainty is stated plainly
above, not hidden.

---

## 2026-08-26 — YOLOV / YOLOV++: multi-frame feature aggregation for video object detection (AAAI 2023 / IJCV 2026), Apache-2.0, built on YOLOX not YOLO11

**Area covered.** Small/low-contrast/camouflaged object detection via
multi-frame and temporal methods — the category the brief calls out
specifically ("motion is what separates a moving clubhead from static
foliage when appearance cannot"). Rotating away from architecture
(RF-DETR, D-FINE) and blur-synthesis (6-DOF camera motion), the last two
areas this log touched on 2026-08-25.

**What it is.** YOLOV and YOLOV++, from `github.com/YuHengsss/YOLOV`
(PyTorch implementation, original YOLOV at AAAI 2023, the newer YOLOV++
paper "Practical Video Object Detection via Feature Selection and
Aggregation" published in *International Journal of Computer Vision*,
2026 — venue confirmed via its `link.springer.com` and `researchgate.net`
listings). Unlike single-frame detectors, YOLOV treats a short video clip
as the unit of inference: it runs a base detector (YOLOX) per frame,
selects a small set of candidate features per frame, then computes
affinity between the current frame's candidates and candidates from
neighboring frames in the clip and aggregates them before the final
classification/box refinement. This is mechanistically distinct from
every temporal method already logged: TrackNetV4 (2026-08-13) fuses a
motion-attention map, Motion-Informed Enhancement (2026-08-18) channel-
encodes motion into an ordinary 3-channel image, DTUM (2026-08-14)
direction-codes a temporal difference module, and SLT-Net/EMIP/SAM-PM
(camouflage-specific VCOD) use short-term motion or optical flow as an
auxiliary appearance cue — YOLOV instead aggregates *detector-level
candidate features* across frames via a learned affinity/attention
mechanism, closer in spirit to video object detection literature
(Flow-Guided Feature Aggregation, Zhu et al. 2017) than to the VCOD or
sports-tracking papers already in this log.

**Reported numbers.** From the repo's own README table (fetched and read
directly, not from a search summary): YOLOV-s reaches 77.3 AP50 at 11.3 ms
on a 2080Ti (batch 1); YOLOV-x reaches 85.5 AP50 at 22.7 ms. YOLOV++
(adds the "selection" half on top of aggregation) reaches up to 92.9 AP50
with a Focal-Large backbone, and 78.7 AP50 at 5.3 ms for its smallest "s"
variant (batch-32 throughput on a 3090, so not directly comparable to a
single-stream on-device latency number). All figures are on ImageNet VID
— a 30-category general-object video benchmark, not sports footage and
not a small/low-contrast-specific slice, so there is no direct evidence
these gains transfer to a single small elongated fast object like a
clubhead.

**Licence.** Apache License 2.0. Verified two ways: the GitHub blob view
of `LICENSE` and a direct fetch of
`raw.githubusercontent.com/YuHengsss/YOLOV/master/LICENSE`, which returned
the genuine text starting "Apache License / Version 2.0, January 2004 /
http://www.apache.org/licenses/ ... TERMS AND CONDITIONS FOR USE,
REPRODUCTION, AND DISTRIBUTION". Commercial use is permitted.

**Why this matters for this model specifically, and the real gaps.**
The mechanism is a plausible fit for the camouflage failure mode: a dark
clubhead against dark clothing or foliage produces zero detections in a
single frame precisely because appearance alone gives the detector
nothing to key on, and aggregating candidate features across several
consecutive frames is a direct, architecturally-native way to let motion
compensate — closer to what the brief is asking for than bolting a motion
channel onto a single-frame YOLO. But three real gaps stand between this
and a usable fix: (1) it is built on YOLOX, an anchor-free detector with
a different backbone/head than YOLO11n — adopting the technique means
either porting the aggregation module onto YOLO11n from scratch (not
something the repo does or documents) or replacing the base detector
entirely, either of which is a substantial engineering project, not a
config change; (2) no CoreML, ONNX, or any mobile/edge export path is
mentioned anywhere in the README — this is a research codebase targeting
GPU batch inference, and the multi-frame aggregation step itself (cross-
frame attention over a buffered clip) is an architecture shape
coremltools and the Neural Engine have not been shown to handle well;
(3) the benchmark domain (ImageNet VID's 30 everyday object categories)
gives no evidence either way for a single small, often-blurred, often-
camouflaged golf club head — the accuracy numbers above cannot be read as
predicting anything about this project's failure modes.

**Effort vs. payoff.** High effort, uncertain payoff. This is a bigger
lift than any temporal-fusion entry already logged (TrackNetV4's motion-
attention map or Motion-Informed Enhancement's channel-encoding are both
adaptable to YOLO11n with modest changes; this is not) — it would need a
from-scratch reimplementation of the aggregation idea against a YOLO11n
backbone plus an unproven CoreML export path before any training
investment could be evaluated at all. Worth logging as a genuinely
different mechanism and a good conceptual reference for *how* to build a
multi-frame aggregation head, but not something to prototype directly;
if this project pursues multi-frame camouflage fixes, the already-logged,
lower-effort options (TrackNetV4-style motion-attention fusion or
Motion-Informed Enhancement's channel encoding, both YOLO-compatible) are
the ones to try first.

**Verification status.** Repository, README performance table, and
licence text: fetched and read directly from `github.com/YuHengsss/YOLOV`
and its raw `LICENSE` file. The paper's own abstract could not be fetched
directly — `arxiv.org` is blocked by this sandbox's egress proxy, the same
standing block this log has hit on every prior arxiv attempt (e.g.
SloMoDeblur 2026-08-19, ReynoldsFlow 2026-08-24). A search-engine snippet
for the paper claimed it targets "degenerated object appearances... such
as motion blur, video defocus, and rare poses," but that phrasing is
generic boilerplate shared across the video-object-detection literature
(it traces to the original 2017 FGFA paper's motivation section) and is
not confirmed as YOLOV/YOLOV++-specific — the GitHub README does not
repeat or substantiate it, so it is deliberately left out of the claims
above rather than reported as fact.

---

## 2026-08-26 (second run) — Temporal-YOLOv8: stacking grayscale frames from different timesteps into the RGB channels of an *unmodified* YOLOv8, CC BY licensed, zero architecture change

**Area covered.** Small/low-contrast/camouflaged object detection via
multi-frame and temporal methods — same category as this morning's
YOLOV/YOLOV++ entry, chosen again deliberately because it turned up a
mechanism materially cheaper than anything already logged in it. A first
attempt this run to find something in the golf-dataset category (bullet 1)
searched specifically for indoor/simulator-bay/low-light golf footage with
a commercial licence; it surfaced only golf-simulator hardware marketing
pages, an unrelated capstone OpenPose repo
(`personableduck/GolfSwing` — 2016-era body pose, not club-specific, no
licence found), and CaddieSet (already ruled out 2026-08-18, no imagery
released) — nothing new and verifiable, so this run pivoted to the
temporal-methods bullet instead.

**What it is.** "Toward Versatile Small Object Detection with
Temporal-YOLOv8" (van Leeuwen et al., TNO Defence, Safety and Security,
The Hague — *Sensors* (MDPI) 24(22):7387, November 2024, DOI
10.3390/s24227387, PMID 39599163, PMCID PMC11598073). The core method,
Temporal-YOLOv8: instead of feeding a standard YOLOv8 a single RGB frame,
feed it three **grayscale** frames sampled at three different timesteps
(e.g. t, t-1, t-2), one per input channel, in place of R, G, and B. The
network architecture, weights shape, and compute cost are all identical to
stock YOLOv8 — this is purely a change to what goes into the existing
3-channel input tensor, not a model change. The paper also reports two
higher-capacity variants that do need an input-layer change: Color-T-YOLO
(9 channels — three full RGB frames stacked, keeping colour *and* time)
and Manyframe-YOLO (11 stacked grayscale frames, for more motion context).
This is mechanistically the same family of idea as this log's
already-logged Motion-Informed Enhancement (2026-08-18) — recolor an
ordinary image with temporal history, feed an unmodified detector — but a
different, independently-arrived-at implementation, validated on YOLOv8
specifically (the direct predecessor, same Ultralytics lineage, to this
project's YOLO11n) rather than YOLOv5, and with no GPL entanglement (see
Licence below).

**Reported numbers.** On the authors' in-house evaluation set of civilian
and military objects (not sports footage — looks to be long-range/aerial
surveillance imagery from the TNO defence-research context, based on the
paper's stated affiliation and object categories; the primary source could
not be read directly to confirm capture conditions, see Verification
status), baseline YOLOv8 mAP of 0.465 improved to 0.839 combining the
temporal input with small-object-tailored augmentation. This is a large
jump, but on a domain and object-size distribution that has no demonstrated
relationship to a single elongated golf clubhead against foliage — treat it
as evidence the general mechanism works, not as a predictor of the size of
gain here.

**Licence.** Sensors is an MDPI open-access journal; multiple independent
search-result snippets state the article is "distributed under the terms
and conditions of the Creative Commons Attribution (CC BY) license" and
that "any part of the article may be reused without permission provided
the original article is clearly cited" — MDPI's standard CC BY posture,
consistent with every other MDPI Sensors article this log has checked.
**Commercial use is permitted**, attribution required. This is a licence on
the *paper*, not on any code — no official code repository was found for
Temporal-YOLOv8 despite a dedicated search (TNO is a defence-research
institute, not a typical open-source publisher); using the technique means
reimplementing the (simple) input-recoloring step ourselves, which has the
practical benefit of leaving no third-party licence to track at all — a
strictly better position than the already-logged Motion-Informed
Enhancement entry, whose reference implementation is GPL-3.0 and explicitly
flagged as not shippable without legal review.

**Why this matters for this model specifically.** This is the cheapest
genuinely-new idea this log's temporal/multi-frame category has surfaced.
Every other multi-frame mechanism logged so far — TrackNetV4's motion-
attention fusion, DTUM's direction-coded temporal module, YOLOV's cross-
frame candidate aggregation, Motion-Informed Enhancement's GPL-encumbered
recolor step — requires either a new architecture block, a ported module,
or accepting a copyleft dependency. The 3-grayscale-frame variant here
requires none of that: the exported CoreML model already takes a 3-channel
image; swapping which three planes fill it (three grayscale timesteps
instead of single-frame R/G/B) touches only the data-engine's frame sampler
at train time and the iOS capture pipeline's frame-buffering at inference
time. No retraining architecture change, no new CoreML export path to
validate, no Neural Engine compatibility risk. That directly targets the
camouflage failure mode the brief describes: a visually sharp, static-
looking frame gives the detector nothing when the clubhead's appearance
blends into clothing or foliage, and three timesteps of grayscale motion
history is a cheap, load-bearing signal the model currently never sees.
Two honest limits: (1) it does not obviously help the motion-blur failure
mode — if all three sampled sub-frames are themselves already blurred,
stacking three blurred grayscale planes does not recover a sharp box, so
this should be read as a camouflage-mode fix, not a blur-mode one; (2) it
is an incremental variant of an idea already in the log (Motion-Informed
Enhancement), not a new category — its value here is specifically removing
that entry's GPL blocker and confirming the recipe transfers to the
Ultralytics YOLOv8/YOLO11 family rather than only YOLOv5.

**Effort vs. payoff.** Low-to-moderate effort, moderate-but-uncertain
payoff. Effort is genuinely small relative to everything else in this
category: no architecture work, no export-path risk, and the change is
confined to two places already under this project's control (the
data-engine's frame sampling and the iOS capture buffer). Payoff is
unproven for this specific domain — the reported mAP jump comes from
aerial/surveillance imagery with no demonstrated bearing on a single fast
clubhead against foliage, and the underlying idea already has a logged
precedent (Motion-Informed Enhancement) that reached a similar conclusion.
Given the low cost, this is a reasonable candidate for an actual small
prototype (train YOLO11n on the existing dataset with a 3-grayscale-frame
input swap, on a held-out slice, and compare to the single-frame baseline)
rather than a documentation-only wait for legal review, which is the
Motion-Informed Enhancement entry's practical status today.

**Verification status.** The paper's existence, exact title, authors,
venue, DOI/PMID/PMCID, and the CC BY licence line were confirmed via
multiple independent search-result snippets (search-engine-rendered
excerpts of the PMC, PubMed, and MDPI listing pages), consistent across
three separate searches. The primary source itself could not be fetched
directly: `www.mdpi.com`, `pmc.ncbi.nlm.nih.gov`, `www.ncbi.nlm.nih.gov`,
`europepmc.org`, and `api.semanticscholar.org` were all rejected by this
sandbox's egress proxy with `EGRESS_BLOCKED` on direct fetch attempts —
the same standing block this log has hit on `arxiv.org` in prior runs
(e.g. YOLOV/YOLOV++ above, SloMoDeblur 2026-08-19), now confirmed to extend
to several other scholarly-publishing domains as well. The method
description (three grayscale frames into RGB channels, zero architecture
change, the Color-T-YOLO/Manyframe-YOLO variants) and the reported mAP
figures are therefore sourced from consistent search-snippet excerpts
rather than a directly-read PDF or HTML body — one notch below full
verification, the same caveat this log has applied to other arxiv-blocked
entries, but corroborated across independent searches rather than resting
on a single snippet.

---

## 2026-08-26 (third run) — OC-SORT: a training-free Kalman-filter tracker built specifically to survive multi-frame detection gaps, as the zero-cost counterpart to the logged InpaintNet recovery entry

**Area covered.** Rotated to bullet 3 (small/low-contrast/camouflaged
object detection, temporal methods), specifically the "recovery" sub-thread
this log opened on 2026-08-17 with TrackNetV3's InpaintNet (a trained
network that inpaints trajectory gaps after detection fails) and has not
revisited since. Searched first for newer video-camouflaged-object-
detection papers (YUV20K, a Mamba-based spatio-frequency method, CamoSAM2)
— all real and recent, but every one is a heavier learned architecture in
the same family already covered five times over (TrackNetV4, DTUM, SLT-
Net, SINet-V2, SAM-PM, CamDiff, EMIP), so none was logged as a new finding.
Pivoted instead to asking a narrower question the log had not yet asked:
is there a *classical, zero-training* way to bridge a run of zero-
detection frames, as a cheaper alternative to InpaintNet's small trained
U-Net? That search surfaced OC-SORT.

**What it is.** "Observation-Centric SORT: Rethinking SORT for Robust
Multi-Object Tracking" (Cao, Pang, Weng, Khirodkar, Kitani — CVPR 2023,
pp. 9686–9696, arXiv:2203.14360) is a Kalman-filter tracking-by-detection
wrapper that sits on top of *any* off-the-shelf per-frame detector's
output boxes — it needs no image input at all, only the (x, y, w, h,
confidence) the detector already emits. Plain SORT-style Kalman trackers
degrade badly across a gap with no observations: the filter keeps
propagating its last-known velocity with nothing to correct it, so its
predicted position drifts away from the truth, and when a real detection
reappears the association fails. OC-SORT's fix is Observation-Centric
Re-Update (ORU): when a track resumes after a gap, it retroactively
re-updates the filter's state using a virtual straight-line trajectory
built between the last observation before the gap and the first
observation after it, instead of trusting the drifted, no-correction
extrapolation the gap produced. This is architecturally the single-object
special case of the exact "camouflage causes a run of zero-candidate
frames, then detection resumes once framing or lighting changes" symptom
the brief describes — a golf swing clip has exactly one clubhead to track,
which is a strict simplification of the crowded multi-object scenes
(MOT17/MOT20/DanceTrack) OC-SORT was built and benchmarked for.

**URL.** Code: https://github.com/noahcao/OC_SORT (`master` branch;
`README.md` and `LICENSE` both fetched directly via `raw.githubusercontent.com`,
HTTP 200, confirmed live — not a stub: the repo ships training/inference
scripts, reported benchmark tables for MOT17/MOT20/DanceTrack/KITTI, and
ONNX/TensorRT/C++ deployment variants). Paper: CVF Open Access page
`openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_...`
and arXiv:2203.14360 — both `openaccess.thecvf.com` and `export.arxiv.org`
returned `EGRESS_BLOCKED` from this sandbox, the same standing restriction
prior runs have hit on scholarly hosts, so the paper's title, authors,
venue, page numbers, and arXiv ID are sourced from consistent search-engine
listings (the CVF page title, the arXiv abstract-page title, and a Medium
technical write-up all agree), not a directly-read PDF. The *mechanism*
description (ORU, the drift problem it fixes) is corroborated identically
across an independent technical summary and the repo's own README/pipeline
diagram, so the mechanism itself is treated as verified; the benchmark
numbers below come directly from the repo's README table, not the paper.

**Licence (verbatim, from the repo's `LICENSE` file, fetched directly via
`raw.githubusercontent.com`, HTTP 200).** MIT License. "Permission is
hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the \"Software\"), to deal in
the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software... subject to the following conditions: The above
copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software." **Commercial use:
permitted.** (The copyright line reads "Copyright (c) 2021 Yifu Zhang" —
OC-SORT's repo is built on the ByteTrack codebase and retains ByteTrack's
original MIT notice; this does not weaken the grant, since MIT permits
reuse regardless of whose name is on the notice, and the README itself
states "OC-SORT, filterpy and ByteTrack are available under MIT License.")
Reported benchmark numbers (README table, not independently reproduced
here): MOT17 HOTA 63.2 / MOTA 78.0; DanceTrack-test MOTA 89.4; KITTI-cars
MOTA 90.3; association-only speed "700+ FPS on a single CPU" when
detections are pre-provided — the relevant number for this use case, since
the app would run its own YOLO11n detector and hand OC-SORT only the
resulting boxes, not run OC-SORT's own detector.

**Which failure mode.** Primarily camouflage, in the same "recovery, not
appearance" sense already logged for InpaintNet (2026-08-17) — it does not
touch *why* a frame produces zero candidates (colour-similarity to
background), it only bridges the resulting gap using motion continuity.
It offers little for motion blur specifically: a blurred-but-visible
clubhead usually still yields a real (if imprecise) detection for the
Kalman filter to condition on, so the gap-bridging mechanism this entry is
about simply doesn't activate for most blur frames the way it does for
true zero-candidate camouflage frames.

**Why it helps this model specifically, and how it differs from the
already-logged InpaintNet entry.** Same target as InpaintNet — the
whole-swing 77% and 85.9% selection-ceiling numbers, not the per-frame
82% — but a meaningfully cheaper mechanism to get there. InpaintNet is a
separately trained small U-Net that must be fit on this project's own
labelled trajectory sequences before it can be used. OC-SORT needs no
training data and no training run at all: it is closed-form Kalman-filter
arithmetic (predict, associate, re-update) applied to the sequence of
boxes the existing on-device YOLO11n already produces per clip. For a
single-object, single-class problem like this one, the "multi-object"
half of OC-SORT's design (Hungarian-algorithm association across several
candidate tracks) is unnecessary complexity that can be dropped entirely —
what's left is essentially "a constant-velocity Kalman filter, plus the
specific ORU correction for handling gaps without drift," which is a
small, self-contained piece of arithmetic straightforward to port natively
into the iOS app's existing per-frame inference loop, with zero CoreML
re-export and zero change to the shipping detector. This is a strictly
lower-effort, zero-training alternative to InpaintNet for the same
recovery problem, at the cost of being a simpler, less expressive
correction (a straight-line virtual trajectory across the gap, vs.
InpaintNet's learned, potentially non-linear inpainting).

**Important honest caveats.** (1) OC-SORT's benchmark numbers are entirely
from crowded multi-object pedestrian/vehicle tracking (MOT17/20, KITTI,
DanceTrack) — none of that transfers as an accuracy estimate for a single
golf clubhead, only the *mechanism* (ORU's gap-correction) transfers by
inspection, not the reported HOTA/MOTA figures. (2) Like InpaintNet, this
degrades to open-loop extrapolation for long gaps: if the camouflage
stretch spans many consecutive frames, the "virtual straight-line
trajectory" ORU corrects onto is a worse approximation of the true
(non-linear, accelerating) clubhead path during a downswing than it would
be for the more ballistic motion OC-SORT's benchmarks are drawn from — the
same honest limitation the InpaintNet entry raised about ballistic vs.
swing motion, applying here too and for the same reason. (3) This produces
a *smoothed position estimate* for display/path purposes, not a
retroactively higher true per-frame detection rate — it should not be
conflated with an appearance-side fix, and should be framed as a UX/
selection-layer mitigation exactly like InpaintNet, not a substitute for
any already-logged appearance or architecture fix.

**Effort vs. payoff.** Low effort, plausible-but-narrow payoff — lower
effort than every other "recovery" entry logged so far because it requires
no training data, no training run, and no new model artifact of any kind,
only porting a well-documented, ~100-line Kalman-filter update rule into
the existing per-clip inference loop. A v0 (single-object constant-
velocity Kalman filter with plain SORT-style prediction, no ORU) is close
to a day's work reusing the eval harness's existing per-frame output. Full
ORU adds a small, well-specified re-update step on top of that v0 once a
detection resumes after a gap. Payoff is capped by the same ceiling
InpaintNet's entry already named honestly: this is a whole-swing/selection
-layer mitigation, not a per-frame accuracy fix, and its real value
depends on whether the app's downstream path-drawing logic benefits more
from gap-smoothing than from raw per-frame recall — unconfirmed from the
docs available. Given it is strictly cheaper than InpaintNet for
functionally the same target, the reasonable next step is to try this
before investing in a trained InpaintNet-style model, not in addition to
it as a separate effort.

---

## 2026-08-27 — JFD3: a dual-branch, feature-consistency blur-robust detector for small/fast infrared targets (AAAI 2026), real partial code but no licence and not YOLO-based (existence-with-caveats result)

**Area covered.** Motion blur -- blur-robust detection architectures. Checked
this log's existing entries in this sub-area first (DFRCP, MoSA-Det, DTUM,
TrackNetV4, channel-stacked multi-frame YOLO, RT-Focuser, Deblur-YOLO) to
avoid a repeat; JFD3 is a distinct mechanism from all of them (see below).

**What it is.** "Blur-Robust Detection via Feature Restoration: An
End-to-End Framework for Prior-Guided Infrared UAV Target Detection"
(IVPLabs, AAAI 2026, arXiv:2511.14371) proposes **JFD3** (Joint
Feature-Domain Deblurring and Detection), aimed at exactly this project's
symptom: motion blur degrading contrast between a small fast-moving target
and its background until detection fails. Mechanism, per the paper's
abstract text as surfaced consistently across independent search snippets:
a dual-branch architecture with **shared weights**, where a "clear" branch
(trained/run on sharp images) supervises a "blurred" branch via a **feature
consistency self-supervised loss**, driving the blurred branch's features
toward the clear branch's representations without needing paired
blurred/sharp ground truth at inference time. A second component, a
**frequency structure guidance module**, extracts a structure prior from a
restoration sub-network and injects it into the detector's shallow layers
to re-inject high-frequency edge information a blur kernel destroys. This
is a different mechanism from every blur entry already logged: DFRCP is a
single-frame pyramid-internal "fuzzy feature" fusion; MoSA-Det conditions
dynamic convolution on inter-frame motion state; DTUM is a direction-coded
temporal module; RT-Focuser and Deblur-YOLO both do explicit image-domain
deblurring as a separate step. JFD3's distinguishing idea is *training-time*
feature-domain supervision from a clear/blurred pair, with no explicit
deblurred image ever produced or needed at inference.

**URL.** Code: https://github.com/IVPLabs/JFD3 -- fetched directly, HTTP
200, real content (not a stub). Paper: arXiv:2511.14371 and an OpenReview
PDF mirror -- both `arxiv.org` and `openreview.net` hit this sandbox's
standing egress block (same recurring failure as `arxiv.org`,
`researchsquare.com`, `scirp.org`, `openaccess.thecvf.com` in prior
entries), so the method description above is search-snippet synthesis, not
a directly-read paper -- treat it at the same verification tier as the
DFRCP and MoSA-Det entries. The repo itself, by contrast, *was* read
directly, and is the source for everything in the next two paragraphs.

**Code and licence status (verbatim from the repo, fetched directly).**
The README states: "We have first compiled the relevant code for the
core contribution points, and the complete code is currently being
compiled." This is genuinely partial, in-progress code, not a full release.
**No LICENSE file and no licence statement of any kind is present in the
repository.** Under default copyright law that means all rights reserved --
**commercial use is NOT permitted** absent an explicit grant, and none
exists here. The repo is built on the **DEIM** codebase
(`github.com/Intellindust-AI-Lab/DEIM`) for non-core components, not on
YOLO -- porting this into the project's YOLO11n/CoreML pipeline would mean
reimplementing the two core modules (feature-consistency loss, frequency
structure guidance) against a different base detector, not dropping in a
compatible checkpoint.

**Dataset (IRBlurUAV) -- not usable for this project regardless of the code
question.** The paper introduces IRBlurUAV, described in search snippets as
30,000 simulated plus 4,118 real infrared UAV target images with motion
blur, released via a Baidu Netdisk link (the Google Drive mirror is marked
"TODO" in the README, not yet live) with no licence -- only an informal
citation request ("If you utilize this dataset in your research, please
consider citing our paper"), which is not a usage grant. Separately and
more fundamentally, it is **infrared**, not RGB -- the sensor modality
mismatch alone would rule it out as training data for a phone-camera RGB
clubhead detector even if the licence were clean.

**Which failure mode.** Motion blur only. The mechanism (restoring
blur-degraded discriminative features via a clear-image-supervised branch)
targets exactly the "small, fast, motion-blurred target loses contrast
against background" symptom this project's blur gap describes -- it does
not address camouflage's static-appearance zero-detection frames, since a
clear/blurred training pair presupposes a moving target, not a
static-looking one hidden by colour and texture alone.

**Why this cannot be logged as a usable finding yet, and what would change
that.** Three independent blockers, any one of which is disqualifying on
its own: (1) no licence on the code -- all rights reserved by default; (2)
no licence on the dataset, and the dataset is the wrong sensor modality
regardless; (3) the code is explicitly incomplete per the authors' own
README, so even a permissive licence today would not yield a working
reference implementation to port from. If the "complete code" the README
promises lands with a permissive licence, this becomes a genuine
architecture lead -- the feature-consistency mechanism is conceptually
portable to any detector (it operates on paired clear/blurred training
images and does not depend on DEIM specifically), so a future run should
re-check this exact repo URL for a licence file and a completed codebase
before reimplementing anything from the paper text alone.

**Effort vs. payoff.** Not assessable yet, same category as this log's
DFRCP and TinyDark-YOLO entries -- zero actionable effort right now because
there is nothing legally or functionally complete to build on. If it
matures (permissive licence + finished code), the effort would be
moderate-to-high: reimplementing a dual-branch training scheme requires
generating this project's own clear/blurred paired training data first
(the exact gap this log's frame-averaging, PSF, and 6-DOF blur-synthesis
entries already aim to close), so this is better read as a second
consumer of that data-synthesis work than as a standalone fix to pursue in
isolation.

---

## 2026-08-27 (second run) — Normalized Wasserstein Distance (NWD) loss: an IoU replacement for tiny/elongated box regression, verified absent from stock Ultralytics YOLO11

**Area covered.** Motion blur -- blur-robust detection architectures /
training-side fix, but at the loss-function level rather than the backbone
or head level, which is a layer this log had not yet checked. Grepped this
log's 71 prior entries for "Wasserstein," "Gaussian" (bounding-box sense),
and the paper's own repo name/author before starting: no hits. Every
architecture entry so far (JFD3, DFRCP, MoSA-Det, D-FINE, RF-DETR, YOLOV,
Temporal-YOLOv8, the P2/4-head entry, etc.) changes the network; this is
the first to check whether the *training objective itself* is well-matched
to this model's box geometry.

**What it is.** "A Normalized Gaussian Wasserstein Distance for Tiny Object
Detection" (Wang, Xu, Yang, Yu, arXiv:2110.13389, 2021; expanded journal
version in *ISPRS Journal of Photogrammetry and Remote Sensing*, 2022;
conference version at ICPR 2021 on the authors' AI-TOD benchmark). It
models each bounding box as a 2D Gaussian distribution (center = mean,
width/height = variance) and measures similarity between predicted and
ground-truth boxes via the Wasserstein distance between their Gaussians,
normalized into a bounded [0,1] similarity score (NWD) that can substitute
for IoU in label assignment, NMS, and the box regression loss.

**URL and licence, verified directly.** Official code:
https://github.com/jwwangchn/NWD — fetched directly: the repo is built on
MMDetection (not Ultralytics/YOLO), includes configs/tools/model code, and
carries an **Apache-2.0 LICENSE file** confirmed by direct fetch. Apache-2.0
permits commercial use. It is not a drop-in dependency for this project
(different framework entirely), but the loss itself is a small, closed-form
formula (compute a 2x2 covariance-style representation per box, Wasserstein
distance between two 2D Gaussians has a known closed form, then
exponential-normalize by a constant C) -- unlike a backbone or attention
module, this is realistically portable by hand into Ultralytics' own loss
code, not something that needs the MMDetection dependency itself.

**Verified this is a real gap in the exact codebase this model already
runs on, not a config flag already available.** Fetched
`ultralytics/utils/loss.py` and `ultralytics/utils/metrics.py` directly
from the `main` branch: `BboxLoss` computes box regression loss via
`bbox_iou(..., CIoU=True)` plus DFL; `bbox_iou`'s signature only exposes
`GIoU`, `DIoU`, `CIoU` boolean flags. There is no NWD, no SIoU, no WIoU, and
no Wasserstein/Gaussian code path anywhere in either file -- confirmed by
reading the actual current source, not inferred from a search-engine
summary (an earlier search this run had claimed YOLO11 already ships
CIoU/SIoU together, which the direct source fetch shows is wrong -- only
GIoU/DIoU/CIoU exist). No maintained, licence-clear NWD patch for
Ultralytics YOLOv8/v11 turned up either -- what exists are Chinese-language
tutorial blog posts (CSDN) walking through a manual `BboxLoss` patch, not a
shipped package -- so this would be a from-scratch implementation against
the NWD paper's formula and the official MMDetection repo as a reference,
not a fork-and-pull.

**Which failure mode it addresses.** Motion blur, specifically the box-shape
problem this log has been assembling data-synthesis fixes for since
2026-08-12 (frame-averaging), 2026-08-14 (PSF-based box expansion), and
2026-08-25 (6-DOF camera-motion blur synthesis) -- all three manufacture
correctly elongated blur-streak training boxes, but none of them touch
*how the loss reacts* to those boxes once they exist. IoU-based losses are
known to be disproportionately unstable for small absolute box sizes (a
few-pixel positional error collapses IoU much faster than the same error on
a large box), and that instability compounds with elongation: for a fixed
pixel offset, a thin/long box loses IoU faster than a square box of equal
area because the offset eats a larger fraction of the box's short axis.
The brief's own numbers describe exactly this shape -- median labelled-box
elongation 1.60, p90 3.01, i.e. genuinely elongated boxes already exist in
the tail of the current 1,308-frame set, before any of the synthesis
entries above even add more. If CIoU's regression signal degrades sharply
on that elongated tail today, training may already be quietly
under-weighting or destabilizing on exactly the examples that matter most
for the blur failure mode, independent of how much more blurred data gets
added. This is orthogonal to camouflage: NWD does not help a network find a
target with zero signal at any confidence threshold, it only changes how
cleanly the network can learn box *shape* once some detection signal
exists.

**Why it helps this model specifically.** It is a training-time-only change
-- the loss function does not touch the model graph, so it carries zero
CoreML export risk, unlike almost every architecture entry in this log
(YOLOV/YOLOX-based, D-FINE/RF-DETR transformer-based, JFD3 DEIM-based, etc.,
all flagged with export-path uncertainty). It is also the natural
complement to this log's own already-logged blur-synthesis entries: those
manufacture more elongated boxes; NWD (or a blended NWD+CIoU loss, which
the original paper itself recommends -- keep IoU for well-sized objects,
blend in NWD for tiny ones) is what would let the model actually learn from
that data cleanly, rather than fighting a loss metric that penalizes
elongated tiny boxes harshest exactly where the training signal is already
sparsest.

**Effort vs. payoff.** Low-moderate effort: this is a self-contained
addition to `BboxLoss`/`bbox_iou` inside the project's own installed
Ultralytics copy (a few dozen lines -- Gaussian-encode a box, compute the
closed-form 2D Wasserstein distance, normalize, blend with existing CIoU by
some ratio per the paper's own ablation), verifiable via reading the
official MMDetection repo's loss implementation as a reference and unit-
testing against known box pairs before touching training. No new data, no
new dependency, no export changes. Payoff: genuinely unverified for this
model -- the NWD paper's own reported gains are on aerial-imagery tiny
objects (AI-TOD), not golf clubheads, and the mechanism has not been tested
here. It is, however, the cheapest and lowest-risk item in this log's
entire motion-blur bucket to try, and the only one that requires touching
neither the architecture nor the dataset -- a reasonable first experiment
would be a single ablation training run (stock CIoU vs. blended NWD+CIoU,
same data, same eval harness) before investing in any of the
heavier architecture or data-synthesis entries already logged.

---

## 2026-08-27 (third run) — Albumentations' `MotionBlur` transform is the cheapest way to inject directional blur into the existing Ultralytics pipeline, but its actively-maintained successor is now AGPL, not MIT

**Area covered.** Ways to synthesise/augment training data for the blur
failure mode, at the training-pipeline level rather than the dataset-
capture or architecture level -- distinct from this log's existing
blur-synthesis entries (frame-averaging 2026-08-12, PSF-based box expansion
2026-08-14, 6-DOF camera blur 2026-08-25, BlenderProc 2026-08-18), all of
which manufacture new source imagery/video offline. This is instead an
in-training augmentation already one `pip install` away from the exact
framework this project uses (Ultralytics YOLO11n), which none of those
entries checked.

**What it is.** Ultralytics' `ultralytics/data/augment.py` has a built-in
`Albumentations` wrapper class that auto-activates during training if the
`albumentations` package is importable -- confirmed by fetching the file
directly from the `ultralytics/ultralytics` `main` branch. Its default
transform set is `A.Blur(p=0.01)`, `A.MedianBlur(p=0.01)`, `A.ToGray(p=0.01)`,
`A.CLAHE(p=0.01)`, plus three transforms disabled at `p=0.0`
(`RandomBrightnessContrast`, `RandomGamma`, `ImageCompression`) -- all
isotropic/generic, none directional. The class's `__init__` signature is
`(p=1.0, transforms: list | None = None, flip_idx=None)`: passing a custom
`transforms` list is an officially supported, documented override, not a
monkey-patch, and this is exactly where `A.MotionBlur(blur_limit=..., p=...)`
(a directional-kernel convolution simulating camera-shake/subject-motion
streaks) would go. This is the same mechanism the Albumentations project's
own `example-ultralytics` doc demonstrates.

**URL and licence, verified directly, including a real trap.** Two
different packages exist and are easy to conflate:
- `albumentations` (PyPI, `github.com/albumentations-team/albumentations`):
  MIT licence, confirmed via the PyPI JSON API (`license: MIT License`,
  latest version `2.0.8`) and via the repo's own `LICENSE` file (MIT,
  copyright Iglovikov/Buslaev/Parinov). The GitHub README states, fetched
  directly: **"This repository is no longer actively maintained. The last
  update was in June 2025, and no further bug fixes, features, or
  compatibility updates will be provided."** It still installs and works
  today (Ultralytics only requires `albumentations>=1.0.3`, checked via
  `check_version` in the same source file) -- it is frozen, not broken.
- `AlbumentationsX` (PyPI package name `albumentationsx`,
  `github.com/albumentations-team/AlbumentationsX`): the actively developed
  successor, advertised as a "100% drop-in replacement" with the same API.
  Its licence is **dual AGPL-3.0 / commercial**, confirmed via the fork's
  own blog post ("AlbumentationsX: A Fork with Dual Licensing") and its
  GitHub `LICENSE` file. Its own docs are explicit: "If your project uses
  MIT, Apache 2.0, or BSD licenses -- even if it's open source -- you
  cannot use AlbumentationsX under AGPL and you need a commercial licence."
  AGPL's network-service copyleft clause is a real risk for a shipped app
  build pipeline, not just a server.

**Which failure mode it addresses.** Motion blur. It does not touch
camouflage at all -- flagging that explicitly since this run's brief asks
for an honest per-mode read, and there is no plausible mechanism by which a
blur transform helps a zero-detection-confidence camouflage failure.

**Why it helps this model specifically, and the real caveat.** The
labeling spec already instructs annotators to box the full motion-blur
streak, but the brief's own elongation stats (median 1.60, p90 3.01) show
that instruction is rarely exercised because genuinely blurred training
examples are scarce -- this is a pipeline-level lever to manufacture more
of them for free, with zero new dependency risk (the MIT-frozen package is
enough; AlbumentationsX is not needed) and zero architecture/export risk
(training-time only, same as the NWD entry logged just above this one).
**But**: `A.MotionBlur` convolves a directional kernel over the *entire*
frame uniformly -- it does not selectively blur only the clubhead region.
For clubhead boxes, which are small in absolute pixel terms, even a modest
kernel (Albumentations' default `blur_limit` range is roughly 3-7px) could
smear the head's true extent past its original tight label box, silently
reintroducing the exact box/visual-extent mismatch this log's PSF and
frame-averaging entries were designed to fix correctly -- global blur is a
cruder, unlabelled approximation of what those entries do with correct box
expansion. This needs a visual sanity pass (render augmented samples,
check the label box still tightly bounds the now-blurred head at the kernel
sizes actually used) before trusting it, not blind enablement.

**Effort vs. payoff.** Very low effort: no new dataset, no architecture
change, no export change -- a `YOLODataset` subclass overriding
`build_transforms` to construct `Albumentations(transforms=[..., A.MotionBlur(...), ...])`
is a few lines, using the already-MIT-licensed `albumentations==2.0.8`
pin (do not let this or any future entry casually recommend "upgrading
albumentations," since the currently-promoted upgrade path is the AGPL
fork). Payoff is unverified and likely modest on its own: it is a coarse,
whole-frame proxy for real per-object blur, not a substitute for the
correctly-elongated synthetic examples this log has already logged, but it
is cheap enough to run as a quick ablation (default augmentations vs.
+MotionBlur, same data/eval harness, visually spot-checked) before or
alongside the heavier data-synthesis entries.

## 2026-08-27 (fourth run) — `VNDetectTrajectoriesRequest`: Apple's own on-device parabolic-motion tracker, and a verified reason it likely does not fit a golf swing

Rotating out of the motion-blur-heavy run of architecture/augmentation entries
(JFD3, NWD, Albumentations MotionBlur, all logged earlier today) into the
temporal/multi-frame category, since the brief flags motion as the thing
that separates a moving clubhead from static camouflage when appearance
can't. This is a different kind of entry: not a paper or dataset, but a
first-party iOS API already sitting in every build of this app, checked
directly against its own stated constraints rather than assumed to fit.

**What it is.** `VNDetectTrajectoriesRequest`, part of Apple's Vision
framework since iOS 14, introduced at WWDC20 session 10099 ("Explore the
Action & Vision app"). It runs on a live `AVFoundation` frame stream,
tracks a moving shape's centroid across frames, and fits the path to a
model the framework calls a parabola. It exposes `frameAnalysisSpacing`
(how often to run), `trajectoryLength` (how many points to require before
accepting a trajectory), and `minimumObjectSize`/`maximumObjectSize` (to
filter noise by expected object scale). It is on-device, real-time, and
ships in the OS -- zero training data, zero export step, zero new
dependency.

**URL.** https://developer.apple.com/documentation/vision/vndetecttrajectoriesrequest
and https://developer.apple.com/videos/play/wwdc2020/10099/ (WWDC session,
fetched directly for the transcript quotes below -- the reference doc page
itself is a JS-rendered SPA that would not return body text through this
sandbox's fetch tool).

**Licence.** Not applicable in the usual sense: this is part of the iOS
SDK covered by the standard Apple Developer Program / Xcode license, the
same terms already covering every other Vision/CoreML API this app uses.
No separate licence risk, no AGPL-style trap to check.

**Which failure mode it addresses, and the verified reason to be skeptical.**
Aimed at both (temporal recovery for camouflage zero-detections; a possible
cross-check for blur-degraded frames), but the WWDC transcript itself
states the hard constraint, quoted directly: *"Objects have to travel on
some kind of a parabola. Now, a straight line is a parabola. That allows us
to filter out spurious movements."* A golf clubhead's path through the
backswing and downswing is a circular arc pivoting around the shoulders --
centripetal, not the constant-one-directional-acceleration motion a
parabola models -- and a follow-up search for prior art applying this API
to swung/rotational sports objects (golf, tennis) returned nothing: no
blog post, sample project, or paper describes anyone using it for a swing
rather than a thrown or kicked object (the WWDC demo itself is a thrown
bean bag). The transcript's second constraint, *"It requires a stable
scene... the phone stabilized on a tripod or otherwise fixated,"* is
probably already satisfied by this app's typical down-the-line/face-on
static camera placement (per `eval/test_set_sources.md`), so that part is
not the blocker -- the motion-model mismatch is.

**Why it might still be worth a look.** The one place a short window of
the swing plausibly *does* look locally straight-ish is the moment right
around impact, which is also exactly where blur and phantom-detection
failures cluster -- if `trajectoryLength` is set very short, the fit only
needs local linearity, not a true global parabola. But this is a guess,
not a finding: nothing found here confirms it works, and Apple's own
framing (ballistic objects, not rotational ones) argues against it holding
up outside that narrow window.

**Effort vs. payoff.** Effort to find out is very low -- add the request
alongside the existing CoreML inference on a handful of already-recorded
swing clips (no new data, no training, no export) and see whether any
accepted trajectory actually tracks the clubhead near impact, or whether
it never fires / fires on the wrong object. Payoff is speculative and,
given the absence of any prior art for rotational motion, more likely
negative than positive -- this is a half-day spike to rule in or out, not
a plan to build on yet.

---

## 2026-08-28 — Closing the 2026-08-17 dead end: "One-Shot Badminton Shuttle Detection for Mobile Robots" now has a public repo, and it is a negative result (AGPL code, no working dataset link)

**What it is.** The 2026-08-17 (third-run) entry flagged this paper
(arXiv:2603.06691v2) as a promising-sounding lead — a 20,510-frame,
11-background indoor/outdoor shuttlecock dataset explicitly stratified by
difficulty, with a badminton-analogue of both this project's failure modes
(small fast object, motion blur mentioned in the abstract) — but could not
find any code or dataset host and explicitly declined to log it. This run
found the code repo (`leggedrobotics/shuttle_detection` on GitHub, fetched
directly) and closes that open thread: it is real and live, but the
dataset itself is not actually obtainable, and the code's licence would be
a problem even if it were.

**URL.** https://github.com/leggedrobotics/shuttle_detection (paper:
arXiv:2603.06691v2, still unreadable directly — arxiv.org is blocked by
this sandbox's egress proxy, the same standing restriction every prior run
has hit; everything below is sourced from the GitHub repo itself, fetched
directly, not the paper).

**Licence — verbatim, from the repo's own `LICENSE` file (fetched via
`raw.githubusercontent.com`, not just the README's claim).** GNU Affero
General Public License, Version 3, 19 November 2007. The README states the
reason directly: "Due to the use of Ultralytics YOLOv8, this project is
licensed under GNU AFFERO GENERAL PUBLIC LICENSE v3.0." **Commercial use
is not flatly prohibited, but AGPL-3.0 is copyleft**: shipping a derivative
of this code (or of Ultralytics YOLOv8 itself, which carries the same
licence) inside a closed-source iOS app would obligate releasing the
app's corresponding source. This is the same licence-shape problem this
log already flagged for AlbumentationsX (2026-08-27, third run) and is
disqualifying for direct reuse in this project without a separate
commercial Ultralytics licence.

**Dataset availability — checked and it does not exist today.** The
README's dataset section reads: "Download the dataset from
[here](https://example.com) to your desired `<DATASET_DIR>/processed`" —
the link is a literal `example.com` placeholder, not a real host. The
repo's Releases page was also checked directly and returns "There aren't
any releases here." So the 20,510-frame dataset the abstract describes is
not downloadable from the one place a reader would expect to find it, and
no mirror, Hugging Face/Kaggle/Zenodo listing, or alternate link was found
by search. This is a stronger and more specific version of the 08-17
entry's "no dataset host found" — it's not that no one has looked, it's
that the authors' own repo ships a dead placeholder.

**Which failure mode.** Motion blur and camouflage, in principle (indoor
badminton is a small, fast, occasionally low-contrast object against
variable backgrounds) — but moot, since nothing here is usable.

**Why it helps this model specifically.** It doesn't, beyond closing an
open question cheaply. The abstract's framing (indoor/outdoor split,
explicit difficulty stratification, semi-automatic annotation from
stationary camera footage) is exactly the shape of dataset this project's
biggest gap (real indoor/low-light footage with genuine motion blur) needs
an analogue of — which is why the 08-17 entry flagged it — but a dataset
that cannot be downloaded and code under a licence incompatible with a
closed-source app contribute nothing actionable. The one reusable idea is
the paper's own method, not its artifacts: "semi-automatic annotation from
stationary camera footage" (per the abstract, not independently verified
beyond the phrase itself) is a labeling-cost technique, not a licensable
asset, so it isn't logged here as a separate finding — it would need its
own verification pass against the actual paper text, which remains
unreadable from this sandbox.

**Effort vs. payoff.** Effort spent: low (one repo fetch, one releases-page
check, one LICENSE fetch — all direct, no speculation). Payoff: zero as a
usable asset, but real as log hygiene — it prevents a future run from
re-discovering the same GitHub repo, getting excited about the abstract
again, and re-spending a cycle before hitting the same placeholder link
and the same AGPL wall. Recommendation: do not pursue this further unless
the actual arXiv PDF becomes reachable and reveals a real dataset link the
GitHub repo omits; low prior on that changing the licence problem either
way, since AGPL is a repo-level declaration independent of where the data
ends up hosted.

---

## 2026-08-28 (second run) — PiTrac: real open-source golf launch monitor, but a separately-licensed non-commercial YOLO model and, as far as verified, ball-only rather than club-head

**What it is.** `github.com/PiTracLM/PiTrac` (moved from the original
`jamespilgrim/PiTrac`), an actively developed, real open-source DIY golf
launch monitor built on Raspberry Pi + camera hardware, independently
documented on Hackaday.io and at docs.pitrac.org. It determines ball
launch speed, angle, and spin from camera images. Chosen for this rotation
(golf-specific pose/club-tracking open-source implementations) because it
is a working, shipping computer-vision pipeline aimed at fast-moving golf
equipment — a different category from the pose-estimation papers (GolfPose,
CADDIE) and the dataset-only projects (GolfDB, CaddieSet) already logged.

**Verification performed.** Fetched the GitHub repo listing directly (not
search snippets): confirmed a real, current, non-stub repo with `LICENSE`,
`LICENSE.MODEL.md`, `LICENSE.ED_LIB.md`, `LICENSE.RPICAM-APPS.md`,
`LICENSE.SHEDSKIN.md` at the root plus `Software/`, `Hardware/`, and
`3D Printed Parts/` directories. Fetched `LICENSE` and `LICENSE.MODEL.md`
verbatim via `raw.githubusercontent.com`. `docs.pitrac.org` and
`hackaday.io` both hit this sandbox's standing egress block (same failure
mode noted for other domains throughout this log), and the `Software/`
subtree did not render enough detail through this sandbox's fetch tool to
inspect the model's class names directly — those two gaps are called out
below rather than papered over.

**Licence — verified; two different licences cover two different things,
and both rule this out.** Code (`LICENSE`): GNU General Public License
Version 2, June 1991 — copyleft, the same commercial-incompatibility shape
already flagged in this log for AGPL projects (the 2026-08-17 badminton
entry, the 2026-08-27 AlbumentationsX entry). Model weights
(`LICENSE.MODEL.md`), separate and explicitly proprietary: it covers
"Model Materials" — trained weights in `.onnx`, `.pt`, `.pth`, `.engine`,
`.tflite`, `.bin`, `.param`, `.safetensors` formats — and explicitly names
"all YOLO-based object detection model weights and ncnn model files
distributed with PiTrac." It grants only a "limited, non-exclusive,
non-transferable, non-sublicensable, revocable, royalty-free license" for
personal, non-commercial "Authorized Use" of the assembled PiTrac system.
Section 3(f) states verbatim that licensees shall NOT "Use the Model
Materials in any commercial product or service... without prior written
permission from PiTracLM," and the same section separately forbids
redistribution, extraction from the PiTrac software, and any derivative
work including "fine-tuning" or "quantization." So even setting the GPL
code licence aside, the model weights carry a harder, more specific block
than ordinary copyleft: not just "your app must also be GPL," but "you may
not fine-tune, quantize, or redistribute this model at all, for any use
outside running stock PiTrac."

**Which failure mode.** Neither directly — like the GolfDB/CaddieSet/
badminton entries, this is a dataset/model-reuse question, not a technique.
Checked because PiTrac is a real, shipping YOLO-based CV system pointed at
fast-moving golf equipment, i.e. exactly the kind of adjacent project that
might have already solved, or at least captured training data for, this
project's motion-blur-at-impact-speed problem.

**Why it helps this model specifically.** It doesn't, as a source of
weights or data — commercial use is explicitly forbidden, and
fine-tuning/extraction are explicitly forbidden even under the personal-use
licence. It is also very likely the wrong target object: every
independent description of PiTrac checked (its own repo description, and
the search-surfaced framing of its Hackaday page and docs site) describes
ball launch speed/angle/spin only; club-head tracking is never mentioned in
any of them, consistent with the 2026-08-18 CaddieSet entry's finding that
launch-monitor systems are typically ball-focused, with any club data
coming from a separate swing sensor rather than vision. This run could not
positively confirm the YOLO model's class list excludes the club (the
`Software/LMSourceCode` tree did not render through this sandbox's fetch
tool), so "very likely wrong domain" is stated as a strong inference from
three independent descriptions, not a confirmed fact. Either way, the
licence terms make the domain question moot for this project.

**Effort vs. payoff.** Low-moderate effort: three direct fetches (repo
root, `LICENSE`, `LICENSE.MODEL.md`), two blocked/unreadable follow-ups
(`docs.pitrac.org`, the `Software/` subtree) called out rather than
guessed around. Payoff: zero as a data or model source, but real as log
hygiene — it closes off "adjacent open-source golf CV hardware projects"
as a place to look for reusable YOLO weights or club imagery, and it
surfaces a licensing pattern this log has not seen before (a separate,
stricter, explicitly anti-fine-tuning "Model Materials" licence layered on
top of GPL code) worth watching for before spending another cycle on
similar camera-based launch-monitor projects (GSPro/E6-compatible units,
etc.).

---

## 2026-08-28 (third run) — `VNGenerateOpticalFlowRequest`: Apple's own on-device dense optical-flow API, and the specific reason it does not share the previous entry's motion-model problem

Rotating back into the temporal/multi-frame category (bullet 3) rather than
a fourth dataset or golf-tracking run today (badminton close-out and PiTrac
already covered those areas this run-cycle). This is a direct follow-up to
this log's most recent entry (2026-08-27 fourth run,
`VNDetectTrajectoriesRequest`), which was ruled out mainly because Vision's
trajectory tracker fits a *parabola* to the moving object's path and a golf
swing is circular/centripetal, not ballistic. Before discarding "first-party
Vision motion APIs" as a category on one data point, this run checked
whether the *other* first-party Vision motion API has the same problem.

**What it is.** `VNGenerateOpticalFlowRequest`, also part of Apple's Vision
framework since iOS 14, introduced in the same WWDC20 cycle as the
trajectory API but in a different session (session 10673, "Explore Computer
Vision APIs" — verified by direct transcript fetch, not search-indexed).
Unlike the trajectory request, it does not track a shape or fit any motion
model at all: it computes **dense, per-pixel optical flow** between two
frames and returns a `VNPixelBufferObservation` — a floating-point image
with interleaved X/Y displacement for every pixel. Quoted directly from the
session transcript: *"Optical Flow... gives me a per pixel flow between X
and Y."* The transcript also explains why this is a meaningfully different
tool than simple frame-differencing or global image registration: *"The
registration will give me the alignment between the two images by telling
me how much the image has moved up and to the right. But I can use the
Optical Flow, because it's going to tell me, for each pixel, how they have
moved"* — i.e. it separates camera motion from independent object motion at
the pixel level, which is exactly what a static down-the-line/face-on
camera (this app's typical setup, per `eval/test_set_sources.md`) needs to
isolate a moving clubhead from a static camouflaged background. A second
revision, `VNGenerateOpticalFlowRequestRevision2`, is documented (per
Apple's top-level API reference page, fetched directly) to use "modern
machine learning under the hood" instead of a classical algorithm, and a
`ComputationAccuracy` option exists to trade speed for precision — but this
run could not fetch that sub-page's actual content (same JS-rendered-SPA
limitation the previous entry hit on its own reference sub-pages) or the
revision-1-vs-2 comparison page, so which revision or accuracy level is
appropriate for real-time on-device use is **not verified**, only known to
be configurable.

**URL.** https://developer.apple.com/documentation/vision/vngenerateopticalflowrequest
(top-level page content fetched directly) and
https://developer.apple.com/videos/play/wwdc2020/10673/ (WWDC session
transcript, fetched directly for the quotes above).

**Licence.** Not applicable in the usual sense, same as the prior Vision-API
entry: part of the iOS SDK under the standard Apple Developer Program /
Xcode license already covering this app's other Vision/CoreML use. No
separate licence risk.

**Which failure mode it addresses.** Primarily camouflage — the zero-candidate-detection
failure. The proposed use is as a fallback trigger, not a replacement for
the CoreML detector: when the appearance-based YOLO model returns no boxes
above threshold on a frame where its neighbours did fire, run optical flow
between that frame and an adjacent one, and treat the pixel region with flow
magnitude/direction inconsistent with the (near-zero, static-camera)
background flow as a candidate ROI — either fed back into the detector as a
crop/zoom or used directly as a bounding region. This is a plausible fix
for exactly the camouflage cases described in the brief (dark clubhead
against dark clothing or foliage, sharp frame, zero detections even at
confidence 0.05) because it does not depend on appearance contrast at all,
only on the clubhead moving relative to a static background — and unlike
the previously-logged trajectory API, it makes no assumption about the
*shape* of that motion (no parabola-fitting, no minimum trajectory length),
so the circular-swing objection that ruled out `VNDetectTrajectoriesRequest`
does not apply here. Secondary, unverified relevance to motion blur: the
magnitude of per-pixel flow could in principle help flag which frames/regions
are blur-streak candidates for labeling QA (similar in spirit to the
already-logged DeFMO entry), but this run found no prior art applying
Vision's optical flow to that purpose and is not claiming it as a finding.

**Effort vs. payoff.** Low-to-moderate effort to try, higher and less
certain effort to productionize. Effort to test: identical shape to the
prior entry's spike — add the request alongside existing CoreML inference
over a handful of already-recorded swing clips, run it only on frames where
the detector abstains, and visually check whether the flagged high-flow
region actually sits on the clubhead or on something else moving (arms,
shirt fabric, background foliage in wind — all plausible false triggers
for a pure-motion signal with no appearance check). That confound is the
main verified risk: dense optical flow will happily flag *any* moving
pixel, and this run found no evidence one way or the other about how
cleanly a swinging clubhead's flow separates from a swinging body's flow at
phone-camera resolution and frame rate. Payoff, if it does separate
cleanly: a zero-training, zero-export, on-device rescue path for exactly
the failure class the brief calls out as producing zero candidates even at
very low confidence — something no architecture change in this log fixes,
since architecture changes still operate on single-frame appearance.
Unverified and not to be assumed: real-time performance of dense per-pixel
flow at this app's target frame rate/resolution, and which
`ComputationAccuracy`/revision setting that requires — both sub-pages
needed to answer that were not fetchable from this sandbox.

---

## 2026-08-28 (fourth run) — YOLO11-OBB: native oriented-box detection, plus a geometric explanation for why axis-aligned boxes understate blur elongation

**Area covered.** Motion blur, architecture/box-representation layer.
Grepped this log's 88 prior entries for "OBB" and "oriented" before
starting: no hits. Every prior architecture entry (JFD3, DFRCP, MoSA-Det,
D-FINE, RF-DETR, YOLOV, Temporal-YOLOv8, the P2/4-head entry, NWD loss)
changes the network or the loss while keeping the box representation
itself axis-aligned. This is the first entry to question the box
representation.

**What it is.** Oriented Bounding Box (OBB) detection is a standard
Ultralytics YOLO task (alongside detect/segment/pose/classify) that
predicts a rotated rectangle — center, width, height, plus an angle — instead
of an axis-aligned box. It ships in the same repo this project's YOLO11n
already comes from, with official pretrained nano weights
(`yolo11n-obb.pt`, 2.7M params, pretrained on DOTAv1) confirmed present in
`ultralytics/docs/en/models/yolo11.md` (fetched directly from
`raw.githubusercontent.com/ultralytics/ultralytics/main`, HTTP 200). Label
format is four corner points per box (`class x1 y1 x2 y2 x3 y3 x4 y4`),
internally represented as `xywhr`. OBB is Ultralytics' standard answer for
elongated objects photographed at arbitrary angles (its usual use cases are
ships, aerial imagery, rotated text) — a description that fits a
motion-blur streak more literally than it fits this project's current
axis-aligned clubhead box.

**Why an axis-aligned box is the wrong representation for a blur streak —
verified by direct derivation, not by a citation.** For a true streak of
length L and width w rotated by angle θ from horizontal, its axis-aligned
bounding box has width = L·cosθ + w·sinθ and height = L·sinθ + w·cosθ (both
terms standard trig for a rotated rectangle's AABB). At θ=0° or 90° this
recovers the true L/w elongation. At θ=45°, width = height = (L+w)/√2 —
**the measured aspect ratio collapses to exactly 1.0 regardless of how
elongated the real streak is.** So an annotator who follows
`labeling-spec.md`'s rule perfectly (box the *full* visible streak) will
still produce a near-square axis-aligned box whenever the streak's
image-plane angle is near 45°, purely from AABB geometry — no under-boxing
required. This is a plausible, previously unstated third explanation for
the brief's median-elongation-1.60 finding, alongside "annotators are
under-drawing" and "blur genuinely is minimal in this test set." It is
**not verified against this project's own data** — the per-frame streak-angle
distribution needed to confirm it was not available to this research task —
but the geometry itself is exact and independently checkable, and a golf
downswing sweeps through a wide range of image-plane angles around impact,
so some fraction of frames landing near the collapse angle is expected on
priors alone.

**URL, verified directly.** Task docs:
`raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/tasks/obb.md`
(HTTP 200) — confirms the label format and that `yolo26n-obb.pt` (the
YOLO26 generation) is the currently-documented default, with YOLO11's own
`yolo11n-obb.pt` still listed on the YOLO11 model page. CoreML/OBB
interaction confirmed by reading
`ultralytics/engine/exporter.py` directly (HTTP 200, `main` branch, not
inferred from a blog post): CoreML export itself does not exclude the OBB
task, but the exporter's own warning states `"'nms=True' is only supported
for detect, segment and pose models. Forcing 'nms=False'."` — i.e. **OBB
models cannot use Ultralytics' embedded CoreML NMS pipeline.** The `NMSModel`
wrapper class explicitly branches on `model.task == 'obb'` with different
(8x) box-decoding multiplier logic, confirming OBB decoding is
structurally different from the detect-task path this project's current
CoreML pipeline presumably relies on.

**Licence.** Same repository, same **AGPL-3.0** licence as the YOLO11
detect task this project already trains against (verified from
`ultralytics/LICENSE` on `main`, HTTP 200) — OBB is not a separately
licensed feature, so this introduces no new licensing question beyond
whatever terms already govern this project's use of Ultralytics YOLO11.

**Which failure mode.** Motion blur only. OBB changes how a *detected*
box is shaped and oriented; it does nothing for the camouflage failure
mode's actual problem (zero candidate detections at any confidence), and
is a different, complementary layer to the already-logged BlurBall entry
(2026-08-16 — an explicit center+length+orientation heatmap auxiliary
head, custom architecture surgery) and NWD-loss entry (2026-08-27 — a
better IoU-replacement loss that still assumes an axis-aligned box).

**Why it helps this model specifically.** Two independent benefits, if the
angle hypothesis above holds even partially: (1) it would let the model
and any downstream loss see the streak's *true* elongation and orientation
instead of an angle-collapsed AABB, which the brief's own p90-of-3.01
figure suggests is already being lost for streaks that happen to fall near
axis-aligned angles too; (2) box orientation is itself a free, physically
meaningful signal (the swing plane at that instant) that the current
axis-aligned pipeline discards entirely — a byproduct with no
extra-labeling cost beyond what OBB annotation itself requires.

**Effort vs. payoff.** High effort, uncertain but structurally sound
payoff. Effort: every training image needs re-annotation in the OBB corner
format (this is not a schema tweak on top of existing axis-aligned labels —
a real rotation angle has to be drawn or recovered, and the existing
54%-Roboflow-sourced boxes almost certainly are not recoverable to OBB
without re-drawing from the original frames); the on-device iOS pipeline
loses Ultralytics' built-in CoreML NMS and needs custom rotated-box
decoding and NMS written by hand, a real engineering cost the confirmed
`exporter.py` warning makes concrete, not hypothetical. Payoff: the
per-streak-angle elongation-collapse mechanism is exact math, not a
guess, but its *practical* contribution to the measured 1.60 median is
unverified — it could be a large fraction of the gap or a minor one,
and this run had no access to per-frame streak-angle data to tell which.
Recommended next step before committing to a relabel is cheap and
falsifiable: from the existing axis-aligned dataset, estimate streak angle
per blurred box (e.g. via a quick optical-flow or gradient-orientation
pass over the labeled crop) and check whether low-elongation boxes cluster
near 45° — that single analysis, not a relabel, is what should decide
whether this entry's hypothesis is worth acting on.

---

## 2026-08-29 — MS-YOLOv11: wavelet/frequency-domain small-object detection, a structurally new (non-appearance, non-motion) mechanism, but with an unverifiable, effectively empty code release

**Area covered.** Bullet 3 (small/low-contrast/camouflaged object detection).
Grepped this log's 89 prior entries for "wavelet", "frequency domain", and
"Fourier" before starting: no hits. Every prior camouflage-adjacent entry in
this log (TrackNetV4, DTUM, SLT-Net, SINet-V2, Motion-Informed Enhancement,
CamDiff, SAM-PM, TOTNet, CAMotion, EMIP, channel-stacked YOLO) attacks the
zero-candidate-detection problem via appearance features, motion/optical
flow, or generative data synthesis. This is the first entry to attack it via
a frequency-domain decomposition of the image instead.

**What it is.** "MS-YOLOv11: A Wavelet-Enhanced Multi-Scale Network for
Small Object Detection in Remote Sensing Images" (Sensors 2025, 25(19),
6008; authors credited to Space Engineering University, Beijing, per
search-indexed abstract). It modifies a YOLOv11 backbone — the same model
family this project's detector is built on — with three additions: (1) a 2D
Haar wavelet decomposition that splits feature maps into frequency
sub-bands to explicitly preserve high-frequency edge/texture detail that
plain convolutional downsampling loses; (2) a lightweight receptive-field
expansion module; (3) adaptive cross-scale feature fusion. Reported mAP50 on
four public remote-sensing benchmarks (DIOR, TGRS-HRRSD, RSOD, NWPU-VHR-10)
ranges 89.0–99.4%, beating baseline YOLOv11 and several other SOTA
detectors on those benchmarks — but note these are aerial/satellite imagery
datasets, not sports or small dark-object-on-dark-background scenes, so the
numbers themselves don't transfer as evidence for this project.

**Verification, and where it stopped.** `arxiv.org`, `doi.org`,
`www.mdpi.com`, and `www.ncbi.nlm.nih.gov` (PMC) are all blocked by this
sandbox's egress proxy, so the paper's full text, methods section, and any
data/code-availability statement could not be read directly — existence and
the summary above rest on convergent search-indexed abstracts from PMC,
NASA ADS, and ResearchGate (consistent title, journal, DOI, and author
affiliation across all three, which is the same standard this log has used
for other unreachable-full-text entries, e.g. iPhoneBlur's initial pass and
the ReynoldsFlow entry). The claimed code release,
`github.com/Axuewu/xuewu`, **was** directly checked via the GitHub API
(`api.github.com/repos/Axuewu/xuewu`, HTTP 200): the repo is 2KB, created
and last pushed on the same day (2026-04-21), `license: null`, and its only
content is a single directory named "main code" that the API's own contents
listing shows as empty. This is not a working implementation — it is, at
best, a placeholder the paper's authors linked and never populated.

**Licence.** None. GitHub reports `license: null` for `Axuewu/xuewu`
directly from the API, and there is no separate LICENSE file to inspect
(the repo has no other content to check). With no license and effectively
no code, there is nothing here to use commercially or otherwise — this is
not a "restrictive license" case, it's a "nothing was actually shipped"
case, same shape as the 2026-08-18 TinyDark-YOLO and 2026-08-24 DFRCP
existence-only entries.

**Which failure mode.** Camouflage / low-contrast small-object detection,
specifically — this is a still-frame, appearance-side (frequency-domain,
not motion-domain) mechanism, structurally unrelated to the elongation/blur
problem. Not motion blur: nothing in the indexed abstract claims blur
robustness, and this entry should not be conflated with the separate,
UAV-survey framing (from this run's own search results, not from this
paper) that mentions motion blur as a generic UAV-imagery nuisance — that
claim belongs to a different paper and is not being carried over here.

**Why it helps this model specifically, in principle.** A dark clubhead
against dark clothing or cluttered foliage is, by definition, a
low-frequency-contrast problem in the RGB domain even when it's perfectly
in focus — exactly the case the brief describes as "visually sharp frames"
producing zero detections. A frequency-domain path that explicitly
preserves high-frequency edge information before it gets smoothed away by
ordinary strided convolution is a plausible, mechanically distinct
complement to the eight-plus motion/appearance mechanisms already logged.
It is also architecturally cheap to reason about in the abstract: it would
insert into or ahead of the backbone of the same YOLO11n this project
already trains, not require a video pipeline or auxiliary heads.

**Effort vs. payoff.** Currently near-zero payoff, because there is nothing
usable to spend effort on: the one code release found is empty and
unlicensed, and the paper's actual implementation details (kernel sizes,
where in the backbone the wavelet module is inserted, training recipe)
could not be read past the abstract due to this sandbox's blocked access to
arxiv/doi/mdpi/PMC. If this idea is worth pursuing, the correct next step is
not to build against `Axuewu/xuewu` — it's to get the actual Sensors PDF
(reachable from a normal network, unlike this sandbox) and check whether the
method is describable well enough from the paper alone to reimplement the
wavelet module as a drop-in addition to the existing Ultralytics YOLO11n
training config, the same way this log's NWD-loss and P2/4-head entries
were scoped as concrete, buildable changes. Until that reading happens, this
entry should be treated as a lead, not an actionable recommendation.

---

## 2026-08-29 (second run) — LOL-Blur / Real-LOL-Blur: a real, verified dataset that pairs low light AND motion blur in the same frames — confirms the joint regime exists, but is non-commercial (S-Lab License 1.0)

**Area covered.** Bullet 1 (commercial-licensable footage, especially low
light / indoor / older phones, where real motion blur appears). This log
has separately logged low-light datasets (Zero-DCE, DEN) and blur datasets
(RealBlur, SloMoBlur, GoPro/REDS, iPhoneBlur), but every one of those treats
low light and blur as independent axes. This is the first entry to check a
dataset that pairs them in the same image — exactly the "overcast, evening
light, older phones" regime the brief flags as where real blur is expected
and has never been measured for this model.

**What it is.** LOL-Blur, from "LEDNet: Joint Low-light Enhancement and
Deblurring in the Dark" (Zhou, Li, Chen, Loy — ECCV 2022). Two parts: (1) a
synthetic set — 200 dynamic dark-scene videos (170 train / 30 test, 60
frames each, 12,000 paired low-light-blurry / normal-light-sharp frames,
indoor and outdoor); (2) Real-LOL-Blur — 1,354 *real* night-time blurry
images (482 from RealBlur-J plus 872 shot on a Sony RX10 IV) with no sharp
ground truth. Code, pretrained weights, and both dataset splits are
confirmed live on GitHub/Google Drive/BaiduPan at
`github.com/sczhou/LEDNet`, not a dead or empty link.

**Verification.** Fetched the LEDNet GitHub repo directly and its `LICENSE`
file. The repo — code, weights, and by the README's own framing the
dataset release alongside it — is under **S-Lab License 1.0**, quoted
verbatim: "Redistribution and use for non-commercial purpose in source and
binary forms, with or without modification, are permitted... In the event
that redistribution and/or use for commercial purpose in source or binary
forms, with or without modification is required, please contact the
contributor(s) of the work." This is an explicit non-commercial license,
same family as several prior ruled-out entries (Zero-DCE original,
CaddieSet-adjacent research licenses) — not silence on commercial use, an
active prohibition without the copyright holder's separate permission.

**Which failure mode.** Both, jointly — this is exactly the caveat in the
brief's evidence section: a dataset built specifically because low light
lengthens exposure and lengthened exposure causes blur, so the two
co-occur in real footage even though this project's 3-clip outdoor test set
(fast shutter, good light) structurally cannot show that.

**Why it helps this model specifically.** Not as training data — it's
licensed out. Its value here is diagnostic and architectural, not a data
source: it confirms the joint low-light/blur regime is real and studied
enough to have 1,354 *real* (not synthetic) paired night images, which is
independent evidence the brief's caveat is right to worry about, and it
names LEDNet's own architecture (a joint enhance-then-deblur pipeline) as a
prior-art shape for a preprocessing stage if this project ever measures
indoor performance and finds blur-in-the-dark is the dominant failure —
which it cannot yet do, since the indoor test set is quarantined per the
README. The synthetic-generation *method* (pairing sharp long-exposure
frames with a physically-modeled dark+blur degradation) is describable from
public summaries and not itself copyrighted, unlike the shipped weights —
so a from-scratch reimplementation of the synthesis pipeline against this
project's own footage would sidestep the license, at the cost of building
it.

**Effort vs. payoff.** Low effort to date (one repo, one license file, both
confirmed quickly) and the payoff is informational, not a usable asset:
this closes off "just use LOL-Blur" as an option, and re-confirms (a third
time, after Zero-DCE and the RealBlur/SloMoBlur entries) that the
recurring blocker in this whole research area is non-commercial licensing
on exactly the real-world dark/blurred imagery this project needs most.
The actionable next step isn't this dataset — it's what the 2026-08-14 PSF
entry and 2026-08-25 6-DOF entry already proposed: synthesize the
degradation directly from this project's own phone footage, which owns its
license outright.

---

## 2026-08-29 (third run) — Vcamba: a Mamba/state-space video camouflaged-object-detection network with explicit dual-domain motion perception — real working code, but unlicensed and CUDA-kernel-locked

**Area covered.** Bullet 3 (small/low-contrast/camouflaged object detection,
temporal methods specifically). This log has logged nine prior VCOD/COD
mechanisms (SLT-Net, SINet-V2, EMIP, SAM-PM, CamDiff, Motion-Informed
Enhancement, DTUM, Channel-stacked multi-frame YOLO, TrackNetV4). Vcamba is
a tenth, and mechanically distinct from all of them: it is the first in
this log built on a Mamba/selective-state-space backbone rather than a
transformer or plain CNN, and the first to fuse motion perception in both
the spatial domain *and* the frequency domain in one network rather than
picking one (the 2026-08-29 first-run MS-YOLOv11 entry today did frequency
only, appearance-only, no motion).

**What it is.** "Video Camouflaged Object Detection via Mamba-based
Spatial-and-Frequency Motion Perception" (Vcamba), Xin Li, Keren Fu, Qijun
Zhao, arXiv:2507.23601 (posted Feb 2026 per search-result metadata; arXiv
itself is blocked from this sandbox's egress, consistent with prior runs'
notes, so the abstract's own performance numbers could not be read
directly — sourced from secondary summaries only, one notch below full
verification). Architecture: a VMamba (vssm1) backbone with three
published variants (tiny/small/base, embed dims 96/128, depths
[2,2,15,2]), plus four VCOD-specific modules confirmed present in the
actual code (not just the abstract) — an adaptive frequency-component
enhancement (AFE) module, space- and frequency-based long-range motion
perception modules (SLMP/FLMP), and a fusion module (SFMF) that combines
them. Evaluated on MoCA-Mask and CAD2016 per `mypath.py`'s dataset
registry — the same MoCA-Mask benchmark this log's 2026-08-15 SLT-Net entry
already found to be a licensing dead end for the *dataset*, which is a
separate question from the *code's* license here.

**Verification.** Cloned `https://github.com/BoydeLi/Vcamba` directly
(`git clone`, not just viewing the page). Confirmed it is a real, working,
non-trivial repo: `train_video_long_term.py` (18KB), a `models/` directory
with `vcamba.py`, `vmamba.py`, `vssblock.py`, etc., and a `kernels/`
directory containing a full custom CUDA extension for the selective-scan
operation (`csrc/selective_scan/`, with separate forward/backward `.cu`
kernels) — this is not an empty or placeholder release like the 2026-08-29
first-run MS-YOLOv11 entry earlier today. Searched the repo for any
`LICENSE`/`LICENCE`/`COPYING` file at any depth: **none exists.** The
`README.md` is two lines (title and authors only) with no license section.
**No license is granted — default copyright applies, meaning no use,
commercial or otherwise, is legally permitted without contacting the
authors.**

**Which failure mode.** Camouflage (primary). Not motion blur — this
detects a camouflaged object across frames using its motion signature, a
different problem from recovering an elongated blur streak within one
frame.

**Why it helps this model specifically, in principle.** The brief's
camouflage frames are described as producing *zero* candidate detections
even at confidence 0.05 on visually sharp footage — an appearance-domain
dead end this log has repeatedly noted motion can solve where texture
cannot (foliage doesn't move with the swing; the clubhead does). Vcamba's
specific contribution over the log's other motion-based entries (DTUM,
Channel-stacked multi-frame YOLO, TrackNetV4) is doing that motion
comparison in the frequency domain as well as the spatial domain in a
single fused representation, which is a mechanically different way of
extracting the same "moving-thing-against-static-background" signal this
model's zero-detection frames need.

**Effort vs. payoff.** Currently zero payoff: the license blocks all use,
so nothing here is buildable today regardless of the model's actual
performance. Even setting the license aside, the architecture itself is a
poor practical fit for this project's deployment target — a Mamba
selective-scan layer needs a custom CUDA kernel to run efficiently (that is
what `kernels/selective_scan/` exists for), and there is no CoreML or
Metal Performance Shaders equivalent shipped or implied anywhere in this
repo; porting a working VMamba backbone to on-device Apple Neural Engine
inference would be a substantial research effort on its own, independent
of this specific paper. Net: interesting confirmation that frequency+motion
fusion is an active, real research direction for exactly this model's
camouflage symptom, but not an actionable lead — the same "get the
authors' permission or don't use it" conclusion as several prior VCOD
entries, compounded here by a genuine mobile-deployment gap the CNN/YOLO
alternatives already logged (Channel-stacked multi-frame YOLO,
Motion-Informed Enhancement) don't have.

---

## 2026-08-29 (fourth run) — YOLO-Ball (tennis, 2026): independent empirical evidence that combining this log's already-logged P2 head and NWD loss gives a real gain on a directly analogous small/blurred/occluded ball-in-sport problem

**Area covered.** Motion blur architectures, with a secondary read on small-
object detection. Two other leads were checked first and discarded before
this one: (1) the Kaggle mirror of GolfDB's raw source videos
(`marcmarais/videos-160`) — not re-checked in depth, since the 2026-08-13
GolfDB entry already establishes the underlying licence (CC BY-NC-4.0,
non-commercial) and curation bias (blur explicitly filtered out) apply
regardless of which mirror serves the files; a different host does not
change either dealbreaker. (2) The HUE Dataset (event-camera + frame
sequences for low-light vision, arXiv:2410.19164) — has genuine indoor/
dim-light sequences, but is built around event-camera sensor data phones
don't have, so it doesn't transfer to this project's RGB-only capture
pipeline; not worth a full entry over that mismatch.

**What it is.** "YOLO-Ball: Real-time tennis ball detection under occlusion
and motion blur" (Ding, Fan, Zhao; SAGE, 2026;
DOI 10.1177/17543371261423768, indexed under *Proceedings of the
Institution of Mechanical Engineers*). Three stated contributions, per
consistent search-engine-indexed abstract excerpts (the DOI resolves to
`journals.sagepub.com`, which — like every SAGE/arXiv/IEEE host checked in
every prior run of this log — returned `EGRESS_BLOCKED` from this sandbox,
so nothing below is a primary-source read): (1) a multi-branch occlusion-
aware attention mechanism for dynamic multi-scale feature fusion; (2) a
"dual-flow shallow fusion pyramid combining P2 features with bidirectional
fusion to enhance small target and blur handling" — mechanically the same
lever as this log's 2026-08-23 P2/4-head entry (add a finer, stride-4
detection head), described independently by a different team on a
different sport; (3) a "dynamic balance loss integrating Normalized
Gaussian Wasserstein Distance (NWD) and IoU with learnable alignment
weights" — the same NWD mechanism as this log's 2026-08-27 entry, here
blended adaptively rather than at a fixed ratio. Reported results on their
own constructed tennis dataset: 82.2% precision, mAP@0.5 70.9%,
"outperforming YOLOv8/v10/v11 by up to 12.5%," with stated generalization
to volleyball and football detection.

**Verification.** Existence of the paper and its abstract is confirmed
across four independent search passes with consistent detail (title,
authors, DOI, contribution list, and result numbers all matched every
time). What is **not** verified: the paper's full text, its actual dataset
composition, and — critically — no GitHub repository, code release, or
dataset download link surfaced in any search performed this run. This is
an existence-only result in this log's established sense: real paper,
plausible and internally consistent claims, zero shippable artifact.

**Which failure mode.** Motion blur (primary, via the P2-pyramid and NWD-
loss combination) and, secondarily, occlusion — which this entry is
careful **not** to conflate with camouflage. Occlusion (a ball hidden
behind a player or racket, needing to be found once it re-emerges) and
camouflage (an object fully visible but blending into background texture)
are different problems with different fixes; the "occlusion-aware
attention" contribution here is closer in spirit to this log's 2026-08-17
InpaintNet and 2026-08-26 OC-SORT entries (recovering identity across a
gap) than to the appearance/motion-based camouflage entries (SINet-V2,
SAM-PM, DTUM, etc.). Only the P2+NWD half of this paper speaks to this
model's brief.

**Why it helps this model specifically.** Neither the P2/4-head entry nor
the NWD-loss entry in this log carried any evidence beyond "well-motivated,
untested on this model" — both closed with an explicit "genuinely
unverified for this model, try it as a first experiment" caveat. YOLO-Ball
is the first thing this log has found that combines both levers together
and reports a result on a problem shape genuinely close to this one: a
small, fast, occasionally-blurred ball-sized object filmed at
sport-action speed, evaluated against the same YOLOv8/v10/v11 family this
project already trains on. A +12.5% lift over stock YOLO baselines, even
on a different sport and an unverified/unreleased dataset, is a real
independent data point in favor of the specific combination (finer
detection head + Wasserstein-based tiny/elongated-box loss) this log
already recommended on first-principles grounds — it doesn't replace
running the ablation on this project's own data, but it raises the prior
that the combination is worth prioritizing over other still-untested
architecture entries in this log's backlog.

**Effort vs. payoff.** Low effort (search-and-verify only; the usual
sandbox egress block prevented a primary-source read of methodology or
dataset details). Payoff is informational, not a usable asset: there is no
code, no dataset, and no camouflage-relevant content here, and the result
is unreplicated outside the authors' own unreleased tennis set. Its value
is narrow and specific — it modestly de-risks running the P2+NWD ablation
this log already proposed, ahead of the heavier, still-more-speculative
architecture entries (JFD3, DFRCP, MoSA-Det) further down the backlog. Not
worth chasing further without a way to actually read the paper or find its
code; a future run should not re-search this specific title again.

---

## 2026-08-30 — YOLO12 (Ultralytics): Area Attention gives a same-family, one-line-checkpoint camouflage lever, but its CoreML export path is not confirmed the way YOLO26's is

**Area covered.** Rotated to bullet 3 (small/low-contrast/camouflaged
object detection, architecture), deliberately avoiding bullet 2 (motion
blur), which the immediately preceding run (2026-08-29, fourth run) used.
Two other leads were checked and discarded before this one: (1) a fresh
web search for golf clubhead/club tracking GitHub projects surfaced only
already-logged dead ends (`onkar-99/Golf-Ball-Tracking`, checked again
directly — still no LICENSE file and no released dataset, exactly as the
2026-08-16 entry already found; not worth re-logging) plus patent filings
and a golf-ball, not clubhead, tracker; nothing new. (2) Meta's CoTracker3
(point tracking) looked promising for surviving detection gaps across
frames, but a first pass found no license terms clearly stated for
commercial use and no obvious way to graft point-tracking output into this
project's frame-by-frame detection+export pipeline without first having a
reliable initial detection to seed a track from — exactly the problem this
model doesn't yet solve — so it was set aside rather than logged.

**What it is.** YOLOv12 ("YOLOv12: Attention-Centric Real-Time Object
Detectors," Tsinghua University, NeurIPS 2025) is now integrated as
"YOLO12" directly into the `ultralytics/ultralytics` repo — the exact same
repo this project's `YOLO11n` and the already-logged `YOLO26` (2026-08-17)
come from, confirmed via `raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/models/yolo12.md`
(fetched directly, HTTP 200). Two architecture changes, quoted from that
doc: (1) **Area Attention (A2)** — "divides feature maps into *l*
equal-sized regions (defaulting to 4), either horizontally or vertically,
avoiding complex operations and maintaining a large effective receptive
field," i.e. windowed self-attention that is cheaper than full
self-attention but still sees much more of the frame than a local
convolution kernel; (2) **R-ELAN** — adds block-level residual connections
with scaling and a redesigned bottleneck-style feature-aggregation path,
aimed at stabilizing training for the added attention layers in larger
variants. Reported metric from the same doc: YOLO12n reaches 40.6% mAP on
COCO val2017 ("+2.1%/-9%" relative to YOLOv10n). No YOLO12n-vs-YOLO11n
accuracy comparison table was found anywhere in this run.

**URL.** Original repo: https://github.com/sunsmarterjie/yolov12 (`LICENSE`
fetched directly via `raw.githubusercontent.com`, HTTP 200). Ultralytics
integration docs: https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo12.md.

**Licence (verbatim, from the original repo's own `LICENSE`, fetched
directly).** "GNU AFFERO GENERAL PUBLIC LICENSE / Version 3, 19 November
2007." Same disposition as the already-logged YOLO26 entry: **commercial
use is not simply "permitted"** — it is permitted only under AGPL-3.0's
copyleft terms (source-disclosure obligations for anything served over a
network, which a detector shipped inside an app arguably is) or by buying
a separate paid Ultralytics Enterprise licence for closed-source
commercial use. This is not a new licensing question — this project's
current `YOLO11n` dependency and the already-logged YOLO26 both ship from
the identical top-level `ultralytics/ultralytics` `LICENSE`, so YOLO12
changes nothing about that already-open question, for better or worse.

**Which failure mode.** Primarily camouflage, and honestly the argument is
the same broad-context intuition already logged for DTUM, SLT-Net, and
FastViT (2026-08-14, 2026-08-15, 2026-08-23): a low-contrast clubhead
against cluttered foliage is easier to separate from its background with
more spatial context than a local convolution kernel sees. What's actually
new here is *where* that context comes from — windowed self-attention
built natively into the same Ultralytics YOLO config this project already
trains, rather than a full backbone swap (FastViT) or a motion-based
signal (DTUM/SLT-Net). Secondarily, weakly, motion blur: an elongated blur
streak spans more of the frame than a small kernel captures at once, so
attention integrating across a wider region could register a streak more
completely — but no author benchmark for blurred or elongated objects was
found, so this is this log's own inference, not YOLO12's authors' claim.

**Why it helps this model specifically, and the gap that limits it.** If
CoreML export works, adopting YOLO12 would cost exactly what the already-
logged YOLO26 checkpoint swap costs — changing `yolo11n.pt` to `yolo12n.pt`
in the existing Ultralytics training config, no new architecture wiring,
no lost COCO-detection pretraining (unlike FastViT, which requires a fresh
backbone integration and loses that transfer-learning head start). That
"if" is the catch this entry exists to flag: this project's own CoreML
integration reference, `raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/integrations/coreml.md`
(fetched directly this run), is written specifically around **YOLO26** as
the export target, mentions YOLO11 only in a comparison aside about NMS
embedding, and does not list YOLO12 anywhere as a supported CoreML export
target. That is a materially weaker verification position than the
already-logged YOLO26 entry, which confirmed CoreML as "a documented
first-class export target" for that model family specifically. It is
plausible the same generic `model.export(format="coreml")` path in the
shared Ultralytics exporter still works for YOLO12 — the export code is
task-generic, not model-specific, in every other family checked — but that
is an inference, not something read from documentation, and should not be
treated as confirmed.

**Effort vs. payoff.** Low effort if CoreML export turns out to work (a
one-line checkpoint substitution plus a re-run of the existing evaluation
harness, exactly like YOLO26); non-trivial risk if it doesn't (this
project has no confirmed path to get a YOLO12-trained model on-device at
all, which would make the whole exercise a dead end before accuracy is
even measurable). Payoff, even in the success case, is the same
low-to-modest, unverified uplift already logged for YOLO26 and FastViT: a
plausible-sounding broader-receptive-field argument for camouflage with no
camouflage-specific benchmark behind it, and no accuracy delta vs. YOLO11n
found anywhere in this run. Recommend: a five-minute smoke test — `yolo
export model=yolo12n.pt format=coreml` — before spending anything further
on this; if that fails, drop it without regret, since every actually-
verified camouflage mechanism already in this log (DTUM, SAM-PM,
Motion-Informed Enhancement) has no CoreML-export uncertainty attached at
all. If it succeeds, treat it exactly like YOLO26: a free, opportunistic
checkpoint choice to make once the project trains a real model, not a
substitute for the higher-conviction, already-logged fixes.

---

## 2026-08-30 (second run) — "Preserve the Hard, Regenerate the Rest": uncertainty-guided diffusion augmentation that keeps the real object and regenerates only the background — the inverse of CamDiff's licence problem, on the opposite side of what to synthesize

**Area covered.** Rotated to bullet 5 (data synthesis/augmentation),
deliberately avoiding bullet 3 (camouflage/small-object architecture),
which the immediately preceding run (2026-08-30, first run — YOLO12) used.
Checked and discarded first: CADDIE (2026-08-21's still-open golf-pose
lead) — retried `openaccess.thecvf.com`, still `EGRESS_BLOCKED`, and a
fresh search surfaced only the authors' affiliation (Fujitsu Research of
America / Colorado State University) with no new code or licence
information, so it stays exactly as open as before and was not re-logged.
BlurBall's motion-blur-direction mechanism was independently re-surfaced
by search but is already logged in full (2026-08-16). A golf-swing MIT
thesis on markerless 3D body-pose tracking ("...Truncation-Robust
Heatmaps", built on MeTRAbs) was checked and set aside: it tracks the
golfer's body joints, not the club, `dspace.mit.edu` is
`EGRESS_BLOCKED`, and "truncation-robust" there means joints leaving the
video frame — nothing to do with clubhead camouflage or blur.

**What it is.** "Preserve the Hard, Regenerate the Rest: Uncertainty-Guided
Synthetic Training Data Augmentation with Diffusion Models" (Röhrich et
al., XITASO GmbH, 2026, arXiv:2606.31603 — abstract/PDF unreachable,
`arxiv.org` egress-blocked as in every prior run, so the mechanism below is
sourced from the repo's own README, fetched directly and successfully via
`raw.githubusercontent.com`, not the paper). The pipeline: train a baseline
model on real images, run it to get per-pixel predictive entropy, aggregate
that uncertainty per ground-truth class, and mark the most-uncertain
classes as **preserved** until their combined area crosses a budget `τ`.
An off-the-shelf diffusion inpainter (SDXL inpainting) then regenerates
everything **outside** the preserved region — i.e. the *easy*,
already-well-classified context — and the original pixels are pasted back
over the preserved region with a feathered boundary. The model is
fine-tuned on the mix of real and regenerated images, with the regenerated
pixels excluded from supervision.

The mechanism is the mirror image of this log's own already-logged CamDiff
entry (2026-08-20, second run): CamDiff keeps the real **background** and
diffusion-generates a new **object** into it; this technique keeps the
real, hardest-to-fake **object region** untouched and diffusion-regenerates
the **surrounding context** instead. For a golf clubhead specifically, that
inversion matters: this project's own already-logged synthesis entries
(BlenderProc, CamDiff) both carry an explicit, unresolved sim-to-real gap
on the object itself — does a rendered or diffusion-generated clubhead look
like a real one? This technique never answers that question because it
never touches the clubhead pixels; it only diversifies what's *behind* the
clubhead — the actual axis of the camouflage failure mode (dark clothing,
foliage), not the object's own appearance.

**URL.** Code: https://github.com/XITASO/Preserve-the-Hard-Regenerate-the-Rest
(confirmed live — `README.md` and `LICENSE.md` both fetched directly via
`raw.githubusercontent.com/.../main/...`, HTTP 200; note the file is
`LICENSE.md`, not `LICENSE` — the bare filename 404s, `LICENSE.md` does
not). Repository contains real, complete code: dataset adapters, an
entropy-based sample-selection module (`guided_generation/sample_selection/`),
a diffusion inpainting + paste-back pipeline (`guided_generation/diffusion/`),
and a full segmenter training/eval harness (`vfm4ss/`) — not a paper
landing page.

**Licence (verbatim, from `LICENSE.md`, fetched directly).** "MIT License
/ Copyright (c) 2026 XITASO GmbH / Permission is hereby granted, free of
charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software... " **Commercial use: permitted**, on the repo's own code — a
real, direct advantage over CamDiff, whose repo this log already confirmed
has no licence file at all (verified-absent, "forbidden by default").
Two caveats the repo's own README states explicitly and this run did not
independently re-verify beyond that: (1) none of the three benchmark
datasets (Cityscapes, BDD100K, UAVID) are redistributed — irrelevant here
since none would be used, this project has its own data; (2) the default
inpainter is `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, whose
Hugging Face model card lists its licence as CreativeML Open RAIL++-M — a
permissive-with-use-restrictions licence that generally allows commercial
use of the model and its generated outputs, but carries its own
use-based restriction clauses that were not read in full this run and
should be checked before relying on SDXL-generated pixels in a shipped
training set.

**Which failure mode.** Camouflage, specifically and only — same scope as
CamDiff. Nothing in the mechanism (entropy-guided background regeneration)
touches blur-streak geometry.

**Why it helps this model specifically.** The camouflage failure mode this
project has measured (a dark clubhead against dark clothing or cluttered
foliage) is a background-diversity problem, not an object-appearance
problem — the failing frames are described as "visually sharp," i.e. the
clubhead itself is rendered fine by the camera, it just sits on a
low-contrast background the training set under-represents. This technique
is a closer mechanical match to that specific gap than either
already-logged synthesis entry: unlike BlenderProc (a full 3D clubhead
render, sim-to-real gap on the object itself, "a multi-week undertaking")
or CamDiff (which requires solving the *inverted* task the paper never
tested, plus building bespoke dark-clothing/foliage prompt and mask logic
from scratch since CamDiff's own repo is unlicensed), this approach starts
from **real clubhead crops already in the training set** and only
generates new backgrounds behind them — a smaller, more bounded generation
problem than synthesizing a whole plausible club head, and one with a
directly reusable, permissively-licensed reference implementation for the
entropy-selection and paste-back mechanics, even though the segmentation-
specific parts (per-pixel masks, ignore-index supervision) would need
translating into this project's box-labeled, YOLO11n training pipeline.

**Effort vs. payoff.** Medium-high effort, plausible but unverified payoff.
Effort: this is not a drop-in — the repo is built entirely around semantic
segmentation (per-pixel entropy over Cityscapes/BDD100K/UAVID-style masks,
a Cityscapes-style ignore-index convention) and would need real adaptation
work to a box-labeled detector: computing an analogous "hard region" signal
from YOLO11n's own per-frame confidence (the failing-frame set this
project's brief already names — zero candidates even at confidence 0.05 —
is a natural substitute for the paper's per-pixel entropy), defining what
"preserve" means for a bounding box rather than a segmentation mask
(plausibly: preserve the labeled clubhead box plus a small margin,
regenerate the rest of the frame), and standing up the SDXL inpainting
dependency with its own licence review. None of that is present in the
repo as shipped. Payoff: a real, mechanism-grounded advantage over this
log's other two synthesis entries (no sim-to-real risk on the object
itself, no unlicensed-code blocker, an automatic hard-example-prioritization
signal instead of manually curated "camouflage-like" background crops) —
but zero evidence this specific technique works for detection rather than
segmentation, and zero evidence for small/dynamic sports objects rather
than static driving/aerial scenes; the paper's own benchmarks don't cover
either. Recommend logging this as the leading candidate the *next* time
this project actually scopes a camouflage data-synthesis spike, ahead of
CamDiff (licence-blocked) and BlenderProc (higher effort, unresolved
sim-to-real risk on the object) — but still as a spike to validate, not a
committed pipeline change.

---

## 2026-08-30 (third run) — RSBlur's licence blocker is resolved: naive frame-averaging blur synthesis is measurably wrong, and there's now a verified, CC BY 4.0 fix recipe

**Area covered.** Bullet 2 (motion blur specifically) — rotated away from
architecture (this log's first 2026-08-30 entry, YOLO12) and data synthesis
for camouflage (the second 2026-08-30 entry), neither of which touched
blur, and toward closing a gap this log already flagged rather than
opening a new thread.

**What changed.** The 2026-08-16 (second run) RealBlur entry logged an
aside: RSBlur (`github.com/rimchang/RSBlur`, Rim et al., ECCV 2022 —
the same POSTECH group and beam-splitter dual-camera lineage as RealBlur,
ECCV 2020) was checked and explicitly **not** logged as usable because "its
README states no licence and no LICENSE file exists in the repo, checked
directly." That is no longer true. Re-fetching
`raw.githubusercontent.com/rimchang/RSBlur/master/README.md` directly today
(HTTP 200) shows the file now ends with an explicit license section:

```
## License

The RSBlur dataset is released under CC BY 4.0 license.
```

There is still no separate `LICENSE` file in the repo (confirmed:
`raw.githubusercontent.com/rimchang/RSBlur/master/LICENSE` → HTTP 404), but
the grant is now stated inline in the authors' own words, in a directly
fetched file — the same evidentiary bar this log has accepted for RealBlur
and other CC BY 4.0 findings. **Commercial use: permitted**, subject to
attribution. Whether the repo was edited sometime in the last two weeks or
the 2026-08-16 run simply missed a section further down the README could
not be determined (GitHub doesn't expose file history through this
sandbox's reachable surface), but the current, verified state is
unambiguous.

**What RSBlur actually is.** Not a source of clubhead footage — it is a
still-image deblurring benchmark. Paired real/synthetic blurred images with
ground-truth sharp images (13,358 pairs), captured via a beam-splitter
dual-camera rig identical in kind to RealBlur's: one sensor exposed long
enough to blur, one short enough to stay sharp, same lens, same instant.
The blur is again camera-shake / long-exposure blur of a scene, not a fast
object streaking across an otherwise sharp frame — so, like RealBlur, it is
not a stand-in for a blurred clubhead. What makes it a different, new
finding rather than a restatement of RealBlur is its actual contribution:
a **measured critique of naive frame-averaging blur synthesis** — exactly
the technique this log's very first entry (2026-08-12, Brooks & Barron) and
the 2026-08-14 PSF-synthesis entry propose for manufacturing synthetic
blurred training examples. RSBlur's own README ablation table (read
directly, not inferred) reports PSNR/SSIM on their real-blur test set for a
deblur model trained on synthetic blur built with increasingly realistic
synthesis steps:

| Synthesis pipeline | PSNR / SSIM |
|---|---|
| Linear-space frame averaging (naive) | 30.12 / 0.7727 |
| + sRGB gamma, + frame interpolation, + saturation-mask synthesis, + Poisson/read noise, + ISP re-application (their full pipeline) | 32.06 / 0.8322 |

The gap is the point: naive averaging omits sensor saturation clipping
(bright highlights streak differently once clipped than a pure linear
average predicts) and photon/read noise, both of which real long-exposure
capture always has and pure synthetic averaging never does by default.

**Which failure mode.** Motion blur — specifically, a methodology warning
for this project's own already-logged synthesis ideas, not a new
architecture or dataset.

**Why it helps this model specifically.** This project has no working
motion-blur augmentation pipeline yet (per every prior run's caveats on the
Brooks & Barron, PSF, and 6-DOF entries — all logged as ideas, none
confirmed implemented). If and when one is built to manufacture blurred
clubhead training examples from the sharp footage this project already
has, RSBlur is direct evidence that the straightforward version of that
idea — average or kernel-blur consecutive sharp frames and call it a
blurred training example — will produce examples that are measurably too
clean relative to real phone-camera blur: no highlight-streak clipping, no
sensor noise. A detector trained on such examples risks learning a
synthetic-blur "tell" (unnaturally smooth gradients, no noise floor) that
does not transfer to the genuinely blurred frames this model actually
needs to handle in dim/indoor light, which is exactly where noise and
dynamic-range limits are worst. RSBlur supplies a concrete, already-coded
correction (CRF → linear space → interpolate → synthesize saturation mask →
inject Poisson/read noise → re-apply CRF/ISP) that any future blur-
augmentation spike for this project should implement rather than the naive
version, plus a paired real/synthetic benchmark to sanity-check the
augmentation's realism against before trusting it on clubhead data.

**Reachability, checked directly (same result as RealBlur on
2026-08-16).** The GitHub repo and README are reachable
(`raw.githubusercontent.com`, HTTP 200). The actual dataset hosts are not,
from this sandbox: `cgdata.postech.ac.kr/sharing/kWA6K6J5G` → connection
failure (curl exit, HTTP code `000`, `CONNECT tunnel failed, response
403`), `cg.postech.ac.kr/research/RSBlur/` → HTTP 403, and the Google Drive
mirror → the same tunnel failure. So: licence confirmed, code confirmed
live, but the dataset itself is unverified as downloadable from this
sandbox — a machine with unrestricted egress would need to confirm the
link before anyone treats "download and use RSBlur's images" as
actionable. (Nothing here requires downloading RSBlur's images to be
useful, though — the synthesis-pipeline methodology and the ablation
numbers are the payoff, not the images themselves.)

**Effort vs. payoff.** Effort to verify: low — one README re-fetch, one
LICENSE 404 check, three reachability probes on already-known URLs from
the 2026-08-16 entry. Payoff: real but entirely conditional and not
implementable today. It is a design note to attach to a future blur-
augmentation spike (2026-08-12 and 2026-08-14 entries), not a standalone
action — this project has not yet built the naive version RSBlur warns
against, so there is nothing to fix yet. Recommend treating this as a
checklist item to apply *when* that spike is scoped ("don't just average
frames — model saturation and noise, per RSBlur"), not as work to schedule
now. Logged primarily because it corrects a specific, previously-recorded
"not usable" finding in this log with a direct, verified contradiction —
log integrity matters as much as new leads.

---

## 2026-08-30 (fourth run) — RefCOD / R2CNet: reference-exemplar-guided camouflaged object detection (TPAMI 2025) — a tenth camouflage mechanism, and the first that fits a single-class detector unusually well

**What it is.** "Referring Camouflaged Object Detection" (Zhang, Yin, Lin,
Zhang, Fan, Cheng — accepted TPAMI 2025), code at
https://github.com/zhangxuying1004/RefCOD. It defines Ref-COD: instead of
segmenting every camouflaged object in a scene from appearance alone, the
network also takes a **reference image** — a salient, non-camouflaged
example of the same object class — and uses it to guide detection of the
camouflaged instance. The architecture (R2CNet) has a reference branch
(pools a reference embedding from a saliency-detection backbone) and a
segmentation branch that fuses that embedding into the target image's
features via an RMG (Reference Matching Guidance) module before decoding,
with further refinement in an RFE (Reference Feature Enrichment) module.
Verified directly from the GitHub README and repo structure (train.py,
test.py, infer.py, an RMG/RFE module layout matching the described
architecture).

**Licence — verified, commercial use permitted.** The repo's `LICENSE` file
(fetched directly,
`raw.githubusercontent.com/zhangxuying1004/RefCOD/main/LICENSE`) is the
standard MIT License, copyright Xuying Zhang 2023: permission "to deal in
the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software," with only the standard attribution and
no-warranty conditions. This covers the code (training/inference scripts
and model definition). **The R2C7K dataset (64 categories, camo + reference
subsets) and the pretrained `r2cnet.pth` checkpoint are hosted on Baidu
Netdisk only**, per the README's own download links — not on GitHub, not
mirrored anywhere else found. Baidu Netdisk carries no separate licence
statement reachable from here, and practically: `curl` against the Baidu
share link from this sandbox returned `HTTP 000` (connection failure,
same failure mode this log has hit repeatedly on China-hosted or
university-hosted dataset mirrors — RSBlur's postech hosts, 2026-08-30
third run). So: **code confirmed MIT and reachable; dataset and checkpoint
unverified as downloadable from here, and their licence terms are whatever
Baidu Netdisk's own hosting terms are, not stated by the authors.**

**Which failure mode.** Camouflage — specifically the zero-candidate-
detection frames (dark clubhead on dark clothing/foliage) this project's
own failure analysis singles out, where the model currently has to infer
"clubhead" purely from local appearance in that one frame and finds
nothing above threshold.

**Why it helps this model specifically — and why it's a better fit than
the other nine logged camouflage mechanisms.** Every prior camouflage
entry in this log (SLT-Net, SINet-V2, DTUM, Motion-Informed Enhancement,
SAM-PM, EMIP, Vcamba, Copy-Paste, YOLO12 Area Attention) either relies on
appearance alone (which is exactly what fails on genuinely low-contrast
frames) or on cross-frame motion (which requires the object to actually be
moving distinctly from the background — not guaranteed at address, in the
backswing pause, or in a slow/deliberate practice swing). Ref-COD's
reference branch is a third, structurally different lever: an explicit
"here is what the object normally looks like" prior injected into the
network, independent of that frame's local contrast or of motion. This
project has an unusual advantage for adapting it: it detects exactly **one
class**, so instead of needing a per-instance reference image at inference
time (the original task's assumption — Ref-COD was built for a many-class,
many-object setting), a single fixed canonical clubhead crop (or a small
fixed set covering driver/iron/wedge head shapes) could serve as a
constant reference embedding baked into the exported model, with no
runtime reference-image lookup needed. That reframes "clubhead in dark
clothing" from a one-shot appearance problem into a matching problem
against a known template, which is a plausible lever specifically for the
zero-detection frames the failure analysis flags as sharp but unmatched.

**Why this is not close to actionable today.** This is an architecture
change (a second, reference-branch input stream plus fusion modules), not
a drop-in fix like a loss swap or an augmentation flag. The published
checkpoint is trained on 64 generic COD categories (animals, objects) and
not on clubheads, so it would need retraining from this project's own
data, using this project's own clubhead crops as reference exemplars — an
unproven adaptation, since Ref-COD's own evaluation never tests a
class-wide fixed prototype in place of per-instance references, so there
is no evidence yet that a single canonical clubhead image generalizes as a
matching template across clubs, camera angles, and lighting the way a
true per-instance reference would. There is also no evidence of a YOLO11n-
compatible or CoreML-exportable version of this architecture; R2CNet is a
dual-branch segmentation network with a ResNet/PVT-style backbone, not
designed for on-device video inference. And the dataset needed to validate
the generic-category baseline before attempting the golf-specific
adaptation is unverified as downloadable (see licence section above).

**Effort vs. payoff.** Effort to verify: low-medium (repo README, LICENSE
file, and README download-link fetches, one Baidu reachability probe — all
direct). Payoff: a genuinely new, structurally distinct camouflage
mechanism worth a research spike (prototype a reference-embedding branch
bolted onto an existing detector, using this project's own well-lit
clubhead crops as fixed exemplars, evaluated only on the already-flagged
zero-detection camouflage frames) — but not an implementation task yet.
Recommend logging this as a candidate for a dedicated small experiment
before any commitment to the full R2CNet architecture: the interesting,
cheap-to-test claim is narrower than "adopt R2CNet" — it's "does
conditioning detection on a fixed reference clubhead crop help on the
frames that currently get zero candidates," which could be tested with a
much smaller architecture change (e.g. a Siamese-style feature comparison
branch) than reproducing R2CNet in full.


---

## 2026-08-31 — FILM (Frame Interpolation for Large Motion, ECCV 2022): the missing piece to run this project's own frame-averaging blur-synthesis idea on standard-frame-rate phone footage instead of requiring slow-mo capture

**Area covered.** Bullet 5 (synthesizing/augmenting training data for the
blur regime), but framed as a *tool* rather than a dataset or architecture.
This log's very first entry (2026-08-12, Brooks & Barron frame-averaging)
and the 2026-08-14 PSF entry both propose synthesizing realistic elongated
blur by combining several real sub-frames of the clubhead in motion — and
both implicitly assume those sub-frames already exist, i.e. that the phone
footage was shot at a high frame rate (240fps slow-mo, as GoPro/REDS were,
per the 2026-08-20 entry). Nothing logged so far addresses the case where
this project's own footage is standard 30/60fps and simply does not have
enough real sub-frames between two consecutive frames to average into a
convincing streak. FILM is a general-purpose tool built exactly to fill in
those missing sub-frames.

**What it is.** "FILM: Frame Interpolation for Large Motion" (Reda,
Kontkanen, Tabellion, Sun, Pantofaru, Curless — Google Research / University
of Washington, ECCV 2022). Code at
`github.com/google-research/frame-interpolation`. A single feed-forward
network (no auxiliary optical-flow or depth network, trainable from frame
triplets alone) that synthesizes intermediate frames between two input
frames, explicitly designed for **large inter-frame motion** rather than
the small-motion regime most interpolation nets assume. It supports
`--times_to_interpolate` for repeated bisection (2, 4, 8, ... intermediate
frames) and `--block_height`/`--block_width` for tiling at high resolution.

**Verification.** Cloned the repo directly
(`git clone --depth 1 https://github.com/google-research/frame-interpolation`)
— this succeeded, unlike the great majority of paper/dataset URLs this log
has hit this month. Read the `LICENSE` file verbatim from the clone: it is
the standard **Apache License, Version 2.0** (full text present, no
modifications). Read `README.md` directly: confirms the ECCV 2022 venue and
author list, confirms pretrained TF2 SavedModels are hosted on Google Drive
(three variants: L1, Style, VGG — a live folder link in the README, not a
placeholder), and confirms the "no auxiliary pretrained network" claim in
the repo's own description. The repo shows as archived (read-only) on
GitHub as of Oct 14, 2025, but archived is not deleted: the code, commit
history, and Google-Drive-hosted weights are all still present and
clonable/downloadable today, same "archived but still usable" situation
this log has treated as usable in other entries.

**Which failure mode.** Motion blur only. This does nothing for
camouflage — it does not change appearance or add contrast, and it
requires the clubhead to already be visible and trackable across the two
real frames it interpolates between, so it is not a fix for zero-detection
frames.

**Why it helps this model specifically.** The project's own footage is the
one training-data source this log has repeatedly noted has no licensing
problem (unlike RealBlur, LOL-Blur, SloMoDeblur, RSBlur, GoPro/REDS, all of
which are either non-commercial or unverified) — the blocker for using it
to fix the blur gap has been that the frame-averaging technique needs
several sub-frames of real sub-exposure motion to blend, which standard
30/60fps capture does not provide. FILM removes that precondition: run it
on this project's own ordinary-frame-rate phone clips to synthesize N
intermediate frames between each real frame pair, then frame-average the
real-plus-interpolated sequence (log entry #1's technique) to produce a
motion-blur streak with a controllable, physically-motivated length — all
from footage the project already owns outright, at any lighting condition
it already has on hand (including, notably, whatever indoor/simulator
footage exists even before the quarantined indoor_test set is resolved,
since this is a data-synthesis step, not a re-use of that set as a labelled
test asset). This directly targets the labeling-spec gap the brief
highlights: median labelled elongation of only 1.60 despite the spec
requiring the full blur streak be boxed, because there is too little
genuinely blurred source material to label in the first place.

**Honest limits.** (1) FILM is validated on natural scenes with photographic
subjects moving at "large" but still bounded motion; a full-speed golf
downswing between two 30fps frames (33ms apart) is an extreme case even by
FILM's own "large motion" standard, and the paper does not test anything
resembling a thin, fast, motion-blurred golf club — this is an untested
extrapolation, not a validated fit. (2) The two real input frames it
interpolates between may themselves already carry some blur at swing speed,
and FILM's own training assumes relatively sharp inputs; feeding it two
already-blurred frames is outside its documented operating conditions and
could produce a plausible-looking but geometrically wrong clubhead position
in the synthesized frames — this would need to be visually spot-checked
per clip, not trusted blindly. (3) It is TensorFlow 2 with a GPU-oriented
install path (CUDA/cuDNN or a provided Docker image) — an offline,
training-time-only tool with no bearing on the on-device CoreML export, but
a new dependency to stand up. (4) It only creates more blur examples for
frames where the clubhead is already visible in both bracketing real
frames; it cannot help label a frame where the club left the visible area
entirely.

**Effort vs. payoff.** Effort to verify: low (one clone, one LICENSE read,
one README read — all direct, no blocked domains). Effort to use: medium —
stand up a TF2/GPU environment, run the interpolator over existing project
clips, then wire the frame-averaging step (already scoped in the
2026-08-12 entry) on top of the interpolated sequence, and manually
spot-check a sample of outputs for the geometric-artifact risk in limit
(2) before trusting any resulting labels. Payoff, if it holds up: directly
increases the supply of genuinely blurred, correctly elongated, fully
license-clean clubhead training examples — the specific shortage this
project's own labeling statistics show — without needing new slow-motion
capture or any third-party dataset. This is the first entry in this log
that is a concrete, licensed, immediately-runnable tool for the "no slow-mo
footage" gap the two prior frame-averaging entries left open; recommend a
small spike (5-10 clips, visually inspect the interpolated + averaged
output before committing to using it as training data) rather than
committing to a full pipeline.

---

## 2026-08-31 (second run) — ID-Blau: diffusion-based reblurring augmentation, checked and ruled out on licence (motion blur area)

**Area covered.** Bullet 2 (motion blur specifically — blur augmentation
techniques). Rotated away from this log's first 2026-08-31 entry (FILM,
which fills the frame-supply gap for frame-averaging synthesis) toward a
different synthesis mechanism entirely: a learned generative model of blur
rather than combining real sub-frames.

**What it is.** ID-Blau ("Image Deblurring by Implicit Diffusion-based
reBLurring AUgmentation," Wu et al., CVPR 2024). URL:
`https://github.com/plusgood-steven/ID-Blau` (paper:
`https://arxiv.org/abs/2312.10998`, CVPR page:
`https://cvpr.thecvf.com/virtual/2024/poster/30526`). It is a conditional
diffusion model that takes a **sharp** image plus a sampled "blur condition
map" (encoding a controllable motion trajectory) and generates a
correspondingly, realistically blurred image — trained on GoPro
(`GOPRO_Large`/`GOPRO_Large_all`) via optical-flow-derived blur conditions.
Unlike this log's frame-averaging and PSF-synthesis entries (2026-08-12,
2026-08-14), which combine *real* sub-frames or convolve a fixed kernel,
ID-Blau samples diverse, varied blur trajectories from a learned
distribution — the README states it "can generate various blurred images
unseen in the training set." A pretrained diffusion checkpoint
(`ID_Blau.pth`) ships in the repo's `weights/` directory, and the README's
"Generating Reblur Dataset" section gives a direct command for running it
over an arbitrary folder of sharp images to produce augmented blurry
outputs — exactly the "feed it our own sharp clubhead crops" use this
project would need.

**Licence, verified directly.** Fetched
`raw.githubusercontent.com/plusgood-steven/ID-Blau/main/README.md`
directly (HTTP 200) and searched the full text for "license", "licence",
"MIT", "Apache", "GPL", "BSD", "CC BY" — none appear anywhere. Fetched
`raw.githubusercontent.com/plusgood-steven/ID-Blau/main/LICENSE` directly →
HTTP 404, confirming no LICENSE file exists in the repo. Per GitHub's own
default-copyright rule (the same bar this log has applied to JFD3, EMIP,
DFRCP, and MS-YOLOv11), an unlicensed public repo grants no rights to use,
modify, or redistribute the code or the pretrained weights — **commercial
use is not permitted**, and neither is non-commercial reuse, technically:
the code is visible on GitHub but not licensed for reuse at all. This is a
straightforward, same-shape finding as those four prior entries, not a new
kind of blocker.

**Which failure mode.** Motion blur.

**Why it would have helped this model, if usable.** A generative,
condition-sampling blur augmenter would have been a genuinely different
mechanism from this log's two prior blur-synthesis entries: it does not
require slow-mo sub-frames (unlike frame-averaging/PSF) or a second real
clip (unlike FILM-then-average) — it only needs the sharp clubhead crops
this project already has in training data, and could synthesize varied
blur trajectories directly onto them. That would attack the exact labeling
statistic driving this failure mode (median box elongation 1.60, most
labelled clubheads near-square) without touching capture logistics at all.
The RSBlur entry (2026-08-30, third run) already warned that naive
frame-averaging under-models sensor noise and highlight clipping; a
learned diffusion model conditioned on real GoPro blur statistics is
arguably a better-motivated fix for that exact gap than hand-engineering a
noise model — which made this worth checking closely before ruling it out.

**Effort vs. payoff.** Effort to verify: low — two direct file fetches
(README, LICENSE 404), no blocked domains, result unambiguous. Payoff:
zero, as things stand — this is a clean negative result, not a
existence-only or "unverifiable" caveat like several prior entries in this
log; the licence question here has a definitive no. Logged so that no
future run re-discovers ID-Blau and re-spends the same verification effort
assuming an unlicensed research repo with a pretrained checkpoint attached
must be usable — a full CVPR-caliber method with real, shippable code is
not a substitute for an actual grant of rights.

---

## 2026-08-31 (third run) — TSM (Temporal Shift Module, ICCV 2019): a channel-shift backbone mechanism, MIT-licensed, with a verified video-object-detection extension aimed at fast-moving objects — but no existing YOLO integration to reuse

**Area covered.** Bullet 3 (small/low-contrast/camouflaged object detection,
multi-frame and temporal methods). Deliberately rotated away from this
log's first two 2026-08-31 entries (FILM, ID-Blau), both motion-blur
synthesis, to avoid three consecutive same-area runs.

**What it is.** TSM ("TSM: Temporal Shift Module for Efficient Video
Understanding," Lin, Gan, Han — MIT, ICCV 2019). URL:
`https://github.com/mit-han-lab/temporal-shift-module` (paper:
`https://arxiv.org/abs/1811.08383`). The core idea: inside an ordinary 2D
CNN, shift a fraction of each intermediate feature map's channels forward
and/or backward along the temporal axis before each convolution — no
extra parameters, no extra FLOPs, implemented as a tensor shift/slice
operation. This lets a 2D backbone reason across a short window of frames
with (per the paper and repo) close to 3D-CNN accuracy at 2D-CNN cost. Two
variants: bi-directional (shifts both future and past channels in, offline
only) and **uni-directional** (shifts only past frames' channels in — the
online-capable, causal variant relevant to a live capture pipeline).

**Licence, verified directly.** Fetched
`raw.githubusercontent.com/mit-han-lab/temporal-shift-module/master/LICENSE`
directly (HTTP 200): "MIT License / Copyright (c) 2021 MIT HAN Lab" plus
the standard MIT permission grant, no additional restriction clauses.
**Commercial use is permitted.**

**Important scope caveat, verified directly.** Fetched the repo's own
`README.md` directly (HTTP 200) and read it in full: the shipped code is
for video **classification/action recognition** (Kinetics-400,
Something-Something V1/V2, an NVIDIA Jetson Nano gesture-recognition demo)
— there is no object-detection code in this repository, and no mention of
"object detection" anywhere in the README. The video-object-detection
result (below) comes from the original paper's own experiments section,
not from shipped, reusable code. This is a materially different
verification state than this log's Temporal-YOLOv8 and Motion-Informed
Enhancement entries, both of which are directly reusable as a
data-pipeline change with no new code to write.

**The video-object-detection claim, corroborated but not directly read.**
Because `arxiv.org` and `openaccess.thecvf.com` are both blocked by this
sandbox's egress proxy (the same standing block prior entries in this log
have hit), the paper itself could not be fetched. Two independent search
queries returned consistent, matching numbers from different snippet
sources: TSM reaches **76.3 mAP on ImageNet-VID** with an R-FCN detection
head on a ResNet-101 backbone, versus 74.7 for plain R-FCN and 75.9 for
FGFA (a purpose-built video-detection method); and injecting
**uni-directional** TSM into the backbone for **online** video object
detection improves mAP by **4.6 points specifically on fast-moving
objects**, without changing the detection head or requiring optical flow.
That "improves most on fast-moving objects, no optical flow needed" shape
is the one that makes this worth logging over a generic backbone tweak —
it is evaluated against exactly the regime (motion, not appearance, as the
detection signal) this project's own camouflage failures point to. Two
caveats stay attached: (1) the backbone tested is ResNet-101/R-FCN, not a
YOLO-family single-stage detector, so the transferability of the exact
mAP numbers to YOLO11n is unproven, not just untested; (2) this is a
snippet-corroborated, not directly-read, result — one notch below full
verification, consistent with how this log has flagged other arxiv-blocked
entries (e.g. Temporal-YOLOv8, 2026-08-26).

**How this differs from what's already logged.** This log already has
three other temporal/multi-frame mechanisms for the camouflage failure
mode: Temporal-YOLOv8 and Motion-Informed Enhancement both *recolor the
existing 3-channel input* with temporal history and touch zero model
architecture; YOLOV/YOLOV++ aggregate detection *candidates/RoI features*
across frames *after* the backbone, as a post-hoc refinement stage. TSM is
a third, structurally distinct mechanism: it modifies what happens
*inside* the backbone's own convolutional layers, at every stage, not just
at the input or after it — closer in spirit to DTUM's direction-coded
temporal module, but implemented as a near-zero-cost shift instead of a
new learned attention/correlation block. It is the most-cited, most
independently-reproduced idea in this category (TSM has spawned known
detection follow-ups per the ImageNet-VID numbers above), which is exactly
why it is worth logging even without a ready YOLO port: it is a
well-validated primitive to build from, not a one-off research trick.

**Why it would help this model specifically.** The uni-directional
(causal) variant fits a live capture pipeline — it only needs past frames,
which the app already has buffered. Because a shifted channel is still
just a channel (not a new input format), CoreML export of the shift
operation itself is not obviously blocked (channel slicing/concatenation
lowers cleanly through ONNX/coremltools) — but *feeding the backbone a
rolling window of past-frame features at inference*, rather than one
independent frame per call, is a real architectural and runtime change:
either the app must keep a small ring buffer of intermediate feature
tensors between calls (a stateful inference loop this project's current
single-frame-per-call design does not have), or CoreML's newer stateful-
model support would need evaluating for this specific op — a genuinely
open question this log has not resolved for TSM or for any of its
already-logged backbone-level competitors (YOLOV, DTUM).

**Effort vs. payoff.** Moderate-to-high effort, uncertain-but-plausible
payoff. Effort: unlike Temporal-YOLOv8/MIE (swap what fills an existing
3-channel tensor, zero architecture change) this requires threading shift
ops into YOLO11n's backbone blocks, building the ring-buffer/stateful
inference path in the iOS app, and validating CoreML export end-to-end —
a real engineering project, not a data-pipeline tweak. Payoff: the
specific "+4.6 mAP on fast-moving objects, no optical flow" result is the
best-targeted evidence this log has found for *why* a motion-only backbone
signal should help exactly this project's camouflage cases, but it was
measured on a different detector family and object domain, so the size of
any gain on a single elongated clubhead against foliage is unproven.
Recommend this only as a research spike (does a shift-augmented YOLO11n
backbone measurably outperform stock YOLO11n on the existing camouflage
failure frames, in an offline PyTorch experiment before touching the iOS
capture pipeline or CoreML export at all) rather than a near-term
shipping candidate — the stateful-inference question alone is enough to
put this behind the input-recoloring entries (Temporal-YOLOv8, MIE)
already in this log, which reach for the same camouflage signal at a
fraction of the engineering cost.

---

## 2026-08-31 (fourth run) — Synthetic-to-Real Camouflaged Object Detection (CSRDA, ACM MM 2025): a domain-adaptation training recipe aimed at the gap this log's own synthetic-camouflage entries have never tested — existence-only result

**Area covered.** Bullet 5 (ways to synthesise or augment training data),
with direct implications for bullet 3 findings already in this log.
Rotated away from bullet 3/temporal methods (this log's prior 2026-08-31
entry, TSM) and away from motion blur (the first two 2026-08-31 entries,
FILM and ID-Blau), to avoid three-in-a-row on the same bucket.

**What it is.** "Synthetic-to-Real Camouflaged Object Detection" (Zhihao
Luo, Luojun Lin, Zheng Lin — Fuzhou University), published in the
**Proceedings of the 33rd ACM International Conference on Multimedia
(ACM MM 2025)**, DOI `10.1145/3746027.3755461`, also posted as
`arxiv.org/abs/2507.18911`. It defines a new task, **S2R-COD**
(Syn(thetic)-to-Real Camouflaged Object Detection), and proposes **CSRDA**
(Cycling Syn-to-Real Domain Adaptation), a student-teacher framework that
trains on **labeled synthetic camouflage images plus a limited pool of
unlabeled real images**, using pseudo-labeling combined with consistency
regularization and a curriculum-learning schedule so pseudo-label
generation and domain-adaptation training improve each other over
training cycles — explicitly aimed at the fact that a model trained only
on synthetic camouflage images does not generalize cleanly to real ones.

**Verification, and its limits.** The paper's existence, venue, authors,
task name, and method name are corroborated consistently across four
independent sources found via search (the arXiv listing page, the ACM
Digital Library page, a paper-tracking snippet site, and secondary
citations) — not a single unconfirmed claim. What could **not** be
verified: the abstract text, the quantitative synthetic-only-vs-CSRDA
performance gap, and whether any code or data was released, because every
host that could answer those questions was unreachable from this sandbox
today — `arxiv.org` (HTTP block, consistent with this log's standing
arXiv block, e.g. the 2026-08-31 TSM entry), `ar5iv.labs.arxiv.org`,
`dl.acm.org`, `huggingface.co`, `api.semanticscholar.org`, and
`paperreading.club` all returned `EGRESS_BLOCKED` on direct fetch. A
targeted search for a companion GitHub repository (`CSRDA`, `S2R-COD`,
author names) returned nothing. This is an existence-only result in the
same evidentiary class as this log's CADDIE, ReynoldsFlow, and DFRCP
entries — real and peer-reviewed, but unread past the title/task
description, and with no code confirmed to exist.

**A separately-verified fact that matters here.** Public COD benchmarks
(COD10K, CAMO, and the rest of the standard evaluation suite) are
overwhelmingly **segmentation-mask** labeled, not bounding-box — confirmed
via a separate search on the benchmark literature, not this paper
specifically. That is consistent with how every other COD-family entry
already in this log (SINet-V2, SLT-Net, RefCOD, Vcamba, EMIP, SAM-PM) has
had to be caveated: none of it is a drop-in fix for a YOLO detector
without adaptation, and this paper is very likely the same, though its own
task formulation was not readable to confirm directly.

**Which failure mode.** Camouflage.

**Why it matters to this project specifically.** This log already carries
several entries that propose *generating* synthetic camouflage training
data for the clubhead detector — CamDiff (2026-08-20, diffusion
scene-inpainting), Preserve-the-Hard/Regenerate-the-Rest (2026-08-30,
uncertainty-guided diffusion), and RefCOD's exemplar-guidance angle
(2026-08-30) — all of which implicitly assume that adding synthetic
camouflaged examples to the training mix straightforwardly helps. None of
those entries, and no earlier entry in this log, has addressed whether a
detector trained on synthetic camouflage actually transfers to real
footage, or what training recipe that transfer needs. A dedicated ACM
MM 2025 paper existing specifically to solve that generalization gap is
itself evidence the naive version of that assumption is not automatically
true in the wider COD literature. The task shape CSRDA targets — labeled
synthetic data plus a pool of *unlabeled* real data — also happens to
match this project's actual data asymmetry unusually well: the README
already documents an abundance of the app's own real phone footage
relative to how much of it is labeled (only ~29% of training data is own
footage at all), so a semi-supervised recipe that puts unlabeled real
clips to use is a plausible fit in principle. But none of that can be
acted on without the paper's actual method details or code, neither of
which this run could reach.

**Effort vs. payoff.** Effort to find and corroborate existence: low.
Effort to verify anything actionable: blocked entirely by sandbox egress,
not by absence of effort. Payoff today: zero — nothing here can be
implemented without the unread paper or unfound code. Payoff as a log
entry: a caution flag on this log's own accumulating pile of synthetic-
camouflage-generation ideas — CamDiff and Preserve-the-Hard should not be
treated as "generate synthetic frames, mix into training set, done" until
whichever of them is actually piloted is validated against genuinely held-
out real camouflage frames, not just inspected qualitatively. Logged so a
future run with better egress, or a human with a browser, can pull the
actual abstract and check for a code release, rather than this log
re-discovering the same paper from scratch.

---

## 2026-09-01 — Enhanced YOLOv11n (MSEAF + ScalCat/Scal3DC + P2 head + SRepD): a published, YOLO11n-native small-object recipe that extends this log's own P2-head entry (existence-only result)

**Area covered.** Bullet 3 (small/low-contrast object detection), the
architecture sub-thread. Rotated away from the last four 2026-08-31 runs
(blur preprocessing/FILM, blur-augmentation-licence/ID-Blau,
temporal-architecture/TSM, camouflage-domain-adaptation/CSRDA), none of
which touched bare YOLO11n architecture surgery — the last entry to do
that was the 2026-08-23 P2/4 head finding.

**What it is.** "Enhanced YOLOv11n for small object detection in UAV
imagery: higher accuracy with fewer parameters" (Zhu, H. & Xie, X.),
*Scientific Reports* (Springer Nature), article `s41598-026-35301-2`,
published 18 January 2026 (preprint on Research Square, 15 October 2025,
`10.21203/rs.3.rs-7553905/v1`; also indexed at IEEE Xplore, document
`11196626`). It proposes a four-part, YOLOv11n-native recipe evaluated on
VisDrone2019 (small, low-contrast objects seen from a moving camera —
architecturally the closest published analogue this log has found to a
dark clubhead against cluttered foliage): (1) **MSEAF** (Multiscale
Edge-Feature Adaptive Selection), a backbone module aimed at weak-edge,
small-object signal; (2) **ScalCat** and **Scal3DC**, neck-reconstruction
modules that add a P2 (stride-4) detection head — the same P2 addition
this log already logged 2026-08-23, but here as one piece of a larger,
empirically-tested recipe rather than a bare config edit; (3) **SRepD**, a
shared, reparameterized lightweight detection head meant to offset the
P2 head's extra compute. Reported result (from search-indexed abstract/
summary text, not a primary-source read — see verification limits below):
**+4.6% mAP50 and +4.6% Precision over the YOLOv11n baseline, with ~8.5%
fewer parameters** — i.e. this is offered as evidence the P2-head compute
cost this log flagged as a risk in the 2026-08-23 entry (~2x GFLOPs for
YOLOv8n-p2) can be substantially offset by pairing it with a lighter head,
not just accepted as a tax.

**Verification performed, and its limits.** This is an existence-only
result, same disposition as this log's DFRCP, ReynoldsFlow, and
TinyDark-YOLO entries. Every primary host attempted this run returned
`EGRESS_BLOCKED` or was unreachable from this sandbox on direct fetch:
`www.nature.com` (the article itself), `assets-eu.researchsquare.com` (the
preprint PDF), `ieeexplore.ieee.org`, `api.semanticscholar.org`, and
`sciety-labs.elifesciences.org` (DNS failure, not a proxy block — a
distinct failure mode from every other host tried). This is consistent
with every prior run's standing restriction on academic-publisher hosts
(arXiv, HuggingFace, ResearchGate, Research Square all previously
blocked); `nature.com` and `ieeexplore.org` are new hosts added to that
observed blocklist by this run. What is corroborated across multiple
independent search results (title, exact author names, journal, article
number, DOI, publication date, and the module names/numbers above) is
consistent and specific enough to treat the paper's **existence** as
solid; the **quantitative claims, the actual architecture diagrams, and
any code/data availability statement are not independently verified** —
no GitHub repository for this specific paper could be found by search,
and unlike the 2026-08-23 P2-head entry (which verified the config file
directly from `raw.githubusercontent.com`), nothing here was read from a
primary source. **Licence: not verified**, but flagged as a reasonable
expectation rather than a confirmed grant — *Scientific Reports* publishes
exclusively open-access under CC BY 4.0 by default (Springer Nature's
standing policy for that journal, not specific to this article), which
would permit commercial use with attribution if the article follows that
default; this was not confirmed against the article's own licence
statement and must not be treated as established.

**Which failure mode.** Camouflage / small-object-primary, same
classification as the 2026-08-23 P2-head entry it extends — MSEAF and the
P2 head target detectability of a small, low-contrast object with weak
edges, not motion elongation. No blur-specific claim is made anywhere in
what could be verified.

**Why it helps this model specifically.** This project already has a
concrete, low-effort candidate experiment logged (2026-08-23: add a P2
head to YOLO11n, retrain, re-export) with one open risk flagged at the
time — roughly doubled on-device compute, unconfirmed against this
project's own latency headroom. This entry is independent, empirical
(if unverified) evidence that a P2 head plus a lightweight reparameterized
head (SRepD) and a small-object-tuned backbone module (MSEAF) recovers
accuracy *and* cuts parameters, on the same base architecture, on a task
that shares this model's core symptom (small, low-contrast, cluttered-
background target). If the numbers hold up under a primary-source read,
this reframes the 2026-08-23 experiment from "P2 head alone, pay the
compute tax" to "P2 head plus the head-side savings this paper reports,"
which is strictly more attractive and directly answers that entry's own
flagged open question. It does not, on its own, change the recommended
next step — the P2-head experiment is still the actionable one — but it
raises the ceiling worth testing for once that experiment is run.

**Effort vs. payoff.** Effort so far: low (search only; five primary hosts
attempted and blocked, consistent with this sandbox's established
limitation rather than a sign the paper is unreal). Effort to actually use
this: someone with unrestricted network access reading the Nature or
Research Square page directly to confirm module details, the exact
ablation numbers, and the licence — a same-day check, per this log's usual
convention for egress-blocked academic sources. Payoff if confirmed:
moderate — it strengthens rather than replaces the already-logged,
already-actionable P2-head experiment; it is not a new independent
direction. Not logged as a strong standalone finding on its own; logged
because it is new information tightly coupled to this log's own highest-
confidence camouflage-side recommendation, and because closing the
"is the compute cost actually a blocker" question the 2026-08-23 entry
left open is worth a cheap follow-up read even without a primary-source
verification yet.

---

## 2026-09-01 (second run) — PAL (Portable Active Learning for Object Detection, CVPR 2026 Highlight): a detector-agnostic labeling-prioritization framework for the data-engine gap, not a blur/camouflage fix (existence-only result)

**Area covered.** Rotated to bullet 5 (synthesizing/augmenting training
data) via its data-engine-efficiency angle, away from the last three runs
which were all camouflage-architecture (TSM, CSRDA, Enhanced YOLOv11n on
2026-08-31/09-01). Also touches the golf-dataset rotation area first this
run: a sweep for new commercially-licensed indoor/low-light/simulator golf
footage (Roboflow Universe listings, a SCIRP pose-estimation paper, a
Hugging Face search) turned up nothing beyond what 2026-08-13/18/21 already
logged and ruled out — not logged as its own entry per the no-padding rule,
noted here only so a future run doesn't re-spend a cycle on the same dead
ends.

**What it is.** "Portable Active Learning for Object Detection" (PAL) —
Rashi Sharma, Justin Timothy C. Bersamin, Karthikk Subramanian (Panasonic
R&D Center Singapore), CVPR 2026, selected as a **Highlight** paper. PAL is
described (consistently across independent sources — see Verification
below) as **detector-agnostic and training-pipeline-agnostic**: it operates
*solely on a trained detector's own inference outputs* on unlabeled
imagery, not on model internals, gradients, or architecture-specific
hooks. It couples two signals: **LIUS** (Logistic-based Instance
Uncertainty Scoring) — lightweight, per-class logistic classifiers trained
to distinguish true from false positives using only two detector-output
features (pre-NMS box count and detection confidence) — with **GUIDE**
(class-weighted image entropy + a rare-class diversity index + a
rank-conditioned similarity penalty) to rank *unlabeled* images by how
much a human label on them would be worth. Reported (via search-indexed
excerpts, not read directly — see caveat) to match prior-SOTA accuracy
while needing roughly 20% less annotation.

**URL.** Paper: https://arxiv.org/abs/2605.10349 (also
https://arxiv.org/html/2605.10349 and
https://arxiv.org/pdf/2605.10349). CVF listing:
https://openaccess.thecvf.com/content/CVPR2026/papers/Sharma_Portable_Active_Learning_for_Object_Detection_CVPR_2026_paper.pdf.
CVPR poster page: https://cvpr.thecvf.com/virtual/2026/poster/38968.
Panasonic press release naming it as one of two CVPR 2026 acceptances:
https://news.panasonic.com/global/press/en260528-3.

**Verification.** `arxiv.org` (all three paths), `cvpr.thecvf.com`,
`openaccess.thecvf.com`, `news.panasonic.com`, and
`api.semanticscholar.org` were all tried directly via fetch this run and
**all five are blocked by this sandbox's egress proxy** — the same
standing restriction every prior run of this log has hit for non-GitHub
academic hosts. A targeted search for a code repository (author name +
"Portable Active Learning" + "github", and independently for the specific
mechanism "class-wise logistic classifier uncertainty diversity active
learning object detection") found **no GitHub repository, project page, or
any code release at all**. What's logged above is reconstructed from
search-engine-indexed excerpts of the abstract/method that agree closely
across multiple independently-worded search results (arXiv's own listing,
the CVF PDF listing, the CVPR poster page, and Panasonic's own press
release identifying the paper by name and authors) — that cross-source
agreement is why this clears this log's bar for "confirmed to exist" (same
standard as the CADDIE and Enhanced-YOLOv11n entries), but the exact
numbers (the "~20% less annotation" figure) and every implementation
detail beyond the two-paragraph summary above should be treated as
unverified until someone with real network access reads the PDF.

**Licence.** Not applicable in the usual sense — there is no dataset or
pretrained model to license, and no code exists to license either. This is
a *method* described in a paper; using it means reimplementing the LIUS/
GUIDE scoring on top of this project's own YOLO11n outputs, which is an
engineering task, not a licence question. (Standard academic norms mean
the paper text itself is presumably all-rights-reserved by CVF/IEEE, but
that restricts redistributing the paper, not implementing the described
algorithm.)

**Which failure mode.** Neither, directly — this is a **data-engine /
labeling-efficiency** technique, not a detection fix. It is being logged
under the "augment/synthesize training data" rotation bullet because its
actual target is this project's stated 29%-own-footage-vs-54%-Roboflow
imbalance: if the team is shooting more of its own phone footage (indoor,
low light, real swings) to close that gap, PAL-style ranking would tell
them which *raw, unlabeled* clips are worth a human annotator's time
first, rather than labeling footage in capture order.

**A specific, important caveat for this model.** The described mechanism
(LIUS) needs the detector to *emit* a box, even a low-confidence one — it
scores uncertainty from pre-NMS box count and confidence. This project's
own camouflage failure mode, per the failure analysis this log's every run
is required to read, is **zero candidate detections even at confidence
0.05** on some genuinely hard frames. On a true zero-detection frame, LIUS
has no per-instance signal to score at all; only the image-level GUIDE
half (entropy/diversity, computed over whatever weak signal exists) could
plausibly still flag such a frame as worth labeling, and that is a weaker,
less-targeted signal than the paper's headline LIUS mechanism. This is not
a reason to discard the idea, but it means PAL should not be assumed, sight
unseen, to specifically solve the "which frames have a totally invisible
clubhead" prioritization problem — that needs verifying against the actual
paper (or a pilot) before relying on it for exactly the hardest camouflage
frames this project cares about most.

**Effort vs. payoff.** Low effort spent this run (search-and-verify only,
~8 queries and 5 blocked fetches, no code found to inspect). Payoff is
speculative but plausibly real and cheap to capture: if the team is
already running the current on-device model over new raw phone footage
before labeling (which the data-engine doc implies is roughly the
workflow), computing two extra numbers per frame — pre-NMS box
count/confidence and an image-entropy score — from outputs that already
exist is a small addition, not a new pipeline. But there is no code to
adopt, so "low effort" only holds for a rough reimplementation of the
idea's spirit (rank unlabeled frames by detector uncertainty +
diversity before sending them to annotators), not the paper's exact,
tuned method. Recommended next step: whoever has real network access
should read the actual PDF to get the precise LIUS/GUIDE formulas and the
ablation showing which component matters most, before spending
implementation time — and specifically check whether the paper's own
results include a "zero detections" edge case, which would resolve the
caveat above either way.

---

## 2026-09-01 (third run) — No new verified finding: Kornia's motion-blur augmentation checked and ruled out as redundant; every other lead blocked by sandbox egress

**What was checked.** Rotating away from this run's earlier two entries
(architecture, data-engine/active-learning), this run searched the
dataset, blur-augmentation, temporal-camouflage, and golf-pose/tracking
areas for something genuinely new. Leads found and their disposition:

- **Kornia's `RandomMotionBlur` / `kornia.filters.motion_blur`**
  (`github.com/kornia/kornia`) — confirmed real via direct fetch of the
  repo page: **Apache-2.0**, commercial use permitted. But its mechanism
  is a straight-line convolution kernel over a static frame — the same
  "naive linear-kernel" approach this log's own 2026-08-27 Albumentations
  entry logged (noting its actively-maintained successor went AGPL) and
  the 2026-08-30 RSBlur entry explicitly flagged as **"measurably wrong"**
  compared to physically-grounded blur (PSF-based or real
  frame-averaging). Kornia is a different library with a real, permissive
  licence, but it is not a *new* technique — it is the same
  already-logged-and-criticized shortcut in a third implementation.
  **Not logged as a standalone entry; ruled out as redundant.**
- **YUV20K** (arXiv 2604.09985, Apr 2026) — a camouflaged-object-detection
  benchmark explicitly targeting "Motion-Induced Appearance Instability"
  under large-displacement/camera motion, which lines up unusually well
  with this project's own camouflage-vs-motion framing. Could not be
  verified beyond the search-indexed abstract: `arxiv.org` (both
  `/abs/` and `/html/` paths) is blocked by this sandbox's egress proxy,
  same as every prior run. No GitHub repo or licence could be found or
  confirmed. Benchmark-only in any case (not training data), so payoff
  would be indirect even if verified.
- **ODGEN** (NeurIPS 2024, Apple + academic co-authors) — a diffusion
  model that generates bounding-box-conditioned synthetic training images
  for object detectors, reporting up to +25.3 mAP on domain-specific
  detection benchmarks. Directly relevant to the "synthesize training
  data" rotation bullet, but a GitHub code search
  (`github.com/search?q=ODGEN...`) returned **zero repositories** — no
  code release found anywhere — and `machinelearning.apple.com` and
  `paperswithcode.com` are both blocked, so even the paper's own claims
  couldn't be re-read past search snippets. Existence-only, and with no
  code, not actionable regardless.
- **HUE Dataset** (arXiv 2410.19164) — a low-light dataset with paired
  event-camera and frame sequences including dim indoor content. Its
  event-camera half doesn't apply to a phone-only capture pipeline, and
  its project page (`ercanburak.github.io`) is blocked, so the
  frame-only subset's licence and standalone usability couldn't be
  checked. Not pursued further given the hardware mismatch.
- **`sunholee1217/golf`** (Hugging Face) — search results describe an
  MIT-licensed golf video dataset, which would be a genuinely useful,
  cheap lead if confirmed. `huggingface.co` is blocked by this sandbox's
  egress proxy, so neither the licence nor the actual content (clip
  count, indoor/outdoor mix, resolution) could be verified. **Left
  unconfirmed — worth a follow-up run with network access to
  huggingface.co, not logged as a finding.**
- Roboflow Universe golf-clubhead projects ("Golf Driver Tracker,"
  "golf-club-tracking," etc., several thousand images each) — same
  problem: `universe.roboflow.com` is blocked, so licence terms (Roboflow
  Universe's per-project licence varies) and image content couldn't be
  checked. Not logged.
- A GitHub search for recently-updated golf-clubhead/YOLO repos and for
  bounding-box motion-blur-synthesis repos hit an HTTP 429 (rate limited,
  `Retry-After: 3600`) partway through this run, before any new
  repository could be found or checked.

**Which failure mode.** N/A — nothing here cleared this log's bar for a
loggable finding.

**Effort vs. payoff.** Moderate effort (roughly a dozen searches plus
~10 fetch attempts across arxiv.org, huggingface.co,
universe.roboflow.com, machinelearning.apple.com, semanticscholar.org,
paperswithcode.com, and ercanburak.github.io — all blocked by this
sandbox's egress proxy — plus two github.com fetches that succeeded
before a third hit rate-limiting). Zero payoff this run: no new
finding cleared verification. The two most promising unconfirmed leads
for a future run with broader network access are **YUV20K**
(camera-motion-instability VCOD benchmark) and **`sunholee1217/golf`**
(claimed MIT-licensed golf video on Hugging Face) — both worth a direct
check, not a fresh search, next time.

---

## 2026-09-01 (fourth run) — CaddieSet (CVPR 2025W): a real, MIT-licensed, indoor-captured golf swing dataset — verified, and verified to be useless for this model (no video, no clubhead annotation)

**Area covered.** Golf-specific pose/tracking + golf dataset rotation
areas (bullets 1 and 4), last touched 2026-08-28 (PiTrac) and 2026-08-29
(YOLO-Ball) respectively. Rotated away from this run's own third-run
predecessor (architecture/blur-augmentation, all blocked). Directly
followed up the third run's own advice to check `sunholee1217/golf` next
rather than re-searching — see Verification below for why that lead is
still unresolved — and independently surfaced CaddieSet via a broader
"golf swing dataset commercial license indoor simulator" search.

**What it is.** CaddieSet — Jung et al. (Dami Lab), "CaddieSet: A Golf
Swing Dataset with Human Joint Features and Ball Information," CVPR 2025
Workshop (CVSPORTS). Per its own README, it covers **1,757 golf shots**
(924 FACEON-view, 833 Down-The-Line-view) from 8 golfers of varying skill,
captured on what the repository describes as a **camera-based launch
monitor system** — i.e. genuinely indoor, controlled-bay footage, the
exact capture regime this project's README flags as never measured. The
published artifact is joint keypoints (extracted via CV across 8 swing
phases), 22 derived biomechanical swing features (shoulder angle, hip
rotation, spine angle, etc.), and ball-flight data (speed, carry, spin,
direction). **No clubhead bounding boxes, and — critically — no
raw video or image files are actually distributed.**

**URL.** Code/data repo: https://github.com/damilab/CaddieSet. Paper (CVF
listing, unreachable this run — see Verification):
https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Jung_CaddieSet_A_Golf_Swing_Dataset_with_Human_Joint_Features_and_CVPRW_2025_paper.html.

**Verification.** Fetched directly, successfully, three times this run:
the repo root, the `LICENSE` file, and the `data/` subdirectory listing —
all via `github.com`/`raw.githubusercontent.com`, which this log's prior
runs have consistently found reachable where `arxiv.org`,
`huggingface.co`, `openaccess.thecvf.com`, and `universe.roboflow.com` are
not (true again this run: the CVF paper page for this exact dataset was
`EGRESS_BLOCKED`). The repo's top level is exactly three items: `data/`,
`LICENSE`, `README.md`. The **`data/` folder contains exactly one file:
`CaddieSet.csv`** — no video, no images, no annotation files of any kind.
The `LICENSE` file's full text was read directly and is the standard MIT
License, `Copyright (c) 2024 damilab`, permitting commercial use,
modification, and redistribution (only a copyright/permission-notice
requirement). The README confirms the CSV-only, no-clubhead-annotation
picture: it lists ball-flight fields and 22 biomechanical joint-derived
features, with no field for clubhead position, bounding box, or any
per-frame image reference. `sunholee1217/golf` (the third run's flagged
follow-up) remains **unconfirmed**: `huggingface.co` is still blocked by
this sandbox's egress proxy on this run too, and unlike CaddieSet it has
no mirroring GitHub repo that a search surfaced, so it stays an open lead
for a future run with different network access, not a finding here.

**Licence.** MIT, quoted in full above from the primary source. Commercial
use is unambiguously permitted — but the licence only covers a CSV of
derived numeric features, not any of the underlying imagery, so the
licence clearance is close to moot for this project's purposes.

**Which failure mode.** Neither, in practice, despite matching the
project's stated indoor-footage gap on paper. It was checked specifically
against the "indoor/low-light/simulator footage where real motion blur
appears" rotation bullet and the README's own note that indoor performance
has never been measured — CaddieSet is exactly the right *kind* of source
(indoor launch-monitor bay, real swings, permissive licence) but delivers
none of the artifact this project could actually use: no frames to look
at, let alone label a clubhead box on, so it does not help camouflage,
motion blur, or the training-data imbalance.

**Effort vs. payoff.** Low-moderate effort (three targeted GitHub fetches
plus two searches, all successful — no egress fights this run once the
CVF paper mirror was abandoned as blocked). Zero payoff for training or
even qualitative frame review: this is a clean, verified negative result,
not a padding entry — it forecloses a specific, plausible-sounding lead
(the dataset's name and abstract both suggest exactly the indoor swing
footage this project lacks) that a future run would otherwise be tempted
to re-discover and re-chase. The one open thread worth another run's time
is still `sunholee1217/golf` on Hugging Face, which — unlike CaddieSet —
was described in search snippets as video content, not derived CSVs; it
needs either direct `huggingface.co` access or a GitHub/other mirror to
resolve.

---

## 2026-09-02 — DoveNet / the bcmi iHarmony4 repo: an MIT-licensed image-harmonization codebase that fixes the exact realism gap this log's own copy-paste entry flagged

**Area covered.** Bullet 5 (ways to synthesize/augment training data), specifically
a direct follow-up to this log's 2026-08-14 Copy-Paste augmentation entry rather
than a fresh area. That entry logged `conradry/copy-paste-aug` (MIT, verified) as
the cheapest camouflage-directed data synthesis idea in this log, but flagged one
concrete, named weakness in its own effort/payoff section: the pasted clubhead
crop is blended with only a feathered-edge alpha mask (Gaussian smoothing), not
gradient-domain or learned harmonization, so "composited realism is inherently
imperfect (lighting/shadow/scale mismatch between the crop's original scene and
the new background)." This run searched specifically for a permissively-licensed
fix to that named gap rather than starting a new area, per this log's own
"first check whether this log's existing entries already cover it" convention.
`sunholee1217/golf` (the open thread from the prior run) was re-attempted first —
`huggingface.co` is still `EGRESS_BLOCKED` this run, consistent with every prior
run's finding, so that thread remains unresolved and is not re-logged here.

**What it is.** `github.com/bcmi/Image-Harmonization-Dataset-iHarmony4` — the
official code release for "DoveNet: Deep Image Harmonization via Domain
Verification" (Cong et al., CVPR 2020), maintained by the BCMI lab that has
produced most of the image-harmonization literature. Image harmonization takes a
composited image (foreground pasted onto a new background, exactly copy-paste
augmentation's output) and adjusts the foreground's color/lighting statistics to
match the background — i.e. it is a direct, purpose-built answer to the
lighting/shadow mismatch the copy-paste entry named as its open weakness, not a
generic image-editing tool. The same repo's README benchmark table also lists
several newer, faster alternatives trained on the same task (HDNet, CDTNet,
PCTNet) with their own repos linked, so DoveNet is a verified entry point into
an actively maintained family, not a single dead-ended model.

**URL.** https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4 (code at
its `DoveNet/` subdirectory: `train.py`, `test.py`, `models/`, `options/`,
`scripts/`, `util/`, plus a pretrained checkpoint linked from the README).

**Verification.** Fetched the repo's raw `LICENSE` file directly (not a badge or
README claim): full MIT License text, BCMI, 2022 — "Permission is hereby granted,
free of charge... without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies," with only the copyright/permission-notice condition and an "as is"
disclaimer. Also fetched the `DoveNet/` subdirectory listing directly and
confirmed it contains real training/inference code, not just documentation:
`train.py`, `test.py`, model definitions under `models/`, plus a pretrained
checkpoint (`latest_net_G.pth`) linked from the README with reported MSE/PSNR
numbers per source dataset. Separately checked the README for what iHarmony4's
*images* are built from (Microsoft COCO, MIT-Adobe FiveK, Flickr, a "day2night"
set) — those source images carry their own separate licences and are *not*
needed for this project's use case, since the intended use here is running the
harmonization code on this project's own already-permissively-labelled crops and
its own negative-frame backgrounds, not redistributing iHarmony4 itself. The
alternative harmonization model checked first, `ZHKKKe/Harmonizer` (ECCV 2022,
lighter-weight at 20MB with a real-time/mobile-oriented claim) was ruled out:
its README states the license verbatim as "Creative Commons Attribution
NonCommercial ShareAlike 4.0" — non-commercial, so DoveNet (or one of the
MIT-covered faster successors in the same repo) is the one to use, not
Harmonizer.

**Licence.** MIT, quoted in full above from the primary source (the code repo).
Commercial use of the code is unambiguously permitted. (`ZHKKKe/Harmonizer`,
checked as the more lightweight alternative, is CC BY-NC-SA 4.0 and therefore
commercially blocked — logged here as a ruled-out alternative, not as a separate
entry, to save a future run from re-discovering and re-checking it.)

**Which failure mode.** Camouflage, indirectly — same as the copy-paste entry it
extends. It does not detect a camouflaged clubhead; it improves the realism of
*synthetic* camouflage-style training examples (dark clubhead crop composited
onto dark clothing/foliage background) that copy-paste already produces cheaply
but with a known lighting/shadow mismatch. It has no bearing on the motion-blur
failure mode.

**Why it helps this model specifically.** This is not a new idea, it is the
missing second half of an idea already in this log. Copy-paste augmentation was
logged as this project's cheapest camouflage-directed lever specifically because
it needs no architecture change and no CoreML re-validation — only the training
set grows — but its own effort/payoff section already conceded that feathered-
alpha blending alone risks producing composites a network learns to detect as
"pasted" rather than as genuinely hard camouflage examples, which would waste
the augmentation's value. DoveNet (or PCTNet/HDNet from the same repo, all under
the same MIT umbrella per the repo's own claim, though only DoveNet's own
LICENSE file was verified directly this run) slots into that exact pipeline
step: composite with copy-paste's box/mask logic, then run the harmonization
network once per composite as an offline preprocessing pass before training,
with zero change to the detector itself.

**Effort vs. payoff.** Low verification effort this run (two GitHub fetches
found the working code and the license; one more ruled out the lighter-weight
alternative on licence). Real integration effort is moderate, not low: it is a
second offline model to run (with its own PyTorch environment and checkpoint
download) inserted between copy-paste's mask-based compositing and training,
and the harmonization model's own domain (COCO/Adobe5k photographic scenes) is
a further distribution shift from golf footage that hasn't been validated
empirically — so, like the copy-paste entry itself, this should be evaluated on
the held-out test set before trusting it to beat plain feathered-edge blending,
not assumed to help. The payoff is bounded to fixing one named, verified
weakness in an already-logged idea, not a new capability on its own, so
priority should stay behind actually shipping copy-paste augmentation first and
only adding this if unharmonized composites measurably underperform.

---

## 2026-09-02 (second run) — YUV20K resolved: the open thread from the 2026-08-31/2026-09-01 runs is a real benchmark, but non-commercial and its code isn't out yet

**Area covered.** Bullet 3 (small/low-contrast/camouflaged object detection,
temporal methods) — specifically closing out a named open lead rather than
starting fresh. Three prior runs (2026-08-31 fourth run's predecessor and
earlier) flagged `YUV20K` as "worth a direct check next time" after search
snippets alone suggested a camera-motion-instability VCOD benchmark, but
`arxiv.org` was blocked every time anyone tried to load the paper itself.
This run re-attempted `arxiv.org/html/2604.09985` directly first — still
`EGRESS_BLOCKED`, consistent with every prior run's finding for that domain —
then found and used a `github.com` mirror instead, which (as with this log's
CaddieSet and DoveNet entries) was reachable. `sunholee1217/golf`, the other
standing open thread, was also re-attempted (`huggingface.co/datasets/...`)
and is still `EGRESS_BLOCKED`; it remains unresolved, not re-logged here.

**What it is.** Liu, Yiyu et al., "YUV20K: A Complexity-Driven Benchmark and
Trajectory-Aware Alignment Model for Video Camouflaged Object Detection,"
arXiv:2604.09985 (2026). A pixel-level-annotated VCOD benchmark — 24,295
frames across 91 scenes, 47 species — built specifically to stress
large-displacement motion and camera motion as complexity axes, paired with
a proposed model (Motion Feature Stabilization + Trajectory-Aware Alignment)
that uses motion, not appearance, to separate camouflaged objects from
background — the same category of temporal mechanism this log's SLT-Net,
SAM-PM, EMIP, and Vcamba entries already cover, and this run's rotation
bullet specifically asks for.

**URL.** Code/dataset repo: https://github.com/K1NSA/YUV20K. Paper:
https://arxiv.org/abs/2604.09985 (unreadable this run, per above).

**Verification.** Fetched the repo's raw `LICENSE` and `README.md` directly
via `raw.githubusercontent.com`. `LICENSE` is the standard MIT License
(K1NSA, 2026) — full permissive grant, copyright/notice condition only. The
README states the **dataset** carries a separate, different license: CC BY-NC
4.0, with the line "any commercial usage of this dataset is strictly
prohibited," and further discloses the source videos are scraped from the
public internet under a fair-use claim for academic research, with copyright
belonging to the original creators and takedown-on-request. The README's own
roadmap section lists source-code release as pending with no date — i.e. the
MFS/TAA model implementation itself is not yet published, only the dataset
and paper.

**Licence.** Two different licences on one repo, both confirmed from primary
source: the repo/code shell is MIT (permits commercial use, but there is no
model code in it yet to use), while the dataset is CC BY-NC 4.0 and
explicitly commercial-use-prohibited — and separately compromised for this
project's purposes regardless of licence, since the source videos are
resold/rehosted internet clips of unclear per-clip provenance, not something
a commercial app should build a training set on even if the wrapper licence
allowed it.

**Which failure mode.** Camouflage. Its "large-displacement motion, camera
motion" complexity axes are inter-frame displacement and viewpoint change —
the thing this project's own brief warns not to conflate with intra-frame
motion blur — so it has no bearing on the blur failure mode; it is one more
data point for the temporal-motion-cue camouflage mechanism this log has now
verified from five independent groups (SLT-Net, SAM-PM, EMIP, Vcamba, and
this one).

**Why it helps this model specifically.** It doesn't, directly, right now:
neither the dataset (non-commercial, unclear provenance) nor the model code
(not released) is usable today. Its value is closing a specific, named
uncertainty three separate prior runs left open, so a future run stops
re-discovering and re-flagging the same lead — the same service the
CaddieSet and Deblur-YOLO entries already provide elsewhere in this log.

**Effort vs. payoff.** Low effort (one blocked `arxiv.org` fetch, one search,
two successful raw-GitHub fetches). Zero payoff for training or architecture
today — both the licence and the code-not-shipped status rule this out — but
it converts a three-times-repeated "worth checking" note into a closed,
citable negative, which is exactly what stops it from costing a fourth run's
time. The one remaining open thread from this log's rotation is
`sunholee1217/golf`, still blocked by this sandbox's `huggingface.co` egress
block on every run that has tried it; it needs either a Hugging Face mirror
on GitHub or a session with different network access, not another identical
retry.

---

## 2026-09-02 (third run) — `onkar-99/Golf-Ball-Tracking`: independent, golf-specific confirmation that motion blur breaks YOLO clubhead detection at inference, plus a training-free optical-flow/tracker fallback — no licence, so it corroborates rather than supplies reusable code

**Rotation note.** The two prior 2026-09-02 runs (DoveNet/iHarmony4, YUV20K)
were both camouflage/VCOD-area findings. This run deliberately switched to
the golf-specific pose/club-tracking rotation slot, which this log's last
few runs had left for datasets and architecture leads instead.

**What it is.** A small, real, MIT-free (see below) GitHub repo,
misleadingly named "Golf-Ball-Tracking" but — verified directly from its own
raw README, fetched via `raw.githubusercontent.com`, not a search snippet —
actually about **clubhead** tracking: "we found the golf head locations in
the video using YOLOv5. The model was trained on custom data... However,
since the golf swing was very fast, the golf club was blurred in most of the
frames and difficult to detect." The author's stated fix is two training-free
fallbacks for frames where the YOLOv5 detector fails: (1) **optical flow**
— threshold the frame-difference image, then pick the contour nearest (by
centroid distance) to the club's last known location; (2) a **Dlib
correlation tracker** seeded from the last confirmed detection, carried
forward while detection keeps failing. The author's own honest conclusion,
quoted verbatim: "Although both these approaches were implemented correctly,
they were not as promising as were expected to be," and under "Areas of
Improvement": "Since the golf club is blurred while downswing, the tracking
and detection fails."

**URL.** https://github.com/onkar-99/Golf-Ball-Tracking (README fetched
directly and verbatim via
`raw.githubusercontent.com/onkar-99/Golf-Ball-Tracking/master/README.md`;
the file listing — `Dlib/`, `Optical Flow/`, `README.md`,
`golf_club_tracking.PNG` — was confirmed by loading the GitHub repo page
itself, not a search index. `api.github.com` is blocked by this sandbox's
egress proxy, same restriction prior runs hit, so repo metadata like
star count and last-push date could not be pulled, but the two content
fetches above did not depend on that endpoint.)

**Licence.** None. There is no `LICENSE` file in the repo's file listing and
no "License" entry in GitHub's sidebar — confirmed by direct inspection of
the repo page, not inferred. The YOLOv5 weights themselves are hosted
externally on Google Drive with no licence terms stated on that link either.
**Commercial use of this repo's code or weights is not established as
permitted** — GitHub's default-licence rule means "no licence" is
all-rights-reserved by the author, not public domain. That rules this out
as a source of code or weights to actually pull into this project's
pipeline. What it is not blocked from being used as: a public, verifiable
fact pattern — the empirical result an independent developer hit and wrote
down, not the specific lines of Python.

**Which failure mode.** Motion blur, primarily, and it is independent
evidence rather than a new mechanism: this project's own repo does not yet
have a measured indoor/low-light blur failure rate (the indoor test set is
quarantined per the README), so this is the first source in this log that
is a *second, unrelated team* independently hitting the identical failure
this project's caveat already predicts — a from-scratch YOLO clubhead
detector failing specifically on the fast-downswing frames where the club is
blurred, described by someone with no connection to this project or its
labeling spec. It is a second, cheap-and-shipped-looking data point for the
recovery-not-appearance tracker mechanism this log already logged from
`OC-SORT` (2026-08-26, training-free Kalman recovery from detection gaps)
and `InpaintNet` (2026-08-17) — but golf-specific and inference-only, not a
generic mechanism from an unrelated sport.

**Why it helps this model specifically.** Two things, neither of which
requires touching this repo's code. First, it is a second confirmation,
from a source with zero incentive to agree with this project's own failure
analysis, that a from-scratch YOLO-family clubhead detector's dominant
failure is exactly what this project's brief already flags: the club
"blurred in most of the frames" during the downswing, not edge cases or
rare poses. That should raise, not lower, confidence in prioritizing the
motion-blur regime over further camouflage-mechanism mining, which is where
this log's last several runs have concentrated. Second, the two fallback
techniques described (optical-flow contour-nearest-centroid; last-detection
correlation tracking) are simple enough to reimplement from the README's
description alone in an afternoon, with no need for the unlicensed code —
they're the same training-free "keep going after a missed detection" idea
as the already-logged OC-SORT entry, just concretely golf-shaped. Combined
with OC-SORT's own citation and MIT licence, that gives a two-source basis
for building a small on-device recovery tracker without touching training
data or the CoreML model at all.

**Important caveats.** (1) The author's own conclusion is negative — both
fallbacks under-performed expectations, and no quantitative before/after
detection-rate numbers are given, only the qualitative "not as promising as
expected." This is corroboration of the *problem*, not evidence that *this
particular fix* works; OC-SORT's own (also unquantified, per this log's
2026-08-26 entry) recovery mechanism is the more rigorously-sourced
candidate to actually build from. (2) No licence means the code itself
cannot be adapted or copied into this project; only the described technique
(optical flow + nearest-centroid; correlation-tracker carry-forward) is
reusable, and it would need to be written from scratch. (3) It is a single
anonymous hobby project — one data point, not a benchmark or paper, and its
own YOLOv5 model and training data are unverified and inaccessible (Google
Drive link, not checked, and out of scope even if it were, given the
licence gap). (4) It says nothing new about camouflage, indoor/low-light
capture, or licensed datasets — it only reinforces the motion-blur framing
already in this project's brief.

**Effort vs. payoff.** Very low effort (two direct GitHub content fetches,
no blocked hosts this run). Payoff is evidentiary, not a ready-to-use asset:
it does not hand this project a dataset, a licensed model, or a novel
technique beyond what OC-SORT already logged — its value is narrowly that
it is independent, real-world confirmation (not this project's own
assumption) that motion blur, not edge-case appearance, is the dominant
failure mode for a YOLO-family clubhead detector once a downswing gets
fast, which is a useful cross-check before committing further research
cycles to camouflage-only leads.

---

## 2026-09-02 (fourth run) — Motion-vector-guided, patch-selective blur synthesis (Bright et al., "Mitigating Motion Blur for Robust 3D Baseball Player Pose Modeling for Pitch Analysis," ACM MMSports 2023): the missing "where to blur" piece for this log's own flagged gap

**Area covered.** Rotated away from golf-specific pose/tracking (this run's
predecessor, `onkar-99/Golf-Ball-Tracking`, was the third run today) back to
bullet 5, "ways to synthesise or augment training data for either failure
regime" — specifically the motion-blur regime. Before settling on this, I
tried the dataset bullet again (a fresh `universe.roboflow.com` fetch for
`club-head-tracking/golf-club-tracking` still returns `EGRESS_BLOCKED`,
same standing wall as every prior run — not logged as new, since Roboflow
inaccessibility is already an established fact per the 2026-08-23 entry)
and a VCOD-mechanism candidate (CamoSAM2, a SAM2-based prompt-refinement
paper with real code at `github.com/zhangxin06/CamoSAM2`) that I ruled out
before writing it up: it uses the full, un-distilled SAM2 encoder, which
the 2026-08-19 SAM-PM entry already closed off as a category for this
on-device deployment target ("cannot run per-frame on an iPhone... unless
the project's deployment target changes") — EdgeTAM is still the only
member of that family this log has found that reopens it, so a ninth
full-SAM2 VCOD paper adds nothing SAM-PM didn't already establish.

**What it is.** Bright, Ilic, Kim, Wu, Poitras, Dickey & Clausi (University
of Waterloo VIP Lab), "Mitigating Motion Blur for Robust 3D Baseball Player
Pose Modeling for Pitch Analysis," presented at the 6th International
Workshop on Multimedia Content Analysis in Sports (MMSports '23, co-located
with ACM Multimedia 2023). The paper's stated problem is, near verbatim
from search-indexed summaries: accessible **broadcast video at 30fps**
produces **partial motion blur during fast athletic action** (a pitcher's
arm/wrist through the delivery), which degrades 2D/3D pose-keypoint
estimation. Their fix is a "motion blur learning module": a synthetic-blur
augmentation pipeline that (1) estimates dense motion-flow vectors between
consecutive frames of already-existing, otherwise-sharp training video, (2)
uses that flow field to identify which **patches/regions** show
significant motion, and (3) synthesizes blur **selectively in those
regions only**, rather than uniformly over the whole frame. Reported
result (search-summary only): a 54.2% and 36.2% loss reduction on 2D and
3D pose estimation respectively versus the un-augmented baseline.

**Why this is a different mechanism from what's already in this log.**
Every existing blur-synthesis entry here is either (a) object-local but
needs a hand-picked or fixed-direction kernel (PSF box-expansion,
2026-08-14: one directional kernel per cropped clubhead), (b) needs real
high-frame-rate/slow-mo source footage to average into a blur (Brooks &
Barron frame-averaging, 2026-08-12; FILM as the standard-frame-rate
enabler for it, 2026-08-31), (c) models whole-frame *camera* shake, not
subject motion (6-DOF, 2026-08-25), or (d) is a whole-frame, generic,
non-selective kernel already flagged as a known problem in this exact log
— the 2026-08-27 (third run) Albumentations `MotionBlur` entry states
explicitly: "`A.MotionBlur` convolves a directional kernel over the
*entire* frame uniformly — it does not selectively blur only the clubhead
region... even a modest kernel... could smear the head's true extent past
its original tight label box, silently reintroducing the exact
box/visual-extent mismatch this log's PSF and frame-averaging entries were
designed to fix." This paper's flow-guided, patch-selective step is a
direct, purpose-built answer to that exact flagged problem: use per-region
motion magnitude (not a hand-set direction/kernel, not a whole-frame
uniform pass) to decide *where* and *how strongly* to blur, on ordinary
30fps footage this project already has — no slow-mo capture requirement at
all, unlike frame-averaging/FILM.

**URL.** https://arxiv.org/abs/2309.01010 (also indexed identically at
`dl.acm.org/doi/10.1145/3606038.3616163`, matching title/authors/venue —
independent-index corroboration of existence). **No primary source was
read.** `arxiv.org`, `dl.acm.org`, `www.researchgate.net`,
`api.semanticscholar.org`, and `paperswithcode.com` were all tried directly
this run and all returned `EGRESS_BLOCKED`, the same standing restriction
noted in nearly every prior entry in this log. Everything above is
reconstructed from search-engine result summaries (three independent
queries, consistent details across all of them: broadcast 30fps → partial
blur → flow-guided patch-selective synthetic augmentation → 54.2%/36.2%
loss reduction), which this log treats as one notch below a directly-read
source — plausible and internally consistent, not independently confirmed.

**Licence — no code or repository found.** I checked the paper's own
GitHub account (`github.com/jerrinbright`, the first author) directly via
fetch: no repository related to motion blur, baseball, or pose estimation
exists there, only unrelated visual-odometry/SLAM/robotics projects. The
author's personal site (`jerrinbright.github.io`) is itself
`EGRESS_BLOCKED`, so it could not be checked for a separate release. One
search result described the arXiv listing's page licence as CC BY 4.0,
which (per this log's established posture on the PSF and 6-DOF entries) is
a statement about the paper text, not a grant over any implementation —
moot here regardless, since no implementation to license was found. The
*idea* (flow-guided, region-selective synthetic blur augmentation) is an
ordinary image/video-processing technique with no original-expression
claim attached; reimplementing it from the description above carries no
licensing exposure, same reasoning this log already applied to the 6-DOF
entry. **Commercial use: not a licensing question** — nothing of the
authors' is being redistributed or run verbatim, but also nothing
shippable exists to save engineering time; this is a from-scratch build.

**Which failure mode.** Motion blur, specifically and only — it says
nothing about camouflage and there's no plausible mechanism by which
region-selective blur synthesis would help a zero-detection frame that is
already sharp.

**Why it helps this model specifically.** This project's own brief already
flags the central gap: median labelled-box elongation of only 1.60 (p90
3.01) despite a labeling spec that instructs annotators to box the full
blur streak, because genuinely blurred training examples are scarce. This
log has three prior routes to manufacturing them (PSF, frame-averaging,
6-DOF) plus one already-flagged-broken shortcut (uniform Albumentations
MotionBlur). This paper adds a fourth route with a property none of the
other three have: it needs no slow-mo capture, no hand-tuned kernel
direction, and — critically — it targets *where* blur should go using
actual per-frame motion rather than a human guess or a uniform pass. Since
the failure symptom in this project is specifically the clubhead (a small,
fast-moving region against a much larger, mostly-static body/background),
a flow-guided selective pass would naturally concentrate synthetic blur
exactly there, which is exactly the region this project's ground-truth
boxes need to grow correctly around. Applied to this project's own
existing sharp, correctly-labelled 29% first-party phone footage, it could
generate additional correctly-elongated training examples without new
capture sessions.

**Effort vs. payoff.** Medium effort, unverified-but-well-motivated payoff.
Effort: this is not a "pip install" like the Albumentations entry — the
paper's own description implies a small standalone pipeline (per-frame
optical flow via any standard method, e.g. OpenCV Farneback or RAFT;
patch/region motion-magnitude thresholding; localized blur-kernel
application; box re-expansion to match, reusing the box-growth logic the
PSF entry already established) with no reference code to start from, so
budget a few days, not an afternoon. Payoff: this is the only blur-
synthesis entry in this log so far that answers "where" as well as "how
much," directly closing the specific gap the Albumentations entry flagged
as unresolved — but every number behind the reported 54.2%/36.2%
improvement comes from an unread source, on a different sport, different
model (3D pose regression, not object detection), and different failure
metric (loss reduction, not mAP or per-frame detection rate), so treat the
mechanism as a promising, well-targeted engineering pattern to prototype
against this project's own eval harness, not as evidence of a specific
expected gain here.

**Verification status.** Paper existence, title, authors, venue (MMSports
'23 / ACM MM 2023 workshop), and problem statement (30fps broadcast video →
partial motion blur → degraded pose estimation): confirmed via two
independent indexes (arXiv listing, ACM DOI record) with matching
metadata. Method description (flow-vector computation → patch-selective
region identification → localized blur synthesis) and the 54.2%/36.2%
figures: from search-engine result summaries only, not the paper text —
flagged as such, not independently confirmed. Code/repository: actively
checked (author's GitHub fetched directly, personal site attempted and
blocked) and confirmed absent, not merely unsearched.
