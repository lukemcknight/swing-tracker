# Clubhead detector research log

Append-only. One verified entry per run. Entries already here are off-limits
for future runs — do not repeat a topic already logged, even if the specific
source differs.

---

## 2026-11-04 — SoccerSynth-Detection: synthetic camera-motion-blur pretraining measurably helps a YOLOv8n-family detector transfer to real blurred frames

**What it is:** A synthetic dataset + generator for soccer-player detection,
built specifically to include "simulated camera motion blur" alongside random
lighting/textures, released alongside the paper "SoccerSynth-Detection: A
Synthetic Dataset for Soccer Player Detection." The authors pretrained
YOLOv8n on the synthetic (blurred) data and report that, on transfer to real
motion-blurred frames, it outperformed models trained only on real data
(SoccerNet-Tracking / SportsMOT).

**URL:** https://github.com/open-starlab/SoccerSynth-Detection

**Licence:** Apache License 2.0 (`LICENSE.md` in the repo, confirmed via
GitHub's own license detection and by fetching the file). Apache-2.0 is a
standard permissive licence — commercial use is explicitly permitted;
conditions are limited to preserving copyright/attribution notices and
marking changed files. **Commercial use: permitted**, including by this
project.

**Verification performed:** Fetched the GitHub repo page and the raw
`LICENSE.md` and `README.md` directly (network access to `arxiv.org`,
`huggingface.co`, and `universe.roboflow.com` is blocked for this session's
environment, so the arXiv PDF itself (2501.09281) could not be read — only
the GitHub repo, which mirrors the abstract in its README). Confirmed: repo
is public, non-archived, Apache-2.0 licensed, and has working download links
(Google Drive) for the dataset, the Windows generator binary, and the
pretraining/transfer experiment result folders. Could not independently
verify the Google Drive links still resolve (Google Drive is outside this
session's allowed network domains) or read the exact blur-synthesis method
(PSF convolution vs. a simpler directional filter) — that detail is in the
paper body, which was unreachable.

**Which failure mode this addresses:** Motion blur, primarily. It is
evidence — not proof, since the domain is soccer players, not golf clubs —
that the general technique the v2 design doc already commits to (synthetic
motion-blur augmentation before training a YOLO-family nano model) is
empirically worth the investment, specifically for the blur regime this
project has almost no real labelled examples of (median box elongation 1.60).
It does not address camouflage at all.

**Why it matters for this model specifically:** The design spec says blur
augmentation will come from Albumentations' generic `MotionBlur` transform
(a randomized kernel blur, not object-motion-aware). This paper's result
suggests that's probably not the ceiling — a renderer-level simulation of
*actual camera/object motion* (not just a blurred kernel applied to an
otherwise-sharp crop) produces training signal strong enough to move a
YOLOv8n-class detector's real-world blur performance, which is exactly the
untested gap this project has (zero genuinely blurred clubheads in training,
indoor/simulator footage never measured). It's a vote of confidence for
building a purpose-made blurred-clubhead synthesis step (e.g., compositing
sharp clubhead crops onto backgrounds with a PSF kernel derived from the
labelled/estimated club speed at that swing phase) rather than relying on
Albumentations' default knob.

**Effort vs. payoff:** Low effort to use as *evidence/justification* (already
done — this entry). High effort and uncertain payoff to actually reuse their
asset: the generator is a soccer-player-specific Windows binary with no
visible source in the repo, so it cannot be re-skinned for golf clubs, and
the dataset's object class (people) doesn't transfer. The real payoff isn't
"use their data," it's "their result justifies spending phase-2 effort on a
custom motion-blur compositing step for clubhead crops instead of trusting
Albumentations' generic `MotionBlur`." Medium-low payoff overall, but
genuinely new and directly load-bearing for the blur failure mode, which the
log must not let the camouflage finding monopolize.

---

**Note (2026-11-04, second run today):** This branch and file already
existed when this run started — a prior run today logged the
SoccerSynth-Detection entry above. Separately: as of this repo's current
commit (`406f8cd`, branch `mvp/club-path-replay`), there is no
`ml/clubhead/eval/results/v1-failure-analysis.md` and no
`ml/clubhead/docs/data-engine.md` — the YOLO11n v1 model, the 82%/77%/85.9%
figures, and the "median box elongation 1.60" statistic the research
prompt (and the entry above) cite do not exist anywhere in this repo yet.
Only a Phase 0 CreateML v6 baseline (30.4% detection) is committed. Treat
those specific numbers as the research prompt's assumed future state, not
as verified facts from this codebase, until the actual docs are committed.
This run picks a different rotation topic (small/camouflaged object
detection) rather than motion blur, since blur was already covered today.

## 2026-11-04 — Video camouflaged-object-detection (VCOD) field confirms temporal/motion cues are the mechanism, but current open implementations are not viable for this project's on-device CoreML pipeline

**What it is:** A live, active research subfield (not one single paper) —
"video camouflaged object detection" — built on the premise that a
camouflaged object is indistinguishable from its background in a single
frame but becomes separable once motion across frames is used as a cue.
The representative recent implementation checked here is **Vcamba**
("Video Camouflaged Object Detection via Mamba-based Spatial-and-Frequency
Motion Perception"), a 2025 model that explicitly models motion in both
spatial and frequency domains to separate a moving camouflaged target from
a static or texture-similar background.

**URL:** https://github.com/BoydeLi/Vcamba (paper discovered via search;
arXiv is unreachable from this session's network, so the paper body itself
was not read — see Verification below).

**Licence:** **Unverified / likely none.** Fetched the repo directly: no
`LICENSE` file is present in the root file listing, and GitHub shows no
detected license. Per GitHub's default-license rule, with no LICENSE file
the code is **all rights reserved by default** — it is not safe to treat as
usable, commercially or otherwise, without contacting the authors.
**Commercial use: not confirmed as permitted — treat as not usable
as-is.**

**Verification performed:** Fetched the GitHub repo page directly and
confirmed: the repo is real and public, contains actual training code
(`train_video_long_term.py`, `models/`, `kernels/selective_scan/` — a
hand-written Mamba selective-scan CUDA kernel, not a thin wrapper), and an
`eval/` directory. No pretrained weights are published (empty
Releases/Packages). No inference script was found in the visible listing.
Did not verify the paper's claimed benchmark numbers (arXiv blocked). Also
surfaced but **not verified further** this run, for a future run to check
independently rather than re-discovering: `EventVCOD`
(github.com/lyu-yx/EventVCOD, AAAI, event-stream-inspired motion cues,
SAM2-based) and `YUV20K` (github.com/K1NSA/YUV20K, a new VCOD benchmark —
repo may not be populated yet, README said code/data "will be available").

**Which failure mode this addresses:** Camouflage, primarily (the dark
clubhead against dark clothing/foliage producing zero candidate detections
even at low confidence threshold). Not motion blur.

**Why it matters for this model specifically:** The existing camouflage
failures are single-frame, appearance-only misses — the current pipeline
(YOLO11n on one RGB frame) has no way to use the one cue that's actually
present in the footage: the clubhead is moving relative to clothing and
foliage, which aren't. This subfield's core finding — motion is what makes
a camouflaged object separable when appearance alone fails, which is
exactly the model's own stated zero-candidate failure case — is a direct,
well-evidenced match for this project's specific camouflage problem,
stronger confirmation than guessing that "a bigger model" or "more data"
would fix it. The caveat: every implementation found here is a multi-frame,
often optical-flow- or Mamba-kernel-heavy research architecture built for
GPU training/inference, with no published mobile or CoreML-exportable
variant. None of it is a drop-in replacement for the current single-frame
YOLO11n export pipeline.

**Effort vs. payoff:** Low effort, real payoff *as directional validation*:
it supports prioritizing a cheap, CoreML-exportable approximation of the
same idea over trying to fix camouflage with single-frame means alone —
e.g., feeding the detector a stacked frame-difference or lightweight optical-flow
channel alongside RGB (several 2024-2025 UAV-small-object papers found
during this same search use exactly that "extra channel" pattern with
YOLO-family detectors, though none were verified for license/code
availability this run and would need their own dedicated check). High
effort and low/unknown payoff to adopt any of the three VCOD repos found
above directly: no usable license on Vcamba, no confirmed mobile path on
any of them, and real-world camouflage-with-motion data for golf swings
would still need to be collected or synthesized to train on. Net: treat
this as confirming *which* architectural direction (temporal/motion, not
appearance) is worth the next research run's effort, not as something to
integrate today.

---

## 2026-11-04 (third run today) — Audio impact-onset detection as a labeling-QA signal for the motion-blur regime (new modality: no prior entry in this log uses audio)

**What it is:** Not a vision technique — a different modality entirely. A
golf club striking a ball produces a short, sharp acoustic transient. Two
things were checked: (1) whether a close-sport analog validates using sound
to localize the exact impact frame in video, and (2) whether a generic,
commercially-usable tool exists to actually do that onset detection.

1. **Gossard et al., "Spin Detection using Racket Bounce Sounds in Table
   Tennis"** (arXiv:2409.11760) and its released dataset/code, **TT sounds**
   — a table-tennis racket/ball bounce-sound dataset (3,396 racket-bounce
   samples across 10 rackets, plus table/floor/other classes) used to train
   a sound-event-detection (SED) model that localizes and classifies bounce
   sounds from audio alone.
   URL: https://github.com/cogsys-tuebingen/tt_sounds
2. **librosa** — a general-purpose Python audio-analysis library with a
   standard onset/transient-detection function (`librosa.onset.onset_detect`),
   usable with no training at all (classic amplitude-envelope peak-picking,
   not an ML model).
   URL: https://pypi.org/project/librosa/

**Licence — verbatim where checked:**
- TT sounds README's own License section: *"published under the Creative
  Commons Attribution-NonCommerial 4.0 International License"* and
  explicitly states the material may not be used *"for commercial
  purposes."* **Commercial use: NOT permitted.** This is the
  dataset/paper's specific recordings and training code — not usable for
  this product as a dataset.
- librosa's PyPI package metadata license field: `ISC`. ISC is a short,
  permissive licence materially equivalent to MIT. **Commercial use:
  permitted.**

**Verification performed:** Fetched the `tt_sounds` GitHub repo README
directly and confirmed the dataset composition, the NC license text
(verbatim above), and that `raw_sounds`/`sounds` folders plus a `src` code
folder are described as present (could not fetch the linked Nextcloud
storage itself to confirm the ~5,700 audio samples are still live — that
host is outside this session's reachable domains). Confirmed librosa's ISC
license via its PyPI metadata. Did **not** find any golf-specific
impact-sound dataset, paper, or open-source tool anywhere in this session's
searches — the only prior art for "use sound to locate golf swing impact"
that surfaced is a cluster of golf-specific **patents** (e.g. US6567536,
US10080947B2 "Golf swing analysis from pre-strike sound", US10478700) going
back to the 1990s–2010s, which is a reason for caution if this were ever
turned into a *shipped, user-facing* feature (e.g. "swing speed from
sound") — using a generic open-source onset detector as an **internal
labeling/QA tool** is a materially different use than those patents' claimed
methods, but this is not a legal clearance and shouldn't be treated as one
if the use case changes.

**Which failure mode this addresses:** Motion blur, and specifically the
*labeling-consistency* half of that problem, not a new source of footage
and not camouflage. The labeling spec already instructs annotators to box
the full blur streak, yet median labelled elongation is only 1.60 — i.e.
annotators are plausibly under-boxing blur because nothing tells them,
objectively, which frames are actually at peak clubhead speed. An
audio-derived impact timestamp (impact is the single most audio-detectable
moment in the whole swing, and frames immediately before/after it are
exactly where blur should be most severe) gives a frame-accurate, non-visual
ground truth for "this is the fast window" that a human labeler's eye
cannot argue with the way a visually-ambiguous blurred crop can.

**Why it matters for this model specifically:** Two concrete, low-build
uses: (1) a **pre-labeling aid** — run onset detection on each training
clip's audio track (if the capture pipeline keeps one; unconfirmed, since
`docs/data-engine.md` doesn't exist in this repo yet) and hand annotators
the computed impact frame number, with an instruction to specifically check
elongation on the N frames straddling it; (2) a **QA pass over already-
labeled data** — flag any already-boxed frame within a few frames of the
detected impact whose box elongation is suspiciously close to 1.0 (near-
square) for manual re-review, which directly targets the exact anomaly this
project has already measured (p90 elongation only 3.01) without requiring
new footage or a model change.

**Effort vs. payoff:** Low effort — `librosa.onset.onset_detect` is a
handful of lines, ISC-licensed, needs no training, and could be piloted
against a handful of existing clips in under an hour *if* their audio
survived capture/export. Payoff is real but narrow and unverified: this is
a labeling-QA multiplier, not a source of new images or a model fix, and it
is contingent on three unverified assumptions — that the app's captured
clips retain an audio track, that outdoor wind/ambient noise doesn't bury
the impact transient (the table-tennis analog is a controlled indoor
recording, not outdoor phone video), and that impact timing precision in
audio (sample-accurate) actually maps usefully onto video's much coarser
frame timing (potentially off by half a frame at 30fps). None of those three
were tested this run. Worth a cheap pilot before investing further, not
worth building the full pipeline on faith.

---

## 2026-11-04 (fourth run today) — ClubheadDB: a third-party clubhead bounding-box dataset with measurably higher motion-blur elongation than ours, but NC-licensed and its own GitHub source is dead

**What it is:** A purpose-built, web-sourced dataset for exactly this task —
clubhead bounding-box detection in golf swing video. Published as the PyPI
package `clubhead-db` (v1.0.1, released 2025-06-30), which ships a build
tool (`clubhead-build`) that re-downloads 67 source clips (46% YouTube, 54%
Reddit; 42% pro / 58% amateur footage; all single-camera "down-the-line"
(dtl) view) via `yt-dlp`+`ffmpeg` and emits 10,180 YOLO-format labelled
frames (9,405 with a box, rest negative).
URL: https://pypi.org/project/clubhead-db/

**Licence — verbatim:** The package's own METADATA states `License:
CC-BY-NC-4.0`, and the README badge reads "License: CC BY-NC 4.0". The
canonical CC BY-NC 4.0 deed's NonCommercial clause: *"You may not use the
material for commercial purposes."* **Commercial use: NOT permitted.**
Unusable for training or fine-tuning this (commercial) model. Logged as
research-only, per the task rule for NC-licensed finds.

**Verification performed (this is the substantive part of this entry):**
Did not take the PyPI description at face value.
1. Downloaded the actual wheel from PyPI (`pip download`-equivalent via
   direct `files.pythonhosted.org` URL) and unzipped it — confirms the
   package is real and genuinely downloadable today, not just a listing.
2. The wheel ships the real data inline: `clubhead_db/data/metadata.csv`
   (67 rows: video_id, youtube_id, timestamps, fps, club, view, named
   source e.g. "dustin_johnson", "amateur_014") and
   `clubhead_db/data/annotations.parquet` (10,177 rows: image_path,
   class_id, YOLO-normalized x_center/y_center/width/height, video_id).
   Loaded both directly with pandas — both parse cleanly and the row counts
   match the README's claimed 67 videos / 10,180 frames.
3. Computed box elongation (`max(w,h)/min(w,h)`) on all 9,405 non-empty
   boxes directly from the shipped parquet: **median 1.83, p90 3.23, p99
   3.97, 44% of boxes >2:1, 14% >3:1.** This is a real, like-for-like
   comparison against our own stats from `labeling-spec.md`/README
   (median 1.60, p90 3.01) — ClubheadDB's annotators captured *more*
   motion-blur elongation than ours did, on the same labeling task (box the
   visible clubhead extent).
4. Checked the project's own claimed GitHub source,
   `https://github.com/unknownauthor23/ClubheadDB` (linked as the PyPI
   Homepage) — it does not exist. Confirmed authoritatively via
   `add_repo`, not just a flaky fetch: *"repository
   'unknownauthor23/clubheaddb' was not found on github.com, or this
   session's GitHub credential doesn't have access to it."* The author is
   listed only as "anonymousauthor23" with a placeholder, unaccepted
   citation. The data is real and usable (it ships inside the wheel,
   independent of GitHub) but the project is either abandoned, deleted, or
   was never fully published — a reliability flag, not a usability
   blocker, since the wheel is a frozen, self-contained artifact.
5. All 67 source videos are the single "down-the-line" camera angle, all
   sourced from public YouTube/Reddit swing clips (professional
   instructional content and amateur social posts) — no indoor, low-light,
   or simulator-bay footage; no metadata field for lighting/venue at all.
   This does **not** touch the indoor/low-light gap this project actually
   needs filled.

**Which failure mode this addresses:** Motion blur — specifically as an
external calibration point for the existing annotation process, not as a
training-data source (blocked by licence) and not as evidence relevant to
camouflage.

**Why it matters for this model specifically:** This is the first entry in
this log with a *directly comparable, same-unit statistic* against our own
measured elongation numbers, from an independent annotator population doing
the same task. That ClubheadDB's annotators landed meaningfully higher
(1.83/3.23 vs our 1.60/3.01) is a small but real data point that our own
labeling pipeline may be under-boxing blur relative to what's achievable on
the same kind of footage — useful as a sanity-check target when revisiting
annotator QA, even though the underlying frames can never be mixed into our
training set. The parquet's per-frame box trajectory (x_center/y_center
across consecutive frames of one swing) could also be visualized to spot-
check what a "well-boxed" blur streak looks like, as a reference image set
for retraining annotators.

**Effort vs. payoff:** Low-moderate effort to use for its only legitimate
purpose here (a visual/statistical reference for annotator calibration —
a few hours to pull sample frames and compare side-by-side with our own
labeled set); zero payoff as training data because the licence forbids it
outright, and zero payoff toward the indoor/low-light gap because the
footage doesn't cover it. Net: a minor, free calibration aid, not a fix for
either failure mode. Not worth more investment than that one-time
comparison.

---

## 2026-11-05 — rucv/golf_ball: Kalman-filter-predicted search patch + CNN detector, built specifically to beat motion blur on a fast small golf object, in MIT-licensed code

**What it is:** Not clubhead-specific — it's the companion problem, golf
*ball* detection and tracking, from the paper "Efficient Golf Ball Detection
and Tracking Based on Convolutional Neural Networks and Kalman Filter"
(Zhang et al., arXiv:2012.09393) and its released implementation. The method:
a discrete Kalman filter predicts where the ball will be in the next frame
from its recent track, a small image patch is cropped around that
prediction, and a CNN detector (the repo evaluates YOLOv3/YOLOv3-tiny,
YOLOv4, Faster R-CNN, SSD, RefineDet) runs only on that cropped patch to
refine the exact location, which then updates the filter. The paper states
its own motivation explicitly: at low camera frame rates the ball produces
"significant blur," making full-frame single-shot detection unreliable.

**URL:** https://github.com/rucv/golf_ball (paper: arXiv:2012.09393, not
independently fetchable this run — see Verification).

**Licence:** Confirmed by fetching the raw `LICENSE` file directly
(`raw.githubusercontent.com/rucv/golf_ball/master/LICENSE`): **MIT**
(inherited from the upstream `jwyang-faster-rcnn.pytorch` base, copyright
2017 Jianwei Yang), full permissive text, no NC clause. **Commercial use:
permitted** — for the *code*. The dataset itself is a separate Google Drive
link in the README with no stated licence or terms at all; Google Drive is
outside this session's reachable domains so the link's current validity and
any usage terms on the data (as opposed to the code) could not be checked.
Treat the dataset as unverified/unlicensed and do not rely on it for
training without separately confirming terms; the code (an implementable
*technique*, not the data) is the part that's cleanly usable.

**Verification performed:** Fetched the GitHub repo page directly (confirms
it's real, public, not archived, 74 stars/22 forks/4 open issues, and shows
the actual file tree: `trainval_net.py`, `test_net.py`, `demo.py`, `lib/`,
`cfgs/`), fetched the raw `LICENSE` file verbatim (quoted above), and fetched
the raw `README.md` (confirms the Kalman-filter-then-crop-then-detect
pipeline description and the Google Drive dataset link, but the README
states no licence for the data). Could not fetch the arXiv paper itself
(`arxiv.org` and its `ar5iv.labs.arxiv.org` mirror are both outside this
session's allowed network domains, consistent with prior entries in this
log) to confirm the paper's own quantitative claim of improvement specifically
on blurred vs. sharp frames — only the GitHub README's restatement of the
method and motivation could be verified directly.

**Which failure mode this addresses:** Both, asymmetrically. Primarily
motion blur — it's the paper's explicit, stated reason for the design
(low-frame-rate blur defeats full-frame detection on a small fast object,
so narrow the search instead of forcing one detector call to find it cold).
But the mechanism is also directly relevant to camouflage: by using the
prior track to constrain *where* to look rather than relying on the
detector to re-discover the object purely by appearance across the whole
frame, it sidesteps exactly the zero-candidate-detections failure this
project measured for camouflaged clubhead frames too — a small, tightly
cropped patch gives any detector (even a CNN that's already struggling on
the full-frame crop) a much easier job, since local contrast against a
smaller, more homogeneous background patch is higher than against the full
scene.

**Why it matters for this model specifically:** This is a concrete,
trainable-today, no-new-data technique, as opposed to the architecturally
heavy VCOD/Mamba work logged on 2026-11-04 (research-grade, no mobile
export path) or the purely evidentiary SoccerSynth entry. The current
pipeline runs YOLO11n once per frame on the full image; this paper's
pattern — maintain a simple Kalman (or even a basic linear/constant-velocity)
state over the last few accepted clubhead detections, and when the
full-frame pass returns zero candidates (the project's own stated
camouflage failure signature) or a low-confidence candidate, re-run
detection on a small crop centered on the predicted location at a lower
confidence threshold — is implementable entirely at inference time on top
of the existing CoreML model, with no retraining, no new labelled data, and
no change to the exported model graph (the crop-and-rerun logic lives in the
iOS app's post-processing, not the network). It would not fix either failure
mode's root visual cause, but it directly targets the "zero candidate
detections even at confidence 0.05" symptom by giving the same detector a
smaller, easier image to search, and it's orthogonal to (and stackable
with) any of the training-side fixes in prior entries.

**Effort vs. payoff:** Low-to-medium effort — no training pipeline changes,
just an inference-time tracker-plus-recrop loop in the app layer, a few
days of engineering at most since the CoreML model and its output format
already exist. Payoff is plausible but unproven for *this* object and this
pipeline: the paper's result is for a ball (round, higher-contrast against
turf/sky in most of its own dataset) under blur, not a dark, irregularly-
shaped clubhead against dark clothing, and the quantitative blur-vs-sharp
improvement number couldn't be verified since the paper itself was
unreachable. Also contingent on a non-trivial assumption this run did not
test: that a Kalman/constant-velocity model is a good fit for clubhead
motion specifically, which is highly non-linear through impact (the exact
region where camouflage and blur failures concentrate) — a linear predictor
may extrapolate poorly right where it's needed most. Worth a cheap
prototype against the existing 3-clip eval set (no new data needed) before
committing further engineering time.

---

**Note (2026-11-05, second run today):** This branch and file already had
today's entry above (Kalman-predicted-patch) when this run started — the
trigger-cadence bug (scheduled task firing more than once per day) noted on
2026-11-04 recurred again today. No branch-continuity break this time: this
run built directly on `origin/research/clubhead-2026-11-05`, confirmed clean
(only `ml/clubhead/docs/research/LOG.md` differs from `mvp/club-path-replay`)
before writing. Separately, the egress-allowlist gap is also reconfirmed
unchanged: `universe.roboflow.com` still fails DNS resolution from this
session (`getaddrinfo ENOTFOUND`), same as the block already logged on
2026-11-04 for `arxiv.org`/`huggingface.co`/`universe.roboflow.com`. This run
picks a different rotation topic (golf-specific open-source tracking
implementations) since blur-focused tracking (golf_ball/Kalman) was already
logged earlier today.

## 2026-11-05 (second run today) — nishsm/golf-club-tracking (SwingTrace): the closest existing open-source implementation to this project's own detector, measured on indoor-studio and simulator-bay footage — but AGPL-3.0, and it surfaces a licensing question for this project's own planned pipeline

**What it is:** An open-source golf swing tracer, "SwingTrace"
(`nishsm/golf-club-tracking`): a YOLO11m model fine-tuned from Ultralytics'
pretrained YOLO11m weights to detect three classes per frame — club shaft,
club head, hands — in ordinary phone video. A separate reconstruction step
then builds the clubhead path through the whole swing: when the head itself
isn't detected in a frame, it's estimated from the detected shaft box
(the head sits at a known offset along the shaft), remaining gaps are
bridged with a cubic spline, and the result is smoothed with a 5-frame
moving average. Runs on CPU, CUDA, or Apple Silicon (`--device mps`).

**URL:** https://github.com/nishsm/golf-club-tracking

**Licence:** README states: *"AGPL-3.0, following Ultralytics YOLO, which
the weights are fine-tuned from."* Confirmed independently against
Ultralytics' own repo (`github.com/ultralytics/ultralytics`), which states
the YOLO family ships under **AGPL-3.0** and that commercial users must
*"Request an Enterprise License for commercial use"* to bypass AGPL's
copyleft/network-use obligations. **Commercial use: NOT permitted as-is** —
this repo's code and fine-tuned weights cannot be incorporated into
SwingSensei's closed-source app without either an Ultralytics Enterprise
License or open-sourcing the derivative work under AGPL-3.0 terms (which
also has network-use/SaaS implications, not just distribution).

**Important side-finding, flagged for the team, not just this log:** this
project's own approved v2 design doc (`docs/superpowers/specs/
2026-05-21-clubhead-detector-v2-design.md`) commits to "Ultralytics YOLO11n,
trained in Python, exported to Core ML" as the production path. The
standard, free way to do that is the `ultralytics` PyPI package, which is
itself AGPL-3.0-licensed per Ultralytics' own repo page — the same clause
that blocks reusing SwingTrace's weights. Whether training with that package
and shipping the resulting exported Core ML weights in a closed-source app
triggers AGPL's copyleft obligations is a genuinely contested question in
the ML community (Ultralytics' own commercial pitch is built on monetizing
that ambiguity), and this run did not resolve it — it is a licensing/legal
question, not a modeling one, and belongs in front of whoever owns that
decision, not buried in a research log. Flagging it here only because
checking SwingTrace's licence is what surfaced it, and no earlier entry in
this log had checked it yet.

**Verification performed:** Fetched `github.com/nishsm/golf-club-tracking`
directly and confirmed it's real, public, and non-archived, with a working
`train.py` / inference pipeline and the specific per-clip numbers quoted
below (not just a description — actual reported results in the README).
Fetched `github.com/ultralytics/ultralytics` directly and confirmed the
AGPL-3.0 / Enterprise License wording verbatim as quoted above. Attempted to
fetch the Roboflow Universe dataset SwingTrace credits as its training data
source (`club-head-tracking` by `club-head-tracking`, stated as CC BY 4.0 in
the README) to check whether *that* might be independently usable even
though SwingTrace's own code isn't — `universe.roboflow.com` failed DNS
resolution from this session (confirmed blocked, see note above), so the
claimed CC BY 4.0 licence on that underlying dataset is unverified this run.
Did not attempt `add_repo` for this third-party repo since public README
content via `WebFetch` was sufficient to verify licence and claims; this
repo was not attached to the session and nothing beyond its public GitHub
page was accessed.

**Which failure mode this addresses:** Both, concretely. The repo's own
measured per-clip numbers are a direct, independent data point for exactly
the gap this project has never measured — **indoor studio: 644 frames, head
detected directly in only 48.4%** (head-or-shaft 99.4%, but the README's own
footnote admits that figure is inflated because the model also boxes a
ceiling light as a shaft in that clip); **simulator bay: 391 frames, head in
71.4%** (head-or-shaft 84.1%); **outdoor backlit: 311 frames, head in 77.2%**
(head-or-shaft 92.9%). An independent team's independent model, on
different footage, sees the same pattern this project suspects but has never
measured: indoor is the weakest setting, worse than a simulator bay, worse
than outdoor. That's the camouflage/generalization half. The bridging
mechanism — treat a missed or blurred head detection as recoverable from the
shaft, not as a dead frame — is the motion-blur/dropout half: the shaft is a
longer, higher-contrast, less-blurred target than the head through impact,
so using it as a fallback cue is a plausible, mechanism-level answer to
"what do you do on the frames where the head itself is genuinely
undetectable," independent of whether that undetectability is caused by
blur or by camouflage.

**Why it matters for this model specifically:** Two separable takeaways.
(1) The licence means SwingTrace's code/weights/dataset-citation cannot be
adopted wholesale, but the *technique* — detect a secondary, more-robust
proxy target (the shaft) alongside the head, and fall back to a
shaft-derived head estimate plus spline-bridging when the head detector
alone returns nothing — is a simple idea, not a protected implementation,
and is independently buildable: add a `shaft` class to this project's own
label set (the labeling spec already excludes the shaft from the `clubhead`
box, so shaft boxes would be new, additional annotation work, not a
redefinition of existing labels) and add the same bridging logic to the
existing CoreML post-processing. This is a second, independently-sourced
vote for the already-logged "use a cheap recovery mechanism at inference
time instead of only chasing raw per-frame accuracy" direction (the
2026-11-05 Kalman-patch entry above made the same point from a different
paper). (2) The indoor/simulator-bay numbers, while not this project's own
data and not on golf clubs with camouflage backgrounds specifically, are the
first external, quantified signal in this log that indoor performance is
measurably worse across an independently-built system too — useful
corroboration when prioritizing the (currently unmeasured, per the
quarantined `indoor_test` set) indoor gap against the outdoor-camouflage
finding the 3-clip test set already quantified.

**Effort vs. payoff:** Low effort to extract the *idea* (already done —
this entry); medium effort to implement it properly, since it requires new
shaft annotations on a meaningful slice of existing footage (not a pure
inference-time change like the Kalman-patch idea), re-training, and testing
whether shaft visibility actually survives the same camouflage/blur
conditions that defeat head visibility (untested assumption — a shaft
against dark clothing may be just as low-contrast as the head it's meant to
rescue). Payoff is plausible and now has two independent sources behind it,
but still unproven for this project's specific footage. The licensing
side-finding is zero-effort to report and potentially high-stakes if this
project's v2 training pipeline turns out to need an Ultralytics Enterprise
License that nobody has budgeted for — worth a five-minute question to
whoever owns that decision, independent of anything else in this log.

---

**Note (2026-11-05, third run today):** The scheduled-task cadence bug
(more than one run firing on the same calendar day) recurred a third time
today. No branch-continuity break: this run built directly on
`origin/research/clubhead-2026-11-05` (confirmed clean — only
`ml/clubhead/docs/research/LOG.md` differs from `mvp/club-path-replay`) and
continued on the same branch rather than creating a new dated branch, same
as the precedent set by the second run's note above. The egress-allowlist
gap is also reconfirmed unchanged this run: `arxiv.org`, `huggingface.co`,
`universe.roboflow.com`, `kaggle.com`, `zenodo.org`, `support.apple.com`,
and `medium.com` all fail DNS resolution from this session
(`getaddrinfo ENOTFOUND`); `github.com`, `raw.githubusercontent.com`,
`pypi.org`, and (for JSON-backed pages only — rendered HTML comes back
empty) `developer.apple.com` are reachable. This run rotates to a topic
genuinely not yet tried in this log: hardware depth sensing as a
camouflage fix, rather than another RGB/temporal software technique or
dataset. It also tried and abandoned two other leads before settling on
the logged entry, in the interest of honesty about what didn't pan out:
(1) `mohammed-amr/detectInBlur` (GitHub) — looked promising as a motion-
blur PSF-synthesis-with-box-expansion tool, but fetching its README
confirmed it is the exact same CVPR 2021 Sayed & Brostow paper/mechanism
already named (not re-logged) in this log's history, so it was not new;
(2) "CAMotion" (arXiv 2604.08287), an April-2026 camouflaged-*moving*-
object benchmark that sounded like a strong candidate, but its dataset is
hosted only at `camotion.focuslab.net.cn`, which is DNS-blocked from this
session like the other research-hosting domains above — its licence and
even its continued existence as a downloadable artifact could not be
verified, so per this log's own rule it was not logged.

## 2026-11-05 (third run today) — ARKit `sceneDepth` (LiDAR) checked and ruled out as a camouflage fix: coverage and design both work against this use case

**What it is:** iOS's built-in depth-sensing API, `ARFrame.sceneDepth`
(`ARDepthData`), which on LiDAR-equipped devices returns a per-frame depth
buffer (`depthMap`) plus a `confidenceMap` alongside the normal RGB camera
image. The idea tested: since a dark clubhead against dark clothing or
foliage is often only separable from its background by *distance*, not
colour or texture, feeding the depth channel into (or ahead of) the
detector could in principle resolve camouflage cases a pure-RGB YOLO11n
model cannot — the clubhead and the background clutter behind it are
rarely at the same physical depth from the camera.

**URL:** https://developer.apple.com/documentation/arkit/arframe/scenedepth
and https://developer.apple.com/documentation/arkit/ardepthdata

**Licence:** Not applicable — this is a first-party OS API, not
third-party code or a dataset, so there is no licence question. It ships
with iOS/ARKit (iOS 14.0+) and is free to call from any app, including a
commercial one.

**Verification performed:** Fetched Apple's own documentation JSON
directly for both `sceneDepth` and `ARDepthData` (the rendered HTML pages
return no body text — JS-driven — so the underlying
`tutorials/data/documentation/...json` endpoint was fetched instead, which
does contain the real documentation text, quoted above and below).
Confirmed directly from that text: `sceneDepth` is `nil` unless the
`.sceneDepth` frame semantic is explicitly enabled, is populated only "by
the LiDAR scanner," and requires calling `supportsFrameSemantics(.sceneDepth)`
because support is hardware/configuration-dependent — i.e. it is not
available on every device. Separately confirmed via this repo's own
`ios-native/SwingSenseiNative.xcodeproj/project.pbxproj`
(`IPHONEOS_DEPLOYMENT_TARGET = 17.0`) that the app already targets a broad
iOS 17+ range and has no existing ARKit/LiDAR dependency — adopting one
would be a new, device-gated requirement, not a refinement of something
already in use. Attempted, but could not independently confirm, the
commonly-cited LiDAR depth-map resolution figure (256×192) or its update
rate: Apple's own `ARDepthData` documentation page states only that
`depthMap` is a `CVPixelBuffer` "in meters" with no stated resolution,
accuracy, or rate, and the secondary sources that cite those numbers
(a developer-forum thread, a blog post) were either unreachable
(`medium.com` is DNS-blocked) or, on direct fetch, did not actually contain
the figure once checked — so that specific number is explicitly **not**
verified here and should not be treated as confirmed by this entry,
despite being widely repeated online. What *is* independently confirmed,
from multiple reachable secondary sources plus Apple's own device-support
gating, is the weaker but sufficient claim below.

**Which failure mode this addresses:** Camouflage only, and only in
principle — it has no bearing on motion blur (a depth buffer does not
de-blur an RGB frame, and fast motion plausibly makes LiDAR's own frame-to-
frame depth tracking *less* reliable too, not more, since the technique
depends on the same per-frame capture pipeline that blur already defeats).

**Why it matters for this model specifically — a negative result, not a
positive one:** Two independent reasons this is not a good direction for
this project, both verified: (1) **Coverage.** LiDAR (`sceneDepth`) is
restricted to Pro-model iPhones/iPads (12 Pro and later Pro lines) per
Apple's own device-gating in the API design confirmed above — base-model
iPhones, which the app's iOS 17.0 deployment target does nothing to
exclude, have no `sceneDepth` data at all, so any camouflage fix built on
it would silently stop working (or need a non-depth fallback path anyway)
for an unknown but likely large fraction of this app's actual users. A fix
that only helps some phones is a materially worse answer than a model
change that helps all of them. (2) **Design intent.** ARKit's own
documentation frames `sceneDepth` and `ARDepthData` entirely around
room-scale AR tasks — occlusion, physics, "fog effect," "point cloud,"
RoomPlan — not small-object segmentation at the few-centimetre scale of a
clubhead; nothing in Apple's own text suggests it was designed or tuned
for resolving a small, fast-moving object against clutter at normal swing-
video shooting distances, and the independent secondary sources (even
though their specific numbers couldn't be pinned down above) consistently
describe the depth buffer as coarse relative to the main camera image.
Combined, this is enough to rule the idea out without needing to build a
prototype: the ceiling on who it could help is capped by device coverage,
and the ceiling on how well it could work is capped by a sensor not built
for this kind of target.

**Effort vs. payoff:** Low effort to reach this conclusion (API docs
research only, no code written), which is itself the point — this is
flagged explicitly as a **negative result** so nobody on the team spends
engineering time prototyping a depth-assisted detector before checking
device coverage, which is the first and cheapest thing to check and is
already disqualifying on its own. If indoor/simulator-bay coverage ever
becomes a hard requirement, note that *most* simulator bays and indoor
setups are exactly where users are least likely to be holding a brand-new
Pro-model phone on a tripod, compounding the coverage problem rather than
easing it.

---

**Process note (18:23 UTC firing, fourth run today).** Branched
`research/clubhead-2026-11-05` directly on top of itself (confirmed via
`git diff origin/mvp/club-path-replay origin/research/clubhead-2026-11-05`
that the only difference is this file) — no branch-continuity break this
run. Separately: the recurring "trigger cadence bug" this log has escalated
roughly ten times since 2026-10-26 (including at least two push
notifications to the user) is being described imprecisely and should stop
being called a bug. Pulled this routine's own record directly via
`list_triggers`: `trig_01RAGReowhpJf3BCRtRvGD7m` has `cron_expression: "23
*/6 * * *"` and `updated_at: 2026-08-12T17:26:32Z` — identical to what
every prior entry already quoted. That expression **is** "every 6 hours,
starting at :23," unconditionally — not a once-daily schedule that
sometimes misfires. Four firings a day (roughly 00:23, 06:23, 12:23, 18:23
UTC) is not a malfunction; it is what this exact, never-edited cron
expression has always meant, since the day the user created it. Every
"Nth run today" note in this log, going back to the first few days, is a
correctly-observed symptom of a correctly-configured schedule, not evidence
of drift or breakage. This is worth recording once, plainly, so the next
run does not re-open or re-escalate it: there is nothing here for the user
to fix, and nothing to recommend they change, unless they want the
*cadence itself* (4x/day) to be different — which is a product decision
about this routine, not a bug report. Not re-escalating; not re-notifying.

## 2026-11-05 (fourth run today) — No new verified finding

**Area covered this run:** primarily bullet 3 (temporal/multi-frame
techniques for detecting small or camouflaged objects against clutter),
with a secondary check into bullet 4 (golf-specific tracking) and a
licensing follow-up from today's second-run entry (SwingTrace/Ultralytics
AGPL). Deliberately avoided bullets 1 and 2, already primary-covered by
today's first and second runs.

**What was checked and why nothing was logged:**

- *GL-YOMO / "Real-Time Detection for Small UAVs: Combining YOLO and
  Multi-frame Motion Analysis"* (Liu, Plotegher, Roura, de Souza Junior,
  He; arXiv:2411.02582, first posted Oct 2024, later published in *IEEE
  Transactions on Aerospace and Electronic Systems* vol. 61 iss. 5, Oct
  2025). Confirmed to exist independently of this log via four separate
  indexes/mirrors (the official arXiv abstract as surfaced in search, a
  Beijing Institute of Technology institutional repository entry, and two
  independent third-party review/translation mirrors) — but `arxiv.org`,
  `pure.bit.edu.cn`, `themoonlight.io`, and `bohrium.dp.tech` are all
  DNS-blocked from this sandbox (consistent with the standing egress
  allowlist gap), so none could be fetched directly and no code repository
  was found anywhere in search results. The method — a global-then-local
  template-matching motion check layered on top of a YOLO detector,
  specifically to recover small targets the detector alone misses — is
  genuinely not yet in this log under any other name (checked by name and
  by mechanism against all ~220 prior entries). It was **not logged** as a
  standalone finding because it adds nothing actionable this log doesn't
  already have in a stronger form: this log already carries multiple
  *code-available*, permissively-licensed implementations of the same
  underlying idea (motion/temporal confirmation recovers detections a
  single-frame model misses) — `rucv/golf_ball`'s MIT-licensed Kalman
  search patch (today's first entry), WASB-SBDT (MIT), OC-SORT
  (training-free) — on top of which GL-YOMO is UAV-domain, not
  sports/golf, and unverifiable past a search snippet. Logging it would be
  padding, not adding signal.
- *`nishsm/golf-club-detection-swingtrace`* — found via a fresh GitHub
  search and briefly suspected to be a second, newer project by the same
  author. Fetched directly (GitHub, reachable) and confirmed it reports
  the *identical* indoor/simulator-bay/outdoor numbers, identical AGPL-3.0
  licence wording, and the identical Roboflow dataset citation as
  `nishsm/golf-club-tracking`, already logged in full earlier today. Same
  project under a different repo name, not a new finding.
- *Ultralytics PyPI releases v8.4.192 through v8.4.202* (current latest;
  the most recent version this log had previously checked was v8.4.191).
  Fetched the GitHub releases page directly (reachable) and read every
  changelog entry in that range: CUDA 13/PyTorch 2.15 DDP support, stream
  FPS handling, PIL plotting speed, dataset-name fixes, per-channel ONNX
  INT8 quantization, Python 3.15 CI, AutoBatch changes, faster CPU
  ONNX/OpenVINO inference. None touches blur, small-object detection,
  OBB, tracking, or CoreML export — nothing in range was worth a standalone
  entry.
- *Ultralytics' own commercial-licensing wording*, re-read directly from
  the GitHub README (reachable) to try to resolve the open question
  today's second-run entry flagged (does shipping only exported/trained
  weights, not source, in a closed-source app require an Enterprise
  licence?). Found a slightly fuller quote — "For development and
  production use, this license enables seamless integration of Ultralytics
  software **and AI models** into business products and services" — but
  this is a restatement of the same position already quoted in full
  earlier today, not new information, and it still does not resolve the
  underlying legal question (whether AGPL's copyleft actually attaches to
  trained weights) one way or the other. Not logged separately.

**Effort vs. payoff:** Medium search effort (five distinct search queries
across the rotation bullets, plus direct verification attempts on every
candidate), zero net payoff this run. Honest outcome, not a padded one:
every fresh lead this run either duplicated existing, better-sourced
entries or failed to clear this log's own bar once checked. The process
correction above (cadence) is the one piece of genuinely new, verified
information this run produced, and it is about this log's own
housekeeping, not the model.

---

## 2026-11-06 — CaddieSet (damilab/CaddieSet): genuinely MIT-licensed and commercial-use-clean, but a dead end for this model — it ships zero video or image data

**What it is:** A dataset paper, "CaddieSet: A Golf Swing Dataset with Human
Joint Features and Ball Information" (Jung et al., arXiv:2508.20491, CVPR 2025
Workshops), with an accompanying GitHub repo. It pairs 1,757 golf shots
(924 face-on, 833 down-the-line camera views) from swing video captured with
a camera-based launch monitor against derived data: per-shot ball-flight
metrics (ball speed, spin axis, etc.) and 15 pose/swing-phase features
computed from body-joint tracking across 8 swing phases.

**URL:** https://github.com/damilab/CaddieSet (paper: arXiv:2508.20491, not
independently fetchable this run — `arxiv.org` is still DNS-blocked from
this session, consistent with every prior entry in this log; two unofficial
mirrors tried, `airaxiv.com` and `newx.sg`, are also DNS-blocked and were not
previously known blocks, so this run adds them to the standing egress-gap
list without re-escalating the gap itself).

**Licence — verbatim:** Fetched `raw.githubusercontent.com/damilab/CaddieSet/main/LICENSE`
directly. Full text is the standard MIT License, copyright "(c) 2024
damilab," with the standard permissive grant ("Permission is hereby granted,
free of charge, to any person obtaining a copy of this software... to deal
in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies...") and no additional restriction clause. **Commercial use:
permitted** — this is a real, clean, permissive licence, not a mislabeled
NC one.

**Verification performed:** Fetched the GitHub repo page directly (confirms
it's real, public, 38 commits, a `README.md`, a `LICENSE` file, and a `data/`
folder — not just a paper citation with no artifact). Fetched the raw
`LICENSE` file verbatim (quoted above). Fetched the `data/` directory listing
directly: it contains exactly **one file, `CaddieSet.csv`** — tabular
ball-flight and joint-angle metrics — and **no video files, no image files,
no bounding boxes, and no frame-level annotations of any kind**. Could not
reach the arXiv paper body (blocked, see above) to confirm the capture
setting (indoor/simulator bay vs. outdoor, lighting, camera model/frame
rate) that the research prompt specifically wants evidence on; this remains
unverified and should not be assumed either way.

**Which failure mode this addresses:** Neither, in practice, despite being
the first dedicated search this log has done for bullet 1 (a commercially-
licensed dataset, especially one covering indoor/low-light/simulator
footage where real motion blur would appear). The underlying capture used a
camera-based launch monitor, which is exactly the kind of fixed, controlled
rig that plausibly operates indoors/in simulator bays — but that's
speculation, not something this run could verify, because the raw
video/images were never published in the repo at all. There is nothing
here to extract a single clubhead frame from, so it cannot help motion
blur or camouflage.

**Why it matters for this model specifically:** It's a clean negative
result worth recording so a future run doesn't re-discover and re-verify
the same dead end. CaddieSet clears the licence bar this log has turned
away ClubheadDB and GolfDB on (both CC-BY-NC) — but a permissively licensed
dataset with no imagery is exactly as useless to a vision model as a
restrictively licensed one with imagery, just for a different reason. The
genuine unresolved question the prompt cares about (does any commercially-
usable dataset contain real indoor/simulator-bay/low-light golf footage
with visible motion blur) is still open after this run.

**Effort vs. payoff:** Low effort (two direct GitHub fetches settled it
conclusively — no ambiguity, no speculation needed on the licence or the
file contents). Zero payoff toward either failure mode, but non-zero payoff
as log hygiene: this closes off a dataset name that would otherwise keep
surfacing near the top of "golf dataset license" searches for every future
run on this rotation bullet. The commercially-clean-dataset-with-real-blur
gap this project actually needs filled remains unfound.

---

## 2026-11-06 (second run today) — WASB-SBDT: a real, MIT-licensed, heatmap-based multi-sport ball tracker, actually verified for the first time — correcting an unverified assumption this log was carrying

**Correction first, substance second:** The 2026-11-05 (fourth run today)
entry justified *not* logging a UAV small-object paper by asserting this
log "already carries multiple code-available, permissively-licensed
implementations of the same underlying idea... WASB-SBDT (MIT), OC-SORT
(training-free)." That was wrong to state as settled fact: neither
WASB-SBDT nor OC-SORT has an actual dedicated entry anywhere in this log —
both were only name-dropped, never fetched, licence-checked, or verified.
This run closes that gap for WASB-SBDT (OC-SORT remains unverified and
should not be treated as confirmed either, until a future run actually
checks it).

**What it is:** `nttcom/WASB-SBDT` — "Widely Applicable Strong Baseline for
Sports Ball Detection and Tracking" (Tarashima et al., BMVC 2023). A
cross-sport ball detection/tracking baseline, with evaluation code and
pretrained weights for soccer, tennis, badminton, volleyball, and
basketball, benchmarked against DeepBall, DeepBall-Large, BallSeg,
TrackNetV2, ResTrackNetV2, and MonoTrack. The `src/models/` directory
contains `hrnet.py`, `unet2d.py`/`unet2d_parts.py`, `resunet2d.py`,
`segbase.py` — HRNet and U-Net-family dense-prediction backbones, not a
YOLO-style box-regression head — and the eval config exposes a
`detector.postprocessor.use_hm_weight` flag, confirming the output is a
**heatmap** (per-pixel confidence/centre map), not a bounding box. This
architecture choice is the standard one in small-ball-tracking literature
(TrackNetV2 and its relatives) specifically because a heatmap only needs
to localise a plausible centre, not agree on tight box edges, which is far
more forgiving when the object's visible extent is ambiguous — exactly the
situation a motion-blur streak creates for a box-regression detector like
this project's YOLO11n.

**URL:** https://github.com/nttcom/WASB-SBDT

**Licence — verbatim:** Fetched `raw.githubusercontent.com/nttcom/WASB-SBDT/main/LICENSE.md`
directly: standard MIT License text, "Copyright (c) 2023 NTT Communications
Corporation," no additional restriction clause. **Commercial use:
permitted.**

**Verification performed:** Fetched the GitHub repo page, the raw
`LICENSE.md`, `MODEL_ZOO.md`, `GET_STARTED.md`, and the `src/` and
`src/models/` directory listings directly (all reachable). Confirmed: repo
is real, public, not archived (208 stars, 41 forks). Confirmed **training
code is explicitly not available** — `GET_STARTED.md`'s Training section
reads only "TBA"; the repo ships evaluation/inference code for pretrained
weights only. Confirmed pretrained weights are distributed via Google Drive
links in `MODEL_ZOO.md`, which could not be followed (Google Drive is
outside this session's reachable domains) — so whether the weights
actually still download, and under what terms (the MIT licence covers the
*code* repo; the weights' own terms are not separately stated anywhere
fetched), is **not verified**. The word "blur" does not appear anywhere in
the README, GET_STARTED.md, or MODEL_ZOO.md — the heatmap-vs-blur
connection above is this run's own architectural inference from the code
structure, not a claim the authors make themselves. Could not fetch the
paper abstract (arxiv.org blocked, as in every prior entry) or two
secondary mirrors tried (`www.emergentmind.com`,
`bmvc2022.mpi-inf.mpg.de`) — both newly confirmed DNS-blocked from this
session, added to the standing egress-gap list.

**Which failure mode this addresses:** Motion blur, as an architectural
pattern — not golf-specific, not camouflage-relevant (heatmaps don't help
when there are zero candidate activations at all, which is the camouflage
failure signature; they only help make localisation of a *present but
blurred* object more forgiving).

**Why it matters for this model specifically:** This is the first entry in
this log that points at the detection *head* design itself, rather than at
data, augmentation, or inference-time recovery tricks, as a lever for the
blur problem. The current pipeline is committed to YOLO11n's native box
head. If a future retrain shows blur-conditioned frames are still missed
or badly localised even after real blurred training data exists (per the
2026-11-04 SoccerSynth entry's recommendation), a heatmap-style auxiliary
head or a heatmap-based small-object detector family is a concretely
different, verified-available, commercially-clean architecture to
benchmark against — not a hypothetical. The caveat is real, though: no
training code exists in this repo to actually produce a clubhead-specific
heatmap model, no golf or clubhead data is involved, and the blur-robustness
argument is this run's inference, not the paper's own claim.

**Effort vs. payoff:** Low-moderate effort to verify (direct fetches only,
no code run); the genuine payoff is narrow — it upgrades an unverified
assumption into a confirmed one, and gives a named, real, licence-clean
architectural reference point for "heatmap vs. box regression" if the blur
problem survives better training data. It is not something to build on
this run: no trainable clubhead model comes out of this, and the
blur-robustness rationale is architectural reasoning, not demonstrated
evidence, so it should not be oversold as more than a lead worth
benchmarking later.
