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

---

## 2026-11-06 (third run today) — EventVCOD: closing the two leads the 2026-11-04 VCOD entry explicitly deferred, plus a real finding — its README claims a licence that the repo does not actually ship

**Follow-up context, not a repeat:** The 2026-11-04 VCOD entry named two repos it
found but deliberately did not verify — `EventVCOD` and `YUV20K` — "for a
future run to check independently rather than re-discovering." This run
closes both, per that entry's own instruction, rather than re-covering VCOD
as a topic from scratch.

**YUV20K, briefly:** `github.com/K1NSA/YUV20K` is real but is a dataset
landing page only — the repo holds just `LICENSE`, `README.md`, and
`.gitignore` (44 commits, no source code; the README's own roadmap lists
"Release source code" for 2026, unchecked). The README's licence section
states verbatim: *"The YUV20K dataset is available under the CC BY-NC 4.0
License"*, with an explicit note that the underlying videos are third-party
web content "for academic research only." **Commercial use: NOT
permitted.** Not pursued further — same NC dead-end pattern as GolfDB and
ClubheadDB, already logged.

**EventVCOD — the substantive finding.** `github.com/lyu-yx/EventVCOD` is
the official implementation of *"Towards Explainable Video Camouflaged
Object Detection: SAM2 with Eventstream-Inspired Data"* (Zhang, Lyu, et
al., Beihang University, **AAAI 2026**, vol. 40 pp. 12511–12519 — confirmed
via the paper's AAAI OJS listing, which this run could reach even though
`arxiv.org` itself remains blocked). Mechanism: instead of Vcamba's
frequency-domain Mamba scanning, this method synthesizes a cheap
event-camera-*like* signal directly from ordinary RGB video — per-pixel
frame differences, thresholded into positive/negative "polarity" (brighter
vs. darker than the previous frame), accumulated over a short time window
— and feeds that motion-only representation into SAM2 through a
memory-augmented dual-branch design (Time Bridge Attention + Visual Bridge
Attention), with a Prompt Embedding Generator that removes the need for a
human-drawn prompt box/point SAM2 normally requires.

**URL:** https://github.com/lyu-yx/EventVCOD (paper: AAAI 2026,
ojs.aaai.org/index.php/AAAI/article/view/38245)

**Licence — the actual finding:** The README states *"This project is
licensed under the Apache License 2.0 - see the LICENSE file for details"*
and links to `/blob/main/LICENSE`. **That file does not exist.** Verified
by fetching `raw.githubusercontent.com/lyu-yx/EventVCOD/main/LICENSE`
directly (HTTP 404), then `.../master/LICENSE` (also 404), then the repo's
own root file listing directly, which lists `assets/`, `checkpoints/`,
`data_manipulate/`, `datasets/`, `eval/`, `sam2/`, `tools/`, `training/`,
`.gitignore`, `README.md`, `__init__.py`, `requirements.txt`,
`sod_metric.py`, `test.py`, `train.py`, `utils.py` — no `LICENSE`,
`LICENSE.md`, or `LICENSE.txt` anywhere in it. A claimed-but-missing
licence is a worse state than Vcamba's silently-absent one (logged
2026-11-04): it's more likely to be taken at face value and copied from
without a second check. Per the same default-copyright reasoning as every
prior unlicensed-repo entry in this log, **treat as not usable
commercially** — the README's stated intent is not a substitute for an
actual licence grant in the repository, and should not be relied on even
informally without the authors adding the file or confirming it elsewhere
(e.g. contacting them directly).

**Verification performed:** Direct fetch of the GitHub repo root listing
(real, non-placeholder — `train.py`/`test.py`/full SAM2 integration code
are present, not a stub); direct fetch attempts on both branch names for
`LICENSE` (both 404, see above); confirmed the AAAI 2026 venue and page
range via the paper's own OJS listing (reachable, unlike arxiv.org).
**Not verified:** the three named datasets (`MoCA-Video-Train_event`,
`MoCA-Video-Test_event`, `CAD2016_event`) and the fine-tuned checkpoints are
hosted on a Baidu-style netdisk (`bhpan.buaa.edu.cn`), outside this
session's reachable domains, so this run could not confirm the links
resolve, whether access requires registration, or what (if any) terms
attach to the data itself as opposed to the code.

**Which failure mode this addresses:** Camouflage, same category as the
2026-11-04 Vcamba entry — not motion blur (a synthetic event/polarity
channel encodes *that* something moved, not *how blurred* it looks within
a frame).

**Why it matters for this model specifically — distinct from the Vcamba
entry, not a restatement of it:** Vcamba's architecture (a hand-written
Mamba selective-scan CUDA kernel) is not something to reimplement
independently of its own code. EventVCOD's core data trick is: thresholded
inter-frame pixel differences, which is a few lines of OpenCV, requires no
exotic architecture, no SAM2 dependency, and no training to *generate* —
only the SAM2/prompt-generator half of their pipeline is heavy and
unlicensed. That means the genuinely reusable idea here — stack a cheap
polarity/frame-difference channel alongside RGB as extra detector input,
so the model has direct access to "this pixel just changed" rather than
inferring motion only implicitly through weights — is realistically
buildable as an experiment on top of this project's own YOLO11n pipeline,
independent of EventVCOD's licence problem, and is a concrete instance of
the "extra motion channel" pattern the 2026-11-04 entry mentioned finding
in UAV small-object papers but never verified. It would need its own
from-scratch implementation and its own clubhead-specific validation —
nothing here proves it works on a golf swing — but it is now a specific,
cheap-to-prototype idea rather than a vague direction.

**Effort vs. payoff:** Low effort to extract the idea (frame-differencing
is trivial to implement, no dependency on the unlicensed SAM2 pipeline);
unproven payoff specific to clubheads — thresholded frame-differencing
degrades on its own fast camera pans/handheld shake (common in phone
footage of a swing) and may fire just as strongly on a swaying tree branch
in foliage as on the clubhead itself, which is exactly the kind of
background clutter the camouflage failures already involve. Worth a cheap
same-day prototype (compute a frame-diff channel over the existing 3-clip
eval set and look at it visually on the known camouflage-failure frames)
before investing in wiring it into training — this run did not build or
test that prototype, only identified and licence-checked the idea.

---

## 2026-11-06 (fourth run today) — YOLOV / YOLOV++: a YOLOX-based *video* detector that aggregates features across frames specifically to recover degraded frames, fully Apache-2.0 end to end (unlike the already-flagged Ultralytics AGPL-3.0 chain)

**Rotation context:** Targets bullet 3 (temporal/multi-frame methods for
detecting objects that are hard to see in a single frame) — the same bullet
Vcamba and EventVCOD covered, but this is a structurally different family:
a one-stage YOLO-lineage detector with a lightweight attention-based
temporal aggregation head, not a Mamba kernel or a SAM2 pipeline, and
licence-clean rather than unlicensed/license-claimed-but-missing like those
two.

**What it is:** `YuHengsss/YOLOV` — the official PyTorch implementation of
**YOLOV** ("YOLOV: Making Still Image Object Detectors Great at Video Object
Detection," Shi, Wang, Guo, AAAI 2023) and its follow-up **YOLOV++**
("Practical Video Object Detection via Feature Selection and Aggregation").
Built directly on **YOLOX** (one-stage, anchor-free — the same detector
family lineage as this project's own YOLO11n, not an unrelated architecture).
The stated motivation, from the paper's own abstract (via search, since
`arxiv.org` remains DNS-blocked from this session — see Verification): video
object detection is hard because of "high variation of object appearance as
well as the diverse deterioration in some frames," and "a frame can borrow
evidence from its neighbors." Mechanism: a Feature Selection Module first
picks a small set of candidate region features per frame (cheap, since it
runs after the one-stage detector head, not before, unlike older two-stage
video-detection aggregation methods); a Feature Aggregation Module then
computes attention-style affinity between the current (target) frame's
candidates and reference frames' candidates, using both classification and
IoU scores, and aggregates across frames before final classification. On
ImageNet VID, the released weights report **87.5% AP50 at 30+ FPS on a single
2080Ti** (YOLOV), with YOLOV++ variants reported higher still.

**URL:** https://github.com/YuHengsss/YOLOV (papers: arXiv:2208.09686 / AAAI
2023, and the YOLOV++ follow-up — not independently fetchable this run, see
Verification).

**Licence — verbatim, checked at both ends of the dependency chain:**
Fetched `raw.githubusercontent.com/YuHengsss/YOLOV/master/LICENSE` directly:
the Apache License, Version 2.0, January 2004, standard text, no added
restriction. Separately fetched the upstream base detector's licence,
`raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE`: also
Apache License 2.0, with the appendix naming "Copyright (c) 2021-2022 Megvii
Inc." **Commercial use: permitted**, for both the video-aggregation code and
the base detector it's built on. This is a direct, useful contrast with the
2026-11-05 SwingTrace entry's licensing side-finding: that entry flagged
Ultralytics YOLO (and this project's own planned `ultralytics` dependency) as
AGPL-3.0, requiring an Enterprise Licence for unambiguous commercial closed-
source use. YOLOX/YOLOV sits in the same one-stage-YOLO lineage but is
entirely outside that AGPL chain — Megvii's YOLOX predates and is legally
unrelated to Ultralytics' licensing choice.

**Verification performed:** Fetched the GitHub repo page directly (real,
public, has a root `LICENSE` file, working `tools/vid_train.py` /
`tools/vid_demo.py` / `tools/yolov_demo_online.py` / `tools/REPPM.py`
scripts, pip-installable, and released pretrained weights for YOLOV-s/l/x
and several YOLOV++ variants, some linked via Hugging Face, with per-variant
ImageNet VID mAP@50 numbers in the README table). Fetched both raw `LICENSE`
files verbatim (quoted above) for YOLOV and its YOLOX base. Could not fetch
the arXiv paper bodies directly (`arxiv.org` still DNS-blocked, consistent
with every prior entry in this log) — the "diverse deterioration in some
frames" / "borrow evidence from its neighbors" motivation and the FSM/FAM
mechanism description above come from secondary summaries of the abstract
found via search, not a direct read of the PDF, so the exact wording and any
explicit mention of motion blur or occlusion specifically (as opposed to
"deterioration" generically) is **not confirmed verbatim**. The README's own
TODO list still shows unchecked items ("Finish Swin-Transformer based
experiments," "Release updated code, model and log"), so treat the repo as
real and usable today but not fully finished per its own authors.

**Which failure mode this addresses:** Both, by mechanism rather than by
explicit golf-specific design — a frame where the clubhead is blurred past
recognition, or briefly camouflaged against clothing/foliage, is exactly the
"deteriorated frame" case the FAM is designed to recover by borrowing
evidence from nearby frames where the same object was clearly detected. This
is mechanistically closer to the already-logged `rucv/golf_ball` Kalman-patch
idea (use temporal context to rescue a bad frame) than to the VCOD/EventVCOD
entries (which specifically target single-frame appearance-camouflage via
motion-as-a-cue, not frame quality in general) — it does not single out
"this pixel moved" the way those do, so it is a weaker, more generic claim
for camouflage specifically than the VCOD line of work, but a comparably
strong claim for motion blur (a genuinely deteriorated frame, not a
same-background-color frame).

**Why it matters for this model specifically:** Three concrete reasons this
is a stronger lead than Vcamba/EventVCOD for this project's actual
constraints, and one real gap. (1) **Licence-clean**, unlike both prior VCOD
entries (Vcamba: no licence; EventVCOD: claimed but missing licence) — usable
today without a legal question mark. (2) **Architecturally closer to the
existing pipeline** — a one-stage YOLO-family detector with an added
attention module, not a hand-written CUDA Mamba kernel or a SAM2 dependency,
so the engineering distance from "YOLO11n single-frame" to "YOLO-family
multi-frame with an aggregation head" is smaller than adopting either VCOD
architecture, even though it would still mean moving off Ultralytics'
specific codebase (or re-implementing just the FSM/FAM idea on top of
YOLO11n's own one-stage head, which is architecturally compatible in
principle). (3) **Reported speed (30+ FPS on a 2080Ti for a reference
GPU)** is in a plausible range to reason about an on-device budget, unlike
VCOD architectures with no stated inference speed at all. **The real gap:**
nothing here is CoreML-exportable or validated on a phone — this is a
PyTorch research codebase for desktop/server GPUs, multi-frame by
construction (needs a short window of reference frames, not one still
image), and no entry in this log, including this one, has verified that a
multi-frame-aggregation head survives Core ML export or runs at acceptable
latency on-device. That is a materially bigger unknown than the model
architecture claims above, and should not be glossed over.

**Effort vs. payoff:** Medium effort to prototype the *idea* (a feature-
selection + cross-frame-attention aggregation head on top of YOLO11n's
existing feature maps is a non-trivial but bounded research experiment, not
a rewrite), high effort and unproven payoff to actually ship it on-device
(multi-frame input changes the Core ML export shape, the Vision integration
in `ClubHeadDetector.swift`, and the inference-time latency budget in ways
none of this log's entries have scoped). Realistic near-term use: a GPU-side
research experiment to test whether frame-aggregation measurably recovers
this project's own blurred/camouflaged failure frames (using the existing
3-clip eval set's known failures as a cheap sanity check) before committing
to any on-device integration work — the licence and architecture are no
longer blockers; the open question is purely whether the payoff clears the
on-device engineering cost, which this run did not attempt to estimate
further.

---

## 2026-11-07 — RT-Focuser: a small, MIT-licensed, CoreML-deployable motion-deblurring network, actually benchmarked on an iPhone — the first on-device deblur-as-preprocessing candidate this log has verified

**Rotation context:** Bullet 2 (motion blur specifically — "video deblurring
usable as preprocessing" sub-bullet). Prior entries covering blur
(SoccerSynth, 2026-11-04; WASB-SBDT, 2026-11-06) addressed training-data
synthesis and detector-head architecture respectively. Neither touched
deblurring as an inference-time preprocessing step ahead of the detector,
and none of this log's entries have a concrete, verified, on-device-deployed
deblur network. This run fills that specific gap.

**What it is:** `ReaganWu/RT-Focuser` — a small CNN trained for single-image
motion deblurring on the GoPro dataset, with a second, newer repo
(`ReaganWu/RT-Focuser-v2`) adding an efficient global-attention module and
four selectable speed/quality "exits." V1 ships a pretrained checkpoint
(`GoPro_RT_Focuser_Standard_256.pth`, 5.85M parameters, 15.76 GMACs) plus
pre-converted ONNX variants (`rt_focuser_wint8_afp16.onnx`,
`rt_focuser_wint8_afp32.onnx`, `rt_focuser_wint8_aint8.onnx`) in its
`Pretrained_Weights/` folder. Its README reports a direct CoreML benchmark:
**146.72 FPS on an iPhone 15 (A16 Bionic)**. V2 is smaller still at its
fastest exit (Y1: 1.25M params, 259.06 FPS on iPhone 15/CoreML per its own
README) but — important caveat below — ships no pretrained weights yet.

**URL:** https://github.com/ReaganWu/RT-Focuser and
https://github.com/ReaganWu/RT-Focuser-v2

**Licence — verbatim:** Fetched `raw.githubusercontent.com/ReaganWu/RT-Focuser/main/LICENSE`
directly: standard MIT License, "Copyright (c) 2025 ReaganWu," no added
restriction. Fetched `raw.githubusercontent.com/ReaganWu/RT-Focuser-v2/main/LICENSE`
directly: also standard MIT License, "Copyright (c) 2026 RT-Focuser-V2
Authors," no added restriction. **Commercial use: permitted**, for both
repos' code and (V1 only) its shipped pretrained weights. The training data
itself — the GoPro deblurring dataset (Nah, Kim, Lee, CVPR 2017) — is
separately confirmed via its own listing page (surfaced by search; the
canonical host, `seungjunnah.github.io`, is DNS-blocked from this session
like every other `*.github.io`/academic-page host this log has hit, so this
specific claim rests on the search engine's synthesis of that page, not a
direct fetch) to be released under **CC BY 4.0** — permissive, commercial
use allowed, attribution required. No NC clause found anywhere in the
training-data chain, unlike most dataset entries this log has logged so far.

**Verification performed:** Fetched the RT-Focuser GitHub repo page directly
(real, public, 29 stars, 28 commits — a small but genuine project, not a
placeholder); fetched its `Pretrained_Weights/` directory listing directly
and confirmed the four weight files actually exist in the repo (not just
named in the README); fetched both raw `LICENSE` files verbatim (quoted
above); fetched the RT-Focuser-v2 repo page directly and confirmed its file
tree (`rt_focuser_v2.py`, `inference.py`, `train.py`, `evaluate.py`,
`data.py`, `losses.py`, no `.mlmodel`/`.mlpackage` artifact — CoreML export
is a code path invoked via `coremltools` on macOS, not a pre-exported file)
and confirmed, directly from its README, that **V2 ships no pretrained
checkpoint** ("Pretrained checkpoints... are not included in this repository
yet") — its 259 FPS figure is an architecture/throughput claim, not
something downloadable and runnable today. Could not fetch the GoPro
dataset's own license page directly (blocked, see above) and could not
independently verify RT-Focuser's claimed 30.67 dB PSNR deblurring quality
figure against any held-out benchmark (no paper/arXiv entry found for this
project — it appears to be an independent engineering project, not a
peer-reviewed publication, which is a reliability flag by itself).

**Which failure mode this addresses:** Motion blur only. Has no bearing on
camouflage — a deblur network sharpens an already-visible-but-smeared
object; it does nothing for a sharp, non-blurred frame where the clubhead
is simply low-contrast against its background (the camouflage failure
signature), and could plausibly do nothing useful on such a frame either
way.

**Why it matters for this model specifically:** Every blur-related entry so
far in this log is either training-side (SoccerSynth's synthetic blur
augmentation) or architecture-side (WASB-SBDT's heatmap head). This is the
first candidate for a third lever: an inference-time preprocessing step —
deblur the frame, then run the existing YOLO11n detector on the sharpened
result — that requires no retraining of the detector at all and is
demonstrated, not hypothesized, to run fast enough on an actual iPhone via
CoreML, the project's own exact deployment target. That makes it a real
alternative (or complement) to the already-logged inference-time-recovery
ideas (`rucv/golf_ball`'s Kalman-patch re-crop, 2026-11-05) which work by
changing *where* the detector looks; this works by changing *what* the
detector sees. Two serious unverified risks, though: (1) RT-Focuser was
trained on GoPro's camera-shake/general-scene blur, which is dominated by
whole-frame or large-object blur from an unstable handheld camera — a golf
clubhead's blur is a small, fast-moving object streaking against a
comparatively stable background, a different blur geometry the network has
never been shown to handle, and deblurring could just as easily erase the
elongated streak shape (the only cue this project's own labeling spec asks
annotators to preserve) as sharpen it usefully; (2) stacking a 5.85M-param
deblur pass ahead of YOLO11n adds real on-device compute even if each piece
individually benchmarks fast alone — nothing here measures the combined
pipeline's actual end-to-end framerate on a phone, only the deblur model in
isolation.

**Effort vs. payoff:** Low effort to pilot — the V1 checkpoint and ONNX
weights are already downloadable and the model is small; a credible first
test is running the existing CoreML export path (or the ONNX weights
directly) over the known blurred/failing frames in the project's own 3-clip
eval set and visually checking whether the output looks like a recognizable
clubhead or a hallucinated artifact, before touching the detector or
training pipeline at all. Payoff is genuinely unproven and specifically
two-sided risk (helps detection vs. destroys the one blur cue the model
might already be partially learning from) rather than a confident
improvement, and V2's higher numbers are not currently usable for anything
beyond "a cheaper version may exist soon" since it has no weights. Worth
the cheap pixel-level pilot described above; not worth wiring into the
production pipeline on the strength of this entry alone.

---

## 2026-11-07 (second run today) — CamDiff: diffusion-inpainting-based object compositing for camouflage data, checked directly — the repo itself is unlicensed and the mechanism doesn't transfer as-is, but it's the first concrete evidence this log has that the *technique* (not this specific artifact) is viable

**Rotation context:** Bullet 5 (synthesize/augment training data for either failure regime). The last three runs today covered bullet 2 (RT-Focuser, blur preprocessing) and bullet 3 (YOLOV/YOLOV++, EventVCOD/YUV20K close-out, both camouflage/temporal); this run deliberately moves to a bullet not yet touched today.

**What it is:** `drlxj/CamDiff` — "CamDiff: Camouflage Image Augmentation via Diffusion Model" (Luo, Wang, Wu, Sakaridis, Cheng, Fan, Van Gool; arXiv:2304.05469, 2023; mostly ETH Zürich). **Important correction to how this could be misread:** it does not synthesize *more camouflaged target objects* — it does the opposite. It uses a latent diffusion inpainting model (`runwayml/stable-diffusion-inpainting`) to insert a new, *salient* (easy-to-see) distractor object into an existing camouflaged-object-detection (COD) training image, verifies the inpainted object actually matches its text prompt via CLIP zero-shot classification (to catch failed generations), and keeps the original image's camouflage ground-truth mask unchanged. The purpose is to make COD training harder by adding a distracting salient object nearby, so the model can't shortcut to "the only interesting thing in this frame" — it has to actually find the camouflaged target despite a competing salient one.

**URL:** https://github.com/drlxj/CamDiff (paper via search summary only — `arxiv.org` and its mirrors remain DNS-blocked from this session, consistent with every prior entry in this log, so the PDF body was not read directly)

**Licence:** Fetched the GitHub repo directly. File listing is `.gitignore`, `README.md`, `clip_classification.py`, `inpaint.sh`, `inpainting_diff.py`, `paper.py`, and an `Imgs/` folder — **no `LICENSE` file anywhere**, and GitHub's own "About" section shows no detected license. Same default-all-rights-reserved situation already logged twice for Vcamba (2026-11-04) and EventVCOD (2026-11-06), except CamDiff's README doesn't even claim a license the way EventVCOD's incorrectly did. **Commercial use: not confirmed as permitted — treat the repo's code and generated `Diff-COD` dataset as not usable as-is.** The method depends on `runwayml/stable-diffusion-inpainting` (Hugging Face) and OpenAI's CLIP; `huggingface.co` is DNS-blocked from this session (newly confirmed blocked this run, added to the standing egress-gap list), so Stable Diffusion inpainting's own licence (commonly CreativeML OpenRAIL-M, which permits commercial use of outputs subject to a use-restriction list) could **not** be verified this run and should not be assumed.

**Verification performed:** Fetched the GitHub repo root listing directly (real, non-archived; confirmed the generation command `python inpainting_diff.py --indir ./Dataset --outdir ./result` and a working Google Drive dataset link, ~5GB, for the generated `Diff-COD` images — Google Drive itself is outside this session's reachable domains so the link's current liveness is unconfirmed). Confirmed via search (not a direct arXiv fetch) the paper's authorship, 2023 date, and mechanism description, cross-checked against the GitHub README's own wording, which agrees. Separately searched for a closer match to what this project actually needs — synthesizing more examples of the hard-to-see *target* itself, not a salient distractor — and found "Quasi-synthetic data generation for camouflaged object detection at edge" (Putatunda et al., SPIE 2023, `journals.spiedigitallibrary.org/.../10.1117/12.2678034`), whose abstract states it uses "image compositing combined with attention-based deep learning harmonization to generate feature-enriched realistic images for camouflaged objects under varying scenarios" and claims this "significantly improve[s] the detection accuracy of camouflaged objects under varying conditions" — a much closer match in intent, but its full text sits behind an SPIE paywall and no GitHub repository was found for it anywhere in search. **Not logged as this run's primary finding** because it could not be verified beyond the abstract (no code, no license, no confirmation the claimed result replicates) — flagged here, by name and venue, for a future run to pursue if SPIE access or a code repo ever becomes reachable, rather than re-discovering it from scratch.

**Which failure mode this addresses:** Camouflage. Has no bearing on motion blur (nothing here touches blur streaks or frame timing).

**Why it matters for this model specifically:** CamDiff itself is not reusable (wrong direction — it adds distractors, doesn't synthesize more hard-to-see targets — and unlicensed regardless), but it is the first entry in this log to directly verify a *mechanism* this project could repurpose: diffusion-model inpainting, with CLIP used only as an automatic quality gate (reject bad generations, no human review needed), to composite a chosen object into a chosen scene convincingly rather than via flat copy-paste. Flipping CamDiff's own direction — instead of inpainting a salient distractor into a camouflage scene, inpaint (or outpaint-composite) a clubhead crop onto backgrounds specifically chosen to match the project's own documented failure conditions (dark clothing, cluttered foliage) — is a plausible, buildable way to generate large numbers of synthetic hard-camouflage training examples with a known, exact ground-truth box, directly targeting the project's stated gap (camouflage failures were found by visual inspection of a 3-clip test set, not engineered into training). This would need its own from-scratch implementation (CamDiff's specific code isn't licensed for reuse, and its default direction is backwards for this use case), its own base-model licence check (Stable Diffusion's licence terms, unverified this run), and real validation that diffusion-inpainted clubheads look plausible enough to help rather than teach the model to detect inpainting artifacts instead of real camouflage.

**Effort vs. payoff:** Medium-high effort to build (a working inpainting pipeline, a prompt/mask strategy for "insert a clubhead convincingly into this backdrop," and a held-out validation loop — not a weekend prototype), payoff genuinely unproven and two-sided: diffusion inpainting is well known to leave subtle statistical fingerprints a CNN can learn to key on instead of the intended visual cue, which would be actively counterproductive for a camouflage fix (the model would learn "detect inpainting artifact," not "detect dark clubhead against dark background"). Lower effort, moderate payoff alternative worth trying first: skip diffusion entirely and prototype plain alpha-composited copy-paste (real clubhead crops pasted onto real dark-clothing/foliage photos, à la the small-object copy-paste augmentation literature from Kisantal et al. 2019) as a cheaper first test of whether *any* synthetic-background compositing helps before paying for a diffusion pipeline. Net: this run's actual deliverable is narrowing the camouflage-augmentation search space (two real techniques identified, one verified-but-wrong-direction-and-unlicensed, one promising-but-paywalled-and-uncoded) rather than a ready-to-use asset — useful to rule things in/out, not something to start building from today.

---

## 2026-11-07 (third run today) — GolfPose: a golf-specific detector/pose model that explicitly includes the club (not just the body), with a dataset EULA that actually permits commercial model training — but access is author-gated and the "+N" club keypoints are unexplained

**Rotation context:** Bullet 4 (golf-specific pose or club tracking papers, benchmarks, or open-source implementations). This run's first attempt was bullet 1 (datasets for low-light/indoor/simulator footage) — `universe.roboflow.com`, `kaggle.com`, and `zenodo.org` are all still DNS-blocked from this session (same standing egress gap every prior run from 2026-11-04 onward has hit), which rules out the dataset hosts that actually carry labelled golf-swing imagery at scale, so nothing in that bullet could be verified today and the rotation moved on rather than logging an unverifiable Roboflow-page summary as if it were checked. Today's three runs now cover bullets 3, 2, 5, and 4 — only bullet 1 (datasets) remains blocked by network, not by absence of candidates; `golf_lowlight` on Roboflow Universe (found by search, not fetched) is flagged here by name for a future run to check the moment that host is reachable.

**What it is:** `MingHanLee/GolfPose` — the official implementation of "GolfPose: From Regular Posture to Golf Swing Posture" (Lee, Zhang, Wu, Tseng; ICPR 2024 proceedings, published by Springer 2025, pp. 387–402; PAIR Lab, National Yang Ming Chiao Tung University, Taiwan). Unlike every pose/tracking paper this log has looked at so far, it is explicitly built around the golf **club**, not just the golfer's body: it ships a 2-class object detector (`golfer`, `club`) and a 1-class detector (`golfer-with-club`), plus 2D pose variants `GolfPose-2D(C)` (club-only extra keypoints) and `GolfPose-2D(GC)` (golfer+club combined) alongside a body-only `GolfPose-2D(G)` baseline, and 3D pose checkpoints labelled "17+0" through "17+5" (17 standard body joints plus up to 5 extra keypoints). Reported AP on the authors' own test split: body-only 2D (G) 0.869–0.887 vs. club-inclusive 2D (GC) 0.915–0.942 depending on backbone (HRNet-w48 / ViTPose-H / DEKR) — i.e. the club-aware variant is reported as *more* accurate than body-only, not a tradeoff.

**URL:** https://github.com/MingHanLee/GolfPose (paper: Springer ICPR 2024/2025 proceedings; not independently fetched — `link.springer.com` was not attempted this run and the repo's own README was treated as the primary source)

**Licence — as summarized from the repo (not independently re-verified byte-for-byte):** The repo's `LICENSE` file is a custom EULA, not a standard open-source license. Per the README directly: *"You are allowed to use the dataset for commercial model training, but strictly prohibited from redistributing or reselling the raw data."* The LICENSE file itself (summarized by the fetch tool, which hit a length limit quoting it character-for-character — flagged honestly rather than presented as a verbatim quote) grants "a non-exclusive, non-transferable, non-sublicensable license for internal research, evaluation, and training machine learning models," states "Models trained on the dataset may be integrated into commercial products or services," and separately prohibits redistributing the raw dataset in any form, selling or monetizing it, using it for facial/identity recognition, or implying endorsement by the authors or university. Attribution (citing the dataset/paper) is required in both academic and commercial product documentation. **Commercial use: permitted for training and shipping a trained model — a real exception in this log, where most entries so far have been NC-licensed, unlicensed, or AGPL.** The one real constraint for this project is attribution in product docs, which is cheap to satisfy.

**Verification performed:** Fetched the GitHub repo page directly (real, public, 50 stars, 8 forks, 28 commits — small but genuine, not a placeholder) and its README in full. Confirmed the LICENSE file exists and contains EULA language (not boilerplate MIT/Apache text) via direct fetch, though the quote above is the fetch tool's summary of that file rather than a raw re-paste — flagged as a limitation, not hidden. Confirmed the dataset itself is **not freely downloadable today**: the README states plainly, "Please email mhlee.cs09@nycu.edu.tw to authorize the dataset download," so unlike every GitHub-hosted code/weights entry logged so far (RT-Focuser, WASB-SBDT, YOLOV), this is gated behind a human granting access, with no stated turnaround time or guarantee of grant. The pretrained-checkpoint download links (gofile.me) are not described as gated the same way, but their actual reachability and whether they work without the dataset license being separately agreed to was not tested this run. Could not determine what the "+1" through "+5" extra 3D keypoints actually are — the README names the checkpoints but never defines the extra joints, and it is plausible they are additional body joints (e.g. head, feet) rather than club-specific points (shaft tip, clubhead), which would materially weaken this entry's relevance if true; this is an open question, not a confirmed fact either way.

**Which failure mode this addresses:** Primarily camouflage, weakly motion blur. The mechanism is not visual-appearance-based at all (unlike every other camouflage entry in this log so far) — it is kinematic: a club-aware pose model gives a predicted club/clubhead position derived from the golfer's body geometry (wrists, shoulders, swing phase) even in a frame where the clubhead itself is invisible to a pure appearance detector, whether because it blends into the background (camouflage) or because it is a blur streak the detector doesn't recognize as a clean bounding box (motion blur). This is the same "give the detector a prior search region instead of relying on raw appearance" idea already logged for `rucv/golf_ball`'s Kalman-patch approach (2026-11-05), but derived from single-frame body pose instead of multi-frame motion history — a genuinely different and complementary signal source, not a restatement of that entry.

**Why it matters for this model specifically:** This is the first entry in this log that is a *golf-specific, club-aware* model with a license this project could actually use commercially, rather than a generic COD/deblur technique borrowed from an unrelated domain. Two concrete, separate uses: (1) fine-tune or distill the 2-class `golfer, club` detector as a second, independent detection signal to ensemble with the existing YOLO11n clubhead model — a frame where YOLO11n finds zero candidates at any confidence (the project's own documented camouflage failure signature) is exactly the case where a differently-trained detector disagreeing usefully would matter; (2) use the pose keypoints (if club-specific ones exist) as a geometric prior to narrow the search region or bias confidence in ambiguous frames, similar in spirit to the already-logged Kalman-patch idea but pose-derived rather than motion-derived. Neither use is demonstrated here — GolfPose was evaluated on its own authors' test set for pose/detection AP, not on this project's camouflage or blur frames, and no comparison to this project's 82%/77%/85.9% numbers exists anywhere.

**Effort vs. payoff:** Non-trivial effort before any payoff is even assessable: emailing an academic author for dataset access with no guaranteed response or timeline is a real blocker (unlike a plain `git clone`), the extra-keypoint definition must be clarified before knowing if "club" keypoints are actually club-specific, and the stack (MMDetection/MMPose/MixSTE) is heavy and not CoreML-native, so any usable piece would need re-export work regardless of which use case is pursued. Payoff is plausible but entirely unproven for this project's two failure modes — the AP numbers show the club-aware variant beats body-only on the authors' own benchmark, which is suggestive that club-context helps pose accuracy in general, but says nothing about camouflage or blur specifically. Worth a single email to request access and a cheap first check of what's actually in the "+N" keypoints before investing further; not worth building an integration around until that question is answered.

---

## 2026-11-07 (fourth run today) — GolfDB checked and ruled out as a public-dataset answer to bullet 1; the one reusable part is a methodology, not the data, and it points at the labeling-spec gap instead of camouflage/blur directly

**Rotation context:** Bullet 1 (datasets, especially low-light/indoor/simulator/older-phone footage). Today's first three runs covered bullets 2 (RT-Focuser), 5 (CamDiff), and 4 (GolfPose); every prior attempt at bullet 1 across this whole log (2026-11-04 ClubheadDB, 2026-11-06 CaddieSet, and the aborted first attempt of the 2026-11-07 GolfPose run) has been blocked or has dead-ended, largely because `universe.roboflow.com`, `kaggle.com`, and `zenodo.org` are DNS-blocked from this session. This run tried a different angle: searching for golf-video datasets reachable through hosts other than those three, rather than retrying the same blocked hosts again.

**What it is:** `wmcnally/golfdb` — "GolfDB: A Video Database for Golf Swing Sequencing" (McNally, Vats, Pinto, Dulhanty, McPhee, Wong; CVPR Workshops 2019, CVSports, DOI 10.1109/CVPRW.2019.00311). It is the best-known, most-cited public golf-video dataset in this space (companion to the `SwingNet` MobileNetV2+BiLSTM baseline, 71.5% PCE on event sequencing). 1,400 golf swing videos sourced from YouTube, each labeled with 8 swing-phase event frames (address, toe-up, mid-backswing, top, mid-downswing, impact, mid-follow-through, finish) plus per-video metadata (a bounding box, player name/gender, club type, view type, frame rate). The repo ships preprocessed clips at **160×160 px**.

**URL:** https://github.com/wmcnally/golfdb (paper: CVPR Workshops 2019, arXiv:1903.06528 — not fetched directly; `arxiv.org`, `openaccess.thecvf.com`, `deepai.org`, `mlanthology.org`, `hyper.ai`, and `api.semanticscholar.org` were all tried this run and are all DNS-blocked from this session, extending the standing egress-gap list every prior entry in this log has hit)

**Licence:** Fetched the GitHub README directly (primary source, not search-summarized). It states: *"The code in this repository is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License."* No separate `LICENSE` file is visible in the repo's file listing, and the README gives the raw video data no license statement distinct from the code's — the NC term is the only one on offer for either. **Commercial use: not permitted.** This is the third golf-video dataset in a row this log has found NC-licensed (after ClubheadDB's CC BY-NC 4.0 and the GolfDB-adjacent search hits for "golf swing" style datasets), and on top of the license, the dataset is distributed as YouTube video URLs the user must download themselves, which layers YouTube's own ToS restrictions on top regardless of the CC term.

**Verification performed:** Fetched the GitHub repo and its README directly and confirmed the license line and the 160×160 clip format verbatim from the primary source. Cross-checked the paper's content (event-frame labels, the per-video "bounding box" field, club type/view type, 1,400 videos) against three independent secondary summaries (web search results surfacing the CVF abstract and a `hyper.ai` mirror page, neither of which this session could fetch directly) — all three agree on the same description, and none of them describe the "bounding box" field as a clubhead box; the paper's own framing lists club/pose keypoint tracking as a *future* extension of the dataset, implying the box that exists today is a player/scene framing box, not a clubhead box. This could not be confirmed with total certainty (the primary abstract itself was unreachable), so it is reported as "most likely, not fully confirmed." Also surfaced two Roboflow Universe "golf swing" projects (`lvs-rd/golf-swing`, `suranaree-university-of-tecnology/golf-swing-detec`) that search describes as CC BY 4.0 (commercially usable) — these remain **unverified** because `universe.roboflow.com` is blocked from this session; flagged by name for a future run to check the moment that host becomes reachable, rather than logging an unfetched license as if it were confirmed.

**Which failure mode this addresses:** Neither, as a data source. Three independent reasons line up against it: (1) its per-video annotation is most likely a player box, not a clubhead box, so it isn't even the right label type for this project's detector; (2) 160×160 px is too low-resolution to usefully train or evaluate clubhead detection, where the clubhead itself is already a small object within a full frame; (3) the source footage is YouTube golf-swing video — almost certainly outdoor, broadcast-quality, well-lit, high-shutter — which does nothing to fill the project's documented indoor/low-light/older-phone blur gap, and is NC-licensed regardless. The one thing extracted from this entry that *is* reusable is a free methodology, not GolfDB's data: its 8-class swing-phase event labeling is a technique this project could apply to its **own** footage (fine-tune a cheap phase classifier on the project's own labeled clips, no GolfDB data needed) to automatically flag which of its own unlabeled frames fall in the impact window — the window where real motion blur, if it exists in the project's footage at all, would actually show up — as a prioritization tool for the labeling-spec gap this task brief itself flags (median box elongation 1.60 despite the spec instructing annotators to box the full streak). That is a motion-blur-adjacent labeling-QA idea, not a camouflage or blur fix on its own.

**Why it matters for this model specifically:** This closes off the most discoverable "golf + video dataset" result in the literature as a non-starter, and for three independent reasons at once (license, resolution, box type) rather than just one — which is stronger negative signal than the earlier ClubheadDB/CaddieSet dead-ends taken alone. Combined with those two, every public golf-swing dataset this log has found so far that is even plausibly relevant is either NC-licensed, missing imagery entirely, or wrong-shaped for clubhead detection — reinforcing that bullet 1's real answer is likely "capture the project's own indoor/low-light footage" rather than "find a licensable public dataset," a conclusion this log has now earned through direct checking of three separate candidates rather than assumed from the start.

**Effort vs. payoff:** Low effort to rule out (one direct GitHub fetch plus cross-checked search corroboration) for a dataset famous enough that skipping the check would have been a real gap in this log. Payoff is informational, not a usable asset: it removes a tempting-looking option from consideration cheaply, and the one reusable nugget (a self-supervised swing-phase classifier on the project's own footage, as a cheap labeling-QA prioritization pass for the blur-elongation gap) costs little to pilot since it needs no external data or license at all. Not worth pursuing GolfDB itself any further; worth a cheap, separate experiment with the phase-classifier idea the next time someone is already looking at the labeling pipeline.

---

## 2026-11-08 — YOLOMG: a verified, code-and-dataset-shipping motion-vs-appearance fusion detector for small moving objects against cluttered backgrounds — but GPL-3.0'd, so only the architecture idea, not the code, is usable here

**Base-branch note:** This run's actual git base (`origin/research/clubhead-2026-11-07`) differs from the branch `research/clubhead-2026-11-08` already pushed to the remote before this run started. That branch's single commit deleted this entire log (all 16 prior entries, 2026-11-04 through 2026-11-07) and replaced it with one new entry that duplicates the already-logged 2026-11-04 ClubheadDB finding almost verbatim — it both lost history and violated the "don't repeat a logged topic" rule in one move. Following this log's own established precedent for a broken daily base (see the 2026-10-21 → `-corrected` chain), this run restores the full log from the last good state (2026-11-07) and pushes as `research/clubhead-2026-11-08-corrected` instead of building on the broken `research/clubhead-2026-11-08`. The broken branch's content is not reproduced here since it isn't new information (same dataset, same conclusion, already covered below under 2026-11-04).

**Rotation context:** The four runs on 2026-11-07 covered bullets 2 (RT-Focuser), 5 (CamDiff), 4 (GolfPose), and 1 (GolfDB). This run covers bullet 3 — techniques for detecting small, low-contrast, or camouflaged objects in video, specifically the multi-frame/temporal angle the task brief calls out directly ("motion is what separates a moving clubhead from static foliage when appearance cannot").

**What it is:** YOLOMG — "Vision-based Drone-to-Drone Detection with Appearance and Pixel-Level Motion Fusion" (arXiv:2503.07115, March 2025). It targets near-identical structural problem to this project's camouflage failures: detecting a *small, fast-moving* object (another drone) against backgrounds where appearance alone is weak or ambiguous. Its mechanism, confirmed from the repo's own file listing (`generate_mask5.py` producing a `mask32` data variant, `models/ARD100_drone_s.yaml`, `dualdetector.py`): it computes a pixel-level motion-difference map between consecutive frames and fuses that map with the RGB appearance frame through a dedicated fusion module before the YOLO detection head, rather than relying on a single RGB frame. The paper also introduces ARD100, an air-to-air drone dataset the authors describe as having the smallest average object size (~0.01 of frame area) among comparable drone-detection datasets (100 videos, 202,467 frames).

**URL:** https://github.com/Irisky123/YOLOMG (paper: https://arxiv.org/abs/2503.07115)

**Licence, verified directly against the repo's own `LICENSE` file (not a search summary):** fetched `raw.githubusercontent.com/Irisky123/YOLOMG/main/LICENSE` directly. It is the **GNU General Public License, Version 3** (GPLv3), the canonical FSF text ("GNU GENERAL PUBLIC LICENSE / Version 3, 29 June 2007 / Copyright (C) 2007 Free Software Foundation, Inc."). GitHub's own repo sidebar independently labels it "GPL-3.0 license," matching the file contents. **Commercial use of GPLv3 code is permitted, but it is copyleft**: any program that incorporates or is built from GPLv3 code must itself be distributed under GPLv3, source included, if distributed at all. For a closed-source commercial iOS app, that means this repo's *code* cannot be adapted or linked into the shipped pipeline without open-sourcing the derived portions — the same trap already flagged for `nishsm/golf-club-tracking` (AGPL-3.0) in the 2026-11-05 entry, now confirmed a second time in this exact research area. The ARD100 dataset itself (irrelevant here regardless — drones, not golf) is distributed via a BaiduYun link with no separate license stated.

**Which failure mode this addresses:** Camouflage, directly and as a concrete architecture rather than a survey (unlike the 2026-11-04 VCOD-field entry, which concluded temporal/motion cues were the right mechanism but found no deployable implementation). YOLOMG is a working demonstration of exactly that mechanism in a structurally similar domain: small object, frame-to-frame motion the dominant discriminative signal, appearance-only detection weak. It does not touch motion blur.

**Why it helps this model specifically:** The task brief's own framing of the camouflage failures — "dark clubhead against dark clothing or cluttered foliage... producing ZERO candidate detections even at confidence 0.05" on "visually sharp frames" — describes exactly the failure mode a single-RGB-frame detector would have against a background it cannot appearance-discriminate, and exactly the case YOLOMG's motion-fusion branch is built to recover, since the clubhead is still moving relative to static clothing/foliage even in a sharp frame. Critically, this project is not working from isolated stills — it runs on continuous phone video — so a frame-difference channel between consecutive frames is already available at zero additional capture cost, unlike e.g. LiDAR depth (ruled out 2026-11-05) which needs hardware most target phones lack. Because the code is GPLv3, nothing here can be copied; the only transferable asset is the *idea* (dual-branch motion+appearance fusion, with the motion branch as a simple consecutive-frame difference map rather than full optical flow) reimplemented from scratch against the project's own YOLO11n.

**Honest caveats:** (1) Unverified in the golf/clubhead domain — this is a drone-to-drone paper, not a sports or small-club-object paper, so the transfer is an analogy, not evidence. (2) The project's current pipeline exports a single-frame YOLO11n to CoreML; adding a second input stream (previous-frame or diff-map) is a real architecture and export change, not a drop-in, and CoreML/on-device latency for a dual-input model is unverified — this is engineering risk, not just a license problem. (3) No code-level detail of the actual fusion module could be inspected beyond file names (reading further into GPLv3 code to copy implementation specifics would itself create derivative-work risk, so this review deliberately stopped at the architecture-level description visible from file names and the paper's own claims, not the source).

**Effort vs. payoff:** Medium-low effort to verify (one repo fetch, one direct LICENSE fetch, cross-referenced against two related papers with no usable code of their own — GreenVCOD, arXiv:2501.10914, is the closest same-task match but has no public code release at all, and the WACV-2023 Corsel et al. tiny-object-via-frame-differencing paper has no confirmed repo either, so neither could be verified as reusable this run). Payoff, if the reimplemented idea transfers to golf footage, is potentially the single most direct lever on the camouflage failure mode this log has found — a concrete, already-demonstrated architecture rather than a license dead-end or a survey conclusion — but it requires nontrivial engineering (dual-frame input, re-export to CoreML, on-device latency validation) before any payoff is realized, so this should be scoped as a prototype experiment, not assumed to transfer.

---

## 2026-11-08 (second run today) — BlurBall: a verified MIT-licensed ball detector that labels blur extent as a separate structured attribute from position, directly targeting this project's own labeling-spec gap for motion blur

**Rotation context:** The first run today (YOLOMG, logged above) covered bullet 3 (small/camouflaged object detection, temporal methods). This run covers bullet 2 — motion blur specifically: blur augmentation, blur-robust architectures, deblur preprocessing, and datasets of fast-moving blurred small objects in sport.

**What it is:** `cogsys-tuebingen/blurball` — the official implementation of "BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking" (Gossard et al., University of Tübingen, arXiv:2509.18387, Sept 2025). It extends the already-logged WASB tracker (2026-11-06 entry) with an explicit blur-estimation head: instead of predicting only a ball-center heatmap, it jointly predicts the ball's center **and** the blur streak's orientation and length as separate outputs, using an HRNet backbone with Squeeze-and-Excitation attention trained on blur-aware heatmaps. The authors also release a new table tennis dataset — per search-engine summaries of the arXiv abstract (the abstract itself could not be fetched directly this run, see Verification), 64,119 frames from 26 online recordings with camera calibration data, built specifically to include genuinely blurred ball frames rather than near-static ones.

**URL:** https://github.com/cogsys-tuebingen/blurball (paper: arXiv:2509.18387)

**Licence, verified directly against the repo's own `LICENSE.md` (not a search summary):** fetched `raw.githubusercontent.com/cogsys-tuebingen/blurball/main/LICENSE.md` directly. Full text is the standard MIT License, copyright "(c) 2023 NTT Communications Corporation" — the copyright holder is inherited from the WASB base the repo forks/extends, not the Tübingen authors, which is consistent with WASB-SBDT's own MIT attribution already confirmed in the 2026-11-06 entry. GitHub's own sidebar independently labels the repo "MIT license," matching the file. **Commercial use: permitted**, no copyleft, no attribution-in-product requirement beyond the standard MIT notice-preservation clause. Repo is public, non-archived, 57 stars, 12 commits — small but genuine, with real code (`src/`), a README describing training/inference commands, and pretrained weights for BlurBall itself plus five baseline trackers (WASB, TrackNetv2, ResTrackNetv2, BallSeg, DeepBall, Monotrack), all hosted on a Tübingen NextCloud instance.

**Verification performed:** Fetched the GitHub repo page and README directly (primary source) and confirmed the file listing, license badge, and dataset/weights links. Fetched `LICENSE.md` directly and quoted it verbatim above — not a summary. Could **not** verify the dataset's own license or confirm the NextCloud download link resolves: `cloud.cs.uni-tuebingen.de` returned a 403 from this session's network proxy, and `arxiv.org` is DNS-blocked from this session (both consistent with the standing egress-gap list every entry since 2026-11-04 has hit). The 64,119-frames/26-recordings figure, the "label at blur-streak center, not leading edge" framing, and the HRNet/SE architecture description come from search-engine summaries of the arXiv listing and the paper's abstract page, cross-checked across three independent search results that agreed — not from reading the paper directly. This is reported honestly as secondary-sourced, not primary-verified, unlike the license and code-repo facts above.

**Which failure mode this addresses:** Motion blur, and specifically the root-cause gap this project's own labeling-spec already has: the spec (`docs/labeling-spec.md`) instructs annotators to box the clubhead's "full visible clubhead including its motion-blur streak" as a single bounding box, conflating position and blur-extent into one annotation. The measured result (median box elongation 1.60, p90 3.01 — most labelled clubheads near-square) suggests that rule is applied inconsistently in practice, likely because a single box gives annotators no structured way to represent "this is blurred, and by how much" separately from "where is it." BlurBall's approach — predict ball center and blur length/orientation as two distinct labeled quantities rather than folding both into one box's aspect ratio — is a concrete alternative labeling and model-output schema that directly targets this exact ambiguity. It does not address camouflage.

**Why it helps this model specifically:** Two separable, differently-sized opportunities. (1) Low effort, process-only: revise the labeling spec to record blur length/orientation as explicit fields alongside (or instead of relying solely on) box aspect ratio — this costs nothing in model architecture, could be piloted on a small re-label of existing frames, and would make it possible to actually audit whether annotators are following the "box the full streak" rule rather than inferring it indirectly from elongation statistics as the failure analysis currently must. This is the first entry in this log that names a concrete fix for *why* the blur-elongation numbers look the way they do, rather than only diagnosing that they do. (2) Higher effort, architecture-level: the HRNet multi-frame blur-aware head is a genuine, MIT-licensed, directly reusable reference implementation (unlike YOLOMG/GPLv3 or SwingTrace/AGPL-3.0) for adding an explicit blur-attribute output to a detector — but it is a different backbone family from this project's YOLO11n, and no CoreML export or on-device latency evidence exists for it anywhere checked so far.

**Effort vs. payoff:** The labeling-spec change (item 1) is cheap to pilot — no new data, no new model, just a revised annotation convention and a small re-label-and-measure experiment to see if explicit blur fields move the needle on consistency — and plausible payoff is real since it targets a gap this project has already measured but not yet explained. The architecture reuse (item 2) is a bigger, unproven lift: different backbone, multi-frame input, table-tennis-ball domain rather than golf-clubhead, and the dataset that would make it directly comparable is unverified and possibly unreachable from this session's network. Recommend piloting (1) before considering (2) at all.

---

## 2026-11-08 (third run today) — RFR / IRSatVideo-LEO: a third distinct architecture for temporal/multi-frame small-target detection (recurrent feature refinement, not frame-differencing or Mamba), checked and ruled out the same way as the other two

**Rotation context:** The first two runs today covered bullet 3 (YOLOMG,
motion-vs-appearance fusion) and bullet 2 (BlurBall, blur-aware labeling).
This run also lands on bullet 3 — multi-frame/temporal methods for
low-contrast small objects — but deliberately pulls from a different
literature stream than any prior entry: infrared/satellite dim-small-target
detection, not video camouflaged-object-detection (VCOD) or UAV-to-UAV
motion fusion. The goal was to check whether a third architectural family
(recurrent temporal refinement across many frames, rather than a 2-frame
difference channel like YOLOMG or frequency-domain Mamba scanning like
Vcamba) changes the license/deployability conclusion this log keeps
reaching, not to re-survey VCOD again.

**What it is:** `XinyiYing/RFR`, the official repo for "Infrared Small
Target Detection in Satellite Videos: A New Dataset and A Novel Recurrent
Feature Refinement Framework" (Ying et al.). It ships two model variants,
`ISTUDNet_RFR` and `ResUNet_RFR`, plus a deformable-convolution (DCN) module,
and introduces `IRSatVideo-LEO`: a semi-simulated dataset (a real satellite
background composited with synthesized target appearance, trajectory, and
intensity) of 200 sequences / 91,021 frames / 218,038 targets with
instance-level segmentation masks. The mechanism — predicting a target mask
per frame by recurrently refining features across a sequence, rather than
differencing or stacking a fixed window of frames — is architecturally
distinct from every temporal method logged so far in this project.

**URL:** https://github.com/XinyiYing/RFR (paper discovered via search;
arXiv remains DNS-blocked from this session per the standing egress-gap
list, so the paper body itself was not read — repo README is the primary
source).

**Licence:** **None.** Fetched the repo directly: the file listing is only
`codes/`, `pics/`, `README.md` — no `LICENSE` file, and GitHub's own "About"
sidebar shows no detected license. Same default-all-rights-reserved
situation already logged three times now (Vcamba and EventVCOD, 2026-11-04
and 2026-11-06; CamDiff, 2026-11-07). **Commercial use: not permitted —
no rights are granted without contacting the authors.** The `IRSatVideo-LEO`
dataset itself is not in the repo at all; access requires submitting a
Microsoft Forms request linked from the README, with no stated turnaround
or guarantee of grant — a stricter gate than WASB-SBDT/BlurBall's open
downloads, closer to GolfPose's author-gated access (2026-11-07).

**Verification performed:** Fetched the GitHub repo page directly and
confirmed it is real, public, and contains actual training/testing code
(`train.py`, `test.py` under `codes/`) rather than a placeholder, plus a DCN
module requiring separate compilation (Python 3.8, PyTorch 1.10.1 — an
older, specific environment, itself a minor deployability flag). Confirmed
the dataset composition (200 sequences / 91,021 frames / 218,038 targets)
and the Microsoft Forms gating directly from the README text. Could not
verify the paper's claimed PA/mIoU/PD-FA benchmark numbers, and the README
does not itself explain the recurrent-refinement mechanism beyond naming it
and pointing at a figure — the exact method (e.g. whether it's LSTM/GRU-style
recurrence or something else) was not confirmed, only that it is not simple
frame-differencing or a Mamba selective-scan kernel (ruling that in/out was
possible from the file listing: no `kernels/selective_scan/` directory,
unlike Vcamba).

**Which failure mode this addresses:** Camouflage — the same "appearance
alone is ambiguous, motion/temporal structure across frames is the
separating signal" premise as every other VCOD-adjacent entry in this log.
Has no bearing on motion blur.

**Why it matters for this model specifically:** It does not change this
log's running conclusion about camouflage from the 2026-11-04 VCOD entry —
if anything it reinforces it a third time: another real, code-shipping,
multi-frame architecture for exactly this class of problem (small,
low-contrast, hard-to-see-in-a-single-frame moving target) exists, and is
again blocked from direct reuse by licensing and a gated dataset rather than
by the idea being unsound. The one new fact this run adds is architectural
breadth, not a new blocker: this project now has three independently
verified, structurally different ways multi-frame temporal information gets
used for exactly this problem elsewhere (frame-differencing fusion —
YOLOMG; frequency-domain Mamba scanning — Vcamba; recurrent feature
refinement — RFR), which narrows what a from-scratch reimplementation for
the clubhead detector would need to choose between, rather than inventing
the design space from nothing. Domain transfer from satellite infrared to
golf-clubhead-on-phone-video is unverified and should be treated as weak
evidence either way — point targets in satellite video have essentially no
shape, unlike a clubhead, so the *segmentation*-style output here is less
directly relevant than YOLOMG's detection-style output.

**Effort vs. payoff:** Low effort to check (one repo fetch, ~10 minutes),
low payoff beyond confirming a pattern this log has now established three
times over: real multi-frame architectures for this exact class of problem
keep existing in the literature, and keep being unusable as-is because of
license or data-access gates, not because the underlying idea is weak.
Given that pattern is now well-established, a future run would get more
value spending bullet-3 time on a from-scratch architecture sketch (reusing
only the *ideas*, already distilled across YOLOMG/Vcamba/RFR) or on
collecting the project's own motion-paired camouflage frames to test any of
them against, rather than continuing to search for a fourth license-clean
implementation of the same mechanism.

---

**Branch-continuity note (2026-11-09):** Two earlier runs today both branched
from `mvp/club-path-replay` (commit `406f8cd`) directly instead of from this
chain (`research/clubhead-2026-11-08-corrected-3`), so each wrote a fresh,
empty `LOG.md` and re-logged topics already closed out above —
`research/clubhead-2026-11-09` re-discovered SoccerSynth-Detection (already
the 2026-11-04 entry) and ClubheadDB (already the 2026-11-04 fourth-run
entry); `research/clubhead-2026-11-09-corrected` added one entry, "Lightweight
Multi-Frame Integration for Robust YOLO Object Detection in Videos" (ECMR
2025, arXiv:2506.20550), that is *not* a duplicate of anything above, but
it's bullet 3 (small/camouflaged/temporal detection), the same bucket as both
of 2026-11-08's logged entries (YOLOMG, RFR) — so, per the rotation rule, it
is noted here rather than copied in, and should be the pick-up point for the
next run due on bullet 3. This run continues properly from
`research/clubhead-2026-11-08-corrected-3` and is pushed as
`research/clubhead-2026-11-09-corrected-2`. Future runs: continue from
whichever of today's three branches is newest by this note, not from
`mvp/club-path-replay` directly — `LOG.md` does not exist on that branch and
never will unless one of these research branches is merged.

## 2026-11-09 — "One-Shot Badminton Shuttle Detection for Mobile Robots" (ETH Zurich RSL) checked and ruled out: real paper and repo, but AGPL-3.0'd and its dataset link is a dead placeholder

**Rotation context:** The three prior runs today (two orphaned, documented
above) covered bullet 3 twice and bullet 3 again. This run targets bullet 2 —
motion blur specifically, via datasets of fast-moving blurred small objects
in sport — after bullet 1 (commercial indoor/low-light golf datasets) and
bullet 4 (golf-specific tracking) were both searched this run and turned up
nothing beyond what's already logged (ClubheadDB, GolfDB, CaddieSet, GolfPose
— all already closed out above; no new golf-specific dataset or tracking
paper was found this run).

**What it is:** A one-shot shuttlecock detector for badminton-playing mobile
robots, built so a single-frame detection can re-initialize tracking after
the robot's camera loses the (small, very fast) shuttle. Authors: Florentin
Dipner, William Talbot, Turcan Tuna, Andrei Cramariuc, Marco Hutter — ETH
Zürich Robotic Systems Lab (RSL), arXiv:2603.06691, submitted 17 March 2026.
Secondary summaries (alphaxiv, Bytez, awesomepapers.io — `arxiv.org` itself
is DNS-blocked from this session, same standing gap this log has hit on
every prior run) describe a dataset of 20,510 semi-automatically annotated
frames across 11 indoor backgrounds, a YOLOv8 detector fine-tuned with
non-maximum suppression constrained to at most one detection per frame, and
a note that small object size and high shuttle velocity are the main
remaining difficulty. I could not independently verify these specific
numbers against the paper body (blocked), so they are reported as secondary,
not confirmed.

**URL:** https://arxiv.org/abs/2603.06691 (paper, unreachable directly this
session); code: https://github.com/leggedrobotics/shuttle_detection (fetched
directly and confirmed real — `leggedrobotics` is ETH RSL's own GitHub org,
matching the paper's affiliation).

**Licence — verbatim:** The repo's `LICENSE` file, fetched directly from
`raw.githubusercontent.com`, is the GNU Affero General Public License,
opening "GNU AFFERO GENERAL PUBLIC LICENSE / Version 3, 19 November 2007".
The README states the reason directly: *"Due to the use of Ultralytics
YOLOv8, this project is licensed under GNU AFFERO GENERAL PUBLIC LICENSE
v3.0."* **Commercial use: permitted under AGPL's own terms, but with the
same network-copyleft obligation this log has already flagged for
`nishsm/golf-club-tracking` (2026-11-05) — any product that embeds this code
or a model trained with it must offer its complete corresponding source to
anyone who interacts with it over a network. That is incompatible with
shipping inside this project's closed-source iOS app as-is**, same
conclusion as the prior AGPL entry.

**Verification performed:** Fetched the GitHub repo page and README
directly (confirmed real: `scripts`, `src/shuttletrack`, `Dockerfile`,
`docker-compose.yaml`, documented `shuttletrack.autogen_labels` /
`.train` / `.eval` / `.predict` commands) and the raw `LICENSE` file
directly. Checked whether the dataset is actually downloadable today, per
this log's standing rule: **it is not** — the README's own download
instruction points to `https://example.com`, a placeholder, not a real
link. The README also claims an "easy/medium/hard" difficulty split and an
automatic/semi-automatic labeling pipeline (`autogen_labels`, output
importable into CVAT for correction), but documents neither the labeling
algorithm (no mention of motion or background subtraction, tracking, or
seed points) nor the easy/medium/hard assignment criteria — so, unlike the
BlurBall entry (2026-11-08), there is nothing concrete here to borrow as a
labeling-QA technique, only a name for one.

**Which failure mode it addresses:** Motion blur, primarily (small, very
fast object in video is exactly our clubhead's problem shape), with
secondary relevance to the indoor-footage gap this project has never
measured — the dataset is explicitly indoor, which none of this log's
prior sport-dataset leads (ClubheadDB, GolfDB, CaddieSet) were. Not
camouflage-relevant: nothing in the verified material addresses
low-contrast/dark-on-dark appearance.

**Why it helps this model specifically:** It doesn't, in practice — this
closes out a lead rather than opening one. The one potentially useful
asset (an indoor, fast-small-object dataset with difficulty tiers that
might correlate with blur severity) is not actually obtainable (dead link),
and the one potentially useful technique (the auto-labeling pipeline, which
could in principle have been adapted to bootstrap more blur-labeled
clubhead frames from our own raw footage, directly targeting the
elongation-1.60 gap the brief describes) ships undocumented and under a
licence this project can't use as-is. The repo is also real evidence that
"indoor + fast + small + blurred" is a problem domain other groups are
actively working on in 2026, which narrows where to look next (robotics labs
building their own one-shot/re-detection pipelines for small flying/thrown
objects, not sports-analytics papers), but that's a pointer, not a result.

**Effort vs. payoff:** Low payoff, low-to-medium effort already spent (one
repo fetch, one license check, one README read — this is now closed, not
worth more time). Logged specifically so no future run re-opens this exact
paper/repo chasing the "11 indoor backgrounds" detail, which turned out to
be unreachable. If this project ever wants an *auto-labeling bootstrap* for
blur-streak boxes specifically, the more promising next step flagged by this
entry is to look at how robotics re-detection papers in this same niche
(one-shot recovery of a lost fast small target) build their seed labels,
rather than this specific repo.

---

## 2026-11-10 — "Lightweight Multi-Frame Integration for Robust YOLO Object Detection in Videos" (Tübingen, ECMR 2025): verified real, but picked it up too early — no code or dataset exists to use yet

**Rotation context:** The 2026-11-09 chain-continuity note flagged this exact
paper as the pickup point for the next bullet-3 run (it was found by an
orphaned same-day branch but never logged, to respect the one-topic-per-run
rule). This run follows that pointer rather than opening a new search area.
Bullet 1 (commercial indoor/low-light golf datasets) and bullet 4
(golf-specific tracking) have both come up empty on recent runs per the
2026-11-09 entry; bullet 2 was just covered last run (shuttle_detection), so
this stays off bullet 2 per the no-consecutive-repeat rule.

**What it is:** arXiv:2506.20550, "Lightweight Multi-Frame Integration for
Robust YOLO Object Detection in Videos," by Yitong Quan, Benjamin Kiefer,
Martin Messmer, and Andreas Zell (University of Tübingen), submitted to
ECMR 2025. The method: stack several consecutive video frames channel-wise
as a single input tensor to an otherwise-unmodified YOLO detector, but
supervise only the output for one target frame (the center of the stack).
No new layers, no added parameters beyond the first conv layer's widened
input, no optical flow, no recurrence — just more channels in, same
architecture otherwise. The paper also introduces BOAT360, a fisheye
boat-video benchmark, and reports (per third-party summaries — see
Verification) gains of roughly +12.8 mAP@0.5 for YOLOv7-tiny and +8.1 for
YOLOv7 over single-frame baselines on BOAT360, under 0.1% added parameters,
and ~53–55 FPS for YOLOv7-tiny on an Orin AGX; a separate summary reports 7
stacked adjacent frames as the best setting on MOT20Det.

**URL:** https://arxiv.org/abs/2506.20550 (arxiv.org itself is DNS-blocked
from this session, consistent with every prior run's network notes — the
paper was confirmed real via independent third-party mirrors/indices
instead: the arXiv PDF/HTML listing, `web3.arxiv.org` (also blocked
directly, but indexed by the search backend), `papers.cool`, `emergentmind`,
`themoonlight.io`'s literature-review summary, and a University of Tübingen
institutional-repository record at `bibliographie.uni-tuebingen.de` citing
the same title/authors/arXiv ID — five independent sources agreeing on
title, authors, and ID is strong evidence the paper is real, even though
none of the pages themselves could be fetched directly this run (all
DNS-blocked). This is corroboration via convergent secondary sources, not a
primary-source read, and is reported as such.

**Licence:** **No code or dataset exists to license.** This is the
substantive finding of this run, and it's why the 2026-11-09 orphaned branch
left this unverified rather than fully logged. Checked specifically: (1) the
arXiv listing page has a "Code, Data and Media" section per search-engine
indexing, but no GitHub or other repository link resolved anywhere in it;
(2) searched GitHub directly for `BOAT360` (`search/code`, `search/
repositories`) — the only hit, `CanvasD/BOAT360`, is a confirmed-empty,
unrelated repo (no README, no files); (3) checked every plausible author
GitHub handle (`benjaminkiefer` — exists but is a different, unrelated
person with two unrelated kata repos; `bkiefer`, `ybquan`, `yitongquan` —
none match or have zero/irrelevant repos); (4) a third-party summary
surfaced by this run's search states plainly that the authors intend to
share data "to support further research... upon paper acceptance" — i.e.
release is explicitly future-conditional, not available now. **Commercial
use: not applicable — there is nothing downloadable to use, commercially or
otherwise, as of this check.**

**Which failure mode this addresses:** Camouflage, primarily — same
temporal/motion-cue premise as every other bullet-3 entry (Vcamba, YOLOMG,
RFR), but architecturally the simplest version of it by a wide margin: no
motion-fusion branch, no frame-differencing module, no recurrent head, just
more channels into an unmodified detector trunk. Not motion-blur-targeted in
itself, though a model that can see 2-3 adjacent frames at once has *some*
chance of disambiguating a blurred streak's true extent from context the
single blurred frame alone doesn't give it — that's speculation, not a
claim the paper makes or this run tested.

**Why it matters for this model specifically — and why it's a genuinely
different entry from YOLOMG/RFR, not a repeat:** Every other bullet-3 entry
in this log so far has been blocked from direct reuse by a licence
(GPL-3.0, AGPL-3.0) or a gated/dead dataset — the architecture idea was
sound but copying or adapting the implementation carried real legal or
practical risk. This technique has no such blocker *because there is
nothing to copy*: "stack N consecutive frames as extra input channels,
widen the first conv layer, supervise the center frame" is fully
describable from the abstract alone and is a few hours of work directly
against this project's own YOLO11n/Ultralytics training pipeline — change
`ch` on the model's first layer, change the dataloader to yield a stacked
tensor instead of one frame, retrain. Zero derivative-work risk, because
nothing of theirs is being used at all, only the idea. That makes it the
cheapest-to-prototype bullet-3 lead this log has found, strictly on
licensing grounds — independent of whether BOAT360's reported gains
(fisheye boat video, not golf) transfer.

**Honest caveats:** (1) All performance numbers above are third-party
summarized, not read from the paper body — the exact baseline mAP values,
the precise frame-stacking order (centered vs. causal/past-only), and how
the first-layer weights are initialized when widening from 3 to N×3
channels are unconfirmed. (2) BOAT360 is fisheye boat video, not
golf-clubhead video; transfer is an analogy. (3) This project's deployment
target is CoreML on-device — a wider first-layer input is export-compatible
in principle (CoreML supports arbitrary input channel counts), but no
on-device latency number for *any* multi-frame YOLO variant has been
verified anywhere in this log (YOLOMG, RFR, and this entry all lack that
number); a 3-frame-stacked YOLO11n's actual iPhone inference cost is
unmeasured. (4) It needs three temporally-adjacent raw frames at inference
time rather than one, which is a real (if small) capture/buffering change
to the app's existing single-frame-in pipeline, not purely a training-time
change.

**Effort vs. payoff:** Low-medium effort to prototype (no new code to
license-check or port, just a training-pipeline change against the
project's existing YOLO11n setup), and — uniquely among this log's
bullet-3 entries — zero licensing risk, which makes it a stronger
practical candidate for an actual experiment than YOLOMG or RFR despite
weaker domain evidence (fisheye boats, not golf) and no primary-source
numbers. Recommend this as the first bullet-3 idea to prototype
in-house, specifically because it's the one bullet-3 lead this log has
found that needs no one else's code or data to start.

---

## 2026-11-10 (second run today) — No ready-made commercial-licensed tool applies motion blur to a single pasted object before compositing; `cap-augmentation` confirms the pattern works but is GPL-3.0'd and ships no blur

**Rotation context:** The first run today covered bullet 3 (Lightweight
Multi-Frame Integration). This run targets bullet 5 (synthesize/augment
training data), specifically to close a gap the 2026-11-07 RT-Focuser entry
opened but didn't follow up on: that entry flagged that GoPro-style blur
(camera-shake, whole-frame or large-object) has a different geometry from
this project's actual problem (a small, fast object streaking against a
comparatively static background), and that the design spec's planned
Albumentations `MotionBlur` is a whole-frame kernel with no per-object
targeting (first raised in the 2026-11-04 SoccerSynth entry). Neither entry
checked whether a tool exists to apply blur to *one object's crop* before
it's placed into a frame — the actually-correct way to synthesize this
project's specific failure geometry. This run checked that directly.

**What it is:** `RocketFlash/cap-augmentation` — a real, published
cut-and-paste augmentation library for object detection/segmentation. It
pastes an object cutout (RGB + alpha mask) onto a destination image and
updates the resulting box/mask automatically, and exposes a documented
per-object hook (`object_transforms`, via `ImageMaskTransform`; an older
`albu_transforms` parameter is deprecated but still works) that runs on
each object's crop *before* it is composited — exactly the architectural
slot a per-object directional-blur step would need, as opposed to
Albumentations' `MotionBlur` which (confirmed in this log's other
2026-11-10 investigation this run, by reading
`albumentations/augmentations/blur/transforms.py` directly) convolves the
entire frame.

**URL:** https://github.com/RocketFlash/cap-augmentation (PyPI: `cap-augmentation`)

**Licence — verbatim:** Fetched `raw.githubusercontent.com/RocketFlash/cap-augmentation/main/LICENSE`
directly: "GNU GENERAL PUBLIC LICENSE / Version 3, 29 June 2007." **Commercial
use: permitted under GPL-3.0's own terms, but with copyleft obligations if
the library's code itself is distributed or embedded** — the same class of
constraint this log has already flagged for AGPL entries, though weaker:
GPL (unlike AGPL) has no network-use trigger, and since this would be used
purely as an offline training-data-generation tool (never shipped inside
the iOS app or distributed to any user), the copyleft obligation most
plausibly would not attach to the shipped product or the trained model
weights — but that is this run's own reasoning, not a legal clearance, and
the project would need to keep this tool's use strictly offline/internal
(e.g. a separate dev-only script, not merged into anything redistributed)
to stay on the safe side of that line.

**Verification performed:** Fetched the GitHub repo page directly (real:
148 stars, 10 forks, PyPI package, CI workflows, test suite, CHANGELOG,
Zenodo DOI — a genuine maintained project, not a stub) and confirmed the
`object_transforms` hook is documented in the README, not just implied.
Checked specifically whether blur or rotation ship as *built-in* named
transforms: **they do not.** The README's only named object-level options
are alpha-blend "ghost" pasting (`blending_coeff`) and histogram matching;
blur/rotation would have to be supplied by the caller (e.g. as a custom
function, or an Albumentations transform passed through the deprecated
hook). Also checked for a built-in or widely-used alternative that *does*
ship object-level blur out of the box — found none in this run's searches
(SPIE's "Copy and restricted paste" paper and the oft-cited Kisantal et al.
2019 "Augmentation for small object detection" both describe copy-paste
small-object augmentation generically, and one summarized source noted
that naively Gaussian-blurring a pasted object's *edge* alone was tested by
a different paper and did not help — but neither ships code, and neither
targets directional motion blur specifically).

**Which failure mode this addresses:** Motion blur only. No bearing on
camouflage.

**Why it matters for this model specifically:** This closes out, rather
than opens, the "is there a reusable tool" question left dangling since
2026-11-04/2026-11-07. There is no commercially-clean, ready-to-use library
that does "crop a real clubhead, blur it directionally to a given
velocity/angle, paste it onto a sharp background frame, emit the
union-of-motion box" — the exact synthesis this project's measured gap
(median labelled elongation 1.60, almost no genuinely streaked boxes) calls
for. `cap-augmentation` is useful only as proof that the general
mechanism (crop → per-object transform → alpha-composite → auto-updated
box) is a working, published pattern worth copying the *idea* of; its own
code is not worth adopting given the GPL-3.0 constraint for what is at most
a few dozen lines of work with plain OpenCV/PIL (crop the labelled
clubhead region using an existing box, apply a directional line-kernel blur
— the same kernel math already verified in Albumentations' `MotionBlur`
source this run, just applied to the crop instead of the full frame —
paste back with alpha, keep the box as the blurred crop's new bounding
extent). This is a build-it-yourself conclusion, not a dead end: it means
the "fix the blur-geometry mismatch" idea flagged twice already in this log
has no shortcut via someone else's library, but also no real blocker,
since the underlying operation is small and license-free to implement from
scratch.

**Effort vs. payoff:** Low-medium effort (a single-purpose script: load a
labelled clubhead crop + its box, apply a directional blur kernel sized to
a target elongation, alpha-composite onto a sharp negative/background
frame, write out the new box) — strictly less effort than integrating any
third-party repo already in this log, since it avoids every licensing
question raised for YOLOMG, the AGPL tracking repos, RT-Focuser's
GoPro-domain mismatch, and now this one. Payoff is plausible but unproven:
it directly manufactures the missing example class (streaked clubhead,
sharp surroundings) the brief identifies as scarce, using the project's
own already-labelled sharp clubhead crops as source material, but nothing
in this run measured whether a detector actually trained better on such
synthetic composites (unlike SoccerSynth, which did measure a transfer-
accuracy gain, albeit for soccer players, not clubheads). Recommend as a
cheap follow-on experiment, not as a confirmed win.

---

## 2026-11-10 (third run today) — RF-DETR: a real, Apache-2.0, actively-maintained alternative to YOLO11n with a higher reported COCO AP at the nano size and an official native CoreML export path — addresses neither failure mode directly, but closes the open AGPL licensing risk this log flagged on 2026-11-05 without needing an Ultralytics Enterprise license

**Rotation context:** Today's first two runs covered bullet 3 (temporal/
multi-frame architectures) and bullet 5 (copy-paste augmentation tooling).
This run went looking at bullet 4 (golf-specific implementations) and
bullet 1 (datasets) first — both came up empty against this log's own
existing entries (see "Dead ends checked, not logged" below) — before this
genuinely new finding surfaced from a licensing angle, not a bullet-list
angle. Logged anyway because it is new, verified, and directly actionable,
even though it doesn't slot cleanly into camouflage or motion blur.

**What it is:** `roboflow/rf-detr` — RF-DETR, a real-time DETR-family
(transformer) object detector from Roboflow (ICLR 2026), built on a DINOv2
ViT backbone with sizes produced via neural architecture search. The
repo's own README benchmark table reports **RF-DETR-N: 48.4 COCO AP50:95 at
2.3ms**, vs. **YOLO11-N: 37.4 COCO AP50:95 at 2.5ms** (RF-DETR measured at
384×384, YOLO11n at 640×640 — not a strictly apples-to-apples resolution
comparison, see caveats below). PyPI shows the `rfdetr` package at v1.12.0,
released 2026-11-06 — four days before this run — so this is an actively
developed, current project, not an abandoned one.

**URL:** https://github.com/roboflow/rf-detr (PyPI: `rfdetr`)

**Licence — verified directly, with an important nuance caught by checking
rather than trusting the headline claim:** Fetched the GitHub repo page and
its root `LICENSE` file directly: standard **Apache License 2.0**,
"Copyright 2025 Roboflow, Inc.," no added restriction — confirmed
verbatim-equivalent (the fetch tool could not reproduce the Apache 2.0 body
text character-for-character but confirmed the header, copyright line, and
every substantive clause: redistribution/attribution conditions, patent
grant and patent-retaliation termination, no warranty). **Commercial use:
permitted** for the Apache-2.0-licensed sizes. The nuance: Roboflow's own
blog post is titled "RF-DETR Is Free to Use Commercially (Apache 2.0)," but
per the model-license page and the `rfdetr-plus` PyPI listing (both reached
only via search snippets — `roboflow.com` and its `rfdetr.roboflow.com` docs
subdomain are both DNS-blocked from this session, consistent with every
other vendor-site block this log has hit), **not every size is Apache-2.0**:
the smaller-than-nano Atto/Femto/Pico sizes and the larger XL/2XL sizes
require the optional `rfdetr[plus]` extra and fall under a separate
"Platform Model License 1.0," which multiple of Roboflow's own pages
describe inconsistently (some say PML covers only XL/2XL, others say it
also covers Atto/Femto/Pico) and which explicitly restricts self-hosted
commercial deployment without an add-on. Standard detection sizes **Nano
through Large, and all segmentation sizes, are Apache-2.0** per every source
checked — and Nano is the size actually benchmarked against YOLO11-N above,
so the specific comparison this entry is built on is not affected by the
PML carve-out. This Apache-2.0-vs-PML split was not obvious from the GitHub
README alone and is exactly the kind of licensing nuance this log has
learned (the hard way, via the NTT-copyright-mismatch and
claimed-but-missing-LICENSE entries) not to take at face value.

**Verification performed:** Fetched the GitHub repo page directly (real,
public, ~10k stars, ~1.3k forks, 1,337+ commits on `develop`, not archived)
and its root `LICENSE` file directly (Apache 2.0, quoted above). Fetched the
README directly and confirmed the benchmark table numbers and methodology
note verbatim ("All latency numbers were measured on an NVIDIA T4 using
TensorRT, FP16, and batch size 1" — i.e. the latency figures are GPU/
TensorRT numbers, not representative of on-device CPU/Neural-Engine
performance, and say nothing about CoreML export cost). Confirmed via direct
PyPI fetch that the `rfdetr` package (v1.12.0, 2026-11-06) genuinely lists a
`coreml` extra alongside `onnx`, `tensorrt`, `tflite`, `executorch`,
`coreai` — the extra exists in the shipped package, not just in marketing
copy — though the PyPI page itself gives no detail on what it contains.
Could not independently fetch Roboflow's own CoreML-export cookbook page
(`rfdetr.roboflow.com`, DNS-blocked) to confirm the exact export call,
output format (`.mlpackage` vs. raw `.mlmodel`), or whether it produces
Vision-compatible `VNRecognizedObjectObservation` output the way this
project's existing CoreML export (`model.export(format='coreml', nms=True)`
per the v2 design doc) does — search-engine snippets describe an official
`rfdetr[coreml]>=1.9.0` install path with an `.mlpackage` output and a
`coreml_precision="float16"` option for Neural-Engine-oriented builds, and a
separate ExecuTorch/CoreML-backend route, but neither was read directly from
Roboflow's own page, so **treat the exact CoreML export mechanics and
Vision-compatibility as unconfirmed**, not just the on-device speed. A
third-party Hugging Face conversion and an unverified Reddit claim of
"120 FPS on iPhone" for RF-DETR-Nano both surfaced in search but are exactly
the kind of unverified secondary claim this log does not log as fact — noted
here only as a lead, not as evidence.

**Which failure mode this addresses:** **Neither, directly.** This is not a
camouflage or motion-blur mechanism the way the VCOD/EventVCOD or
BlurBall/WASB-SBDT entries are — RF-DETR's reported advantage is general
COCO detection accuracy and a licence-clean path, not anything specific to
low-contrast targets or motion-blurred small objects. It is logged anyway
because of what it's relevant *to*, not because it targets either symptom.

**Why it matters for this model specifically:** Two separate, concrete
reasons, both already open questions in this log rather than new worries.
(1) **Licensing.** The 2026-11-05 SwingTrace entry flagged, and left
unresolved, that this project's own approved v2 design doc commits to
training with the `ultralytics` package, which is itself AGPL-3.0 — the
same clause that already blocked reusing SwingTrace's weights — and that
whether shipping only exported CoreML weights (not source) in a closed-
source app triggers AGPL's copyleft obligations is a genuinely contested,
unresolved legal question for this project. RF-DETR-Nano is a concrete,
real, actively-maintained, cleanly Apache-2.0 alternative at a comparable
size class, with its own first-party CoreML export path (mechanics
unconfirmed, existence confirmed) — i.e., it is a legitimate candidate to
benchmark *instead of* resolving the YOLO/AGPL question, not just another
entry in the pile. (2) **General detection quality**, weakly: a model
reporting meaningfully higher COCO AP at a comparable or better latency
class is, in the same spirit as the already-logged SoccerSynth entry's
"vote of confidence" framing, a plausible (not demonstrated) lever on both
failure modes at once — tighter, more confident general-purpose detection
plausibly helps localize low-contrast objects better and produce tighter
boxes around partially-blurred ones — but this is architecture-general
reasoning, not a camouflage- or blur-specific claim, and COCO performance on
large common objects is a weak proxy for a small, elongated, often-blurred
clubhead against clutter. Treat reason (2) as speculative upside, reason
(1) as the actual, load-bearing reason this entry is worth having.

**Dead ends checked, not logged (bullets 1 and 4, this run):** A
Roboflow-hosted golf model (`swingfeed/golf-49wbh-spfor`, RF-DETR-nano,
person/club/clubhead, 7,842 images, dated Feb 2026) surfaced and is what led
to checking RF-DETR itself — but `universe.roboflow.com` remains DNS-blocked
from this session (consistent with every prior entry in this log back to
2026-11-04), so its license and continued existence could not be verified
and it is **not** logged as a finding, same as this log's standing practice
for every other unreachable Roboflow Universe listing. `phillt3/ClubUp`
(fetched directly on GitHub) turned out to be an unrelated club-recommendation
iOS app (slope/wind/altitude distance calculator), not a vision project —
checked and ruled out, not a new lead. No new golf-specific dataset or
tracking implementation beyond what this log already carries (ClubheadDB,
CaddieSet, GolfDB, SwingTrace, rucv/golf_ball, GolfPose) was found this run.

**Effort vs. payoff:** Low effort to benchmark — the project's own eval
harness (`ml/clubhead/src/chdet/evaluate.py` per the README) is already
designed to score "any model, old CreateML or new YOLO," and RF-DETR-Nano
is pip-installable and trainable on the same existing dataset/labels without
new annotation work, so a side-by-side comparison against the already-
planned YOLO11n is a contained experiment, not a pipeline rewrite. Payoff
is real but two-sided and currently unquantified for this project: the
licensing win is unambiguous and immediate (removes a legal open question
with zero new risk introduced), but the CoreML export path's actual
mechanics, on-device latency, and Vision-framework compatibility are all
unconfirmed, and a transformer-family detector historically carries more
mobile-export risk (dynamic shapes, attention ops, NMS handling) than a
CNN-based YOLO head even when a first-party export path exists on paper.
Worth a cheap first step — attempt the documented CoreML export on a
stock/pretrained RF-DETR-Nano checkpoint and confirm it loads in Xcode and
produces `VNRecognizedObjectObservation`-compatible output — before
investing in a full retrain-and-compare cycle.

## 2026-11-10 (fourth run today) — FMODetect: a real, MIT-licensed architecture that detects fast-moving objects *from their blur streak itself*, rather than treating blur as noise to deblur or tolerate — but its background-subtraction design assumes a largely static camera, which a handheld phone swing video is not

**Rotation context:** Today's first three runs covered bullet 3 (Lightweight
Multi-Frame Integration), bullet 5 (cap-augmentation), and a licensing find
(RF-DETR) that fell out of attempted bullets 1/4. This run targets bullet 2
— motion blur specifically, the "blur-robust detection architectures" angle,
which no run today has yet covered and which this log's existing bullet-2
entries (SoccerSynth pretraining, RT-Focuser preprocessing, BlurBall
labeling, shuttle_detection dead-link) have not addressed from the
architecture side.

**What it is:** `rozumden/FMODetect` — the official code release for
"FMODetect: Robust Detection of Fast Moving Objects" (Rozumnyi, Matas,
Sroubek, Pollefeys, Oswald; arXiv:2012.08216, published at ICCV 2021). The
method treats a fast-moving object (FMO) — one that crosses many pixels
within a single exposure, appearing as a blur streak rather than a sharp
shape — as the primary detection target, not an edge case: it estimates a
running-median background from the last few frames, concatenates the
current frame with that background as a 6-channel input, and a CNN
(TensorFlow/Keras, `net_model.py`, `FMODetect.h5`) predicts a heatmap of the
blur region directly. This is the architectural inverse of deblur-then-
detect (RT-Focuser, 2026-11-07) or blur-augment-then-train (SoccerSynth,
2026-11-04): here the blur *is* the feature the network is trained to find.

**URL:** https://github.com/rozumden/FMODetect (code); the paper is on
`arxiv.org`, DNS-blocked from this session like every prior run — read via
the repo's own README/bibtex and search-engine summaries instead (noted
below where unverified).

**Licence — verbatim:** Fetched `LICENSE` directly from
`raw.githubusercontent.com` (the `github.com` HTML page itself returns 403
from this session, same standing gap as every prior entry — only the raw/
API-less content routes work): *"MIT License / Copyright (c) 2022 Denys
Rozumnyi"*, followed by the standard MIT permission/warranty text in full.
**Commercial use: permitted**, no copyleft, no field-of-use restriction.

**Verification performed:** Fetched `README.md`, `LICENSE`, `run.py`,
`dataset/generate_dataset.sh`, `dataset/generate_patterns.py`, and
`dataset/process_dataset_traj.py` directly from
`raw.githubusercontent.com` — all return real file content, confirming the
repo is genuinely populated, not a stub. Read `run.py` in full: it confirms
the background-subtraction design described above (`B = np.median(Ims, 0)`
over the last `--median` frames, concatenated with the current frame as
model input) rather than inferring it from the README's prose alone. The
README's "clone --recursive" instruction corresponds to a real submodule
(`dataset/trackdat` — 404 as a plain file, consistent with being a git
submodule rather than missing) that downloads the VOT2016 tracking
benchmark as background plates for synthetic training-data generation.
**Not verified:** the pretrained-model download link
(`polybox.ethz.ch/index.php/s/X3J41G9DFuwQOeY`) — `polybox.ethz.ch` is
blocked by this session's egress proxy (403 on CONNECT, same failure shape
as every academic host this log has hit), so whether that link still
resolves is unknown. No `requirements.txt` exists in the repo (checked:
404), so exact TensorFlow/Keras/OpenCV versions needed to run it today are
undocumented — getting it running would mean reverse-engineering the
dependency set from 2021-era TF/Keras API usage in `net_model.py`. The repo
has had no commits since ~2022 per third-party aggregator data (not
independently re-verified via the blocked `github.com` HTML page, so stated
as secondary).

**Which failure mode this addresses:** **Motion blur, directly and
specifically** — this is a detection-architecture answer to the exact gap
the brief describes (training data short on genuinely blurred examples,
median elongation 1.60), because it is a published, working precedent that
a network *can* be trained to find a target from its blur shape alone,
using synthetically generated blur streaks (composited foreground patterns
dragged across real backgrounds) rather than requiring thousands of
hand-labeled real blurred clubheads. Not camouflage-relevant: nothing here
helps find a static, unmoving dark object against clutter — if anything, a
camouflaged *and* blurred clubhead would still need the appearance cues
this method does not model.

**Why it helps this model specifically — and the caveat that limits it:**
The dataset-generation pipeline (`generate_patterns.py` +
`process_dataset_traj.py`, composite a moving foreground pattern onto a
real background with a simulated blur trajectory) is a second, independent
precedent for synthetic blur-streak generation, alongside the
copy-paste/compositing pattern already logged for bullet 5
(`cap-augmentation`, 2026-11-10) — reinforcing that "synthesize the blur we
lack" is a viable, multiply-attempted approach, not a one-off idea. But the
*detection* half has a load-bearing assumption this project's footage
violates: the running-median background estimate requires the camera (and
background) to be near-static across the few frames it medians over. Phone
swing footage is explicitly **not** static — the README's own task brief
notes the app is run handheld, and even a down-the-line tripod shot has
the golfer's own body and club moving through most of the frame, not just
isolated in front of a fixed background. A moving-camera clip would corrupt
the median background estimate and likely degrade or break this specific
architecture's core mechanism. This is a real, concrete limitation, not a
hedge: unlike RT-Focuser (works on arbitrary single frames) or SoccerSynth
(blur augmentation, camera-agnostic), FMODetect's detection network is
built for a fixed-or-slow-moving-camera regime (it was trained on VOT
tracking sequences and benchmarked on falling-object/table-top clips), which
is a materially different shooting condition than this project's handheld
golf footage.

**Effort vs. payoff:** Low-to-medium effort to even try — the inference
code runs standalone (`run.py --video ...`), but getting it running at all
requires resolving undocumented dependencies and a pretrained-model link
that could not be confirmed alive from this session, before any test against
real footage is possible. Payoff, if it ran, is uncertain rather than
clearly high: the core architectural idea (learn blur-streak shape directly)
is sound and the licence is clean, but the background-subtraction
precondition is a poor match for handheld phone video, so a direct port is
unlikely to work without first replacing the background-estimation step
with something that tolerates camera motion (e.g., optical-flow-compensated
background, or dropping the background channel and training a single-frame
blur-heatmap head instead). The more actionable takeaway is the *dataset
generation* technique, not the detector as shipped — logged as a second,
independent confirmation that synthetic blur-streak compositing is a
real, implemented pattern worth building on, with this repo's
`generate_patterns.py`/`process_dataset_traj.py` as a concrete reference
implementation to read (not reuse as-is, given its VOT/TensorFlow-specific
plumbing).

---

## 2026-11-11 — MonoClub: a brand-new (Aug 2026) preprint doing pose-and-shaft-conditioned joint clubhead+shaft localisation — the first bullet-4 finding that structurally targets *both* failure modes at once, but it has no code, no confirmed license, and could not be read directly

**Rotation context:** The last four runs (2026-11-10 x3, plus the
2026-11-10-corrected FMODetect addition) covered bullets 2, 3, and 5, plus
one licensing/architecture side-investigation (RF-DETR). This run returns
to bullet 4 (golf-specific pose/club tracking), last touched 2026-11-07
(GolfPose, GolfDB).

**What it is:** "MonoClub: Pose-Guided Joint Localisation of Golf Clubhead
and Shaft in Monocular Swing Video" — a preprint by Zheng Zhang, Zhixin
Zhou, Wen Jung Li, and Rui Luo, dated August 2026 on Research Square
(DOI-style ID `rs-10817615`, v1). It estimates a 2D club configuration
(clubhead location, shaft centreline, and an explicit visibility/withhold
decision) from a single frame by scoring candidate hypotheses against
multiple evidence channels together: shaft-strip evidence, crop-level
clubhead segmentation, shaft-endpoint evidence, grip-compatibility with
detected body pose, and an optional temporal-consistency term across
frames. Reported metrics (from the abstract, cross-confirmed across two
independent search results quoting the same numbers): proposal recall
96.2% at 5 candidates / 98.1% at 10, clubhead segmentation Dice 0.84 / IoU
0.73, end-to-end mean localisation error 9.1 px, PCK@10 93.5%, shaft-angle
error 3.6°. Ablations test pose conditioning, joint (clubhead+shaft)
coupling, and the temporal-consistency term separately.

**URL:** https://www.researchsquare.com/article/rs-10817615/v1 —
`researchsquare.com` is DNS-blocked from this session's egress proxy (403
on CONNECT, confirmed directly with `curl`), the same standing allowlist
gap every prior entry in this log has hit for academic hosts (`arxiv.org`,
`universe.roboflow.com`, `kaggle.com`, `huggingface.co`,
`paperswithcode.com`, `github.io`, `wikipedia.org` were also re-checked
this run and are all still blocked — no change to report). The paper's
existence, exact title, full author list, and the specific numeric results
above were **not read directly from the source** — they come from
cross-referencing two independent web-search result sets that both quoted
the same title, authors, and metrics verbatim, which is strong but not
first-party evidence.

**Licence:** **Not confirmed.** Research Square's platform default for
preprints is CC BY 4.0 unless an author opts out at submission, but that
is general platform knowledge, not something read off this paper's own
landing page — the page could not be fetched. Treat the licence as
**unknown** until someone with unblocked access opens the page directly
and reads the licence badge.

**Verification performed:** Confirmed the title/authors/DOI-style ID
resolve consistently across two separate search queries (one direct title
search, one broader query that surfaced it as a related result). Searched
separately for an arXiv mirror, a GitHub implementation, and author names
plus "code"/"github" — **found none of the three.** This is a preprint
only: no peer review, no code release, no dataset release could be
confirmed as existing today. The specific numeric results could be
fabricated noise from the search layer rather than genuine abstract text,
and that risk could not be fully closed from this session.

**Which failure mode this addresses:** **Both, by design, which is why
it's a new kind of entry for this log.** Every prior architecture logged
here (FMODetect, RT-Focuser, YOLOMG, WASB-SBDT, VCOD/EventVCOD, etc.) picks
one failure mode's mechanism — blur-as-signal, deblur-then-detect,
motion-vs-appearance fusion, temporal aggregation — and is silent on the
other. MonoClub's grip-compatibility and shaft-strip evidence channels are
load-bearing on the golfer's body pose and the shaft line, not on the
clubhead's own appearance — exactly the information that survives both
when the head itself is camouflaged against dark clothing/foliage (zero
appearance signal, per this project's camouflage finding) and when it is
smeared into a blur streak (zero clean-box signal, per this project's blur
gap). A detector that can propose "the head is somewhere along this
tracked shaft line, near this tracked grip position" has a fallback when
the appearance channel alone returns nothing.

**Why it helps this model specifically — and the caveats that limit that:**
This project's v1 failure analysis found camouflage frames with *zero*
candidate detections even at confidence 0.05 — a pure appearance-channel
failure that a kinematic/pose prior could plausibly rescue, similar in
spirit to the GolfPose entry logged 2026-11-07, but MonoClub is explicitly
built around club geometry (shaft + head jointly) rather than general body
pose, which is a tighter fit. However: (1) it operates on a single frame's
club *configuration*, not the frame-to-frame clubhead *box* this project's
YOLO11n pipeline and CoreML export are built around — adopting it would
mean a different model family and output representation, not a drop-in
training-data or architecture swap; (2) with no code, no license, and
metrics unverified first-party, this is a lead to watch, not something
actionable this quarter; (3) the "visibility withhold" decision is a
concept worth borrowing on its own even without the rest of the paper —
this project's labeling spec already has a negative-frame convention, and
an explicit low-confidence-withhold head (vs. just thresholding
objectness) is a smaller, standalone idea that doesn't require the paper
to be reproducible to be useful.

**Effort vs. payoff:** High effort, uncertain payoff, if pursued as a
model swap — there is nothing to run today. Low effort, plausible payoff,
if the takeaway is narrowed to just the "condition clubhead-box confidence
on a tracked shaft-line/body-pose prior, with an explicit withhold state"
idea, which could be prototyped against this project's own existing pose
tooling without needing MonoClub's code at all. Recommend: watch for a
public release (code, arXiv mirror, or venue publication) rather than
attempting to reproduce from the abstract alone; do not cite the specific
numeric results above as settled fact in any decision document, since they
were never read from the primary source.

---

## 2026-11-11 (second run today) — SHOP: a real deblur→detect→pose-filter pipeline for small handheld objects in blurry video, itself unlicensed, but its BSD-licensed deblurring component (DeblurGANv2) and its hand/pose-region filtering idea are independently usable

**Rotation context:** The first run today (MonoClub) covered bullet 4
(golf-specific pose/club tracking). This run returns to bullet 2 (motion
blur: deblurring preprocessing) and bullet 3 (pose/region-based filtering
as a substitute for appearance when appearance alone is unreliable),
cross-checking against the deblur-as-preprocessing work already logged
2026-11-07 (RT-Focuser).

**What it is:** "SHOP: A Deep Learning Based Pipeline for near Real-Time
Detection of Small Handheld Objects Present in Blurry Video" (Ganguly,
Gandhi, E, Chang, Hudson; IEEE SoutheastCon 2022, arXiv:2203.15228,
submitted 29 Mar 2022). A three-stage pipeline built for exactly this
project's blur problem in miniature: (1) deblur the frame with
DeblurGANv2, (2) run YOLOv5 object detection on the deblurred frame, (3)
filter detections using pose estimation (OpenPose/tf-pose-estimation) to
keep only objects near a detected hand region and reject ones too large to
plausibly be a small handheld object. Reported result (from the paper
abstract via search snippets, not read first-party since arXiv is blocked
— see below): a 70% drop in false positives for a 17% drop in true
positives on their strongest configuration, measured on a handheld subset
of MS COCO that the authors also released as a byproduct dataset.

**URL:** Paper: https://arxiv.org/abs/2203.15228 (arxiv.org is
DNS-blocked from this session's egress proxy — same standing gap every
prior entry has hit — so the abstract numbers above are from search
snippets, not the PDF, and are one notch below full verification). Code
and dataset: **https://github.com/spider-sense/SHOP — this was fetched
directly and confirmed to exist**: public repo, 41 commits, 3 stars, 2
forks, containing `SHOP.py`, a bundled `DeblurGANv2/` subfolder, a
`pose_estimation/`/`tf_pose_estimation/` folder, a `validation/` folder,
and a README that links a tagged dataset release (v0.1.0) plus model
weights hosted on Google Drive. This is a real, non-dead project, which is
not true of every lead this log has chased.

**Licence:** **Checked directly — `spider-sense/SHOP` ships no LICENSE
file and its GitHub "About" panel shows no license.** Under GitHub's and
US copyright default, that means all rights reserved by the authors: the
SHOP pipeline code and its released handheld-object dataset are **not
commercially usable** without contacting the authors for explicit
permission. However, the bundled DeblurGANv2 component is a vendored copy
of a separately-licensed project: `VITA-Group/DeblurGANv2`'s own LICENSE
file (fetched directly) contains three BSD-style licence blocks — the main
software and the DeblurGANv2-specific code (Kupyn/Martyniuk/Wu/Wang,
2019), plus a vendored DCGAN component (Facebook, 2015, BSD with a
non-endorsement clause). All three are permissive, **commercial use
permitted**, subject to retaining copyright/disclaimer notices. So: the
deblurring *model* this pipeline relies on is clean; the handheld-object
*pipeline code and dataset* built on top of it is not.

**Which failure mode:** Motion blur (primary) — deblur-then-detect is the
same mechanism as RT-Focuser (2026-11-07), but SHOP is a second,
independently-verified real implementation of that mechanism, which
corroborates it rather than just repeating it. The pose/hand-region
filtering stage is also a camouflage-adjacent idea: it is a second,
non-appearance signal (a tracked anchor region) used to keep or reject a
candidate detection, in the same spirit as MonoClub's shaft/grip
conditioning logged just above — but using a 2022, actually-working, MIT
Python stack (OpenPose-based pose estimation) rather than an unread 2026
preprint.

**Why it helps this model specifically:** Two separable, reusable ideas,
neither requiring SHOP's own code or dataset licence: (1) DeblurGANv2
itself is confirmed commercially clean and could be evaluated as a
preprocessing step ahead of this project's YOLO11n detector on blurred
frames, as a second deblur candidate alongside RT-Focuser — RT-Focuser is
smaller and already CoreML-benchmarked on iPhone, so it remains the
stronger on-device candidate, but DeblurGANv2 is a useful heavier baseline
to sanity-check against during offline evaluation, where on-device latency
doesn't matter. (2) The "filter candidate detections by proximity to a
tracked hand/grip region" idea is directly applicable to the camouflage
failure mode this project measured (zero detections at confidence 0.05
against dark clothing/foliage): if a body-pose model can localise the
golfer's hands/grip even when the clubhead itself is invisible, that
region becomes a strong spatial prior for re-scoring or re-running
detection at a lower threshold specifically near the grip — cheaper to
prototype than MonoClub's full shaft-conditioned architecture, since
off-the-shelf pose estimation (e.g. Apple's own Vision body-pose API,
already on-device and licence-free for this project) could stand in for
OpenPose.

**Effort vs. payoff:** Low-to-medium effort, plausible payoff, but only
for the *ideas*, not the artifact — SHOP's own code/dataset cannot be
used commercially as-is, so nothing here is a drop-in import. Prototyping
"detect near the tracked grip region" as a post-filter or re-scoring pass
on top of the existing YOLO11n output is a small, testable experiment
against this project's own test clips, independent of any of SHOP's code.
Evaluating DeblurGANv2 as an offline preprocessing baseline is also cheap
(BSD-licensed, pip-installable) but is lower priority than RT-Focuser
specifically for the on-device path, since RT-Focuser is lighter and
already has iPhone latency numbers. Recommend: treat this entry as
corroboration that deblur-then-detect is a validated pattern (two
independent implementations now logged), and as the source of the
hand/grip-region-filtering idea — not as a dataset or code source.

---

## 2026-11-11 (third run today) — Real-frame-averaging blur synthesis (the GOPRO/REDS dataset-construction method): a way to manufacture genuinely-blurred, correctly-boxed clubhead training examples from this project's *own* high-frame-rate footage, with no third-party licence involved at all

**Rotation context:** Today's first two runs covered bullet 4 (MonoClub)
and bullets 2/3 (SHOP). This run first re-attempted bullet 1 (commercial
indoor/low-light golf datasets): `universe.roboflow.com` (the
`golf_lowlight` lead this log flagged by name on 2026-11-07 without
fetching it) and a second candidate host, `affinedrift.com`, were both
tried directly via WebFetch this run and both came back DNS-blocked from
this session's egress — the same standing gap every bullet-1 attempt since
2026-11-04 has hit, confirmed again rather than assumed. Per this log's own
rule (don't log an unfetched page summary as if it were checked), bullet 1
is deferred again and this run moved to bullet 5 — synthesizing training
data for the motion-blur regime — specifically to test an angle no prior
bullet-5 entry has checked: not a kernel-blur-on-a-crop technique (already
covered 2026-11-10, `cap-augmentation`) and not a composited-foreground
technique (already covered 2026-11-08 FMODetect's `generate_patterns.py`),
but *real* optical blur assembled from real sharp frames.

**What it is:** The exact method used to construct the GOPRO and REDS
deblurring benchmarks (Nah, Kim, Lee, "Deep Multi-scale Convolutional
Neural Network for Dynamic Scene Deblurring," CVPR 2017, arXiv:1612.02177;
and the follow-up NTIRE 2019 REDS dataset): shoot a scene at a high frame
rate, then average a short run of consecutive sharp frames (7–13 frames in
the original GOPRO paper, shot at 240fps on a GoPro Hero 4 Black) to
synthesize a longer-effective-exposure blurry frame, applying an inverse
gamma correction before averaging and a forward gamma after (to avoid
non-linear camera-response-curve artifacts), with the middle frame kept as
the paired sharp ground truth. REDS extended this by first running an
off-the-shelf video-frame-interpolation model to upsample 120fps source to
1920fps *before* averaging, specifically to remove staircase artifacts
that plain averaging produces when the native frame rate isn't high
enough relative to the motion speed and output resolution.

Applied to this project: take existing (or newly captured) high-frame-rate
phone slow-motion footage of a swing — modern iPhones natively shoot
1080p/240fps and some models 1080p/960fps slow-mo — and average runs of N
consecutive sharp frames around the fast downswing/impact window to
synthesize realistic, physically-correct motion-blur streaks of the actual
clubhead, not an invented kernel. Because the clubhead's position in each
of the N source frames is already known (either from this project's
existing per-frame labels, or from running the current trained model on
each sharp frame as a pre-label, per the data-engine's own model-assisted
labeling loop), the ground-truth box for the synthesized blurry composite
is simply the union of the N per-frame boxes — no new manual labeling
required, and the box is exactly correct by construction rather than
estimated.

**URL:** Primary source: https://arxiv.org/abs/1612.02177 (paper) —
`arxiv.org` is DNS-blocked from this session's egress proxy, the same
standing gap every prior entry in this log has hit for that host, so this
could not be fetched first-party. Verified instead via convergent
independent search-engine summaries of the paper's own methods section
(multiple distinct sources — arXiv mirror listings, a paper-review site,
and an NTIRE workshop paper comparing GOPRO's and REDS's construction —
agreeing on the same specific numbers: 240fps GoPro Hero 4 Black source,
7–13 frame averaging window, inverse-gamma correction, middle frame as
sharp ground truth), which is weaker than reading the PDF directly but
stronger than a single unverified snippet. The REDS interpolate-then-average
refinement is separately corroborated by the NTIRE 2019 workshop paper
summary. This is a widely-cited, foundational dataset-construction method
in the deblurring literature (GOPRO/REDS are the standard benchmarks nearly
every deblurring paper already logged in this log — RT-Focuser, SHOP's
DeblurGANv2 — trains or evaluates against), not an obscure or unverifiable
claim.

**Licence:** Not applicable in the usual sense — this is a *technique*,
not a dataset or a code artifact being reused. No third-party licence is
implicated at all if applied to this project's own footage with its own
tooling: the averaging step is a few lines of NumPy/OpenCV array math
(`cv2`/`numpy` are already project dependencies), and even the optional
interpolation refinement (if native frame rate proves insufficient) has a
commercially-clean option already available: `ffmpeg` (installed in this
environment, LGPL/GPL build tooling used purely offline, never
redistributed) ships `minterpolate`/`tblend` filters for exactly this. No
need to adopt RIFE or any other third-party interpolation model and its
licence questions. **Commercial use: fully permitted** — the method itself
carries no licence restriction, and the one piece of source material this
project would actually average (its own captured footage) is the
project's own.

**Verification performed:** Confirmed via multiple independent
search-engine summaries that this is the real, specific method both
GOPRO and REDS used (not a loose paraphrase — the 7–13 frame count, 240fps
source rate, and inverse-gamma step are consistent across sources).
Confirmed `ffmpeg` is installed and available in this environment as a
fallback interpolation tool if needed. Could not fetch the arXiv PDF or an
ar5iv mirror directly (both DNS-blocked) to quote the methods section
verbatim, and did not locate or run a reference implementation of the
exact averaging script — this entry is a verified *technique*, not a
verified *tool*, and that distinction matters for the effort estimate
below. Did not verify whether this project's existing own-swing training
clips (the ~29% app-captured footage) were shot in a high-frame-rate slow-
motion mode at all; if they were all captured at standard 30/60fps, this
technique requires new, deliberately-shot slow-mo capture rather than
reprocessing footage already on hand — a real open question this run could
not resolve from inside `ml/clubhead/docs/research/`.

**Which failure mode this addresses:** Motion blur, directly and
specifically. No bearing on camouflage.

**Why it matters for this model specifically:** This is a materially
different and arguably stronger answer than the two synthesis approaches
already logged for bullet 5. `cap-augmentation` (2026-11-10) and
FMODetect's pattern-compositing (2026-11-08) both produce *synthetic*
blur — an invented kernel or a dragged foreground pattern — applied to
material that wasn't really moving. Frame-averaging instead produces
*real* optical blur of the real clubhead on its real swing arc (curved,
accelerating, exactly as fast as it actually moves), which should transfer
to real footage better than an artificial straight-line kernel, and it
produces a provably-correct ground-truth box as a free byproduct of the
averaging window rather than a hand-tuned or estimated one. It also slots
directly into the data engine this project has already committed to
(model-assisted pre-labeling + confidence auto-accept): the per-frame boxes
needed to compute the union box can come from the current model's own
predictions on the sharp source frames, meaning this technique could be
implemented as a label-generation step inside the existing pipeline rather
than a separate one-off script.

**Effort vs. payoff:** Low implementation effort for the averaging step
itself (a short NumPy script: gamma-decode N consecutive sharp frames,
average, gamma-encode, union the N boxes) — on par with or simpler than
the kernel-blur approach already logged, and with no licence diligence
needed at all. The real cost is upstream and unverified: it depends on
having (or capturing) genuinely high-frame-rate source footage of swings,
which this run could not confirm the project already has. If new capture
is required, that is a data-collection cost, not an engineering one, and
smaller than most of this log's other open leads (it needs a phone's
built-in slow-mo mode, not special equipment). Payoff is plausible and,
unlike the kernel-blur entry, has a direct literature precedent of working
at scale (GOPRO/REDS are the field's standard deblurring benchmarks, built
exactly this way) — but no entry in this log, including this one, has yet
measured whether *this specific project's* detector trains better on such
composites, so it remains a recommended experiment, not a confirmed win.
Recommend as the next bullet-5 experiment to prototype, ahead of the
kernel-blur idea, specifically because the ground-truth box comes for
free and the realism is physically grounded rather than approximated.

---

## 2026-11-11 (fourth run today) — STF: a real, Apache-2.0, inspectable two-frame attention+fusion neck whose own stated target is occluded, blurred, *and* small objects — a concrete architecture blueprint, not another unreadable preprint

**Rotation context:** Today's prior three runs covered bullet 4 (MonoClub),
bullets 2/3 (SHOP), and bullet 5 (frame-averaging). This run picks bullet 3
(temporal/multi-frame methods for small, low-contrast objects), specifically
looking for something that improves on this log's existing bullet-3 entries
(VCOD/Vcamba, EventVCOD, YOLOV/YOLOV++, RFR/IRSatVideo-LEO, YOLOMG, the
Tübingen lightweight-multi-frame paper) by actually being readable,
small, and non-exotic (no Mamba kernels, no diffusion, no AGPL).

**What it is:** "STF: Spatio-Temporal Fusion Module for Improving Video
Object Detection" (Anwar, Bilodeau, Bouachir — arXiv:2402.10752). It is a
plug-in *neck* module (in object-detector terms: the stage between
backbone and head) built from two pieces: a **Multi-Frame Attention (MFA)**
block that learns adaptive per-channel weights to blend the current
frame's backbone features with the previous frame's, and a **Dual-Frame
Fusion** block that aligns and merges the two frames' features using
deformable convolutions. Per independent search-engine summaries of the
abstract (arXiv itself is DNS-blocked — see Verification), it is explicitly
built "to better detect objects that are occluded, blurred, or small," and
is evaluated on KITTI MOT, Cityscapes, and UAVDT (a small-object aerial
dataset — the same small/cluttered-background regime this project's
camouflage failures sit in).

**URL:** Paper: https://arxiv.org/abs/2402.10752 (DNS-blocked from this
session, same standing gap as every prior entry — not fetched directly).
Code: **https://github.com/noreenanwar/STF-module — fetched directly and
inspected.**

**Licence:** Fetched `LICENSE` directly — verbatim opening:
```
Copyright 2018-2023 OpenMMLab. All rights reserved.

                                 Apache License
                           Version 2.0, January 2004
```
**Apache-2.0. Commercial use: permitted**, subject to the standard
attribution/notice-preservation terms. Caveat worth stating plainly: the
repo is a fork of MMDetection (2,764 commits total, 8 stars, 1 fork, no
GitHub Releases/pretrained weights), so the Apache-2.0 grant is the
repository's own blanket licence, not a bespoke licence the paper authors
wrote for their specific contribution — the new file itself only carries
mmdetection's generic boilerplate header, not the authors' own copyright
line. That doesn't weaken the licence (the repo's root `LICENSE` governs
every file in it, this file included), but it does mean there is no
separate, authors-signed statement that *this specific module* is
Apache-2.0 beyond the repo's own terms.

**Verification performed:** This run almost logged a false lead:
fetching the repo's `projects/` directory first showed only stock
MMDetection example projects (AlignDETR, CO-DETR, DiffusionDet, ViTDet,
etc.) with nothing named STF — i.e., the obvious place to look for the
contribution was a dead end. Used `gh api search/code -f
q='stf repo:noreenanwar/STF-module'` to search inside the repo directly,
which surfaced `mmdet/models/necks/custom_neck.py` as the one
non-stock file touched. Fetched that file's contents directly (via raw
GitHub) and confirmed it is real, substantive PyTorch code — a
`MultiFrameAttention` class (adaptive-weighted current+previous-frame
blending, referencing "Equation (4) in STF" in its own comment — i.e. it
matches the paper it's named for) and a `DualFrameFusion` class
(deformable-convolution-based alignment), registered into MMDetection's
`MODELS` registry as a neck. Also confirmed via `gh api
repos/.../commits?path=...` that this file is a genuine addition, not
boilerplate (blocked by this session's repo-scoping before full commit
history could be pulled, so exact commit dates are unconfirmed). Did not
and could not verify the paper's own numeric results (mAP deltas on
KITTI/Cityscapes/UAVDT) — arXiv is blocked and the README reports no
numbers, only output images and an unlabeled loss curve.

**Which failure mode this addresses:** Both, by the authors' own stated
design goal (occluded/blurred/small), making it the second bullet-3/4
entry in this log (after MonoClub) to target both regimes at once — but
unlike MonoClub, this one has real, readable, inspected code today.

**Why it matters for this model specifically:** Every prior temporal/
camouflage entry in this log was either architecturally exotic and hard to
port (Vcamba's Mamba selective-scan CUDA kernels, diffusion-based CamDiff),
licence-blocked (YOLOMG GPL-3.0, the AGPL badminton-shuttle repo), or a
paper with no code at all (the Tübingen lightweight-multi-frame work,
MonoClub). STF's actual mechanism — two small attention blocks plus one
deformable-conv fusion block, wired as a neck between any backbone and any
head — is simple enough to reimplement from scratch as a bolt-on layer
ahead of YOLO11n's existing head, feeding it the previous frame's backbone
features alongside the current frame's. That is directly aimed at the
camouflage failure's own evidence (zero detections at confidence 0.05 on
a single sharp frame, where the one cue the model never gets to see is
that the clubhead moved and the foliage didn't) and, per the authors'
claim, the blur regime too — a double-purpose architecture change rather
than two separate fixes.

**Effort vs. payoff:** High effort, genuinely uncertain payoff. This is
not a drop-in: integrating it means (1) reimplementing the two modules
outside MMDetection's framework to fit the Ultralytics/YOLO11n training
and CoreML-export pipeline, and (2) changing the on-device inference
contract from one frame in to two consecutive frames in, which touches the
iOS app's capture/inference loop, not just the model file — a materially
bigger change than anything else logged so far (bigger than swapping in
RF-DETR, bigger than adding a deblur preprocessing pass). There are no
pretrained weights, no sport/golf evaluation, and the repo itself has
minimal external validation (8 stars). Recommend treating this as the
clearest *architecture reference* this log has found for a genuine
two-frame temporal fusion prototype — worth a small standalone experiment
(bolt MFA + DualFrameFusion onto YOLO11n's neck, train on a handful of
labelled clip pairs, measure recall specifically on the existing
zero-candidate camouflage frames) before committing to the much larger
app-architecture change a two-frame model implies.

## 2026-11-12 — TTNet (table tennis ball tracking): a real, first-hand-verified coarse-to-fine two-stage temporal detector for a small, fast, blurry ball — but both the code and the dataset are blocked for commercial use

**What it is:** TTNet ("Real-time temporal and spatial video analysis of
table tennis", Voeikov et al., CVPR 2020 Workshops, OSAI/osai.ai) and its
most-used open implementation,
`maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch`. It
takes a sliding window of **9 consecutive frames** (confirmed first-hand:
`configs.num_frames_sequence = 9` in `src/config/config.py`) and runs a
**coarse-to-fine two-stage** detector: a "global stage" CNN produces a
low-confidence ball-position heatmap over the full 1920x1080 frame, then
the model crops a small patch around that coarse prediction
(`__crop_original_batch__` in `src/models/TTNet.py`, confirmed first-hand)
and re-runs a second "local stage" detector on just that patch at full
resolution to refine the position. An "events spotting" head (bounce/net
hit) and a segmentation head are trained jointly, multi-task-style, on top
of the same features. The authors report <4px RMSE (on the full-HD frame)
for ball position and ~97% event-spotting accuracy — i.e. this is a
real, benchmarked architecture for exactly this project's problem shape:
a single small, fast-moving, often-blurred object that is easy to lose
track of frame-to-frame.

**URL:** Paper: arXiv:2004.09927 (arXiv is DNS-blocked from this session,
the same standing gap as every prior entry — not fetched directly; paper
details below come from the implementation's README/code, which restate
them, and from search-engine summaries of the abstract). Code (cloned and
inspected first-hand, not just rendered via WebFetch):
https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch
Dataset: OpenTTGames, downloaded by the repo's own
`prepare_dataset/download_dataset.py` from
`https://lab.osai.ai/datasets/openttgames/data/` (confirmed first-hand by
reading that script; `lab.osai.ai` itself is DNS-blocked from this
session, so the host was not reached directly).

**Licence — code: none, dataset: non-commercial, neither usable here.**
Cloned the full repo (`git clone --depth 1`, not just a rendered page) and
searched the entire tree: **there is no LICENSE file anywhere in it** —
confirmed by direct directory listing (`docs/`, `prepare_dataset/`,
`src/`, `.gitignore`, `README.md`, `requirement.txt`, nothing else) and by
`find -iname '*licen*'` returning zero matches. Also checked `main` and
`master` branches of `raw.githubusercontent.com/.../LICENSE` directly —
both 404. Under GitHub's own terms, a public repo with no LICENSE file
grants no rights beyond viewing the source on GitHub; by default **all
rights are reserved and the code is not usable, commercially or
otherwise, in this project.** Separately, the OpenTTGames dataset itself
is licensed **CC BY-NC-SA 4.0** — confirmed via two independent secondary
sources (a 2025 "Extended OpenTTGames Dataset" paper/repo that explicitly
states it reuses "the original ball-position annotations unchanged" under
"Attribution-NonCommercial-ShareAlike 4.0 International License (CC
BY-NC-SA 4.0)", and a third-party paper-review summary repeating the same
license and terms); the primary osai.ai source could not be reached
directly (DNS-blocked), so this rests on the secondary citations, same
caveat as every arXiv-blocked entry in this log. **NC = commercial use is
not permitted**, so even if the code were usable, the one dataset it is
built to train on is not.

**Verification performed:** This is the deepest first-hand verification
in this log's "licence rules it out" category to date — rather than
relying only on a rendered GitHub page (which can omit files from its
truncated listing, as several prior entries in this log have had to
caveat), the repo was fully cloned via `add_repo` + `git clone` and its
file tree inspected directly with `find`/`ls`, and the two frame-count and
crop-and-refine architecture claims were confirmed by reading the actual
Python source (`config.py`, `TTNet.py`), not the README's images (the
README itself only shows the architecture as unreadable `.PNG` diagrams).

**Which failure mode this addresses:** Motion blur / small-fast-object
tracking, primarily — the coarse-to-fine global→local crop-and-refine
mechanism is a different technique from every multi-frame fusion entry
already logged (YOLOV/YOLOV++, RFR, STF all fuse features across frames
at a fixed resolution; TTNet instead uses one frame-sequence pass to get a
rough location, then re-examines a tight crop at full resolution — a
zoom-and-refine step that is largely orthogonal to temporal fusion and
could in principle be stacked with it). Only secondarily relevant to
camouflage: TTNet's global stage still needs the ball to be visible in at
least a diffuse sense in the full frame, so it does not address the
zero-candidate-at-any-confidence camouflage case directly.

**Why it matters for this model specifically:** The crop-and-refine idea
is the useful transferable piece, independent of the blocked code/dataset
— it's a plausible lightweight answer to a problem this project's own
labeling spec implicitly has (boxes should capture the full blur streak,
but a single fixed-resolution pass over a full phone frame gives a blurred
clubhead very few pixels to be recognized by). A two-pass global-coarse
then local-refine design, run on YOLO11n's own output as the "global"
stage and a second light pass on the cropped region as "local," is a
smaller, more tractable architecture change than the STF/YOLOV two-frame
fusion options already logged, because it needs no second frame and no
change to the iOS app's capture contract — only a second, cheap inference
pass per frame. That said, the *code itself* cannot be used (no licence)
and the *dataset itself* cannot be used (NC), so this is purely an
idea/techniques takeaway, reimplemented from scratch against YOLO11n, the
same posture this log already took toward YOLOMG (GPL-3.0) and the ETH
Zurich badminton-shuttle paper (AGPL-3.0).

**Effort vs. payoff:** Medium effort, speculative payoff, lower risk than
the STF/two-frame options already logged. Implementing a from-scratch
global→crop→local second pass on top of the existing YOLO11n pipeline is
a self-contained addition (no iOS capture-loop change, no second-frame
dependency) that could be prototyped and measured against the existing
camouflage/blur failure frames without touching the app at all — genuinely
cheaper to try than the two-frame fusion entries in this log. The
counter-weight: TTNet's own reported precision (<4px RMSE) comes from a
static, tripod-mounted, fixed-court camera with a known table geometry,
which is a much easier localization problem than a handheld phone
following a swinging club — so the headline numbers should not be taken
as evidence this approach will transfer, only that the architecture
pattern itself is real, inspectable, and legitimately applicable in
theory.

---

## 2026-11-12 (second run today) — `VNTrackObjectRequest`: Apple's own first-party, zero-licence, already-unused Vision-framework object tracker is a free way to bridge the exact camouflage gap (zero candidates at any confidence) by propagating the last confident detection instead of re-detecting every frame

**Rotation context:** This run's first attempt was bullet 1 (datasets, especially low-light/indoor/simulator/older-phone footage), rotating away from the previous run's bullet 2/3 (TTNet). It hit the same standing wall every prior bullet-1 attempt in this log has hit: `universe.roboflow.com`, `huggingface.co`, and `zenodo.org` are all still DNS-blocked from this session (same gap documented since 2026-11-04), and every GitHub-reachable lead found by search (GolfDB, CaddieSet, GolfPose, YouTube-sourced "golf swing dataset" hits) is a topic already logged and ruled out. Rather than log a fourth "still blocked" restatement, the run pivoted to bullet 3 (small/camouflaged-object temporal methods) with a different kind of candidate: not a paper or a third-party repo, but a capability already sitting unused in this project's own iOS code.

**What it is:** `VNTrackObjectRequest`, part of Apple's Vision framework (iOS 11.0+, confirmed present in the actively-targeted iOS 17+ deployment range — see Verification). It is a built-in, on-device, model-free object tracker: seeded with a `VNDetectedObjectObservation` (a bounding box) from any prior frame, it is run frame-by-frame through a `VNSequenceRequestHandler` and attempts to relocate that same object by appearance/correlation in each subsequent frame, returning an updated bounding box plus a `confidence` score, independent of whatever produced the seed box. Apple's own sample code ("Tracking Multiple Objects or Rectangles in Video") describes exactly the loop this project would need: seed a tracker from a detector's output, feed each frame's result back in as the next frame's input, and periodically re-run the detector (the sample's own choice is every 10 frames) so objects the tracker loses are re-acquired.

**URL:** https://developer.apple.com/documentation/vision/vntrackobjectrequest and https://developer.apple.com/documentation/vision/tracking-multiple-objects-or-rectangles-in-video (both JS-rendered pages return no body text directly — confirmed by fetching the underlying `tutorials/data/documentation/vision/....json` endpoints instead, the same workaround the 2026-11-05 ARKit `sceneDepth` entry used and documented; quotes below are from that JSON, not paraphrased from training knowledge). Supporting evidence: Apple Developer Forums thread https://developer.apple.com/forums/thread/780037 (fetched directly, real thread, dated April 2025).

**Licence:** Not applicable — first-party OS API shipped with iOS/Vision, free to call from any app including a commercial one, exactly the same posture as the already-logged ARKit `sceneDepth` entry. No third-party code, no dataset, no attribution requirement.

**Verification performed:** Fetched the class-reference JSON directly and confirmed verbatim: abstract — "An image-analysis request that tracks the movement of a previously identified object across multiple images or video frames"; it inherits `VNTrackingRequest`; its only initializers take a `detectedObjectObservation` (optionally with a completion handler); platforms listed as iOS 11.0+, iPadOS 11.0+, Mac Catalyst 13.1+, macOS 10.13+, tvOS 11.0+, visionOS 1.0+. Fetched the "Tracking Multiple Objects or Rectangles in Video" sample JSON directly and confirmed verbatim: tracking runs through `VNSequenceRequestHandler` (not `VNImageRequestHandler`, which is for detection on still images); the sample re-detects every 10 frames to pick up new objects; each result is a `VNDetectedObjectObservation` with a `confidence` value, and the sample's own code comment treats confidence above 0.5 as a successful track (explicitly flagged in the sample as that author's threshold choice, not a documented rule); the article states plainly that "the quality of the initial selection strongly affects tracking" and gives no specific guidance for small or fast-moving objects. Fetched the real Apple Developer Forums thread 780037 directly (not search-summarized) and confirmed a reported edge case: intermittent `Code=9 "Internal error: unexpected tracked object bounding box size"` errors, with an Apple Frameworks Engineer's own reply quoted verbatim: "We are checking that the width * height of the bounding box to be tracked is bigger than FLT_MIN. So it looks like that your detected object has a zero width or height or at least very close to that." — i.e. the error fires only on genuinely degenerate (near-zero-area) boxes, not merely small-but-valid ones, so a clubhead-sized box should not inherently trigger it as long as the seeding detection itself is sane. Separately confirmed, by reading this project's own `ios-native/SwingSenseiNative/ClubHeadDetector.swift` directly, that the app's current detection path uses `VNCoreMLRequest` against the YOLO11n CoreML model on every frame independently and contains no `VNTrackObjectRequest`, `VNSequenceRequestHandler`, or any other cross-frame tracking call anywhere in that file — this capability is unused today, not already tried and rejected. Could not verify: documented behavior under fast motion or genuine motion blur (neither the class reference nor the sample article states anything about either, confirmed absent from both JSON fetches, not just unsearched); no numeric guidance on `trackingLevel`'s `.fast` vs `.accurate` tradeoff beyond the enum existing (the properties page itself was not reachable in JSON form this run).

**Which failure mode this addresses:** Primarily camouflage, secondarily motion blur, via a different mechanism from every other entry in this log: it is not a better detector and not better training data — it is a frame-bridging mitigation at the inference/selection layer, the same layer the project's own documented "software selection ceiling of 85.9%" already lives at. On a frame where YOLO11n returns zero candidates at any confidence (the project's own documented camouflage signature), a tracker seeded from the last confident detection a few frames earlier can still propose a box by appearance-correlation rather than relying on the CNN's own (failing) features — directly analogous to, but architecturally simpler than, the already-logged rucv/golf_ball Kalman-patch idea (2026-11-05, motion-history-based search-region prediction) and GolfPose's pose-derived prior (2026-11-07, body-geometry-based prior): those both require building or fine-tuning a separate model; this requires zero model work, only application code, because the capability already ships in the OS the app already targets. The real caveat, stated honestly: `VNTrackObjectRequest` is itself an appearance/correlation-based tracker, so on a severely motion-blurred *and* low-contrast frame it may fail for the same underlying reason the detector does — this is a plausible bridge for short camouflage gaps, not a guaranteed fix, and nothing here measures how well it survives either failure mode on this project's own clips.

**Why it matters for this model specifically:** This is the cheapest lever this entire log has found to date for the camouflage failure mode — cheaper than every other entry logged under bullet 3 or bullet 4, because it needs no new training data, no architecture change, no third-party licence review, and no export/CoreML-compatibility work (it is not a model at all). It slots directly into the existing per-frame `VNCoreMLRequest` loop in `ClubHeadDetector.swift`: on a frame where the detector returns no candidate above threshold, feed the last confident `VNDetectedObjectObservation` into a `VNTrackObjectRequest` for that frame instead of accepting "no detection," and keep re-running the real detector on a cadence (Apple's own sample uses 10 frames) to catch cases where the tracker itself has drifted or lost the object. This would smooth over exactly the kind of sharp, isolated camouflage frame the 82%/77% per-frame/through-swing gap describes, without touching the model or its training data at all — a change orthogonal to, and combinable with, every blur/camouflage model-side idea already in this log.

**Effort vs. payoff:** Low effort, plausible but unmeasured payoff. This is a small, self-contained iOS change (add a tracking-request fallback path to the existing per-frame detection loop) that could be prototyped and run against the project's own 3-clip outdoor eval set in well under a day, with no model retraining, no new licence review, and no dependency on any blocked host — strictly less engineering risk than any dataset, augmentation, or architecture entry logged so far. The honest limit: nothing here demonstrates it actually recovers any of the project's own documented camouflage failures, since a correlation tracker shares the same "needs some distinguishing visual signal" weakness as the detector it would be covering for, and it does nothing for a camouflage failure that persists for the tracker's entire bridging window. Worth a same-day prototype against the existing failing frames before any further investment; this run did not build or test one.

---

## 2026-11-12 (third run today) — SAHI (slicing-aided hyper inference): a real, actively-maintained, MIT-licensed, model-agnostic inference-time technique that improves small-object detection by tiling the frame — checked directly against this project's own zero-candidate camouflage signature

**Rotation context:** The first two runs today covered bullet 3-adjacent ground via TTNet (blocked licence) and `VNTrackObjectRequest` (free, but a frame-bridging fallback, not a detection-accuracy fix). This run stayed in bullet 3 (small/low-contrast/camouflaged object detection) but looked for a different class of lever: not a new architecture, not a new training signal, but a drop-in change to how the *existing* YOLO11n model is queried at inference time — a category this log had not yet checked.

**What it is:** `obss/sahi` — "Slicing Aided Hyper Inference," a framework-agnostic Python library that slices a full input image into overlapping tiles, runs the detector separately on each tile (optionally plus the full image), maps each tile's boxes back into full-image coordinates, and merges everything with NMS. It works with "any ultralytics / mmdet / huggingface / torchvision model" through a generic adapter (`AutoDetectionModel.from_pretrained(model_type="ultralytics", ...)`), not a model-specific reimplementation.

**URL:** https://github.com/obss/sahi (PyPI: https://pypi.org/project/sahi/). Underlying paper: Akyon, Altinuc & Temizel, "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection," ICIP 2022 (arXiv 2202.06934).

**Licence — verbatim (fetched `raw.githubusercontent.com/obss/sahi/main/LICENSE` directly):** "MIT License / Copyright (c) 2020 obss / Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software... " **Commercial use: permitted, no copyleft, no attribution-in-product requirement beyond keeping the notice in the software's own copy** — the cleanest licence posture of any third-party code logged so far in this file.

**Verification performed:** Fetched the GitHub repo and README directly (confirmed real: 873 commits, non-stub, documents a CLI, COCO utilities, FiftyOne visualization, and an evaluation/error-analysis toolkit, not just the slicing core). Fetched the PyPI project page directly: latest release is **0.14.0, dated Nov 1, 2026** — eleven days before this run, confirming the project is actively maintained today, not an abandoned 2022 research artifact. Confirmed the README's own `pip install` guidance pins `ultralytics>=8.3.161` and that its documented notebooks name YOLO26, YOLO26-SEG/OBB, and YOLOE specifically — it does **not** name YOLO11 by string anywhere I could fetch, but the integration point is the generic Ultralytics model-loading interface (not a YOLO-version-specific code path), and YOLO11 is itself an Ultralytics-family model that predates YOLO26 in the same API, so compatibility with this project's YOLO11n is a reasonable inference from the adapter design, not a verbatim confirmed claim — flagged honestly as the one unverified link in this chain. Could not fetch the ICIP 2022 paper itself: `arxiv.org` and `ar5iv.arxiv.org` are both still DNS-blocked from this session (same standing gap as every other arXiv entry in this log since 2026-11-04); the AP figures below come from a WebSearch results summary of the abstract, not a direct read of the PDF, and were not independently cross-checked against the paper's tables — same caveat this log has applied to every other arXiv-blocked entry. Did **not** find any mention of CoreML, "on-device," or mobile export anywhere in the repo or README — SAHI itself is a server/desktop Python inference wrapper, not something that ships into an iOS app as-is.

**Reported numbers (secondary source, not independently verified):** Per the abstract summary, slicing-aided inference alone raised AP by 6.8% (FCOS), 5.1% (VFNet), and 5.3% (TOOD) on VisDrone/xView-style aerial small-object benchmarks; adding slicing-aware fine-tuning raised the cumulative gain to 12.7%/13.4%/14.5% AP; a later revision reportedly adds a further ~2.9% AP from 25% tile overlap. These are dense-aerial-imagery numbers (many small objects per frame), a different domain from a single handheld-phone clubhead, so the *magnitude* should not be assumed to transfer — only the mechanism is the transferable claim (see below).

**Which failure mode this addresses:** Primarily camouflage. The project's own documented failure signature is zero candidate detections at any confidence (down to 0.05) on sharp, well-lit frames — a classic symptom of an object occupying too few effective pixels after a CNN backbone's downsampling has diluted its features, independent of lighting or contrast. Slicing the frame into overlapping tiles before each pass gives the clubhead a larger fraction of each tile's resolution than it has in the full letterboxed frame, which is exactly the lever the cited AP gains are measuring (small-object recall specifically, not general AP). Secondarily relevant to motion blur for the same reason: a blur streak is small relative to a full phone frame even when it is not small relative to the clubhead's sharp size.

**Why it matters for this model specifically:** This is the first entry in this log that needs **no training data, no licence negotiation, and no architecture change** to get a first answer — it is a pure inference-time wrapper around the already-exported model. It could be tested directly against the existing Python eval harness (`chdet.evaluate`) and the project's own 3-clip outdoor test set this week: run the current `.mlmodel`/checkpoint through SAHI's sliced-prediction path instead of a single full-frame pass, and see whether any of the documented zero-candidate camouflage frames produce a detection. That is a much faster experiment than anything else logged under bullet 3 (all of which require new training data, a new architecture, or a third-party model integration). The real cost shows up only if it works and needs to ship: SAHI itself is Python-only with no CoreML/on-device story, so productionizing this on-device means reimplementing the tiling-plus-NMS-merge loop in Swift around the existing `VNCoreMLRequest` call in `ClubHeadDetector.swift` (geometry + NMS, not a new model — similar in kind to the from-scratch reimplementation this log already concluded was reasonable for `cap-augmentation`'s pattern on 2026-11-10) — and it multiplies per-frame inference cost by roughly the tile count plus one (for the full-frame pass), which competes directly with the real-time on-device latency budget this app needs. The paper's own tradeoff note (compute scales linearly with slice count, memory stays flat) confirms this is a real cost, not a free lunch.

**Effort vs. payoff:** Low effort for the first, decisive test — a pure-Python offline experiment against the existing eval harness and the already-identified failing frames, with zero licence risk and zero training cost, answerable in well under a day. This is the cheapest *detection-accuracy* (not just inference-layer bridging, as `VNTrackObjectRequest` was) experiment this log has found to date. Payoff is genuinely unknown in the one direction that matters: the cited AP gains are from dense aerial small-object benchmarks, not a single handheld clubhead, so a direct test against this project's own failing frames is necessary before trusting the technique at all — but that test is cheap enough that it should be run before investing further in any of the heavier architecture or dataset options already logged. If it does recover camouflage frames, the on-device port is a real but bounded second phase (geometry/NMS code, no new model), gated on a real latency measurement against the app's frame-rate budget, not assumed free.
