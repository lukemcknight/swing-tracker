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
