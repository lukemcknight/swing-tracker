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
