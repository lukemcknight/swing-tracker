# Clubhead detector — research log

Append-only. Each run adds exactly one verified entry. Entries already here
are off-limits for future runs — do not repeat them.

---

## 2026-10-22 — ClubheadDB (web-sourced golf clubhead dataset)

**What:** `clubhead-db`, a PyPI-distributed, reproducible dataset for golf
clubhead detection. Not a bundle of images — it ships metadata
(`metadata/metadata.csv`: source video URLs + timestamps) and pre-made
annotations (`annotations/annotations.parquet`, YOLO-format boxes), plus a
`clubhead-build` CLI that uses `yt-dlp` + `ffmpeg` to re-download the source
clips and re-extract the exact frame set. Contents, per the package's PyPI
page: 10,180 hand-annotated "down-the-line" frames (one box per frame, zero
for null/negative examples), 67+ unique golfers, split 47/10/10 train/val/test
by video, skill mix Pros 42% / Amateurs 58%, club mix Irons 70% / Drivers 27%
/ Woods 3%. Source composition: YouTube 46%, Reddit 54% — i.e. casual,
web-sourced public footage, not studio production.

**URL:** https://pypi.org/project/clubhead-db/ (install via `pip install
clubhead-db`). Fetched and read directly this run; the page, the install
command, and the described build pipeline are real and present today. Could
not find an accompanying GitHub repo in search — PyPI is the only confirmed
source.

**Licence:** The page states the licence as `CC-BY-NC-4.0` (Creative Commons
Attribution-NonCommercial 4.0) — verbatim identifier as shown on the page; I
could not reach creativecommons.org this run to pull the full legal-code text
verbatim, but CC BY-NC's NonCommercial clause is unambiguous and standard:
it forbids use "primarily intended for or directed towards commercial
advantage or monetary compensation." **Commercial use is NOT permitted.**
This is a research-only dataset for this product. Do not train the shipped
on-device model on it, and do not use it to tune/validate a commercial
release without legal sign-off — "noncommercial" licensing on an eval set
used to justify a commercial model's quality claims is a genuine gray area,
not a loophole.

**Failure mode addressed:** Camouflage, primarily. The current training mix
is ~54% curated third-party Roboflow imagery and ~29% the app's own phone
footage — both comparatively clean and controlled. ClubheadDB's casual
YouTube/Reddit sourcing (67+ golfers, varied clothing/backgrounds/clubs,
no studio lighting) is a closer match to the "dark clubhead against dark
clothing or cluttered foliage" camouflage scenario than anything already in
the mix. Motion blur: **unconfirmed** — the page gives no information on
shutter speed, frame rate, or blur content, and down-the-line swing-analysis
clips posted publicly skew toward higher-fps/slow-mo capture intended to be
legible, which would work against blur. I did not find evidence either way
and am not claiming this helps blur.

**Why it matters for this model specifically:** It's a free, independent,
already-annotated source of diverse real-world clubhead appearances at the
exact low-effort end (pip install, one CLI command) of a problem — getting
more non-own-swing, non-Roboflow footage — that `eval/test_set_sources.md`
already flags as the top data-engine priority. It would be useful as a
**held-out evaluation set** to stress-test camouflage generalization beyond
the current 3-clip outdoor test set, which is itself flagged as too narrow.

**Effort vs. payoff:** Low effort to obtain and inspect (minutes). Low
near-term payoff for the shipped model because of the NC licence — it can't
go into the training set or be used to validate release-quality claims
without legal review. Moderate payoff as a free, quick way to *diagnose*
whether camouflage failures generalize beyond the current tiny test set,
which is cheap to do and informs whether camouflage work is still worth
prioritizing over motion blur. Recommend: use for offline diagnostic eval
only, flag to legal/product before touching training or official metrics.

---

## 2026-10-22 — Temporal/tracklet consistency for small-object detection (WASB-SBDT)

**Rotation area:** techniques for detecting small, low-contrast, or camouflaged
objects in video (multi-frame/temporal methods).

**What it is:** "Widely Applicable Strong Baseline for Sports Ball Detection
and Tracking" (WASB), Tarashima et al., BMVC 2023 (arXiv:2311.05237). It's a
general small-fast-ball detection/tracking baseline (soccer, tennis,
badminton, volleyball, basketball) built on an HRNet-style heatmap detector,
with a tracker/inference stage that enforces temporal consistency across
frames (a motion model that links detections into tracklets across time and
suppresses spurious single-frame false positives/negatives). Code and
pretrained weights: https://github.com/nttcom/WASB-SBDT

**Licence (verified verbatim from `LICENSE.md` on the repo's `main` branch,
fetched directly):**

> MIT License
>
> Copyright (c) 2023 NTT Communications Corporation
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions: ...

MIT — **commercial use permitted.**

**Which failure mode it addresses:** primarily **camouflage**. The core idea —
don't decide per-frame in isolation; carry a motion/tracklet model across
frames so a frame with zero (or low-confidence) single-frame detections can
still be resolved from where the object was and how it was moving a moment
before — is exactly the lever the camouflage failure needs: our 0.05-confidence
floor still produces zero candidates on some sharp, well-lit frames because
the clubhead is genuinely camouflaged against clothing/foliage in that single
frame; appearance alone can't win those, but motion across frames can. It's a
secondary help for **motion blur** too (tracking a blurred streak frame-to-frame
is an easier problem than classifying it from scratch each frame), but that's
not the paper's focus and isn't verified here.

**Why it's relevant to this model specifically:** our detector is run frame-by-
frame on-device with no temporal state — the eval harness and (as far as
README.md describes) the CoreML inference path treat each frame
independently. WASB's inference-time tracklet/motion-consistency layer is a
*post-processing* addition, not a retrain — it sits on top of an existing
per-frame detector's heatmap/box outputs and uses the frame sequence the app
already has (it's a video, not isolated images). That makes it one of the few
techniques here that could plausibly move the "~82% per-frame → ~77% through
swing" gap without touching the existing YOLO11n weights at all.

**Caveats / what I could NOT verify:** network access in this environment
could reach github.com/raw.githubusercontent.com but not arxiv.org,
paperswithcode.com, huggingface.co, or project pages — so I read the repo's
README.md and LICENSE.md directly, but not the paper itself. The repo and
README do not describe the motion-model/tracklet algorithm in implementation
detail (that's in the paper, which I couldn't fetch), and the repo contains
no golf or clubhead data — it's a generalist ball-sports baseline, not
something to import as-is. My description of the temporal-consistency
mechanism above is reconstructed from paper abstracts/search summaries, not
read first-hand from the paper text, so treat the *existence, licence, and
code* as verified, and the *algorithmic detail* as secondhand until someone
reads arXiv:2311.05237 directly.

**Effort vs. payoff:** Medium-high effort, uncertain payoff. This is not a
drop-in fix — it requires designing and building a lightweight temporal
post-processor (e.g., Kalman/optical-flow-based tracklet linking across the
existing per-frame YOLO detections, with a fallback re-detection window when
confidence craters) and validating it doesn't regress latency on-device. No
golf-specific prior art exists to lean on. But it's the only candidate so far
that attacks the camouflage failure mode without needing more labeled data or
a model swap, and it composes with everything else (better data, better
labels) rather than competing with it. Recommend a small time-boxed spike —
run WASB's released pretrained tracker logic (not retrained, just the
tracklet-consistency math) over our existing per-frame YOLO confidence maps on
the 3-clip outdoor failing set to see if any zero-candidate frames become
recoverable — before committing to a full implementation.
