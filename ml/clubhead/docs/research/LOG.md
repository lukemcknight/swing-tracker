# Clubhead detector research log

Append-only. Each entry is one verified, new thing. Don't repeat an entry
already logged below — treat everything here as off-limits for future runs.

---

## 2026-11-28 — BlurBall: blur-centroid labeling + joint blur-orientation/length prediction (table tennis)

**What it is.** A 2025 paper and open-source repo from the University of
Tübingen's Cognitive Systems group: a table-tennis ball detector that (a)
relabels ball position at the **center of the motion-blur streak** (not the
leading edge, which was the prior convention in racket-sports ball tracking),
and (b) adds blur **orientation** and **length/extent** as explicit auxiliary
regression targets the network predicts jointly with position, instead of
letting the blur be absorbed implicitly into a bigger bounding box.

- Paper: "BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis
  Ball Tracking," Gossard, Radovic, Ziegler, Zell (arXiv 2509.18387, CVPRW
  2026). I could not fetch arxiv.org from this sandbox (DNS blocked by the
  network policy — confirmed, not assumed) so the quantitative
  accuracy-improvement numbers in the paper itself are **unverified** by me.
- Code + project page: https://github.com/cogsys-tuebingen/blurball
  (also https://cogsys-tuebingen.github.io/blurball, same content).
- Dataset download (verified link resolves today, NextCloud, University of
  Tübingen): https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7

**Licence.**
- Code: `LICENSE.md` in the repo is the MIT License (verified by fetching the
  raw file). Verbatim: "Permission is hereby granted, free of charge, to any
  person obtaining a copy of this software and associated documentation
  files (the "Software"), to deal in the Software without restriction,
  including without limitation the rights to use, copy, modify, merge,
  publish, distribute, sublicense, and/or sell copies of the Software..."
  **Commercial use of the code/method is permitted.** Note: the copyright
  header reads "Copyright (c) 2023 NTT Communications Corporation" — this
  repo builds on NTT's WASB-SBDT ball-tracking codebase, so the MIT grant
  traces back to that upstream project, not solely to the BlurBall authors.
- Dataset (video + annotations): **no license file or statement found** in
  the README or repo for the dataset itself. Earlier search results indicate
  the source videos were pulled from "publicly available online sources
  under a Creative Commons license" to sidestep copyright, but the specific
  CC variant (BY vs BY-NC vs BY-NC-ND) is not stated anywhere I could read.
  **Do not treat the raw BlurBall dataset as cleared for commercial use** —
  it is unverified, not confirmed-restricted. The downloadable annotation
  format and convention, however, are not copyrightable facts and are free
  to reuse.

**Which failure mode: motion blur.**

**Why it matters for this model specifically.** This is the closest
published analog to our exact problem: a small, fast-moving object in sport
that becomes a directional blur streak on the frames that matter most. Our
own labeling-spec.md already independently adopted BlurBall's core idea —
"box the full visible clubhead including its motion-blur streak" — but our
measured elongation stats (median 1.60, p90 3.01, i.e. mostly near-square
boxes) show we have very few real blurred examples to learn from, and even
where we do, the detector gets *only* a bigger/rotated axis-aligned box as
signal — it has no explicit target that says "this shape is blur, pointing
this way, this long." BlurBall's reframe — predict blur orientation + extent
as its own head, and anchor position at the blur centroid rather than
wherever an annotator happened to center a loose box — is a concrete,
testable architecture change for exactly the regime (fast downswing/impact)
where our model currently drops out, independent of whether we ever get more
blurred training frames. It would also give us, for free, a usable
per-frame "how blurred is this" signal that could drive exposure/shutter
guidance in the capture UI.

**Effort vs. payoff.**
- Low effort, do first: audit our existing blur-streak-labeled boxes against
  the blur-centroid convention (are our annotators centering on the streak's
  midpoint, or anchoring to one end?) — this is a labeling QA pass on data
  we already have, roughly free.
- Medium effort: add blur-orientation/length as auxiliary regression outputs
  on top of YOLO11n's detection head (not a stock Ultralytics config knob —
  needs a custom head and loss term, plausibly 1-2 days of engineering plus
  retraining).
- Payoff is plausible, not proven: the mechanism is sound and the domain
  match is unusually tight, but I could not read the paper's own ablation
  numbers (arxiv blocked), so there is no verified quantitative evidence yet
  that this convention change moves detection accuracy versus just having
  more blurred examples in the box-the-whole-streak convention we already
  use. Treat as a promising architecture experiment, not a confirmed win.

---

## 2026-11-28 — `albumentations.MotionBlur` (MIT) as a cheap synthetic source of motion-blur streaks

**What it is:** `A.MotionBlur`, a transform in the original `albumentations`
Python package. It convolves the image with a randomly-sized, randomly-angled
line-shaped kernel to synthesize a directional motion-blur streak — the
stated purpose is simulating camera or object motion during capture.
Parameters in the stable (MIT) release: `blur_limit` (kernel size range,
default up to 7px) and `allow_shifted` (whether the kernel can be off-center,
for asymmetric streaks), applied with probability `p`. A newer, more
expressive API (`blur_range`, `angle_range`, `direction_range`) exists but
only in the AGPL/commercial successor described below — it is not available
under MIT.

**URL:**
- Package / license: https://github.com/albumentations-team/albumentations (MIT)
- Transform docs: https://albumentations.ai/explore/transform/MotionBlur/

**Licence:** The original `albumentations` repo and its PyPI releases
(`pip install albumentations`, last released ~2.0.8) are MIT-licensed —
commercial use is permitted, no attribution requirement beyond the standard
MIT notice. **Important caveat:** this package is frozen/unmaintained; all
new development has moved to a separate, differently-licensed package,
`albumentationsx` (a "drop-in replacement"), which is dual-licensed
AGPL-3.0-only / paid commercial
(https://github.com/albumentations-team/AlbumentationsX,
https://albumentations.ai/blog/2025/01-albumentationsx-dual-licensing/). The
AGPL option is not safe to use in this app without either open-sourcing
linked code or buying the commercial licence — **do not install
`albumentationsx` for this project; use the MIT `albumentations` package
specifically.** This also means the fancier `angle_range`/`direction_range`
API is off-limits unless we pay for it; only the older `blur_limit`/
`allow_shifted` API is free to use.

**Which failure mode: motion blur** (does not touch camouflage).

**Why it matters for this model specifically:** our own stats say the
training set is short of genuinely blurred clubheads — median labelled box
elongation is 1.60 (p90 3.01), i.e. near-square, despite the labeling spec
explicitly requiring the full motion-blur streak to be boxed (same gap the
BlurBall entry above addresses from the labeling/architecture side). There is
currently no mechanism in this repo (checked: no augmentation config exists
under `ml/clubhead/`) that synthesizes directional blur — nothing here
manufactures the high-elongation examples the spec calls for. `MotionBlur` is
a direct, free way to synthesize exactly that: take existing sharp,
correctly-boxed clubhead frames (most of the dataset) and apply a directional
kernel at training time, pushing some fraction of training examples toward
realistic streak shapes without collecting a single new clip. Ultralytics'
YOLO trainer already auto-applies a *different*, unrelated Albumentations
transform by default (a generic isotropic `Blur`/`MedianBlur`/`CLAHE` set at
p=0.01 when `albumentations` is installed) — that default does not produce
directional streaks and is not a substitute for this. Note this is
complementary to, not a replacement for, the BlurBall entry above: BlurBall
changes what the model predicts and how boxes are anchored; this changes what
the model is trained to see more of. They could be combined.

**Effort vs. payoff.**
- Low effort: `pip install albumentations` (not `albumentationsx`), add
  `A.MotionBlur(blur_limit=7, allow_shifted=True, p=~0.3–0.5)` to the training
  augmentation pipeline or an offline augmentation pass — hours, not days,
  and worth trying before any new data collection.
- Payoff is **unproven for this use case**: I found no published benchmark
  specifically showing synthetic `MotionBlur`-style augmentation closes a
  real-world detection gap for thin, fast, small objects like a golf club
  head (the closest related work, Sayed & Brostow CVPR 2020 on blur-robust
  detection, evaluates on COCO-scale objects, not clubhead-sized ones).
  Synthetic motion blur is also not a perfect stand-in for real sensor
  blur — it won't reproduce rolling-shutter artifacts, exposure-dependent
  noise, or compression interacting with blur the way a genuinely
  long-exposure indoor frame would. Treat as a cheap, quick first experiment
  (train with vs. without, measure recall specifically on the existing
  high-elongation subset of labels) rather than a fix — it does not reduce
  the need for real low-shutter/indoor footage, and it does nothing for the
  camouflage failure mode.
