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
