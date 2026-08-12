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
