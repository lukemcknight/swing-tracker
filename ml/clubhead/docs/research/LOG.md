# Clubhead detector research log

One verified, actionable finding per entry. Newest first. Entries are
off-limits for repeat research runs — don't re-log the same source.

---

## 2026-10-21 — Albumentations `MotionBlur` is commercially usable, but naive use will make the blur problem worse, not better

**What it is:** [Albumentations](https://github.com/albumentations-team/albumentations)
is an open-source image augmentation library with a built-in `MotionBlur`
transform (directional blur kernel, random angle/size). Confirmed by direct
read of the GitHub repo today:

- **License:** MIT, confirmed verbatim on the repo page ("Continue Using
  Albumentations (MIT License)"). **Commercial use is permitted.**
- **MotionBlur exists** and is listed under the library's **pixel-level**
  transforms (confirmed from the repo's transform listing).

**Which failure mode:** Motion blur (failure mode 2) — directly targets the
elongation gap (median labelled box elongation 1.60, p90 3.01, despite the
labeling spec requiring the full blur streak to be boxed).

**Why it matters for this model, including the catch:** Pixel-level
transforms in Albumentations, by the library's own design distinction
(pixel-level vs. spatial-level), modify pixels only — they do **not** resize
or dilate bounding boxes. That means running `MotionBlur` over the existing
training set as-is would smear each already-near-square labeled clubhead
across a wider streak while leaving the box at its original tight size. Per
the labeling spec's own rule ("box the full visible clubhead including its
motion-blur streak"), that is now a *mislabeled* example — it teaches the
model a tight box on a blurred object, the opposite of the stated
convention, and would likely make the small-elongation bias worse, not
better.

To be useful here, `MotionBlur` can't be dropped in as a stock augmentation.
It would need to be paired with a bbox-dilation step keyed to the same
kernel (grow the box along the blur axis by ~half the kernel length) so the
synthetic blur and its label stay consistent with the labeling spec. That
pairing is not something Albumentations does for you — it would be custom
code in the data engine, not a flag you flip.

**Effort vs. payoff:** Low effort to get the library in (MIT, pip-installable,
already has an official Ultralytics/YOLO integration per their docs) — but
near-zero payoff, or negative payoff, if applied naively. Real payoff
requires writing the box-dilation pairing logic ourselves, which is a small
but non-trivial data-engine change (medium effort), and even then this only
produces *synthetic* blur on pixels that were originally sharp — it's not a
substitute for real exposure-length blur from actual indoor/low-light
capture, which is still unmeasured in this model. Treat this as a cheap way
to add blur-diversity to the *existing* 29% own-footage + 54% Roboflow mix,
not as a fix for the indoor-validation gap.

**Related, not independently verified:** Tim Brooks & Jonathan Barron,
"Learning to Synthesize Motion Blur" (CVPR 2019,
https://github.com/timothybrooks/motion-blur) uses frame-interpolation
between consecutive sharp video frames to synthesize physically-grounded
blur rather than a generic kernel — closer to what we'd want since we
already have high-fps phone video and frame-to-frame box positions to
derive true motion direction/magnitude from. Confirmed the repo exists and
is reference code for that paper, but **the repo publishes no license** (no
LICENSE file, nothing stated on the page), so their *code* is not safe to
reuse commercially — only the general frame-blending technique as an idea,
reimplemented from scratch on our own footage, is available to us. Did not
get far enough to verify implementation details (couldn't fetch arxiv.org
or the paper PDF from this environment — DNS/network blocked those
domains while github.com resolved fine). Logging as a lead for a future run
to chase properly, not as a verified finding in its own right.
