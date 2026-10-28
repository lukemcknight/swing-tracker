# Clubhead detector — research log

One verified finding per run. Entries are append-only; once something is
logged it is off-limits for future runs (don't re-research it, do act on it
or explicitly decide not to). Each entry says which failure mode it targets:
**camouflage** (zero detections on dark/cluttered frames) or **motion blur**
(near-square labelled boxes, p50 elongation 1.60, despite a labeling spec
that calls for boxing the full blur streak).

---

## 2026-10-28 — Knowledge-distillation deblurring subnet for small-object detection (YOLOv8-based)

**What it is:** A published technique (not a dataset) for detecting small
objects in motion-blurred frames: a teacher/student knowledge-distillation
setup where a teacher network is trained on sharp images and a student
network — built on a YOLOv8 detector plus a lightweight "deblurring subnet"
with attention modules — is trained on *synthetically motion-blurred*
versions of the same images, with the student distilling features from the
teacher. At inference the student (detector + deblur subnet) runs alone, no
separate deblur-then-detect pipeline or teacher network needed. Reported
gains: +4.3% accuracy on a synthetic-motion-blur version of VisDrone and
+4.6% on synthetic blur applied to the authors' own small-object UAV dataset
("SOD-Dataset").

Paper: Jobaer, Tang, Zhang, Li, Ahmed — "A novel knowledge distillation
framework for enhancing small object detection in blurry environments with
unmanned aerial vehicle-assisted images," *Complex & Intelligent Systems*,
published 2024-12-04. DOI: https://doi.org/10.1007/s40747-024-01676-w
(article page: https://link.springer.com/article/10.1007/s40747-024-01676-w)

**Licence:** Open access under **CC BY-NC-ND 4.0** (confirmed via the
OpenAlex and DOAJ bibliographic records, which both list the article's
license as `cc-by-nc-nd`). That licence governs redistribution of the
*paper itself* (the PDF/text/figures) — it does not cover the underlying
method. Algorithms and techniques described in a paper are not copyrightable
in themselves (only the author's specific expression of them is, and no
patent is mentioned anywhere I could find), so **re-implementing the
described teacher/student distillation + deblur-subnet architecture in our
own code for commercial use is not blocked by this licence.** What *would*
be blocked is copying the paper's text/figures/PDF verbatim, or using any
dataset of theirs that itself carries a non-commercial term (I found no
public release of their SOD-Dataset or code — no GitHub repo under these
author names turned up a code/data release, so this is a from-the-paper
reimplementation, not a drop-in asset).

**Failure mode addressed:** Motion blur (primary). Not camouflage.

**Why it matters for this model specifically:** It is the first concrete,
published technique I found that targets exactly our documented gap: a
detector that has to find a small fast-moving object in a frame where the
object itself is a motion-blur streak, built on the same YOLO lineage
(YOLOv8) we're one version away from (YOLO11n). Two things line up neatly
with our failure analysis:
1. Their synthetic-blur generation step is a reusable recipe for turning our
   existing *sharp*, near-square-labelled clips into genuinely blurred
   training frames — directly answering the data-engine gap that median
   labelled-box elongation is only 1.60 despite the labeling spec asking
   annotators to box the full blur streak. We don't need new footage to try
   this, only a blur-synthesis augmentation step on what we already have.
2. The deblur subnet runs as part of the student network at inference, with
   no separate preprocessing pass — this is the right shape for an
   on-device, CoreML-exportable pipeline, unlike approaches that bolt a full
   deblurring GAN in front of the detector.

**Verification caveats (read before acting):** This session's network
policy blocks direct fetches to link.springer.com, doi.org, api.openalex.org,
proquest.com, and library.kab.ac.ug — I could not pull the primary PDF or
read the full methodology myself. What's above is corroborated by multiple
independent bibliographic sources converging on the same title, authors,
DOI, venue, date, licence, architecture summary, and the two reported
percentage gains (OpenAlex, DOAJ listing, X-MOL abstract, and the Springer
search snippet all agree) — so I'm confident the paper exists, is real, and
says roughly what's summarized above. But I have NOT independently read the
attention-module design, the exact blur-kernel/synthesis parameters, or the
baseline architecture details, and could not locate a code or dataset
release to inspect directly. Treat the architecture summary as "verified to
exist and roughly match this description," not "verified down to
implementation detail."

**Effort vs. payoff:** Medium-high effort, honestly — there is no code or
dataset to drop in; this means reading the full paper (needs network access
to Springer, or a university/library proxy) and reimplementing a
teacher-student training loop, which is meaningfully more complex than our
current single-model training pipeline. Payoff could be high *if* the blur
hypothesis in the failure analysis is right (which is plausible but, per the
caveat in our own failure analysis, unproven — our test set structurally
can't surface blur failures). Recommended next step is cheap and doesn't
require implementing the full distillation architecture: borrow just their
blur-synthesis step, apply it to our existing sharp own-footage clips to
generate elongated-box training examples, and see if that alone moves the
needle before investing in the full teacher/student training setup.

**Also checked and ruled out (saving a future run the trip):** GolfDB
(https://github.com/wmcnally/golfdb, arXiv:1903.06528) — confirmed via its
own GitHub README: "The code in this repository is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License."
Non-commercial, source videos are also uncleared YouTube content. Not usable
for this product. Do not re-log this.
