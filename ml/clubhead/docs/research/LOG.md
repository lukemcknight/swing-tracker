# Clubhead detector — research log

One verified, non-repeated finding per run. Entries are append-only; nothing
here should be re-researched in a later run. Each run searches ONE area
(dataset-with-licence, blur-augmentation/architecture, camouflage/temporal
detection, golf-specific tracking, or synthetic-data technique) and rotates
across the list so consecutive runs don't overlap.

---

## 2026-11-21 — SHOP pipeline + dataset (small handheld object detection in blurry video)

**Area searched:** motion blur (augmentation techniques / blur-robust
architectures / deblurring preprocessing).

**What it is:** "SHOP: A Deep Learning Based Pipeline for near Real-Time
Detection of Small Handheld Objects Present in Blurry Video" (Ganguly et al.,
IEEE SoutheastCon 2022). Three-stage pipeline: (1) deblur the frame with
DeblurGANv2, (2) run a single-stage detector (YOLOv5) on the deblurred frame,
(3) filter detections using pose-derived hand regions-of-interest. They also
built a companion dataset: a handheld-object subset of MS COCO, with a
**blurry test set made by applying synthetic rotational + linear motion blur**
to the sharp images, plus published code for generating that blur.

- Paper: https://arxiv.org/abs/2203.15228 (also https://ieeexplore.ieee.org/document/9763890)
- Code + dataset: https://github.com/spider-sense/SHOP
- Deblur backbone used: https://github.com/VITA-Group/DeblurGANv2

**Verification done:** Confirmed the paper exists and is indexed at both
arXiv and IEEE Xplore (title/author/venue match across both). Fetched the
GitHub repo directly — it is public, has 41 commits, a linked dataset release
(tag `0.1.0`), and folders for DeblurGANv2/YOLOv5/pose estimation, matching
the paper's description. Fetched the DeblurGANv2 repo directly — public,
pretrained weights (`fpn_inception.h5`, `fpn_mobilenet.h5`) are linked and
downloadable via Google Drive; no mobile/CoreML export is provided.

**Licence — read directly, not assumed:** The `spider-sense/SHOP` repo has
**no LICENSE file**. Under GitHub's own default, that means all rights
reserved — the public code and dataset are **not** licensed for reuse,
commercial or otherwise, despite being visible. The paper's arXiv listing
grants arXiv a non-exclusive distribution licence; it does **not** grant
downstream reuse rights to the dataset. **Verdict: do not pull their code or
dataset into this project.** I could not confirm DeblurGANv2's exact licence
text (the repo links a "License" file but the page didn't surface its
contents) — treat it as unverified and re-check before using the weights.

**Which failure mode:** Motion blur (primary). Marginally relevant to
camouflage only insofar as "deblur before detect" changes the precision/
recall tradeoff in general, but their failure mode (blur hiding an
otherwise-camouflage-free handheld object) isn't the same as ours (zero
candidates on a *sharp* frame), so don't over-read the overlap.

**Why it helps this model specifically:** It doesn't give us data or code we
can use directly (licence blocks that), but it gives two validated, reusable
*ideas*, independent of their implementation:

1. Synthesizing linear/rotational motion blur on top of sharp labelled crops
   is a published, working technique for this exact problem class (small,
   fast/handheld object, blur under-represented in the source data). That's
   directly applicable to the gap this repo's data-engine already
   identified: median labelled-box elongation is only 1.60 even though the
   labeling spec calls for boxing the full blur streak — i.e. real blurred
   examples are scarce, not just unlabeled. A synthetic directional-blur
   augmentation step (e.g. `cv2.filter2D` with a line kernel, angle sampled
   along the local swing-plane direction, length parameterized) applied to
   existing sharp clubhead crops at train time would manufacture exactly the
   missing elongated/blurred examples, without new footage or new labels —
   the box only needs to grow by a known amount along the blur direction.
2. Their ablation (deblur-then-detect cut false positives ~70% at a cost of
   ~17% true positives in their strongest configuration) is evidence that a
   deblur pre-filter measurably shifts the precision/recall operating point
   for small blurred objects — useful context if a deblur-preprocessing
   experiment is tried later, but see the effort note below before going
   that route.

**Effort vs. payoff:**
- **Synthetic blur augmentation (idea only, reimplemented in-house): LOW
  effort, plausible payoff.** No licensing issue (we'd write our own kernel
  code, not use theirs), no new capture needed, directly targets the
  measured elongation gap. Worth prototyping against the existing training
  crops in a day or two and checking whether augmented-blur training moves
  the failing-frame rate on blur-heavy footage (once such footage exists to
  test against — see the indoor/simulator caveat already on file).
- **DeblurGANv2 as an on-device pre-filter: HIGH effort, LOW payoff for this
  project.** It's a second heavy model bolted in front of an already
  size/latency-constrained on-device CoreML pipeline; their reported speed
  and quality numbers are desktop/GoPro-benchmark, not mobile; no CoreML or
  mobile export exists for it; and its own licence is still unverified. Not
  recommended to pursue further right now.
