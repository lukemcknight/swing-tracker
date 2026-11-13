# Clubhead detector research log

Append-only. Each entry is one verified finding. Do not repeat anything
already logged below — check before researching.

---

## 2026-11-13

**What it is:** A verified, concrete *technique* (not a dataset) for synthesizing
realistic motion-blur training examples, plus a verified warning against the
"obvious" fix.

First, the trap: the standard off-the-shelf tool anyone would reach for is
Albumentations' `MotionBlur` transform
(https://github.com/albumentations-team/albumentations/blob/main/albumentations/augmentations/blur/transforms.py).
Read its source directly (`MotionBlur.apply()` calls
`fpixel.convolve(img, kernel=kernel)` with one directional kernel built by
`create_motion_kernel`). It is an `ImageOnlyTransform`: it convolves the
**entire frame** with one kernel, does not distinguish foreground object from
background, and does not touch bounding boxes. Same story for imgaug's and
Kornia's motion-blur augmenters (all whole-image, uniform-kernel, no bbox
interaction) — confirmed via their public docs/source. Turning this on in a
YOLO training config would blur the whole frame uniformly (grass, body,
background all blur too), which does not match real clubhead-swing footage
(fast-moving head, comparatively sharp background/body), and would leave
labeled boxes at their current near-square size (median elongation 1.60) —
it does not address the measured elongation gap at all, since the box itself
never grows to cover a streak that isn't there.

The matching real technique is **local/compositing motion blur**: build a
sub-frame trajectory for the object only, render N intermediate positions,
average them, and composite the resulting blurred object crop onto an
otherwise-sharp background — ground-truth box = the full compositing extent,
which is exactly what `docs/labeling-spec.md`'s motion-blur rule already
specifies. This is published and well-described: **LODE: Deep Local
Deblurring and A New Benchmark** (Zhang et al., Tsinghua/OPPO/Aberystwyth,
arXiv:2109.09149, https://arxiv.org/abs/2109.09149; mirrors verified at
https://arxiv.org/pdf/2109.09149 and https://ar5iv.labs.arxiv.org/html/2109.09149).
It builds its LOcal-DEblur dataset by exactly this compositing method
(multi-step trajectory average of a moving object crop over a sharp
background), explicitly to separate "local, object-motion" blur from global
camera-shake blur — which is the same distinction this project's own data is
missing (p90 elongation only 3.01; the labeling spec calls for boxing full
streaks but graders rarely see one).

**Caveat on reuse:** LODE's own dataset/code release status could not be
confirmed — the 2021 arXiv abstract says the authors "will publish" the data
and code, and no GitHub/release page turned up in search. **Do not depend on
their asset being downloadable.** The value here is the *method*, which is
cheap to reimplement from this project's own labeled data: take existing
near-square clubhead crops (already have clean masks via the bbox), run a
short synthetic trajectory (3-7 sub-steps, speed calibrated against this
project's own measured impact-frame displacement stats, which already exist
per the task brief's 54.8% impact-speed figure) over an unrelated sharp
background frame from the existing dataset, average, and derive the new bbox
from the swept extent. No external license question at all — it only touches
in-house footage already cleared for training.

**Licence:** N/A (method only; no third-party data or code asset is being
imported). Commercial use: unaffected, since nothing external is redistributed.

**Which failure mode:** Motion blur, directly. Does not touch camouflage.

**Why it helps this model specifically:** The repo's own measured gap is
precise — median box elongation 1.60 vs. a labeling spec that calls for full
streaks — i.e. the training distribution is close to blur-free even though
the spec asks for blur boxes. This technique manufactures correctly-elongated,
correctly-labeled synthetic examples without new footage or licensing risk,
directly targeting that exact gap. It also sidesteps the misleading inter-frame
"54.8% impact-speed" stat (displacement, not blur) by using it instead as a
calibration input for trajectory speed.

**Effort vs. payoff:** Medium-high effort, not a config flag. Building a
correct compositor (trajectory synthesis, alpha-aware edge blending so the
streak doesn't have hard edges, bbox-extent computation, avoiding an
obvious sim2real seam) is a multi-day task, and it needs validation that
models trained on it actually improve on real blurred frames (none of which
exist in any current test set — the camouflage-only outdoor test set has a
fast shutter and the indoor set is quarantined, so this can't even be
evaluated until real blurred eval data exists). High payoff *if* done well,
because it is the only verified technique found so far that attacks the
elongation gap directly rather than through whole-image blur that doesn't
match the failure geometry. Recommend pairing with a small real-footage
collection pass (indoor/evening) before fully trusting results, since no
synthetic blur technique can be validated without at least some real blurred
ground truth to check against.
