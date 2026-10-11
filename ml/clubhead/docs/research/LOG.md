# Clubhead detector — research log

Append-only. One verified entry per research run. Entries are never repeated
or revisited by later runs — each run picks a new angle.

**Note on provenance (2026-10-11):** the research brief for this log describes
a later repo state (YOLO11n trained, CoreML-exported, 82%/77%/85.9% eval
numbers, a `v1-failure-analysis.md` and `data-engine.md`) that does not yet
exist in this checkout. The actual state at the time of this first entry is
Phase 0 of `docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`:
only the CreateML v6 baseline is captured (30.4% detection rate on the 3-clip
held-out test set); YOLO11n training hasn't happened yet. This entry is
written against the v2 design doc and the labeling spec, which do exist and
do establish the two failure modes (camouflage, motion-blur under-representation)
as real, intended targets.

---

## 2026-10-11 — Ultralytics' default Albumentations wrapper does not include MotionBlur

**What it is.** The clubhead-detector-v2 design doc commits to "explicit
motion blur augmentation (via Albumentations)" as the fix for missed frames
on the fast, blurred downswing (`docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`,
sections 4 and 7). I checked what Ultralytics' YOLO training pipeline
actually does with Albumentations when no custom transform list is supplied,
by downloading the live PyPI packages and reading the source directly
(not docs or blog posts, which I could not reach from this sandbox — see
caveat below):

- `ultralytics==8.4.175` (current release as of 2026-10-11),
  `ultralytics/data/augment.py`, class `Albumentations.__init__`
  (`transforms: list | None = None`): when `transforms` is `None` it falls
  back to a **hardcoded default list**:
  ```python
  T = [
      A.Blur(p=0.01),
      A.MedianBlur(p=0.01),
      A.ToGray(p=0.01),
      A.CLAHE(p=0.01),
      A.RandomBrightnessContrast(p=0.0),
      A.RandomGamma(p=0.0),
      A.ImageCompression(quality_range=(75, 100), p=0.0),
  ]
  ```
  `A.Blur` and `A.MedianBlur` are **isotropic** kernel blurs (no direction),
  applied with **1% probability**, kernel size 3–7px. There is **no
  `MotionBlur` in the default list at all**, and brightness/contrast/gamma
  jitter are disabled (`p=0.0`) by default too.
- `albumentations==2.0.8`, `albumentations/augmentations/blur/transforms.py`,
  class `MotionBlur(Blur)` — the directional transform that actually exists
  and is the right tool: `blur_limit` (kernel size, default `(3, 7)`),
  `angle_range` (default `(0, 360)`), `direction_range` (default `(-1.0,
  1.0)`, forward/backward bias along the angle), `allow_shifted`, `p`
  (default `0.5`). It is not used anywhere in Ultralytics' defaults.

**URLs:**
- https://pypi.org/project/ultralytics/ (package metadata; source read from
  the downloaded wheel, `ultralytics/data/augment.py` ~line 2164)
- https://pypi.org/project/albumentations/ (package metadata; source read
  from the downloaded wheel, `albumentations/augmentations/blur/transforms.py`
  ~line 191)

**Licence.** `albumentations` is **MIT** (verified from the PyPI package's
`METADATA`: "Copyright (c) 2017 Vladimir Iglovikov, Alexander Buslaev,
Alexander Parinov... Permission is hereby granted, free of charge... to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies...").
Commercial use is permitted without restriction.
`ultralytics` itself is **AGPL-3.0** per its PyPI metadata — already the
project's accepted dependency (decided in the v2 design doc), not new
information, but worth flagging once: AGPL has copyleft obligations distinct
from the MIT-licensed `albumentations` dependency this entry is about, and
Ultralytics sells a commercial/enterprise licence specifically to route
around them. Not this entry's focus; mentioning so it isn't missed.

**Failure mode addressed:** motion blur.

**Why it matters for this model.** The project's stated root cause for the
blur failure mode is that training data is short on genuinely blurred
clubheads (median labelled elongation 1.60). The design doc's planned fix —
"explicit motion blur via Albumentations" — will silently not happen if
`train.py`/`hyp.yaml` (not yet written; this is Phase 0) ends up calling
Ultralytics' Albumentations integration with the default transform list,
since that list has no directional motion-blur transform at all, only a
1%-probability isotropic blur. The team could easily ship a config that
*looks* like it added motion-blur augmentation (because Albumentations is a
listed dependency) while actually adding none. The concrete fix: pass an
explicit `transforms` list through the `augmentations` hyperparameter
(supported directly by `Albumentations.__init__`) that includes
`A.MotionBlur(blur_limit=(15, 41), angle_range=(0, 360), direction_range=(-1.0, 1.0), p=0.5)`
or similar — kernel sizes tuned to match the blur-streak lengths the
labeling spec already asks annotators to box, not the tiny 3–7px default.

**Effort vs. payoff.** Very low effort — this is a few lines in whatever
builds the Albumentations transform list for `train.py`, no new data, no new
infra, and the library is already a planned dependency. Payoff is
directly on the stated root cause of the motion-blur failure mode, but it
only helps once `train.py` exists (Phase 1) — right now it's a "don't repeat
this mistake when you write it" note rather than something to apply today.
It does not address camouflage at all, and it does not by itself prove real
blurred clubheads will be well-represented in validation/eval — it only
fixes the training-time augmentation gap; the eval/test-set blur gap (no
indoor/low-light footage, as the brief's caveat notes) is a separate,
unaddressed problem.

**Verification notes / limitations.** Confirmed by downloading and reading
the actual package source for the current PyPI releases (`pip download
--no-deps`), not by trusting docs or blog summaries. I could not reach
`docs.ultralytics.com`, `arxiv.org`, `link.springer.com`, `ms.copernicus.org`,
or `universe.roboflow.com` from this sandbox (outbound network policy denies
these hosts; `pypi.org`/`files.pythonhosted.org` are allow-listed), which
ruled out several other candidate findings this run (a CC BY 4.0 2026 paper
on YOLOv8 motion-blur augmentation for kart racing I could only verify at the
abstract level via search, and a CC-BY-NC-4.0 10k-frame golf-swing dataset
on PyPI, `clubhead-db`, which is genuinely downloadable today — annotations
and metadata are bundled in the wheel, no dependency on its GitHub repo which
404s — but is explicitly non-commercial and so excluded here).
