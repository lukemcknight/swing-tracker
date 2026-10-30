# Clubhead detector research log

Append-only. One verified entry per research run. Entries already here are
off limits for future runs — do not repeat them.

---

## 2026-10-30 — FMODetect: synthetic motion-blur data generation for fast-moving-object detection

**What it is:** FMODetect (Rozumnyi, Matas, Šroubek, Pollefeys, Oswald — ICCV
2021) is a learning-based detector for "fast moving objects": things that
cross a large distance within a single exposure and appear as a heavy,
elongated blur streak rather than a sharp shape. It represents the object as
a truncated distance function to its motion trajectory and is trained
entirely on **synthetically generated blurred data** — a 3D object model is
rendered along a simulated trajectory and composited onto real video
backgrounds to produce the blur streak plus matching ground truth, instead of
relying on hand-labelled real blur footage (which is exactly what we are
short of).

- Paper: https://arxiv.org/abs/2012.08216 (also https://openaccess.thecvf.com/content/ICCV2021/html/Rozumnyi_FMODetect_Robust_Detection_of_Fast_Moving_Objects_ICCV_2021_paper.html)
- Code: https://github.com/rozumden/FMODetect
- Verified by fetching the repo directly: `run.py` (inference), `train.py`/`train.sh`
  (training), `download_datasets.sh`, and a `dataset/generate_dataset.sh`
  script the README points to for building the synthetic training set. Files
  are genuinely present in the repo today, not just referenced in the paper.

**Licence (verbatim, from `LICENSE` at the repo root, fetched directly):**

> MIT License
>
> Copyright (c) 2022 Denys Rozumnyi
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions: [...]
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND [...]

MIT — **commercial use permitted.** (Note: the pretrained model weights are
hosted on a separate polybox link mentioned in the README; I did not fetch or
verify terms on that specific binary. The code and license I verified only
cover the repo itself. The synthetic dataset-generation pipeline likely
depends on a 3D-model submodule (ShapeNet-style assets, used in the authors'
companion DeFMO work) that is cloned separately and would need its own
licence check before reuse — I have not verified that sub-dependency.)

**Which failure mode:** motion blur (failure mode 2) — directly. Not relevant
to camouflage.

**Why it matters for this model specifically:** our own labeling spec
(`docs/labeling-spec.md`) already instructs annotators to box the full
motion-blur streak, but the measured box elongation (median 1.60, p90 3.01)
shows we have almost no genuinely blurred examples to label in the first
place — the gap is upstream of labeling, in the source footage. FMODetect's
contribution that transfers here is not its detector (it outputs a
distance-to-trajectory map, a different output paradigm from our YOLO11n
bounding-box head, so the architecture doesn't drop in) but its **approach**
to the data gap: synthesize the blur instead of waiting to film it. The same
idea applied to us would be: take sharp clubhead crops from our existing
(mostly near-square) labels, simulate a plausible swing trajectory and
per-frame displacement, render the composited blur streak onto real swing
backgrounds, and emit YOLO bounding boxes that already match the full streak
per our own labeling convention. That gives genuinely blurred positive
examples with correct boxes, without needing new footage or re-labeling.

**Effort vs. payoff:** medium-high effort, honestly assessed. Reusing
FMODetect's actual code/model buys little (wrong output format for our
pipeline). What's valuable is reimplementing their *idea* — trajectory
simulation + blur compositing — which is a build-it-yourself job, not a
download. Rough shape of the work: estimate clubhead speed/path per swing
phase (we may already have some of this from swing analysis), render a
directional blur kernel over existing sharp crops, composite onto real
backgrounds, and validate box elongation distributions shift toward
realistic (p90 well above 3.01) before trusting it. Payoff is real if motion
blur is in fact a live failure mode (unverified — see caveat in this log's
originating task about the indoor/low-light blind spot), but this is weeks of
work, not a quick augmentation flag. Worth scoping as a project, not a patch.
