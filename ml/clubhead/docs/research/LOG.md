# Clubhead detector research log

One verified, new finding per run. Entries are append-only; never repeat
something already logged below.

**Note (2026-11-16):** The task brief for this log references
`ml/clubhead/eval/results/v1-failure-analysis.md` and
`ml/clubhead/docs/data-engine.md`, and describes a trained YOLO11n model with
measured 82%/77%/85.9% detection numbers and a 54%/29% Roboflow/own-footage
split. None of those files or numbers exist in this repo as of this run — the
actual state per `docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`
is Phase 0/1: a CreateML baseline (mAP@IoU50 0.63, varied-IoU 0.18) with YOLO11n
not yet trained. Flagging this so future runs (and whoever reads this log)
know the specific stats in the task brief are unverified against the current
repo, not confirmed ground truth. Proceeding with the research task on its own
merits since it doesn't depend on those stats being true.

---

## 2026-11-16 — GolfPose (ICPR 2024/2025): golfer+club pose dataset and models, commercial-use license

**What it is:** `GolfPose: From Regular Posture to Golf Swing Posture`
(Lee, Zhang, Wu, Tseng — ICPR 2024, published Springer 2025), from NYCU's PAIR
Lab. A pipeline of (1) a golfer+club object detector (Faster R-CNN / YOLOX-s,
2-class "golfer, club", AP 0.916–0.984), (2) a 2D pose model that treats the
**club as its own keypoint-estimation target** (HRNet-w48 / ViTPose-H / DEKR,
AP 0.857–0.870 on the club class), and (3) a 2D→3D lifting model (MixSTE).
Ships a "GolfSwing" dataset (broadcast-style clips, split into groups
G1–G6) plus pretrained checkpoints for every stage.

- **Code:** https://github.com/MingHanLee/GolfPose (verified via
  `raw.githubusercontent.com/MingHanLee/GolfPose/main/README.md` — fetched and
  read directly, not just search snippets)
- **Paper:** https://minghanlee.github.io/papers/ICPR_2024_GolfPose.pdf
  (link found via search; could not fetch directly — `minghanlee.github.io`
  is outside this session's allowed network hosts. Citation confirms venue/
  authors/year via the repo's own `## Citation` block.)
- **Dataset access:** gated — must email `mhlee.cs09@nycu.edu.tw` for
  authorization before the download link (`gofile.me/...`) works. Could not
  complete the download myself (no email auth), so I have NOT verified the
  dataset is actually downloadable today, only that the request mechanism and
  checkpoints are real and the repo is live.
- **Licence (verbatim, from the repo's `LICENSE` file, a custom EULA):**
  > "**Commercialization of Derived Models**: You are permitted to integrate
  > the machine learning models trained on this Dataset into commercial
  > products or services."
  > "**No Redistribution**: You may not publish, share, distribute, or make
  > the raw Dataset ... available to any third party ..."
  > "**No Resale** ... **No Re-identification** ..."
  The README echoes this directly: *"You are allowed to use the dataset for
  commercial model training, but strictly prohibited from redistributing or
  reselling the raw data."*
  **Commercial use: permitted for models trained on it; the raw dataset
  itself may not be redistributed or resold.** This is a real, specific grant
  — stronger than "research only" — but confirm it still applies if the repo
  or EULA changes before anyone actually pulls the data.

**Which failure mode this addresses:** Neither camouflage nor motion blur
directly — be honest about this. The footage looks like broadcast/instructional
golf video (similar shooting conditions to our existing well-lit, fast-shutter
3-clip test set), so there's no specific reason to expect it contains the
indoor/low-light/blurred frames this project actually lacks. What it *does*
offer:
1. **A second source of genuinely independent club-location data** (different
   players, cameras, backgrounds) to dilute the 54%-Roboflow/29%-own-footage
   imbalance the brief describes — general generalization, not either target
   failure mode specifically.
2. **A working precedent for keypoint-based club localization instead of
   box regression.** The v2 design doc already flags "keypoint / pose model"
   as an out-of-scope Phase 2 idea "if measured wobble persists after boxes
   are tight." GolfPose is existence proof that treating the clubhead as a
   pose keypoint (rather than a box with a motion-blur streak baked into its
   edges) is workable and gets reasonable AP (0.857–0.870) — relevant to BOTH
   failure modes, since a keypoint has no "elongation" ambiguity under blur,
   and a temporal pose model naturally uses motion context the way a static
   box detector can't, which is the same lever the camouflage caveat calls
   for ("motion is what separates a moving clubhead from static foliage").

**Effort vs. payoff:** Low effort to investigate further (email the authors,
inspect a few sample frames) but likely **low near-term payoff** for this
project specifically — adopting it would mean changing the detection paradigm
(box → keypoint), which is a bigger architecture change than the current
Phase 1 YOLO11n box detector, and the dataset's shooting conditions probably
don't fill the blur/low-light gap this project actually needs filled. Worth
revisiting only if/when the project seriously considers the keypoint-model
Phase 2 idea already named in the design doc — not an action item now.
