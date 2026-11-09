# Clubhead detector research log

One verified, new-to-this-log finding per entry. Nothing here should repeat
an earlier entry. Entries are appended in run order (newest at bottom).

---

## 2026-11-09

**Note on sources:** this run's brief pointed at `ml/clubhead/eval/results/v1-failure-analysis.md`
and `ml/clubhead/docs/data-engine.md` for ground truth. Neither file exists at the
current `HEAD` (detached at `mvp/club-path-replay`) — the repo's clubhead work is
at an earlier CreateML-baseline stage (`ml/clubhead/eval/baseline/createml-v6.md`:
30.4% detection, 80% phantom rate on the 3-clip held-out set), not the
YOLO11n/CoreML/82%-detection state the brief describes. `labeling-spec.md` does
exist and does confirm the "box the full motion-blur streak" convention the brief
cites. I could not re-verify the specific 82%/85.9%/54%-Roboflow/elongation-1.60
figures against repo files, so this entry evaluates the finding on its own merits
rather than against those numbers.

**What it is:** `SoccerSynth-Detection` (Qin, Yeung, Umemoto, Fujii — Nagoya
University / Fujii Lab, 2024), a synthetic data generator and companion dataset
for soccer-player detection that explicitly simulates camera motion blur, random
lighting and textures. Trained/fine-tuned with YOLOv8n and benchmarked against
real datasets (SoccerNet-Tracking, SportsMOT): the paper reports the synthetic
set "matched the performance of real datasets and significantly outperformed
them in images with motion blur," and improved results further when used as a
pre-training set ahead of fine-tuning on real data.

- URL: https://github.com/open-starlab/SoccerSynth-Detection
- Licence (verified by reading `LICENSE.md` in the repo directly): Apache
  License, Version 2.0. Apache-2.0 is permissive and explicitly allows
  commercial use, modification and redistribution, subject to attribution and
  marking changed files (Section 4) and the standard "AS IS" warranty
  disclaimer (Section 7). **Commercial use: permitted.**
- The repo links a Windows data generator and the generated dataset via
  Google Drive; I could not confirm those specific Drive links are still live
  (did not attempt the download), only that the repo, its README claims and
  its licence file are real and current.

**Which failure mode it addresses:** motion blur (primarily), with secondary
relevance to camouflage/low-contrast via its "broad range of random lighting
and textures."

**Why it helps this model specifically:** This is not a golf dataset — it's
evidence that a *synthetic motion-blur generation pipeline* measurably closes
the blur gap for a small, fast-moving sports object detected with a YOLO
model, which is close to our exact situation (YOLO11n, clubhead vs. soccer
player — different object, same blur/lighting problem class). Our own blur
gap is a labeling/sourcing problem, not a modeling one: real footage with
genuinely blurred, labeled clubheads is scarce (median box elongation ~1.6
per the brief), and acquiring more real blurred footage is slow. A generator
that can render a 3D clubhead (or even a simple elongated proxy mesh) swinging
through a scene with simulated camera/object motion blur, varied lighting,
and auto-generated tight bounding boxes would let us manufacture exactly the
class of examples we're short on, at whatever volume and blur-severity
distribution we choose, with ground-truth boxes that don't depend on a human
annotator's eye for "where does the streak end."

**Effort vs. payoff:** High effort, potentially high payoff. The tool itself
doesn't hand us golf data — it's a precedent and possibly an adaptable
pipeline (engine/blur method aren't documented on the repo page, so the
generator internals would need to be inspected directly, and a golf club head
+ swing motion would need to be modeled from scratch; soccer players are a
very different asset to author than a thin elongated metal/graphite object
whose blur streak geometry is the entire point). Realistic path: treat this
as a reference implementation to study (how they generate the blur kernel,
how they randomize lighting/texture, how they auto-label) rather than a
drop-in tool, then build a minimal custom synthetic-swing renderer scoped
narrowly to clubhead + blur streak, which is a much smaller asset-authoring
problem than full synthetic soccer scenes. Not a quick win, but it's the
first concrete, licence-clean technique found so far that directly targets
manufacturing the motion-blur examples the training set is short on, rather
than just hoping for more real footage.

---

## 2026-11-09 (run 2)

**Note on sources:** same discrepancy as the entry above — `v1-failure-analysis.md`
and `data-engine.md` do not exist at the current `HEAD`. The "median elongation
1.60" figure below is taken from this run's brief, not re-derived from a repo
file; everything else in this entry was computed directly from data I
downloaded and inspected in this session.

**What it is:** `clubhead-db` (PyPI, v1.0.1, published 2025-06-30) — a
hand-annotated golf clubhead tracking dataset shipped as a pip package. It
bundles metadata for 67 source clips (19 YouTube, 12 YouTube Shorts, 36
Reddit; down-the-line view; pros and amateurs) and a 10,177-row
`annotations.parquet` of YOLO-format clubhead boxes. 40 of the 67 clips are
tagged `fullspeed` (not slow-motion) in the metadata, i.e. footage where real
motion blur is physically present.

- URL: https://pypi.org/project/clubhead-db/
- Licence: `CC-BY-NC-4.0`, read directly from the PyPI JSON API's
  machine-readable `info.license` field (not inferred from a badge).
  **Commercial use: NOT permitted.**
- Verification performed, not just page-reading: ran `pip download
  clubhead-db` (succeeded, produced a real 285 KB wheel), unzipped it, and
  confirmed `metadata.csv` and `annotations.parquet` contain real data, not
  placeholders. The package's listed GitHub homepage,
  `github.com/unknownauthor23/ClubheadDB`, returns a 404; the author account
  exists but its one public repo (`hilospot`) is unrelated, so the project
  has no traceable paper/canonical repo — the PyPI wheel is the only
  verifiable source, and the frame-download step depends on `yt-dlp`
  pulling the 67 source videos from YouTube/Reddit URLs in `metadata.csv`,
  which will rot over time independent of the dataset's own licence.
- Loaded `annotations.parquet` and computed box elongation
  (`max(w,h)/min(w,h)`) split by the metadata's `playback_speed` field:

  | subset | n | median elong | p90 | p95 |
  |---|---|---|---|---|
  | slowmo | 3,279 | 1.67 | 2.96 | 3.22 |
  | fullspeed | 6,126 | 1.95 | 3.43 | 3.70 |
  | all | 9,405 | 1.83 | 3.23 | 3.63 |

**Which failure mode it addresses:** motion blur.

**Why it helps this model specifically:** This is an independent dataset,
labeled by different annotators under a similar "box the visible clubhead"
instruction, and its `fullspeed` subset alone has a higher median box
elongation (1.95) than this run's brief states for our *entire* training
set (1.60) — including our own fast-swing frames. That's evidence for a
labeling-consistency explanation alongside (not instead of) a
data-scarcity explanation: independent annotators given a similar
blur-boxing rule drew visibly more elongated boxes on real-speed footage
than appears in our own training labels. Before spending effort sourcing
more blurred footage (real or synthetic, cf. the SoccerSynth-Detection
entry above), it is cheap to first spot-check a sample of this project's
own fastest-swing labeled frames against the `labeling-spec.md` rule
("box the full visible clubhead including its motion-blur streak") to see
whether annotators are under-boxing blur that is already present in the
source video, rather than blur being absent from the footage.

**Can we use the data itself?** No — the NC licence rules out training the
shipping model on it. The payoff here is the elongation comparison above,
which only needed the parquet's numeric box metadata (not the licensed
video frames) and is already captured in this entry; there is no reason to
build the actual video dataset given the licence blocks production use.

**Effort vs. payoff:** Low effort (one `pip download` + unzip + pandas,
about 20 minutes) for a diagnostic finding rather than a fix. It doesn't
hand us usable training data, but it reframes the blur gap as possibly
partly a labeling-QA issue, which is a much cheaper thing to check (audit
~20-30 existing frames) than sourcing or synthesizing new blurred footage.
