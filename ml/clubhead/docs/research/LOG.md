# Clubhead detector research log

Append-only. Each entry is one verified, new finding. Do not repeat anything
already logged below — check before researching.

---

## 2026-11-08

**Before this entry: a note on repo state.** This research task's brief
describes a trained YOLO11n / CoreML model with measured results
(~82% per-frame detection, ~77% through-swing, an 85.9% selection ceiling,
camouflage producing zero candidate detections, a median label elongation of
1.60) and cites `ml/clubhead/eval/results/v1-failure-analysis.md` and
`ml/clubhead/docs/data-engine.md` as sources. **Neither file exists anywhere
in this repo's history, and no YOLO11n model has been trained yet.** The repo
is at Phase 0 of `docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`:
the only captured baseline is the old CreateML model
(`eval/baseline/createml-v6.md`: 30.4% detection rate, 80% phantom rate, on a
196-frame/3-clip held-out test set). The design spec does independently name
motion blur ("missed frames... especially the fast, motion-blurred
downswing") and generalization/background clutter ("fails on new clips",
"locks onto hands, ball, or background clutter") as real, named concerns, so
researching those two areas is still directionally sound — but the specific
numbers in this task's brief are not grounded in anything in this repo today.
Logging this so whoever reads this log knows the brief and the repo have
drifted apart; worth fixing the scheduled task's prompt or writing the actual
`v1-failure-analysis.md` once v1 is trained.

**Area:** golf swing datasets usable for commercial training (focus: footage
where real motion blur would appear).

**What it is:** `ClubheadDB`, a PyPI package (`pip install clubhead-db`,
v1.0.1, uploaded 2025-06-30) that ships metadata + YOLO-format annotations
for 67 down-the-line golf-swing clips (10,180 annotated frames, 9,405 with a
box) and reconstructs the actual frames on your machine via `yt-dlp` +
`ffmpeg` from the YouTube/Reddit source videos named in its metadata.

- PyPI: https://pypi.org/project/clubhead-db/
- Declared homepage, `https://github.com/unknownauthor23/ClubheadDB`, is
  **dead (404)** as of this check — likely an anonymized repo from a paper
  submission that was never made public or was taken down. This does not
  block the data: the sdist (`clubhead_db-1.0.1.tar.gz`, downloaded and
  inspected directly from PyPI) bundles `metadata.csv` and
  `annotations.parquet` itself, so the annotations and video index are real
  and in hand regardless of the dead homepage. One embedded image path leaks
  the real author's local username (`/Users/sebh/Desktop/ClubheadDB/...`),
  consistent with an anonymized-for-review repo.
- **Licence, verbatim from the package's own README badge:** `License: CC
  BY-NC 4.0`. CC BY-NC 4.0's legal text (creativecommons.org/licenses/by-nc/4.0/legalcode
  — could not fetch directly, this sandbox's network policy doesn't allow
  creativecommons.org; quoting the clause as surfaced via search) defines
  NonCommercial as: *"not primarily intended for or directed towards
  commercial advantage or monetary compensation."* **Commercial use is not
  permitted.** SwingSensei is a commercial product, so this dataset **cannot
  be used to train the shipped model** without a separate agreement from the
  authors. Logging it anyway per the task's research-only allowance.
- Underlying videos are YouTube (46%) and Reddit (54%) clips of pro and
  amateur golfers — a second, independent licensing problem even if the CC
  BY-NC term weren't there (no redistribution rights over the original
  footage are granted by this package).

**Why it's relevant to the motion-blur failure mode specifically (verified,
not assumed):** the metadata has a `playback_speed` column with two values,
`fullspeed` (40/67 clips) and `slowmo` (27/67) — slow-motion golf-instruction
footage is shot at high frame rates with correspondingly short effective
exposure per displayed frame, which suppresses motion blur, while real-speed
footage does not. I downloaded the package, loaded `annotations.parquet`
with pyarrow, and computed each box's elongation (`max(w,h)/min(w,h)`) split
by that column:

| playback_speed | n boxes | median elongation | p90 elongation | max |
|---|---|---|---|---|
| fullspeed | 6,126 | 1.945 | 3.431 | 5.383 |
| slowmo | 3,279 | 1.670 | 2.964 | 5.119 |
| **all boxes** | **9,405** | **1.833** | **3.225** | **5.383** |

This is a real, measured gap, not a guess: `fullspeed` clips carry visibly
more elongated (blur-streak-shaped) clubhead boxes than `slowmo` clips in the
same corpus, labeled under the same box-the-blur-streak convention this
project's own `labeling-spec.md` uses. (Null/unannotated-frame rate is
similar either way — 7.2% fullspeed vs 8.3% slowmo — so the elongation gap
isn't an artifact of different amounts of missing data.)

**What this is actually useful for, given the licence blocks training use:**
not as training data, but as a **validation signal and a cheap sourcing
heuristic**:
1. As a research-only benchmark: it's a second, independent, already-labeled
   source of real motion-blur clubhead boxes that could be used (non-
   commercially) to sanity-check whether a blur-robust detector or blur
   augmentation actually generalizes to blur the project didn't generate
   itself.
2. More valuable long-term: `playback_speed` (real-speed vs slow-motion) is
   exactly the kind of tag that's usually visible from a YouTube video's own
   title/description ("super slo-mo", "100fps", "real speed", "regular
   speed") *before* downloading it. This project's own data engine
   (`docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`, §5)
   already plans to pull swing footage from YouTube; this confirms that
   filtering candidate source videos by real-speed-vs-slowmo framing in the
   title/description is a cheap, verifiable proxy for "will this clip
   actually contain motion blur," worth building into the source-selection
   step of the data engine rather than discovering it after labeling.

**Effort vs. payoff:** low effort to verify (one `pip download`, one parquet
read, ~20 minutes), near-zero payoff as a training-data source (non-
commercial licence + no rights over the underlying video), modest payoff as
a confirmed heuristic for the project's own future YouTube sourcing pass and
as a non-commercial benchmark to validate blur-robustness work against.
Does not touch the camouflage failure mode at all — all 67 clips are
well-lit, down-the-line instructional/pro footage, not indoor/low-light.
