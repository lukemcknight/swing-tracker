# Clubhead detector research log

Append-only. Each run (see scheduled task) adds exactly one new, verified
entry. Entries already here are off limits for future runs — don't repeat.

---

## 2026-10-24

### Note on repo state vs. this research brief

The research brief this log is run under describes a YOLO11n → CoreML
model at ~82% per-frame / ~77% through-swing detection, trained on a mix
that is ~54% third-party Roboflow imagery, with a documented camouflage
failure mode (`v1-failure-analysis.md`) and a `data-engine.md`. None of
that exists in this repo as checked out right now (branch
`mvp/club-path-replay`, HEAD `406f8cd`): the only model on record is the
CreateML v6 baseline (30.4% detection rate, 80% phantom rate, 196 test
frames, see `ml/clubhead/eval/baseline/createml-v6.md`), there is no
Roboflow data anywhere in the dataset provenance
(`ml/clubhead/eval/test_set_sources.md` — training data is 63 own-swing
clips + 5 YouTube clips, ~1,112 frames), and `v1-failure-analysis.md` /
`data-engine.md` don't exist. The labeling spec's motion-blur convention
("box the full visible clubhead including its motion-blur streak",
`ml/clubhead/docs/labeling-spec.md`) is the one claim from the brief that
does check out against this repo.

I'm flagging this once rather than re-deriving it every run. Given the
mismatch, I'm treating "more real, diverse clubhead training data" as the
higher-value target right now than re-litigating camouflage vs. blur
specifics that aren't yet measured in this codebase, and picked the
dataset-sourcing research area for this entry.

### Finding: ClubheadDB (golf-clubhead-specific detection dataset)

- **What it is**: A purpose-built video dataset for golf clubhead
  detection/tracking, hand-annotated with a single bounding box per frame
  (empty label for frames with no visible clubhead — true negatives).
  10,180 annotated frames from 67+ unique golfers (47 train / 10 val / 10
  test video-level split), sourced 46% YouTube / 54% Reddit, 42% pro / 58%
  amateur golfers, 70% irons / 27% drivers / 3% woods. Labels ship in YOLO
  format already (`frames/swing_xxx/labels/*.txt`).
- **URL**: https://pypi.org/project/clubhead-db/ (installed via
  `pip install clubhead-db`, built locally via the bundled `clubhead-build`
  CLI, which uses `yt-dlp` + `ffmpeg` to re-download the source clips and
  re-extract the exact frames — nothing is redistributed directly).
  Declared source repo: https://github.com/unknownauthor23/ClubheadDB
  (not independently fetched — GitHub access for this arbitrary third-party
  repo isn't in this session's scope; relying on the PyPI package metadata,
  which is a primary source for the package itself).
- **Licence**: The package's own README states, verbatim as a badge line:
  `License: CC BY-NC 4.0`. CC BY-NC 4.0's own definition of the
  restriction (public, standard Creative Commons legal text): *"NonCommercial
  means not primarily intended for or directed towards commercial advantage
  or monetary compensation."* **Commercial use is NOT permitted.** SwingSensei
  is a commercial app, so this dataset cannot be used to train the shipped
  model's weights, full stop.
- **Which failure mode**: Primarily a training-data-diversity dataset — it
  doesn't document motion blur or camouflage targeting explicitly, so it
  speaks to both only indirectly: more varied amateur/Reddit-sourced footage
  (different cameras, swings, backgrounds) plausibly contains more
  camouflage-adjacent and blur-adjacent examples than this repo's current
  63-own-swing-clip corpus, but that's not a verified claim from the
  dataset's own documentation — the README doesn't break down lighting
  conditions, indoor/outdoor split, or blur amount.
- **Why it would matter here, if it weren't for the licence**: it is the
  closest thing I could find to an off-the-shelf, clubhead-specific,
  bounding-box-labeled, YOLO-format dataset — unlike GolfDB (swing-phase +
  person boxes, no clubhead boxes) or CaddieSet (joint pose + ball, no
  clubhead) which came up in the same search and don't label the clubhead
  at all. It also already includes true negative frames, matching this
  repo's own labeling convention for phantom-detection training. If this
  model is actually still at the CreateML v6 / ~1,100-training-frame stage
  the repo shows, 10,180 more annotated clubhead frames of any flavor would
  be a meaningfully larger jump than most augmentation tricks.
- **Effort vs. payoff**: Trivial to try (`pip install clubhead-db`,
  one CLI command) for an *offline-only* feasibility check — e.g. does
  adding this data reduce the phantom-detection rate in a throwaway
  experiment — but it has **zero path to production** as-is: the licence
  blocks training the shipped model on it, and because the dataset is
  reconstructed from individually-copyrighted YouTube/Reddit posts rather
  than released footage, there's no way to "re-clear" the same clips under
  a commercial licence later. Positive signal from this dataset would only
  justify spending effort sourcing equivalent *commercially clean* footage
  (own capture, licensed stock, or a dataset with an actual commercial
  grant), not using the dataset itself downstream. Also worth noting: its
  citation is a placeholder for an unpublished/anonymous workshop
  submission, so there's no independent review of label quality — spot-check
  before trusting it even for the offline experiment.
