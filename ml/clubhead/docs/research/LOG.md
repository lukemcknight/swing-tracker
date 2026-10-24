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

---

## 2026-10-24 (second run) — BlurBall / WASB-SBDT: blur-aware heatmap tracking instead of box-IoU detection

Rotated to the motion-blur / small-object-tracking research area (dataset
sourcing is covered by the entry above; don't repeat it).

**What it is:** *BlurBall: Joint Ball and Motion Blur Estimation for Table
Tennis Ball Tracking* (Gossard, Radovic, Ziegler, Zell — Uni Tübingen,
partial Sony AI funding), arXiv 2509.18387, also at CVPR Workshops 2026
(CVSports). It tracks a small, very fast, frequently blurred ball (table
tennis) by regressing a per-frame heatmap of ball location *plus* an
explicit blur vector (orientation + half-length of the motion streak) as a
second output, over a short window of consecutive frames with
Squeeze-and-Excitation attention across the frame stack. It builds on
**WASB-SBDT** ("Widely Applicable Strong Baseline for Sports Ball
Detection and Tracking," Tarashima et al., BMVC 2023, arXiv 2311.05237),
an HRNet-based multi-frame heatmap tracker already shown to generalize
across several ball sports.

**URLs:**
- Paper: https://arxiv.org/abs/2509.18387 (found via web search; arxiv.org
  itself is denied by this session's network policy — confirmed via the
  agent-proxy status endpoint — so the paper's full text was read only via
  search-result summaries, not fetched directly).
- BlurBall project/dataset page: https://cogsys-tuebingen.github.io/blurball/
  — exists per search results (project page + per-video CSVs of ball
  position and blur orientation/half-length), but **could not be fetched**
  this run (`github.io` is also denied by network policy, `getaddrinfo
  ENOTFOUND`). BlurBall's own code release status is therefore
  **unconfirmed** — don't assume it's downloadable until someone checks
  from an unrestricted network.
- Base code (independently fetched and confirmed): https://github.com/nttcom/WASB-SBDT
  — real, public, MIT-licensed (`LICENSE.md`), 201 stars / 40 forks. Has a
  Dockerfile, `GET_STARTED.md`, and a `MODEL_ZOO.md` with pretrained
  weights for evaluation. Whether full *training* code (not just eval of
  pretrained models) is included wasn't independently confirmed — check
  `src/` before committing effort.

**Licence:** WASB-SBDT is **MIT — commercial use permitted**, verified by
fetching the repo directly. BlurBall's own licence is unknown (page
unreachable this run); its dataset is table-tennis-specific anyway, so it
isn't a training-data candidate for this project regardless of licence —
the value here is the *method*, not their data.

**Which failure mode: motion blur, primarily** (with a secondary tie to
camouflage via its multi-frame design). This is directly relevant to the
diagnosed blur-labeling problem in this repo's `docs/labeling-spec.md`:
annotators are told to box the full motion-blur streak, but boxes come out
near-square anyway (per the brief's elongation stats) because drawing a
tight, consistent box around a smeared streak is a genuinely hard manual
task — people default to a box around the visible "core." BlurBall's
approach sidesteps that labeling problem entirely: instead of asking for
one box that has to simultaneously encode position *and* blur extent,
it labels and predicts **center position and blur vector as separate
targets**. That's an easier, more consistent thing for a human to label
(mark the streak's center, then draw a line along its length) and a more
natural thing for a network to regress than box corners. The multi-frame
input also directly targets **camouflage**: a static dark clubhead against
dark clothing/foliage has near-zero single-frame signal (hence the
brief's zero-candidate-at-0.05-confidence failures), but it still *moves*
frame to frame, which a heatmap tracker conditioned on a short frame stack
can exploit in a way a single-frame YOLO detector structurally cannot.

**Effort vs. payoff:** High effort, uncertain-but-potentially-high payoff
— this is an architecture swap, not a tweak. Moving from single-frame
YOLO11n box detection to a multi-frame heatmap tracker means: a different
training pipeline, different label format (center + blur vector instead of
xyxy boxes — would need re-deriving labels from the existing box
annotations, not a full re-label), and an unproven CoreML export/on-device
latency path (YOLO11n→CoreML is working today per this repo's README;
HRNet-based multi-frame models are heavier and the multi-frame input
changes the on-device buffering story — nothing here confirms it would
still hit real-time on an iPhone). Don't start a rewrite on this alone.
What's cheap to try first: (1) adopt WASB-SBDT/BlurBall's blur-vector
*labeling convention* for a small re-labeled subset to see if it's more
consistent across annotators than box-elongation, without changing the
model yet; (2) read WASB-SBDT's actual multi-frame heatmap loss and see if
a lightweight auxiliary blur-vector head could bolt onto the existing
YOLO11n backbone instead of a full architecture swap. Both are
order-of-a-day spikes; the full tracker swap is not. As with the first
entry in this log: this is still aimed at a problem (blur) that hasn't
been measured indoors yet, so treat it as a candidate to revisit once
real indoor/low-light clubhead footage actually shows blurred failures,
not as something to build against today's camouflage-only evidence.
