# Clubhead detector research log

Append-only. One entry per research run. Each entry's "off limits" content
(sources, datasets, techniques already logged) must not be repeated in a
later run — check this file before researching.

---

## 2026-11-01

**Note on this run's premise:** the scheduled task prompt for this research
loop describes measured results (82% per-frame detection, 77% through-swing,
85.9% selection ceiling, a camouflage failure mode found via visual
inspection, a 1.60 median box-elongation stat) that live in
`ml/clubhead/eval/results/v1-failure-analysis.md` and
`ml/clubhead/docs/data-engine.md`. **Neither file exists in this repo** at
the commit this run checked out (`406f8cd`, "Capture CreateML v6 clubhead
detector baseline"). The repo is still at Phase 0 of the v2 design doc
(`docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`): only
the CreateML v6 baseline has been scored (30.4% detection rate, 80% phantom
rate), no YOLO11n model has been trained yet, and there is no failure
analysis or data-engine doc on disk or in git history anywhere in this repo.
Someone should check whether this scheduled task's prompt needs updating, or
whether the referenced v2 work exists in a different repo/branch than the one
this session has access to. This run proceeded with general research anyway
since the two named failure modes (camouflage, motion-blur training gap) are
plausible regardless, but the specific numbers above are unconfirmed in this
codebase.

**Area searched (rotation 1/5):** commercially-licensed golf swing/clubhead
datasets, specifically seeking indoor/simulator-bay/low-light/older-phone
footage where real motion blur would appear.

**Finding: nothing new clears the bar.** Two concrete, verifiable datasets
exist and were checked directly, both blocked for commercial use:

- **GolfDB** — https://github.com/wmcnally/golfdb (paper:
  https://arxiv.org/abs/1903.06528). README states verbatim: "The code in
  this repository is licensed under a Creative Commons Attribution-
  NonCommercial 4.0 International License." **CC BY-NC 4.0 — commercial use
  NOT permitted.** Outdoor/range footage only, no indoor content.
- **ClubheadDB** — https://pypi.org/project/clubhead-db/ (~10k down-the-line
  frames). Listed license **CC BY-NC 4.0 — commercial use NOT permitted.**
  Sourced from public YouTube clips, which raises a separate rights question
  even setting the dataset's own license aside.
- **Roboflow Universe** golf-club datasets (e.g. `golf-club-head-tracer`)
  could not be verified — `universe.roboflow.com` did not resolve from this
  session's network, so license/content/indoor-split could not be confirmed
  first-hand. Not logged as a finding either way. Also not "new" to this
  project regardless, since 54% of current training data is already
  Roboflow-sourced.
- Commercial stock footage (Storyblocks/Pond5/Getty) of indoor golf
  simulators exists but none of the reachable license pages state terms for
  AI/ML training-data mining one way or the other — ordinary "commercial
  video production" licenses are not evidence that TDM use is permitted.
  Unverified, not logged as usable.

**Failure mode addressed:** motion blur (the indoor/low-light gap this
rotation targeted). Not applicable to camouflage.

**Why it would matter if it existed:** indoor/simulator-bay footage is
exactly where longer exposure produces real motion-blur streaks, which the
training set is short of — and it's untested territory for this model (no
usable indoor test set exists per the README). A licensed dataset here would
directly target the blur gap.

**Effort vs. payoff:** effort to search/verify was low (web search only, no
download/integration attempted). Payoff is a negative result, but a useful
one: it rules out two concrete datasets and establishes that the public,
commercially-usable, indoor/blurred golf-swing dataset this project wants
does not currently appear to exist. The real paths forward are (a) shoot
proprietary indoor/simulator footage in-house, matching the labeling spec's
existing motion-blur-streak convention, or (b) license stock footage with an
explicit written TDM/AI-training clause confirmed with the vendor — neither
of which is a "go download this" action, so no immediate action item beyond
ruling this search path out for future runs.
