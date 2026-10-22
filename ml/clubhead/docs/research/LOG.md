# Clubhead detector — research log

Append-only. Each run adds exactly one verified entry. Entries already here
are off-limits for future runs — do not repeat them.

---

## 2026-10-22 — ClubheadDB (web-sourced golf clubhead dataset)

**What:** `clubhead-db`, a PyPI-distributed, reproducible dataset for golf
clubhead detection. Not a bundle of images — it ships metadata
(`metadata/metadata.csv`: source video URLs + timestamps) and pre-made
annotations (`annotations/annotations.parquet`, YOLO-format boxes), plus a
`clubhead-build` CLI that uses `yt-dlp` + `ffmpeg` to re-download the source
clips and re-extract the exact frame set. Contents, per the package's PyPI
page: 10,180 hand-annotated "down-the-line" frames (one box per frame, zero
for null/negative examples), 67+ unique golfers, split 47/10/10 train/val/test
by video, skill mix Pros 42% / Amateurs 58%, club mix Irons 70% / Drivers 27%
/ Woods 3%. Source composition: YouTube 46%, Reddit 54% — i.e. casual,
web-sourced public footage, not studio production.

**URL:** https://pypi.org/project/clubhead-db/ (install via `pip install
clubhead-db`). Fetched and read directly this run; the page, the install
command, and the described build pipeline are real and present today. Could
not find an accompanying GitHub repo in search — PyPI is the only confirmed
source.

**Licence:** The page states the licence as `CC-BY-NC-4.0` (Creative Commons
Attribution-NonCommercial 4.0) — verbatim identifier as shown on the page; I
could not reach creativecommons.org this run to pull the full legal-code text
verbatim, but CC BY-NC's NonCommercial clause is unambiguous and standard:
it forbids use "primarily intended for or directed towards commercial
advantage or monetary compensation." **Commercial use is NOT permitted.**
This is a research-only dataset for this product. Do not train the shipped
on-device model on it, and do not use it to tune/validate a commercial
release without legal sign-off — "noncommercial" licensing on an eval set
used to justify a commercial model's quality claims is a genuine gray area,
not a loophole.

**Failure mode addressed:** Camouflage, primarily. The current training mix
is ~54% curated third-party Roboflow imagery and ~29% the app's own phone
footage — both comparatively clean and controlled. ClubheadDB's casual
YouTube/Reddit sourcing (67+ golfers, varied clothing/backgrounds/clubs,
no studio lighting) is a closer match to the "dark clubhead against dark
clothing or cluttered foliage" camouflage scenario than anything already in
the mix. Motion blur: **unconfirmed** — the page gives no information on
shutter speed, frame rate, or blur content, and down-the-line swing-analysis
clips posted publicly skew toward higher-fps/slow-mo capture intended to be
legible, which would work against blur. I did not find evidence either way
and am not claiming this helps blur.

**Why it matters for this model specifically:** It's a free, independent,
already-annotated source of diverse real-world clubhead appearances at the
exact low-effort end (pip install, one CLI command) of a problem — getting
more non-own-swing, non-Roboflow footage — that `eval/test_set_sources.md`
already flags as the top data-engine priority. It would be useful as a
**held-out evaluation set** to stress-test camouflage generalization beyond
the current 3-clip outdoor test set, which is itself flagged as too narrow.

**Effort vs. payoff:** Low effort to obtain and inspect (minutes). Low
near-term payoff for the shipped model because of the NC licence — it can't
go into the training set or be used to validate release-quality claims
without legal review. Moderate payoff as a free, quick way to *diagnose*
whether camouflage failures generalize beyond the current tiny test set,
which is cheap to do and informs whether camouflage work is still worth
prioritizing over motion blur. Recommend: use for offline diagnostic eval
only, flag to legal/product before touching training or official metrics.
