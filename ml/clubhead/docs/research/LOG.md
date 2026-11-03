# Clubhead detector — research log

Append-only. Each entry is one verified, genuinely-new finding. Do not repeat
prior entries' topics in the next run — rotate across areas.

---

## 2026-11-03 — ClubheadDB (PyPI/GitHub) — verified, but not usable for this product

**What it is:** `ClubheadDB` is a reproducible-dataset package (not a direct
download — it's a build pipeline): `pip install clubhead-db` then run
`clubhead-build`, which uses `yt-dlp` to re-download the source clips and
`ffmpeg` to extract frames, then writes YOLO-format single-box clubhead
labels (empty `.txt` for null/no-clubhead frames — same convention as this
repo's negative-frame handling). Per its own README: 10,180 annotated
frames, 67+ golfers (42% pro / 58% amateur), down-the-line angle only,
clubs mixed irons 70% / drivers 27% / woods 3%, sourced 46% YouTube / 54%
Reddit.

**URLs:**
- Package (verified live, real wheel + sdist on PyPI today):
  https://pypi.org/project/clubhead-db/
- Claimed homepage (checked today, returns **HTTP 404** — not reachable):
  https://github.com/unknownauthor23/ClubheadDB

**Licence — verbatim from the PyPI metadata (`license` field) and the
README badge:** `License: CC BY-NC 4.0` (`CC-BY-NC-4.0`).
**Commercial use is NOT permitted.** This is a non-commercial-only licence,
full stop — do not train the shipping model on this data.

There's a second, independent problem beyond the stated licence: the
dataset doesn't ship images, it ships a *recipe* (video URLs + timestamps)
that re-scrapes frames from arbitrary public YouTube/Reddit posts at build
time. The CC-BY-NC-4.0 grant covers the dataset authors' own annotations/
metadata/code — it says nothing about the underlying video footage, whose
copyright still belongs to whoever posted it. Nothing in the README
indicates the source clips were filtered for a redistributable licence.
So even a non-commercial research use of this data carries copyright
exposure independent of the CC-BY-NC terms, and because videos are fetched
live rather than bundled, `metadata.csv` is exposed to ordinary link rot
(deleted YouTube videos, removed Reddit posts) — reproducibility isn't
guaranteed either.

**Which failure mode this addresses:** Neither cleanly. It's a general
diversity/volume dataset (more golfers, more backgrounds, a chunk of
casual Reddit footage that might include more varied lighting/clothing
than our current 54%-Roboflow mix) — closer to camouflage-adjacent than
blur-specific. **The README makes no mention of motion blur, indoor
footage, or shutter speed at all**, so this should NOT be read as evidence
it fills the motion-blur gap — I'm flagging that absence explicitly rather
than assuming it helps.

**Why it's relevant to this model specifically:** The label *shape* is an
exact match for our pipeline — single clubhead-only box per frame, YOLO
format, empty label file for negative frames — so structurally it's the
right shape of data. That's the only thing worth taking from it.

**Effort vs. payoff:** Verification was cheap (PyPI metadata + README
inspection only; did not run the build pipeline, since doing so means
bulk-downloading other people's videos via `yt-dlp`, which is exactly the
copyright exposure flagged above). Payoff for this product is **zero** —
blocked twice over (explicit NC licence, plus unclear rights on the
re-scraped source video). The only transferable value is the *methodology*
(scrape → yt-dlp → ffmpeg → hand-annotate YOLO boxes, with explicit empty
labels for negatives) as a template to apply to our own rights-cleared
footage — not the dataset itself. Logging this mainly to close off this
lead so a future run doesn't re-discover it.
