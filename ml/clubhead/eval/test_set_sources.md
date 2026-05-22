# Held-out test set — source clips

The permanent held-out test set for the clubhead detector. These clips are
**excluded from all training** — every model (the CreateML baseline and every
future YOLO model) is scored on them by the eval harness.

## Provenance

Carved from the Label Studio export `project-1-at-2026-05-22-16-31-ff31f44a.json`
(1,308 labeled frames across 71 clips, exported 2026-05-22). A copy of that
export is kept at `data/labeling/clubhead/label_studio_export_2026-05-22.json`.

Of the 71 clips, only 8 are genuinely diverse (YouTube swings — other golfers,
cameras, angles); the other 63 are the developer's own swings. To keep diverse
data in *both* training and test, the 8 YouTube clips were split 3 test / 5
train. The 63 own-swing clips all go to training.

## Test clips (3 — held out, ~196 frames)

| Clip | Frames | Character |
|---|---|---|
| `yt-test-DrPORihFauA` | 64 | Down-the-line, iron, overcast |
| `yt-test-5f5iA7JacYk` | 49 | Down-the-line, driver, bright |
| `yt-test-mNW6zLNfabI` | 83 | Face-on |

Imported to `ml/clubhead/dataset/test/` via `scripts/import_test_set.py`
(gitignored — rebuild from the export if needed).

## Training clips (Phase 1)

Everything else in the export: the other 5 YouTube clips
(`yt-test-BueoWXZwrs0`, `yt-test-gmNrmMDcl9c`, `yt-test-LPCwNzciG0`,
`yt-test-0d6KeUFluvE`, `yt-test-lzueEtc6PPM`) plus all 63 own-swing clips
(`swing-*`, `range2-IMG_*`, `impact-IMG_*`) — ~1,112 frames.

## Known caveat

The dataset's diversity rests on 8 YouTube clips. A v1 model may still
generalize weakly to the 3 held-out clips — that is expected and is exactly
what the harness quantifies. The top data-engine priority is more
non-own-swing footage.
