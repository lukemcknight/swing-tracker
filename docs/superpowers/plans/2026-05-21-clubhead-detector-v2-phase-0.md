# Clubhead Detector v2 — Phase 0: Foundations & Eval Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `ml/clubhead/` Python pipeline and a reusable eval harness that scores any Core ML clubhead model on a held-out test set, then capture the baseline numbers for the current CreateML model.

**Architecture:** A small Python package (`chdet`) under `ml/clubhead/src/`. Pure-logic modules — box geometry, metrics, label parsing, output decoding — are built test-first with `pytest`. A CLI harness (`chdet.evaluate`) loads a Core ML model via `coremltools`, runs it over a directory of test-swing frames with aspect-correct letterboxing, compares predictions to YOLO-format ground-truth labels, and writes a Markdown + JSON report. Phase 0 ships the harness and a measured baseline; it trains nothing.

**Tech Stack:** Python 3.12 (required — `coremltools` does not support 3.14), `coremltools`, `numpy`, `Pillow`, `pytest`. Virtualenv at `ml/.venv`.

---

## Context for the implementer

- This is Phase 0 of the design in `docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`. Read §2, §4, §8, §10 of that spec.
- The current model under test is a CreateML object detector at `swing-tracker-ml.mlproj/Models/swing-tracker-ml 6.mlmodel` (single `clubhead` class). Its outputs after built-in NMS are two arrays: `coordinates` (N×4, normalized `[cx, cy, w, h]`) and `confidence` (N×C).
- The repo root is `/Users/lukemck/Development/swing-tracker`. All paths below are relative to it.
- `data/` and `swing-tracker-ml.mlproj/` are gitignored; `ml/` is **not** — code under `ml/` is tracked, so this plan adds an `ml/.gitignore` for the venv, datasets, and generated reports.
- Work happens on the current branch `mvp/club-path-replay`.
- `python3.12` is available at `/opt/homebrew/bin/python3.12`.

## Canonical data types (used across every task)

These names are fixed. Every task below uses them consistently.

- `Box(cx, cy, w, h)` — frozen dataclass, normalized `[0, 1]` coordinates, **center form**, origin top-left. The single internal box representation.
- `Detection(box: Box, confidence: float)` — a model prediction.
- A "frame" in metrics is a `(prediction, ground_truth)` pair where each side is `Box | None` (`None` = no clubhead).

## File structure

```
ml/
  .venv/                         # virtualenv (gitignored)
  .gitignore                     # NEW — Task 1
  clubhead/
    pyproject.toml               # NEW — Task 1
    requirements.txt             # NEW — Task 1
    README.md                    # NEW — Task 1
    src/chdet/
      __init__.py                # NEW — Task 1
      geometry.py                # NEW — Task 2 (Box, conversions, iou, unletterbox)
      metrics.py                 # NEW — Task 3 (detection rate, center error, jitter)
      labels.py                  # NEW — Task 4 (YOLO + Label Studio parsing)
      detector.py                # NEW — Task 5 (Detector protocol, decode, Core ML adapter)
      evaluate.py                # NEW — Task 6 (eval harness + CLI)
      dataset.py                 # NEW — Task 7 (Label Studio export -> test-set layout)
    scripts/
      import_test_set.py         # NEW — Task 7 (thin CLI over dataset.py)
    tests/
      test_geometry.py           # NEW — Task 2
      test_metrics.py            # NEW — Task 3
      test_labels.py             # NEW — Task 4
      test_detector.py           # NEW — Task 5
      test_evaluate.py           # NEW — Task 6
      test_dataset.py            # NEW — Task 7
    dataset/test/                # test-set frames + labels (gitignored) — Task 8
    eval/
      baseline/                  # committed baseline reports — Task 9
      reports/                   # generated reports (gitignored)
    docs/
      labeling-spec.md           # NEW — Task 10
```

---

## Task 1: Project skeleton & environment

**Files:**
- Create: `ml/.gitignore`
- Create: `ml/clubhead/pyproject.toml`
- Create: `ml/clubhead/requirements.txt`
- Create: `ml/clubhead/README.md`
- Create: `ml/clubhead/src/chdet/__init__.py`

- [ ] **Step 1: Create the directory tree and files**

Create `ml/.gitignore`:

```gitignore
.venv/
**/__pycache__/
*.egg-info/
.pytest_cache/
clubhead/dataset/
clubhead/eval/reports/
```

Create `ml/clubhead/pyproject.toml`:

```toml
[project]
name = "chdet"
version = "0.1.0"
description = "SwingSensei clubhead detector — training & evaluation pipeline"
requires-python = ">=3.12,<3.13"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create `ml/clubhead/requirements.txt`:

```text
coremltools>=8.1
numpy>=1.26
Pillow>=10.0
pytest>=8.0
```

Create `ml/clubhead/README.md`:

```markdown
# chdet — SwingSensei clubhead detector pipeline

Python pipeline for training and evaluating the on-device clubhead detector.
See `docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Test

```bash
source ../.venv/bin/activate && pytest
```

## Phase 0: eval harness

```bash
python -m chdet.evaluate \
  --model "../../swing-tracker-ml.mlproj/Models/swing-tracker-ml 6.mlmodel" \
  --test-set dataset/test \
  --out eval/reports/createml-v6.md
```
```

Create `ml/clubhead/src/chdet/__init__.py` (empty file).

- [ ] **Step 2: Create the virtualenv and install dependencies**

Run:
```bash
/opt/homebrew/bin/python3.12 -m venv ml/.venv
source ml/.venv/bin/activate
pip install -r ml/clubhead/requirements.txt
pip install -e ml/clubhead
```
Expected: all packages install with no error; `pip show coremltools` reports a version `>= 8.1`.

- [ ] **Step 3: Verify the toolchain works**

Run:
```bash
source ml/.venv/bin/activate && cd ml/clubhead && python -c "import chdet, coremltools, numpy, PIL; print('ok')" && pytest
```
Expected: prints `ok`, then pytest reports `no tests ran` (exit code 5 is fine — there are no tests yet).

- [ ] **Step 4: Commit**

```bash
git add ml/.gitignore ml/clubhead/pyproject.toml ml/clubhead/requirements.txt ml/clubhead/README.md ml/clubhead/src/chdet/__init__.py
git commit -m "Scaffold chdet clubhead ML pipeline package"
```

---

## Task 2: Box geometry & coordinate conversions

**Files:**
- Create: `ml/clubhead/src/chdet/geometry.py`
- Test: `ml/clubhead/tests/test_geometry.py`

All commands below run from `ml/clubhead/` with the venv active (`source ../.venv/bin/activate`).

- [ ] **Step 1: Write the failing tests**

Create `ml/clubhead/tests/test_geometry.py`:

```python
import math

from chdet.geometry import (
    Box, iou, from_createml, from_label_studio, from_yolo_line, to_yolo_line,
    unletterbox_box,
)


def test_box_corners():
    assert Box(0.5, 0.5, 0.2, 0.4).corners() == (0.4, 0.3, 0.6, 0.7)


def test_iou_identical_boxes_is_one():
    b = Box(0.5, 0.5, 0.2, 0.2)
    assert iou(b, b) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou(Box(0.1, 0.1, 0.1, 0.1), Box(0.9, 0.9, 0.1, 0.1)) == 0.0


def test_iou_half_overlap():
    a = Box(0.25, 0.5, 0.5, 0.5)   # x in [0.0, 0.5]
    b = Box(0.50, 0.5, 0.5, 0.5)   # x in [0.25, 0.75]
    # intersection 0.25*0.5=0.125, union 0.25+0.25-0.125=0.375
    assert math.isclose(iou(a, b), 0.125 / 0.375, rel_tol=1e-9)


def test_from_createml_pixel_center_form():
    box = from_createml({"x": 200.0, "y": 100.0, "width": 40.0, "height": 20.0},
                        img_w=400, img_h=200)
    assert box == Box(0.5, 0.5, 0.1, 0.1)


def test_from_label_studio_percent_topleft_form():
    # Label Studio stores top-left x/y and size as percentages.
    box = from_label_studio({"x": 40.0, "y": 30.0, "width": 20.0, "height": 40.0})
    assert math.isclose(box.cx, 0.5, rel_tol=1e-9)
    assert math.isclose(box.cy, 0.5, rel_tol=1e-9)
    assert math.isclose(box.w, 0.2, rel_tol=1e-9)
    assert math.isclose(box.h, 0.4, rel_tol=1e-9)


def test_yolo_line_roundtrip():
    box = Box(0.123456, 0.654321, 0.1, 0.2)
    parsed = from_yolo_line(to_yolo_line(box, class_id=0))
    assert math.isclose(parsed.cx, box.cx, abs_tol=1e-6)
    assert math.isclose(parsed.cy, box.cy, abs_tol=1e-6)


def test_unletterbox_box_inverts_centered_padding():
    # A 400x200 image letterboxed into a 320x320 square:
    # scale = min(320/400, 320/200) = 0.8 -> resized 320x160, pad_y = 80, pad_x = 0.
    # A box at the resized-image center maps back to the original center.
    box_in_canvas = Box(160 / 320, 160 / 320, 80 / 320, 40 / 320)
    box = unletterbox_box(box_in_canvas, scale=0.8, pad_x=0, pad_y=80,
                          target_w=320, target_h=320, orig_w=400, orig_h=200)
    assert math.isclose(box.cx, 0.5, abs_tol=1e-9)
    assert math.isclose(box.cy, 0.5, abs_tol=1e-9)
    assert math.isclose(box.w, 80 / 0.8 / 400, abs_tol=1e-9)
    assert math.isclose(box.h, 40 / 0.8 / 200, abs_tol=1e-9)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chdet.geometry'`.

- [ ] **Step 3: Write the implementation**

Create `ml/clubhead/src/chdet/geometry.py`:

```python
"""Box geometry and coordinate conversions for the clubhead pipeline.

Canonical internal form: `Box` in normalized [0, 1] coordinates, center form
(cx, cy, w, h), origin top-left. Every other format converts to/from this.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    w: float
    h: float

    def corners(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2): top-left and bottom-right corners."""
        return (
            self.cx - self.w / 2,
            self.cy - self.h / 2,
            self.cx + self.w / 2,
            self.cy + self.h / 2,
        )


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a.corners()
    bx1, by1, bx2, by2 = b.corners()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def from_createml(coords: dict, img_w: int, img_h: int) -> Box:
    """Convert a CreateML annotation (pixel center form) to a normalized Box."""
    return Box(
        coords["x"] / img_w,
        coords["y"] / img_h,
        coords["width"] / img_w,
        coords["height"] / img_h,
    )


def from_label_studio(value: dict) -> Box:
    """Convert a Label Studio rectanglelabels value (percent, top-left) to a Box."""
    w = value["width"] / 100.0
    h = value["height"] / 100.0
    return Box(value["x"] / 100.0 + w / 2, value["y"] / 100.0 + h / 2, w, h)


def from_yolo_line(line: str) -> Box:
    """Parse a YOLO label line `class cx cy w h` (normalized) into a Box."""
    _, cx, cy, w, h = line.split()[:5]
    return Box(float(cx), float(cy), float(w), float(h))


def to_yolo_line(box: Box, class_id: int = 0) -> str:
    """Render a Box as a YOLO label line."""
    return f"{class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}"


def unletterbox_box(
    box: Box,
    scale: float,
    pad_x: int,
    pad_y: int,
    target_w: int,
    target_h: int,
    orig_w: int,
    orig_h: int,
) -> Box:
    """Map a Box detected on a letterboxed canvas back to original-image coords.

    `scale`, `pad_x`, `pad_y` are the values returned by `detector.letterbox`.
    """
    cx_px = box.cx * target_w - pad_x
    cy_px = box.cy * target_h - pad_y
    return Box(
        cx_px / scale / orig_w,
        cy_px / scale / orig_h,
        box.w * target_w / scale / orig_w,
        box.h * target_h / scale / orig_h,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_geometry.py -v`
Expected: PASS — 8 passed.

- [ ] **Step 5: Commit**

```bash
git add ml/clubhead/src/chdet/geometry.py ml/clubhead/tests/test_geometry.py
git commit -m "Add box geometry and coordinate conversions"
```

---

## Task 3: Evaluation metrics

**Files:**
- Create: `ml/clubhead/src/chdet/metrics.py`
- Test: `ml/clubhead/tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `ml/clubhead/tests/test_metrics.py`:

```python
import math

import pytest

from chdet.geometry import Box
from chdet.metrics import (
    center_error_pct, percentile, detection_rate, phantom_rate, path_jitter,
)


def test_center_error_pct_zero_for_identical_centers():
    b = Box(0.5, 0.5, 0.1, 0.1)
    assert center_error_pct(b, b, aspect=16 / 9) == 0.0


def test_center_error_pct_pure_horizontal_offset():
    pred = Box(0.52, 0.5, 0.1, 0.1)
    gt = Box(0.50, 0.5, 0.1, 0.1)
    # dx = 0.02 -> 2% of frame width.
    assert math.isclose(center_error_pct(pred, gt, aspect=16 / 9), 2.0, abs_tol=1e-9)


def test_center_error_pct_rescales_vertical_offset_by_aspect():
    pred = Box(0.5, 0.6, 0.1, 0.1)
    gt = Box(0.5, 0.5, 0.1, 0.1)
    # dy = 0.1 of height; aspect 2.0 -> 0.05 of width -> 5%.
    assert math.isclose(center_error_pct(pred, gt, aspect=2.0), 5.0, abs_tol=1e-9)


def test_percentile_median_and_p90():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 50) == 3.0
    assert math.isclose(percentile(values, 90), 4.6, abs_tol=1e-9)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_detection_rate_counts_only_clubhead_present_frames():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    hit = Box(0.5, 0.5, 0.1, 0.1)
    miss = Box(0.9, 0.9, 0.1, 0.1)
    frames = [
        (hit, gt),      # hit
        (miss, gt),     # miss (IoU below threshold)
        (None, gt),     # miss (no prediction)
        (hit, None),    # negative frame — excluded from denominator
    ]
    assert math.isclose(detection_rate(frames, min_iou=0.3), 1 / 3, rel_tol=1e-9)


def test_detection_rate_no_present_frames_raises():
    with pytest.raises(ValueError):
        detection_rate([(None, None)], min_iou=0.3)


def test_phantom_rate_fraction_of_negatives_with_a_detection():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    frames = [
        (gt, None),     # phantom
        (None, None),   # correct (no detection on a negative frame)
        (gt, gt),       # positive frame — excluded
    ]
    assert phantom_rate(frames) == 0.5


def test_phantom_rate_no_negatives_is_zero():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    assert phantom_rate([(gt, gt)]) == 0.0


def test_path_jitter_zero_for_straight_constant_motion():
    # Constant velocity -> second difference is zero -> no jitter.
    centers = [(0.1, 0.5), (0.2, 0.5), (0.3, 0.5), (0.4, 0.5)]
    assert path_jitter(centers, aspect=1.0) == 0.0


def test_path_jitter_detects_a_kink():
    # A single back-and-forth kink in an otherwise still path.
    centers = [(0.5, 0.5), (0.6, 0.5), (0.5, 0.5)]
    # second diff dx = 0.5 - 2*0.6 + 0.5 = -0.2 -> 20% of width.
    assert math.isclose(path_jitter(centers, aspect=1.0), 20.0, abs_tol=1e-9)


def test_path_jitter_skips_triples_with_a_missing_frame():
    centers = [(0.5, 0.5), None, (0.5, 0.5), (0.6, 0.5), (0.5, 0.5)]
    # Only the last triple (indices 2,3,4) is fully present.
    assert math.isclose(path_jitter(centers, aspect=1.0), 20.0, abs_tol=1e-9)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chdet.metrics'`.

- [ ] **Step 3: Write the implementation**

Create `ml/clubhead/src/chdet/metrics.py`:

```python
"""Evaluation metrics for clubhead detection.

A "frame" is a (prediction, ground_truth) pair; each side is `Box | None`.
`aspect` is always frame_width / frame_height; vertical offsets (fractions of
height) are divided by it so every distance is expressed in fractions of width.
"""
from __future__ import annotations

import math

from .geometry import Box, iou


def center_error_pct(pred: Box, gt: Box, aspect: float) -> float:
    """Euclidean center distance as a percentage of frame width."""
    dx = pred.cx - gt.cx
    dy = (pred.cy - gt.cy) / aspect
    return 100.0 * math.hypot(dx, dy)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile; `p` in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return s[lo] * (1 - (rank - lo)) + s[hi] * (rank - lo)


def detection_rate(
    frames: list[tuple[Box | None, Box | None]], min_iou: float = 0.3
) -> float:
    """Fraction of clubhead-present frames where the model found the clubhead.

    Only frames with a ground-truth box count toward the denominator. A frame
    is a hit when a prediction exists and overlaps the ground truth at or above
    `min_iou`.
    """
    present = [(p, g) for p, g in frames if g is not None]
    if not present:
        raise ValueError("no clubhead-present frames")
    hits = sum(1 for p, g in present if p is not None and iou(p, g) >= min_iou)
    return hits / len(present)


def phantom_rate(frames: list[tuple[Box | None, Box | None]]) -> float:
    """Fraction of no-clubhead frames where the model produced a detection."""
    negatives = [p for p, g in frames if g is None]
    if not negatives:
        return 0.0
    return sum(1 for p in negatives if p is not None) / len(negatives)


def path_jitter(centers: list[tuple[float, float] | None], aspect: float) -> float:
    """Mean second-difference magnitude of the center path, in % of frame width.

    A smooth arc has a small second difference; frame-to-frame jitter inflates
    it. Only triples of three consecutive present frames contribute.
    """
    diffs: list[float] = []
    for i in range(1, len(centers) - 1):
        a, b, c = centers[i - 1], centers[i], centers[i + 1]
        if a is None or b is None or c is None:
            continue
        ddx = a[0] - 2 * b[0] + c[0]
        ddy = (a[1] - 2 * b[1] + c[1]) / aspect
        diffs.append(100.0 * math.hypot(ddx, ddy))
    return sum(diffs) / len(diffs) if diffs else 0.0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS — 12 passed.

- [ ] **Step 5: Commit**

```bash
git add ml/clubhead/src/chdet/metrics.py ml/clubhead/tests/test_metrics.py
git commit -m "Add clubhead detection eval metrics"
```

---

## Task 4: Label parsing

**Files:**
- Create: `ml/clubhead/src/chdet/labels.py`
- Test: `ml/clubhead/tests/test_labels.py`

`labels.py` reads ground truth from two formats: YOLO `.txt` files (what the harness consumes) and Label Studio JSON exports (what the test-set importer in Task 7 consumes).

- [ ] **Step 1: Write the failing tests**

Create `ml/clubhead/tests/test_labels.py`:

```python
import json
import math

from chdet.geometry import Box
from chdet.labels import load_yolo_label, load_label_studio_export


def test_load_yolo_label_reads_a_box(tmp_path):
    f = tmp_path / "frame-0001.txt"
    f.write_text("0 0.5 0.5 0.1 0.2\n")
    box = load_yolo_label(f)
    assert box == Box(0.5, 0.5, 0.1, 0.2)


def test_load_yolo_label_empty_file_is_negative_frame(tmp_path):
    f = tmp_path / "frame-0002.txt"
    f.write_text("")
    assert load_yolo_label(f) is None


def test_load_label_studio_export_parses_annotated_task(tmp_path):
    export = [
        {
            "data": {"image": "/data/local-files/?d=images/frame-0001.jpg"},
            "annotations": [
                {"result": [
                    {"value": {"x": 40.0, "y": 30.0, "width": 20.0,
                               "height": 40.0, "rectanglelabels": ["clubhead"]}}
                ]}
            ],
        }
    ]
    f = tmp_path / "export.json"
    f.write_text(json.dumps(export))
    result = load_label_studio_export(f)
    box = result["frame-0001.jpg"]
    assert box is not None
    assert math.isclose(box.cx, 0.5, abs_tol=1e-9)
    assert math.isclose(box.cy, 0.5, abs_tol=1e-9)


def test_load_label_studio_export_task_with_no_annotation_is_negative(tmp_path):
    export = [
        {"data": {"image": "/data/local-files/?d=images/frame-0009.jpg"},
         "annotations": [{"result": []}]}
    ]
    f = tmp_path / "export.json"
    f.write_text(json.dumps(export))
    result = load_label_studio_export(f)
    assert result == {"frame-0009.jpg": None}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chdet.labels'`.

- [ ] **Step 3: Write the implementation**

Create `ml/clubhead/src/chdet/labels.py`:

```python
"""Load ground-truth clubhead labels from the formats the pipeline uses."""
from __future__ import annotations

import json
from pathlib import Path

from . import geometry
from .geometry import Box


def load_yolo_label(path: Path) -> Box | None:
    """Read a single-object YOLO label file.

    An empty file means a negative frame (clubhead not present) and returns
    None. Otherwise the first line is parsed.
    """
    text = path.read_text().strip()
    if not text:
        return None
    return geometry.from_yolo_line(text.splitlines()[0])


def load_label_studio_export(path: Path) -> dict[str, Box | None]:
    """Parse a Label Studio JSON export.

    Returns {image_filename: Box or None}. A task with no annotation result
    maps to None (a negative frame). Only the first result of the first
    annotation is used — the clubhead is a single object per frame.
    """
    tasks = json.loads(path.read_text())
    out: dict[str, Box | None] = {}
    for task in tasks:
        name = Path(task["data"]["image"]).name
        annotations = task.get("annotations") or []
        results = annotations[0]["result"] if annotations else []
        out[name] = geometry.from_label_studio(results[0]["value"]) if results else None
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_labels.py -v`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
git add ml/clubhead/src/chdet/labels.py ml/clubhead/tests/test_labels.py
git commit -m "Add YOLO and Label Studio label parsing"
```

---

## Task 5: Core ML detector adapter

**Files:**
- Create: `ml/clubhead/src/chdet/detector.py`
- Test: `ml/clubhead/tests/test_detector.py`

This task has a pure, test-first core (`decode_createml_outputs`, `letterbox`) and a thin Core ML wrapper (`CoreMLClubheadDetector`) verified against the real model in Step 6.

- [ ] **Step 1: Write the failing tests**

Create `ml/clubhead/tests/test_detector.py`:

```python
import numpy as np
from PIL import Image

from chdet.detector import Detection, decode_createml_outputs, letterbox
from chdet.geometry import Box


def test_decode_picks_highest_confidence_above_threshold():
    coords = np.array([[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.05, 0.05]])
    confidence = np.array([[0.40], [0.90]])
    det = decode_createml_outputs(coords, confidence, threshold=0.25)
    assert det == Detection(Box(0.2, 0.2, 0.05, 0.05), 0.90)


def test_decode_returns_none_when_all_below_threshold():
    coords = np.array([[0.5, 0.5, 0.1, 0.1]])
    confidence = np.array([[0.10]])
    assert decode_createml_outputs(coords, confidence, threshold=0.25) is None


def test_decode_returns_none_for_no_rows():
    assert decode_createml_outputs(
        np.zeros((0, 4)), np.zeros((0, 1)), threshold=0.25
    ) is None


def test_letterbox_preserves_aspect_and_centers():
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    padded, scale, pad_x, pad_y = letterbox(img, 320, 320)
    assert padded.size == (320, 320)
    assert scale == 0.8          # min(320/400, 320/200)
    assert pad_x == 0
    assert pad_y == 80           # (320 - 160) / 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chdet.detector'`.

- [ ] **Step 3: Write the implementation**

Create `ml/clubhead/src/chdet/detector.py`:

```python
"""Detector interface and the Core ML adapter for the current CreateML model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .geometry import Box, unletterbox_box


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float


class Detector(Protocol):
    """Anything that turns an image into at most one clubhead Detection."""

    def detect(self, image: Image.Image) -> Detection | None:
        ...


def decode_createml_outputs(
    coordinates: np.ndarray, confidence: np.ndarray, threshold: float
) -> Detection | None:
    """Decode CreateML object-detector outputs into the single best Detection.

    `coordinates` is (N, 4) normalized [cx, cy, w, h]; `confidence` is (N, C).
    Returns the highest-confidence detection at or above `threshold`, or None.
    """
    best: Detection | None = None
    for i in range(coordinates.shape[0]):
        conf = float(confidence[i].max())
        if conf < threshold:
            continue
        if best is None or conf > best.confidence:
            cx, cy, w, h = (float(v) for v in coordinates[i])
            best = Detection(Box(cx, cy, w, h), conf)
    return best


def letterbox(
    image: Image.Image, target_w: int, target_h: int
) -> tuple[Image.Image, float, int, int]:
    """Resize `image` into a target canvas, preserving aspect ratio with padding.

    Returns (padded_image, scale, pad_x, pad_y). This matches the aspect-correct
    scaling Vision uses with `.scaleFit`, so harness measurements are faithful.
    """
    iw, ih = image.size
    scale = min(target_w / iw, target_h / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    resized = image.resize((nw, nh))
    padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    pad_x, pad_y = (target_w - nw) // 2, (target_h - nh) // 2
    padded.paste(resized, (pad_x, pad_y))
    return padded, scale, pad_x, pad_y


class CoreMLClubheadDetector:
    """Runs a CreateML object-detector `.mlmodel` over images via coremltools."""

    def __init__(self, model_path: Path, threshold: float = 0.25) -> None:
        import coremltools as ct

        self.threshold = threshold
        self.model = ct.models.MLModel(str(model_path))
        image_input = self.model.get_spec().description.input[0]
        self.input_name = image_input.name
        self.input_w = image_input.type.imageType.width
        self.input_h = image_input.type.imageType.height

    def detect(self, image: Image.Image) -> Detection | None:
        padded, scale, pad_x, pad_y = letterbox(
            image.convert("RGB"), self.input_w, self.input_h
        )
        out = self.model.predict({self.input_name: padded})
        det = decode_createml_outputs(
            np.asarray(out["coordinates"]),
            np.asarray(out["confidence"]),
            self.threshold,
        )
        if det is None:
            return None
        orig_w, orig_h = image.size
        return Detection(
            unletterbox_box(det.box, scale, pad_x, pad_y,
                            self.input_w, self.input_h, orig_w, orig_h),
            det.confidence,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_detector.py -v`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
git add ml/clubhead/src/chdet/detector.py ml/clubhead/tests/test_detector.py
git commit -m "Add Core ML detector adapter with letterbox preprocessing"
```

- [ ] **Step 6: Smoke-test the real Core ML model**

Run from `ml/clubhead/` with the venv active:
```bash
python -c "
from pathlib import Path
from PIL import Image
from chdet.detector import CoreMLClubheadDetector
d = CoreMLClubheadDetector(Path('../../swing-tracker-ml.mlproj/Models/swing-tracker-ml 6.mlmodel'))
print('input size:', d.input_w, 'x', d.input_h)
img = next(Path('../../data/training/clubhead/images').glob('*.jpg'))
print('detection on', img.name, ':', d.detect(Image.open(img)))
"
```
Expected: prints the model input size and a `Detection(...)` (or `None`) with no exception. If `out["coordinates"]` / `out["confidence"]` raises a `KeyError`, inspect the real output names with `print(d.model.get_spec().description.output)` and update the two keys in `detect()` accordingly, then re-run Step 4 and this step. This is a verification-only step — no commit.

---

## Task 6: Eval harness

**Files:**
- Create: `ml/clubhead/src/chdet/evaluate.py`
- Test: `ml/clubhead/tests/test_evaluate.py`

The harness reads a test set laid out as `dataset/test/images/<source>/<frame>.jpg` with a matching `dataset/test/labels/<source>/<frame>.txt` for every image (empty `.txt` = negative frame). The pure computation (`compute_report`, `render_markdown`) is test-first; `collect_frames` is tested with `tmp_path`; the CLI `main` is verified in Task 9.

- [ ] **Step 1: Write the failing tests**

Create `ml/clubhead/tests/test_evaluate.py`:

```python
from chdet.detector import Detection
from chdet.evaluate import (
    FrameResult, compute_report, render_markdown, collect_frames,
)
from chdet.geometry import Box


def _hit(cx=0.5, cy=0.5):
    return Detection(Box(cx, cy, 0.1, 0.1), 0.9)


def test_compute_report_overall_detection_rate():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    results = [
        FrameResult("swingA", 0, _hit(), gt),
        FrameResult("swingA", 1, None, gt),       # miss
        FrameResult("swingA", 2, _hit(), gt),
    ]
    report = compute_report("fake", results, aspect=1.0)
    assert report.total_frames == 3
    assert report.detection_rate == 2 / 3


def test_compute_report_phantom_rate_uses_negative_frames():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    results = [
        FrameResult("swingA", 0, _hit(), gt),     # positive — excluded
        FrameResult("swingA", 1, _hit(), None),   # phantom
        FrameResult("swingA", 2, None, None),     # correct
    ]
    report = compute_report("fake", results, aspect=1.0)
    assert report.phantom_rate == 0.5


def test_compute_report_groups_per_source():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    results = [
        FrameResult("swingA", 0, _hit(), gt),
        FrameResult("swingB", 0, None, gt),
    ]
    report = compute_report("fake", results, aspect=1.0)
    sources = {s.source: s for s in report.per_source}
    assert sources["swingA"].detection_rate == 1.0
    assert sources["swingB"].detection_rate == 0.0


def test_render_markdown_contains_headline_numbers():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    report = compute_report("createml-v6",
                            [FrameResult("swingA", 0, _hit(), gt)], aspect=1.0)
    md = render_markdown(report)
    assert "createml-v6" in md
    assert "Detection rate" in md
    assert "swingA" in md


def test_collect_frames_pairs_images_with_labels(tmp_path):
    img_dir = tmp_path / "images" / "swingA"
    lbl_dir = tmp_path / "labels" / "swingA"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in (1, 2):
        (img_dir / f"frame-{i:04d}.jpg").write_bytes(b"")
        (lbl_dir / f"frame-{i:04d}.txt").write_text("")
    frames = collect_frames(tmp_path)
    assert [f.source for f in frames] == ["swingA", "swingA"]
    assert [f.index for f in frames] == [0, 1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chdet.evaluate'`.

- [ ] **Step 3: Write the implementation**

Create `ml/clubhead/src/chdet/evaluate.py`:

```python
"""Eval harness: score a clubhead detector on a held-out test set.

Test-set layout:
    <root>/images/<source>/<frame>.jpg
    <root>/labels/<source>/<frame>.txt   (empty file = negative frame)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from . import metrics
from .detector import CoreMLClubheadDetector, Detection, Detector
from .geometry import Box
from .labels import load_yolo_label


@dataclass
class FrameSpec:
    source: str
    index: int
    image_path: Path
    label_path: Path


@dataclass
class FrameResult:
    source: str
    index: int
    prediction: Detection | None
    ground_truth: Box | None


@dataclass
class SourceMetrics:
    source: str
    frames: int
    detection_rate: float
    median_center_error: float
    p90_center_error: float
    path_jitter: float


@dataclass
class Report:
    model_name: str
    total_frames: int
    detection_rate: float
    median_center_error: float
    p90_center_error: float
    phantom_rate: float
    path_jitter: float
    per_source: list[SourceMetrics]


def collect_frames(root: Path) -> list[FrameSpec]:
    """Discover every image in the test set, paired with its label file."""
    specs: list[FrameSpec] = []
    images_root = root / "images"
    for source_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        images = sorted(source_dir.glob("*.jpg"))
        for index, image_path in enumerate(images):
            label_path = root / "labels" / source_dir.name / f"{image_path.stem}.txt"
            specs.append(FrameSpec(source_dir.name, index, image_path, label_path))
    return specs


def run_detector(detector: Detector, specs: list[FrameSpec]) -> list[FrameResult]:
    """Run the detector over every frame and pair it with ground truth."""
    results: list[FrameResult] = []
    for spec in specs:
        with Image.open(spec.image_path) as img:
            prediction = detector.detect(img)
        ground_truth = (
            load_yolo_label(spec.label_path) if spec.label_path.exists() else None
        )
        results.append(
            FrameResult(spec.source, spec.index, prediction, ground_truth)
        )
    return results


def compute_report(
    model_name: str, results: list[FrameResult], aspect: float
) -> Report:
    """Aggregate per-frame results into per-source and overall metrics."""
    by_source: dict[str, list[FrameResult]] = {}
    for r in results:
        by_source.setdefault(r.source, []).append(r)

    per_source: list[SourceMetrics] = []
    all_pairs: list[tuple[Box | None, Box | None]] = []
    all_errors: list[float] = []

    for source, frs in sorted(by_source.items()):
        frs = sorted(frs, key=lambda r: r.index)
        pairs = [
            (r.prediction.box if r.prediction else None, r.ground_truth)
            for r in frs
        ]
        all_pairs.extend(pairs)
        errors = [
            metrics.center_error_pct(p, g, aspect)
            for p, g in pairs
            if p is not None and g is not None
        ]
        all_errors.extend(errors)
        centers = [
            (r.prediction.box.cx, r.prediction.box.cy) if r.prediction else None
            for r in frs
        ]
        has_present = any(g is not None for _, g in pairs)
        per_source.append(
            SourceMetrics(
                source=source,
                frames=len(frs),
                detection_rate=metrics.detection_rate(pairs) if has_present else 0.0,
                median_center_error=metrics.percentile(errors, 50) if errors else 0.0,
                p90_center_error=metrics.percentile(errors, 90) if errors else 0.0,
                path_jitter=metrics.path_jitter(centers, aspect),
            )
        )

    has_present = any(g is not None for _, g in all_pairs)
    return Report(
        model_name=model_name,
        total_frames=len(all_pairs),
        detection_rate=metrics.detection_rate(all_pairs) if has_present else 0.0,
        median_center_error=metrics.percentile(all_errors, 50) if all_errors else 0.0,
        p90_center_error=metrics.percentile(all_errors, 90) if all_errors else 0.0,
        phantom_rate=metrics.phantom_rate(all_pairs),
        path_jitter=(
            sum(s.path_jitter for s in per_source) / len(per_source)
            if per_source else 0.0
        ),
        per_source=per_source,
    )


def render_markdown(report: Report) -> str:
    """Render a Report as a human-readable Markdown document."""
    lines = [
        f"# Clubhead detector eval — {report.model_name}",
        "",
        f"- Total frames: {report.total_frames}",
        f"- **Detection rate:** {report.detection_rate:.1%}",
        f"- **Median center error:** {report.median_center_error:.2f}% of frame width",
        f"- **p90 center error:** {report.p90_center_error:.2f}% of frame width",
        f"- **Phantom rate:** {report.phantom_rate:.1%}",
        f"- **Mean path jitter:** {report.path_jitter:.3f}% of frame width",
        "",
        "## Per source",
        "",
        "| Source | Frames | Detection rate | Median err % | p90 err % | Jitter % |",
        "|---|---|---|---|---|---|",
    ]
    for s in report.per_source:
        lines.append(
            f"| {s.source} | {s.frames} | {s.detection_rate:.1%} "
            f"| {s.median_center_error:.2f} | {s.p90_center_error:.2f} "
            f"| {s.path_jitter:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a clubhead detector.")
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to a Core ML .mlmodel / .mlpackage")
    parser.add_argument("--test-set", required=True, type=Path,
                        help="Test-set root (contains images/ and labels/)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Markdown report output path (.json written alongside)")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args(argv)

    detector = CoreMLClubheadDetector(args.model, threshold=args.threshold)
    specs = collect_frames(args.test_set)
    if not specs:
        raise SystemExit(f"no frames found under {args.test_set}/images")

    with Image.open(specs[0].image_path) as first:
        aspect = first.width / first.height

    results = run_detector(detector, specs)
    report = compute_report(args.model.name, results, aspect)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report))
    args.out.with_suffix(".json").write_text(json.dumps(asdict(report), indent=2))
    print(f"wrote {args.out}")
    print(f"detection rate {report.detection_rate:.1%}, "
          f"median center error {report.median_center_error:.2f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS — 5 passed.

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS — all tests from Tasks 2–6 pass (33 total).

- [ ] **Step 6: Commit**

```bash
git add ml/clubhead/src/chdet/evaluate.py ml/clubhead/tests/test_evaluate.py
git commit -m "Add clubhead detector eval harness"
```

---

## Task 7: Test-set importer

**Files:**
- Create: `ml/clubhead/src/chdet/dataset.py`
- Create: `ml/clubhead/scripts/import_test_set.py`
- Test: `ml/clubhead/tests/test_dataset.py`

This converts a Label Studio export plus a directory of images into the
test-set layout the harness expects: it copies images under
`images/<source>/` and writes a YOLO `.txt` (empty for negative frames) under
`labels/<source>/`.

- [ ] **Step 1: Write the failing tests**

Create `ml/clubhead/tests/test_dataset.py`:

```python
import json

from chdet.dataset import import_label_studio_export


def _write_jpg(path):
    # Minimal valid file content; the importer copies bytes, it does not decode.
    path.write_bytes(b"\xff\xd8\xff\xd9")


def test_import_writes_images_and_labels_under_source(tmp_path):
    src_images = tmp_path / "src"
    src_images.mkdir()
    _write_jpg(src_images / "frame-0001.jpg")
    _write_jpg(src_images / "frame-0002.jpg")

    export = [
        {"data": {"image": "/data/local-files/?d=images/frame-0001.jpg"},
         "annotations": [{"result": [
             {"value": {"x": 40.0, "y": 30.0, "width": 20.0, "height": 40.0,
                        "rectanglelabels": ["clubhead"]}}]}]},
        {"data": {"image": "/data/local-files/?d=images/frame-0002.jpg"},
         "annotations": [{"result": []}]},
    ]
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export))

    dest = tmp_path / "dataset" / "test"
    count = import_label_studio_export(export_path, src_images, "swingX", dest)

    assert count == 2
    assert (dest / "images" / "swingX" / "frame-0001.jpg").exists()
    label = (dest / "labels" / "swingX" / "frame-0001.txt").read_text().strip()
    assert label.startswith("0 0.5")            # annotated frame
    negative = (dest / "labels" / "swingX" / "frame-0002.txt").read_text()
    assert negative == ""                       # negative frame -> empty label


def test_import_skips_export_entry_with_missing_image(tmp_path):
    src_images = tmp_path / "src"
    src_images.mkdir()
    export = [
        {"data": {"image": "/data/local-files/?d=images/missing.jpg"},
         "annotations": [{"result": []}]},
    ]
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export))
    dest = tmp_path / "dataset" / "test"
    count = import_label_studio_export(export_path, src_images, "swingX", dest)
    assert count == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chdet.dataset'`.

- [ ] **Step 3: Write the implementation**

Create `ml/clubhead/src/chdet/dataset.py`:

```python
"""Build the harness test-set layout from a Label Studio export."""
from __future__ import annotations

import shutil
from pathlib import Path

from .geometry import to_yolo_line
from .labels import load_label_studio_export


def import_label_studio_export(
    export_path: Path,
    images_src_dir: Path,
    source_name: str,
    dest_root: Path,
) -> int:
    """Materialise one labeled source into the test-set layout.

    For each task in the export whose image exists in `images_src_dir`, copies
    the image to `dest_root/images/<source_name>/` and writes a YOLO label to
    `dest_root/labels/<source_name>/` (empty file for a negative frame).
    Returns the number of frames imported.
    """
    labels = load_label_studio_export(export_path)
    images_dest = dest_root / "images" / source_name
    labels_dest = dest_root / "labels" / source_name
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)

    imported = 0
    for filename, box in labels.items():
        src = images_src_dir / filename
        if not src.exists():
            continue
        shutil.copy2(src, images_dest / filename)
        label_text = "" if box is None else to_yolo_line(box) + "\n"
        (labels_dest / f"{Path(filename).stem}.txt").write_text(label_text)
        imported += 1
    return imported
```

Create `ml/clubhead/scripts/import_test_set.py`:

```python
"""CLI: import a Label Studio export into the eval test-set layout.

Usage:
    python scripts/import_test_set.py \\
        --export path/to/export.json \\
        --images path/to/source/images \\
        --source swing-name \\
        --dest dataset/test
"""
from __future__ import annotations

import argparse
from pathlib import Path

from chdet.dataset import import_label_studio_export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True, type=Path)
    args = parser.parse_args()
    count = import_label_studio_export(
        args.export, args.images, args.source, args.dest
    )
    print(f"imported {count} frames for source '{args.source}' into {args.dest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add ml/clubhead/src/chdet/dataset.py ml/clubhead/scripts/import_test_set.py ml/clubhead/tests/test_dataset.py
git commit -m "Add Label Studio test-set importer"
```

---

## Task 8: Assemble & label the held-out test set

This task is **manual data work** — there is no code and no commit of code. It produces the labeled test set the baseline run needs. The deliverable is a populated `ml/clubhead/dataset/test/` directory (gitignored).

- [ ] **Step 1: Choose held-out source clips**

Pick **5–8 complete swing clips** that are **not** represented in `data/training/clubhead/` — different phones, players, lighting, and backgrounds (down-the-line and face-on). The candidate pool already extracted is the YouTube test swings referenced by `data/labeling/clubhead/extract_yt_test_swings_all_frames.py` and `~/Downloads/yt-test-swings/`. Record the chosen clip names in a new file `ml/clubhead/eval/test_set_sources.md` (one clip name per line, with a one-line note on why each is diverse).

- [ ] **Step 2: Extract frames for each clip**

For each chosen clip, extract its frames to a per-clip folder of `.jpg` files. Reuse the existing extraction script `data/labeling/clubhead/dump_all_frames.sh` (ffmpeg-based) or `extract_yt_test_swings_all_frames.py`. Keep each clip's frames in a separate directory.

- [ ] **Step 3: Label every frame in Label Studio**

Load each clip's frames into Label Studio and draw a tight `clubhead` box per the conventions in `ml/clubhead/docs/labeling-spec.md` (written in Task 10 — if doing tasks in order, do Task 10 first). Label **every** frame across the full swing, and leave frames with no visible clubhead **unannotated** — they become negative frames. Export each clip as Label Studio **JSON**.

- [ ] **Step 4: Import each clip into the test-set layout**

For each clip, run (from `ml/clubhead/`, venv active):
```bash
python scripts/import_test_set.py \
  --export <clip-export>.json \
  --images <clip-frames-dir> \
  --source <clip-name> \
  --dest dataset/test
```
Expected: prints `imported N frames for source '<clip-name>'`.

- [ ] **Step 5: Verify the test set is well-formed**

Run from `ml/clubhead/`:
```bash
python -c "
from pathlib import Path
from chdet.evaluate import collect_frames
specs = collect_frames(Path('dataset/test'))
sources = sorted({s.source for s in specs})
print(f'{len(specs)} frames across {len(sources)} sources: {sources}')
missing = [s.image_path.name for s in specs if not s.label_path.exists()]
assert not missing, f'images without a label file: {missing}'
print('every image has a label file — ok')
"
```
Expected: prints the frame and source counts and `every image has a label file — ok`.

- [ ] **Step 6: Commit the test-set source list**

```bash
git add ml/clubhead/eval/test_set_sources.md
git commit -m "Record held-out clubhead test-set source clips"
```
(The frames and labels under `dataset/` stay gitignored — only the source list is tracked.)

---

## Task 9: Capture the baseline

**Files:**
- Create: `ml/clubhead/eval/baseline/createml-v6.md` (and `.json`)

- [ ] **Step 1: Run the harness against the current CreateML model**

From `ml/clubhead/` with the venv active:
```bash
python -m chdet.evaluate \
  --model "../../swing-tracker-ml.mlproj/Models/swing-tracker-ml 6.mlmodel" \
  --test-set dataset/test \
  --out eval/baseline/createml-v6.md
```
Expected: prints `wrote eval/baseline/createml-v6.md` and a one-line detection-rate / center-error summary. Both `eval/baseline/createml-v6.md` and `eval/baseline/createml-v6.json` are written.

- [ ] **Step 2: Sanity-check the baseline numbers**

Open `eval/baseline/createml-v6.md`. Confirm: detection rate is a plausible fraction (not 0% or 100%), per-source rows are present for every test clip, and center-error / jitter numbers are finite. If detection rate is 0%, the most likely cause is wrong output keys in `detector.py` (revisit Task 5 Step 6) or a coordinate-convention mismatch — investigate before continuing.

- [ ] **Step 3: Commit the baseline (force-add past the gitignore)**

`eval/baseline/` is inside the gitignored `clubhead/eval/reports/` exclusion only for `reports/`, but to be safe add explicitly:
```bash
git add ml/clubhead/eval/baseline/createml-v6.md ml/clubhead/eval/baseline/createml-v6.json
git commit -m "Capture CreateML v6 clubhead detector baseline"
```
Expected: both files are committed. (If git reports them ignored, run `git add -f` on the two paths — the baseline must be tracked as the reference point for every future model.)

---

## Task 10: Write the labeling spec

**Files:**
- Create: `ml/clubhead/docs/labeling-spec.md`

- [ ] **Step 1: Write the labeling spec document**

Create `ml/clubhead/docs/labeling-spec.md`:

```markdown
# Clubhead labeling spec

How to draw `clubhead` boxes for the SwingSensei detector. Consistency here is
what fixes loose boxes (the low varied-IoU score). Apply every rule the same
way on every frame, in training data and test data alike.

## What "clubhead" means

The box encloses the **club head only** — the mass at the end of the shaft that
strikes the ball. It does **not** include the shaft or the grip.

- The box is the tightest axis-aligned rectangle that contains the whole head.
- For irons and wedges: the blade/face plus the hosel where it meets the head.
- For drivers/woods: the whole head volume.
- Include the head's full visible extent — toe, heel, sole, crown.

## The motion-blur rule (decided convention)

During the fast parts of the swing the clubhead is a blurred streak.

**Rule: box the full visible clubhead including its motion-blur streak.** The
rectangle covers the entire smeared region the head occupies in that frame.

Rationale: it is the most consistent, least subjective call and it matches what
is actually on screen. (This convention is flagged for revisit in phase 2 if it
turns out to drive path wobble.)

## Partial occlusion

If the clubhead is partially hidden (behind the body, ball, or frame edge):

- Box only the **visible** portion.
- If essentially all of the head is hidden, leave the frame **unannotated** —
  it becomes a negative frame.

## Negative frames

Leave a frame unannotated when no clubhead is meaningfully visible — for
example, follow-through frames where the club has left the frame, or non-swing
footage. Negative frames are first-class training data against phantom
detections; do not skip or delete them.

## One box per frame

There is exactly one clubhead. Never draw more than one `clubhead` box on a
frame. If two candidates seem plausible, pick the real club head and box only
that.
```

- [ ] **Step 2: Commit**

```bash
git add ml/clubhead/docs/labeling-spec.md
git commit -m "Add clubhead labeling spec"
```

---

## Done — Phase 0 complete

At this point: the `chdet` package is installed and fully tested, the eval
harness scores any Core ML clubhead model on the held-out test set, the
CreateML v6 **baseline is captured and committed**, and the labeling spec is
written. Every later phase is now measurable against `eval/baseline/`.

**Next:** Phase 1 (YOLO training pipeline + data engine) gets its own plan,
written when Phase 0 is verified complete.

## Self-review notes (for the plan author — not an execution step)

- **Spec coverage:** Phase 0 of the design spec (§10) calls for: project
  skeleton (Task 1), eval harness (Tasks 2–6), held-out test set (Task 8),
  baseline run (Task 9), labeling spec (Task 10). All covered. The harness
  metrics (Task 3) match spec §8: detection rate, center-point error
  (median + p90), phantom rate, path jitter. Training, the data engine, and
  iOS integration are correctly deferred to later-phase plans.
- **Type consistency:** `Box`, `Detection`, `FrameResult`, `FrameSpec`,
  `SourceMetrics`, `Report` are defined once and referenced with identical
  field names throughout. `decode_createml_outputs`, `letterbox`,
  `unletterbox_box`, `collect_frames`, `run_detector`, `compute_report`,
  `render_markdown`, `import_label_studio_export` keep one signature each.
- **Known risk surfaced in-plan:** the CreateML model's exact output key names
  are verified against the real model in Task 5 Step 6 before the harness
  depends on them.
```
