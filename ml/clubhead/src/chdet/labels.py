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
