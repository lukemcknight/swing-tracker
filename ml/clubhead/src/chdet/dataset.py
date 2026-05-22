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
