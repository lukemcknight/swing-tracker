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
