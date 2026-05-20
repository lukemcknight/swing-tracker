"""
Export labeled training frames for CoreML object detectors (club head, ball).

Reads an iOS-format `swings.json` and per-swing videos, then writes JPEGs +
CreateML-compatible annotations bootstrapped from the existing backend
analysis output:
  - Club head: pulled directly from analysis.frames[i].clubHead (always present
    for completed swings).
  - Ball: re-runs the analyzer's post-impact ball tracker (off by default; pass
    --include-ball to enable).

Output layout (under --output-dir):
  clubhead/
    images/<swing-id>-<frame-index>.jpg
    train_annotations.json     (CreateML object detection format)
    test_annotations.json
  ball/
    images/...
    train_annotations.json
    test_annotations.json

CreateML object-detection annotation entry:
  {
    "image": "<filename>",
    "annotations": [
      {"label": "club_head", "coordinates": {"x": 0.5, "y": 0.7, "width": 0.08, "height": 0.08}}
    ]
  }
Coordinates are normalized [0, 1]; x/y are box CENTER.

Defaults assume you run this from the repo root with the backend virtualenv
activated, pointing at the iPhone 17 simulator's SwingSensei container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2

# Add backend/ to sys.path so we can import the analyzer's helpers.
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from swing_analyzer import (  # noqa: E402  (after sys.path mutation)
    apply_video_edit,
    open_video_capture,
    track_ball_flight_from_video,
)


DEFAULT_SWINGS_JSON = (
    "/Users/lukemck/Library/Developer/CoreSimulator/Devices/"
    "0F35FF08-B8B4-409C-8081-F7BE03D3DEE2/data/Containers/Data/Application/"
    "934DD8D7-9CC5-4903-8922-95E4C8154D4C/Documents/SwingSensei/swings.json"
)
DEFAULT_VIDEOS_DIR = (
    "/Users/lukemck/Library/Developer/CoreSimulator/Devices/"
    "0F35FF08-B8B4-409C-8081-F7BE03D3DEE2/data/Containers/Data/Application/"
    "934DD8D7-9CC5-4903-8922-95E4C8154D4C/Documents/SwingSensei/videos"
)
DEFAULT_OUTPUT_DIR = BACKEND_DIR.parent / "data" / "training"

# Box sizing (fraction of min(frame_width, frame_height))
CLUB_HEAD_BOX_FRACTION = 0.07
BALL_BOX_FRACTION = 0.025

# Holdout fraction for the test set, by stable swing-ID hash.
TEST_HOLDOUT_FRACTION = 0.20


def main() -> int:
    args = parse_args()
    swings = load_swings(Path(args.swings_json))
    print(f"Loaded {len(swings)} swings from {args.swings_json}")
    videos_dir = Path(args.videos_dir) if args.videos_dir else None
    if videos_dir is not None:
        print(f"Falling back to videos under {videos_dir} by filename")

    clubhead_out = Path(args.output_dir) / "clubhead"
    ball_out = Path(args.output_dir) / "ball"
    clubhead_out.joinpath("images").mkdir(parents=True, exist_ok=True)
    if args.include_ball:
        ball_out.joinpath("images").mkdir(parents=True, exist_ok=True)

    clubhead_train: list[dict] = []
    clubhead_test: list[dict] = []
    ball_train: list[dict] = []
    ball_test: list[dict] = []

    eligible = [s for s in swings if is_eligible(s)]
    print(f"{len(eligible)} eligible (status=complete, has analysis, has video)")

    for swing in eligible:
        swing_id = swing["id"]
        video_path = resolve_video_path(swing, args.swings_json, videos_dir)
        if not video_path or not video_path.exists():
            print(f"  [skip] {swing_id}: video not found at {video_path}")
            continue

        analysis = swing["analysis"]
        rotation = (swing.get("videoEditState") or {}).get("rotationDegrees")
        mirror = bool((swing.get("videoEditState") or {}).get("isMirroredHorizontally"))
        fps = float(analysis.get("fps") or 0)
        if fps <= 0:
            print(f"  [skip] {swing_id}: missing fps")
            continue

        is_test = is_test_swing(swing_id)
        clubhead_bucket = clubhead_test if is_test else clubhead_train
        ball_bucket = ball_test if is_test else ball_train

        # --- Club head extraction (from existing analysis JSON) ---
        club_frames_exported = export_clubhead_frames(
            swing_id=swing_id,
            video_path=video_path,
            analysis=analysis,
            rotation=rotation,
            mirror=mirror,
            images_dir=clubhead_out / "images",
            bucket=clubhead_bucket,
        )

        # --- Ball extraction (re-runs analyzer's tracker) ---
        ball_frames_exported = 0
        if args.include_ball:
            ball_frames_exported = export_ball_frames(
                swing_id=swing_id,
                video_path=video_path,
                analysis=analysis,
                rotation=rotation,
                mirror=mirror,
                images_dir=ball_out / "images",
                bucket=ball_bucket,
            )

        print(
            f"  {swing_id}: {club_frames_exported} club-head, "
            f"{ball_frames_exported} ball ({'test' if is_test else 'train'})"
        )

    write_annotations(clubhead_out / "train_annotations.json", clubhead_train)
    write_annotations(clubhead_out / "test_annotations.json", clubhead_test)
    print(
        f"Wrote {len(clubhead_train)} train + {len(clubhead_test)} test "
        f"club-head annotations to {clubhead_out}"
    )

    if args.include_ball:
        write_annotations(ball_out / "train_annotations.json", ball_train)
        write_annotations(ball_out / "test_annotations.json", ball_test)
        print(
            f"Wrote {len(ball_train)} train + {len(ball_test)} test "
            f"ball annotations to {ball_out}"
        )

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--swings-json",
        default=DEFAULT_SWINGS_JSON,
        help="Path to iOS-format swings.json (default: iPhone 17 simulator path)",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write images/ and annotations.json (default: data/training)",
    )
    p.add_argument(
        "--videos-dir",
        default=DEFAULT_VIDEOS_DIR,
        help=(
            "Fallback directory for video lookup by filename. Used when the "
            "videoURL in swings.json points at a stale app-container UUID, "
            "which is the common case in iOS simulator development."
        ),
    )
    p.add_argument(
        "--include-ball",
        action="store_true",
        help="Also re-run the ball tracker and export ball training frames (slower).",
    )
    return p.parse_args()


def load_swings(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"swings.json not found: {path}")
    return json.loads(path.read_text())


def is_eligible(swing: dict) -> bool:
    return (
        swing.get("status") == "complete"
        and bool(swing.get("analysis"))
        and bool(swing.get("videoURL"))
    )


def resolve_video_path(
    swing: dict,
    swings_json_path: str,
    videos_dir: Path | None,
) -> Path | None:
    raw = swing.get("videoURL", "")
    candidate: Path | None = None
    if raw.startswith("file://"):
        candidate = Path(raw[7:])
    elif raw:
        candidate = Path(swings_json_path).parent / raw

    if candidate is not None and candidate.exists():
        return candidate

    # Fallback: look up by filename in --videos-dir. Handles the common case
    # of stale app-container UUIDs in simulator-written swings.json paths.
    if videos_dir is not None and candidate is not None:
        fallback = videos_dir / candidate.name
        if fallback.exists():
            return fallback

    return candidate


def is_test_swing(swing_id: str) -> bool:
    """Deterministic test/train split: ~20% by SHA-256 hash of the swing ID.

    Stable across re-runs so labels don't leak between train/test."""
    digest = hashlib.sha256(swing_id.encode("utf-8")).digest()
    # First two bytes → integer in [0, 65535]; threshold at 20%.
    bucket = (digest[0] << 8) | digest[1]
    return bucket < int(0xFFFF * TEST_HOLDOUT_FRACTION)


def export_clubhead_frames(
    swing_id: str,
    video_path: Path,
    analysis: dict,
    rotation: int | None,
    mirror: bool,
    images_dir: Path,
    bucket: list[dict],
) -> int:
    fps = float(analysis["fps"])
    capture = open_video_capture(video_path)
    if not capture.isOpened():
        return 0

    exported = 0
    try:
        for frame_entry in analysis.get("frames", []):
            club = frame_entry.get("clubHead")
            if not club:
                continue
            t_ms = int(frame_entry.get("t", -1))
            if t_ms < 0:
                continue
            frame_idx = int(round(t_ms / 1000.0 * fps))
            image = read_frame_at(capture, frame_idx, rotation, mirror)
            if image is None:
                continue

            x = float(club["x"])
            y = float(club["y"])
            entry = make_annotation_entry(
                image=image,
                label="club_head",
                center_x=x,
                center_y=y,
                box_fraction=CLUB_HEAD_BOX_FRACTION,
                image_dir=images_dir,
                filename=f"{swing_id}-{frame_idx:04d}.jpg",
            )
            if entry is not None:
                bucket.append(entry)
                exported += 1
    finally:
        capture.release()

    return exported


def export_ball_frames(
    swing_id: str,
    video_path: Path,
    analysis: dict,
    rotation: int | None,
    mirror: bool,
    images_dir: Path,
    bucket: list[dict],
) -> int:
    fps = float(analysis["fps"])
    positions = analysis.get("positions") or {}
    address_index = positions.get("address")
    impact_index = positions.get("impact")
    frames = analysis.get("frames") or []

    if address_index is None or impact_index is None:
        return 0
    if not (0 <= impact_index < len(frames)):
        return 0

    # Address-frame ball anchor: average club-head position in a small window
    # around the address frame (proxy for ball location since the club rests on
    # the ball at address). Mirrors analyzer.address_ball_anchor() logic but
    # tolerant of different window sizes.
    anchor = compute_address_ball_anchor(frames, address_index)
    if anchor is None:
        return 0

    impact_frame = frames[impact_index]
    impact_time_ms = int(impact_frame.get("t", -1))
    if impact_time_ms < 0:
        return 0

    ball_points = track_ball_flight_from_video(
        video_path=video_path,
        impact_time_ms=impact_time_ms,
        source_fps=fps,
        ball_anchor=anchor,
        anchor_frame=frames[address_index],
        rotation_degrees=rotation,
        mirror_horizontal=mirror,
    )
    if not ball_points:
        return 0

    capture = open_video_capture(video_path)
    if not capture.isOpened():
        return 0

    exported = 0
    try:
        for point in ball_points:
            frame_idx = int(point.get("frame", -1))
            if frame_idx < 0:
                # Some entries don't carry frame number; derive from offset_ms.
                offset_ms = float(point.get("offsetMs", 0))
                frame_idx = int(round((impact_time_ms + offset_ms) / 1000.0 * fps))
            image = read_frame_at(capture, frame_idx, rotation, mirror)
            if image is None:
                continue

            entry = make_annotation_entry(
                image=image,
                label="golf_ball",
                center_x=float(point["x"]),
                center_y=float(point["y"]),
                box_fraction=BALL_BOX_FRACTION,
                image_dir=images_dir,
                filename=f"{swing_id}-{frame_idx:04d}.jpg",
            )
            if entry is not None:
                bucket.append(entry)
                exported += 1
    finally:
        capture.release()

    return exported


def compute_address_ball_anchor(
    frames: list[dict], address_index: int
) -> tuple[float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for offset in range(-2, 3):
        idx = address_index + offset
        if not (0 <= idx < len(frames)):
            continue
        club = frames[idx].get("clubHead")
        if not club:
            continue
        xs.append(float(club["x"]))
        ys.append(float(club["y"]))
    if len(xs) < 2:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def read_frame_at(
    capture: cv2.VideoCapture,
    frame_idx: int,
    rotation: int | None,
    mirror: bool,
):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = capture.read()
    if not success:
        return None
    return apply_video_edit(frame, rotation_degrees=rotation, mirror_horizontal=mirror)


def make_annotation_entry(
    image,
    label: str,
    center_x: float,
    center_y: float,
    box_fraction: float,
    image_dir: Path,
    filename: str,
) -> dict | None:
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return None

    # Clamp center to image bounds.
    cx = max(0.0, min(1.0, center_x))
    cy = max(0.0, min(1.0, center_y))

    # Box size as a fraction of the shorter side, then normalized per axis.
    base_size = min(width, height) * box_fraction
    box_w = base_size / width
    box_h = base_size / height

    # Ensure the box fits inside the image.
    cx = max(box_w / 2, min(1.0 - box_w / 2, cx))
    cy = max(box_h / 2, min(1.0 - box_h / 2, cy))

    output_path = image_dir / filename
    if not cv2.imwrite(str(output_path), image):
        return None

    # CreateML's Object Detection template expects PIXEL coordinates with the
    # box CENTER at (x, y). Normalized coords look like sub-pixel garbage to
    # CreateML and trigger a misleading "could not identify filename" error.
    return {
        "image": filename,
        "annotations": [
            {
                "label": label,
                "coordinates": {
                    "x": round(cx * width, 2),
                    "y": round(cy * height, 2),
                    "width": round(box_w * width, 2),
                    "height": round(box_h * height, 2),
                },
            }
        ],
    }


def write_annotations(path: Path, entries: Iterable[dict]) -> None:
    path.write_text(json.dumps(list(entries), indent=2))


if __name__ == "__main__":
    sys.exit(main())
