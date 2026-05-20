"""
Detect swings in long range videos and export training frames in one pass.

Run:
    python backend/scripts/analyze_long_video.py path/to/video.mov [more paths...]
    python backend/scripts/analyze_long_video.py ~/Desktop/range_session/

For each long video:
  1. Pre-pass: run MediaPipe pose (lite, every Nth frame) on the entire video.
  2. Compute lead-wrist velocity over time.
  3. Find velocity peaks (each peak ≈ one swing impact).
  4. For each peak, run the existing swing analyzer on a ~3.5s window.
  5. Drop windows where phase detection fails (auto-rejects ball pickup, walking).
  6. Append surviving swings' clubHead positions as labeled training frames in
     the same format as export_training_frames.py.

Appends to existing data/training/clubhead/ — re-running is safe (per-swing IDs
are derived from the video filename so duplicate runs overwrite same labels).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# Avoid double-importing swing_analyzer (heavy) before the script-internal helpers.
from swing_analyzer import analyze_video, open_video_capture  # noqa: E402

# Reuse training-frame export helpers from the sibling script.
from export_training_frames import (  # noqa: E402
    export_clubhead_frames,
    is_test_swing,
    write_annotations,
)


# Pre-pass tunables
SAMPLE_STRIDE = 4              # every Nth frame in the pre-pass (smaller = slower, more precise)
MIN_SWING_SPACING_SEC = 2.5    # two peaks must be at least this far apart
PRE_IMPACT_WINDOW_SEC = 2.5    # capture 2.5 s before the peak
POST_IMPACT_WINDOW_SEC = 1.2   # capture 1.2 s after the peak
MIN_WRIST_VISIBILITY = 0.30    # MediaPipe wrist confidence floor
MIN_PEAK_VELOCITY = 0.10       # normalized coords/sec; below this is body sway, not a swing
MIN_PROMINENCE_RATIO = 0.35    # peak must exceed its local minimum by this fraction of global max

PoseLandmark = mp.solutions.pose.PoseLandmark


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir) / "clubhead"
    out_dir.joinpath("images").mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_annotations.json"
    test_path = out_dir / "test_annotations.json"

    train_anns = load_existing(train_path)
    test_anns = load_existing(test_path)
    starting_train = len(train_anns)
    starting_test = len(test_anns)

    videos = collect_videos(args.inputs)
    if not videos:
        print("No videos found in the given paths.", file=sys.stderr)
        return 1

    print(f"Processing {len(videos)} video(s).")
    print(f"Existing annotations: {starting_train} train, {starting_test} test.\n")

    for video_path in videos:
        process_video(
            video_path=video_path,
            images_dir=out_dir / "images",
            train_bucket=train_anns,
            test_bucket=test_anns,
            sample_stride=args.sample_stride,
        )

    write_annotations(train_path, train_anns)
    write_annotations(test_path, test_anns)

    added_train = len(train_anns) - starting_train
    added_test = len(test_anns) - starting_test
    print(
        f"\nDone. Added {added_train} train + {added_test} test annotations "
        f"({added_train + added_test} new total)."
    )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "inputs",
        nargs="+",
        help="Video files or directories containing .mov/.mp4 files.",
    )
    p.add_argument(
        "--output-dir",
        default=str(BACKEND_DIR.parent / "data" / "training"),
        help="Where to append training frames (default: data/training).",
    )
    p.add_argument(
        "--sample-stride",
        type=int,
        default=SAMPLE_STRIDE,
        help=f"Sample every Nth frame during impact detection (default: {SAMPLE_STRIDE}).",
    )
    return p.parse_args()


def collect_videos(inputs: list[str]) -> list[Path]:
    videos: list[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            for ext in ("*.mov", "*.MOV", "*.mp4", "*.MP4"):
                videos.extend(sorted(p.glob(ext)))
        elif p.is_file():
            videos.append(p)
        else:
            print(f"warning: {p} not found, skipping")
    return videos


def process_video(
    video_path: Path,
    images_dir: Path,
    train_bucket: list[dict],
    test_bucket: list[dict],
    sample_stride: int,
) -> None:
    print(f"=== {video_path.name} ===")

    impacts_ms, fps, duration_ms = detect_swing_impacts(video_path, sample_stride)
    if not impacts_ms:
        print("  no impact candidates found; skipping\n")
        return

    print(f"  {len(impacts_ms)} impact candidates at {fps:.1f} fps "
          f"({duration_ms / 1000:.1f}s of footage)")

    accepted = 0
    rejected = 0
    no_phase = 0
    analyzer_error = 0
    too_short = 0

    for index, impact_ms in enumerate(impacts_ms):
        trim_start = max(0, impact_ms - int(PRE_IMPACT_WINDOW_SEC * 1000))
        trim_end = min(duration_ms, impact_ms + int(POST_IMPACT_WINDOW_SEC * 1000))
        if trim_end - trim_start < 1500:
            too_short += 1
            continue

        swing_id = f"{video_path.stem}-{index:02d}"

        try:
            analysis = analyze_video(
                swing_id=swing_id,
                video_path=video_path,
                trim_start_ms=trim_start,
                trim_end_ms=trim_end,
            )
        except Exception as exc:
            print(f"  swing {index:02d}: analyzer error ({exc.__class__.__name__})")
            analyzer_error += 1
            continue

        positions = analysis.get("positions") or {}
        if "impact" not in positions or "address" not in positions:
            no_phase += 1
            continue

        bucket = test_bucket if is_test_swing(swing_id) else train_bucket
        exported = export_clubhead_frames(
            swing_id=swing_id,
            video_path=video_path,
            analysis=analysis,
            rotation=None,
            mirror=False,
            images_dir=images_dir,
            bucket=bucket,
        )

        if exported > 0:
            accepted += 1
        else:
            rejected += 1

    print(
        f"  accepted={accepted}  rejected={rejected}  "
        f"no_phase={no_phase}  analyzer_error={analyzer_error}  too_short={too_short}\n"
    )


def detect_swing_impacts(
    video_path: Path,
    sample_stride: int,
) -> tuple[list[int], float, int]:
    """Run a lite pose pre-pass; return (impact_times_ms, source_fps, duration_ms)."""
    capture = open_video_capture(video_path)
    if not capture.isOpened():
        return [], 0.0, 0

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_ms = int(total_frames / fps * 1000) if total_frames > 0 else 0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,            # lite — enough for wrist tracking
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    samples: list[tuple[int, float, float, float, float]] = []
    # (t_ms, lw_x, lw_y, rw_x, rw_y)

    frame_idx = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_idx % sample_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    lm = result.pose_landmarks.landmark
                    lw = lm[PoseLandmark.LEFT_WRIST]
                    rw = lm[PoseLandmark.RIGHT_WRIST]
                    if lw.visibility > MIN_WRIST_VISIBILITY and rw.visibility > MIN_WRIST_VISIBILITY:
                        t_ms = frame_idx / fps * 1000.0
                        samples.append((t_ms, lw.x, lw.y, rw.x, rw.y))
            frame_idx += 1
    finally:
        capture.release()
        pose.close()

    if len(samples) < 10:
        return [], fps, duration_ms

    # Compute max-wrist linear velocity (normalized coords per second).
    velocities: list[float] = []
    times: list[float] = []
    for i in range(1, len(samples)):
        dt = (samples[i][0] - samples[i - 1][0]) / 1000.0
        if dt <= 0:
            continue
        lv = float(np.hypot(samples[i][1] - samples[i - 1][1], samples[i][2] - samples[i - 1][2]) / dt)
        rv = float(np.hypot(samples[i][3] - samples[i - 1][3], samples[i][4] - samples[i - 1][4]) / dt)
        velocities.append(max(lv, rv))
        times.append(samples[i][0])

    if not velocities:
        return [], fps, duration_ms

    peaks = find_velocity_peaks(
        np.asarray(velocities),
        min_distance_samples=max(2, int(MIN_SWING_SPACING_SEC * fps / sample_stride)),
    )

    impacts_ms = sorted(int(times[p]) for p in peaks)
    return impacts_ms, fps, duration_ms


def find_velocity_peaks(values: np.ndarray, min_distance_samples: int) -> list[int]:
    """Greedy non-max suppression peak finder.

    A peak must:
      * be a local maximum (strictly greater than its immediate neighbors),
      * exceed MIN_PEAK_VELOCITY,
      * have a prominence at least MIN_PROMINENCE_RATIO * max(values),
      * be at least min_distance_samples away from any already-accepted peak.
    """
    if len(values) == 0:
        return []

    global_max = float(values.max())
    if global_max <= MIN_PEAK_VELOCITY:
        return []

    prominence_floor = MIN_PROMINENCE_RATIO * global_max
    candidates: list[int] = []
    for i in range(1, len(values) - 1):
        v = float(values[i])
        if v <= values[i - 1] or v <= values[i + 1]:
            continue
        if v < MIN_PEAK_VELOCITY:
            continue
        if v < prominence_floor:
            continue
        candidates.append(i)

    if not candidates:
        return []

    candidates.sort(key=lambda i: -float(values[i]))

    accepted: list[int] = []
    for c in candidates:
        if all(abs(c - a) >= min_distance_samples for a in accepted):
            accepted.append(c)

    accepted.sort()
    return accepted


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


if __name__ == "__main__":
    sys.exit(main())
