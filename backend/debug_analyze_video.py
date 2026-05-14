from __future__ import annotations

import argparse
import json
from pathlib import Path

from swing_analyzer import analyze_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local swing analysis and emit debug artifacts.")
    parser.add_argument("video", type=Path, help="Path to a local video file.")
    parser.add_argument("--swing-id", default=None, help="Stable id used for debug artifact filenames.")
    parser.add_argument("--duration-ms", type=int, default=None, help="Client-reported playback duration in milliseconds.")
    parser.add_argument("--trim-start-ms", type=int, default=None)
    parser.add_argument("--trim-end-ms", type=int, default=None)
    parser.add_argument("--rotation-degrees", type=int, default=None)
    parser.add_argument("--mirror-horizontal", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=Path(__file__).resolve().parent / "debug")
    parser.add_argument("--debug-url-path", default=None)
    parser.add_argument("--full-json", action="store_true", help="Print the full analysis payload instead of a compact summary.")
    args = parser.parse_args()

    swing_id = args.swing_id or args.video.stem
    analysis = analyze_video(
        swing_id=swing_id,
        video_path=args.video,
        client_duration_ms=args.duration_ms,
        debug_dir=args.debug_dir,
        debug_url_path=args.debug_url_path,
        trim_start_ms=args.trim_start_ms,
        trim_end_ms=args.trim_end_ms,
        rotation_degrees=args.rotation_degrees,
        mirror_horizontal=args.mirror_horizontal,
    )

    if args.full_json:
        print(json.dumps(analysis, indent=2))
        return

    summary = {
        "id": analysis["id"],
        "analysisVersion": analysis.get("analysisVersion"),
        "durationMs": analysis.get("durationMs"),
        "analysisQuality": analysis.get("analysisQuality"),
        "phaseConfidence": analysis.get("phaseConfidence"),
        "positionsMs": {
            name: analysis["frames"][index]["t"]
            for name, index in analysis.get("positions", {}).items()
            if 0 <= index < len(analysis.get("frames", []))
        },
        "rawSenseiScore": analysis.get("rawSenseiScore"),
        "senseiScore": analysis.get("senseiScore"),
        "debug": analysis.get("debug"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
