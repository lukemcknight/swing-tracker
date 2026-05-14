import asyncio
import json
import os
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemini_analyzer import (
    GeminiResponseError,
    GeminiTimeoutError,
    build_gemini_prompt,
    parse_gemini_payload,
    phase_timestamps,
    prepare_gemini_video_clip,
    wait_for_gemini_file_active,
)
from main import ai_analysis


class FakeGeminiFile:
    def __init__(self, state: str | None, name: str = "files/video-1", error=None):
        self.name = name
        self.state = state
        self.error = error


class FakeGeminiFilesClient:
    def __init__(self, files: list[FakeGeminiFile]):
        self.files = files
        self.get_names: list[str] = []

    def get(self, *, name: str) -> FakeGeminiFile:
        self.get_names.append(name)
        return self.files.pop(0)


class FakeGeminiClient:
    def __init__(self, files: list[FakeGeminiFile]):
        self.files = FakeGeminiFilesClient(files)


def make_analysis() -> dict:
    return {
        "analysisVersion": 21,
        "durationMs": 1400,
        "senseiScore": 7.1,
        "rawSenseiScore": 7.4,
        "positions": {
            "address": 0,
            "takeaway": 1,
            "top": 2,
            "impact": 3,
            "finish": 4,
        },
        "frames": [
            {"t": 0, "keypoints": []},
            {"t": 300, "keypoints": []},
            {"t": 700, "keypoints": []},
            {"t": 950, "keypoints": []},
            {"t": 1300, "keypoints": []},
        ],
        "metrics": {
            "headDisplacement": 0.08,
            "shoulderTurn": 31,
            "hipSway": 0.05,
            "postureLoss": 9,
            "tempo": 1.4,
        },
        "analysisQuality": {
            "status": "warning",
            "cameraAngle": "down_the_line",
            "captureConfidence": 0.7,
            "scoreConfidence": 0.66,
            "warnings": ["provisional_analysis"],
        },
        "phaseConfidence": {
            "impact": 0.62,
        },
        "feedback": [
            {
                "position": "impact",
                "label": "Posture loss",
                "status": "warning",
                "detail": "Your chest rises before impact.",
            }
        ],
    }


def write_test_video(path: Path, frame_count: int = 12) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (6, 4))
    if not writer.isOpened():
        raise RuntimeError("Could not create test video.")

    try:
        for index in range(frame_count):
            frame = np.zeros((4, 6, 3), dtype=np.uint8)
            frame[:, :3] = (index * 10, 20, 40)
            frame[:, 3:] = (200, index * 10, 80)
            writer.write(frame)
    finally:
        writer.release()


class GeminiAnalyzerTests(unittest.TestCase):
    def test_missing_gemini_api_key_returns_503(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(
                    ai_analysis(
                        swing_id="swing-1",
                        club="7",
                        analysis_json=UploadFile(
                            filename="analysis.json",
                            file=BytesIO(json.dumps(make_analysis()).encode("utf-8")),
                        ),
                        video=UploadFile(filename="swing.mp4", file=BytesIO(b"placeholder")),
                    )
                )

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("GEMINI_API_KEY", context.exception.detail)

    def test_prompt_includes_club_score_phase_quality_and_feedback(self):
        prompt = build_gemini_prompt("swing-1", "7", make_analysis())

        self.assertIn("Selected club: 7", prompt)
        self.assertIn('"senseiScore": 7.1', prompt)
        self.assertIn('"impact": 950', prompt)
        self.assertIn("provisional_analysis", prompt)
        self.assertIn("Your chest rises before impact.", prompt)
        self.assertIn("Do not infer ball flight", prompt)

    def test_phase_timestamps_uses_position_indices(self):
        timestamps = phase_timestamps(make_analysis())

        self.assertEqual(timestamps["address"], 0)
        self.assertEqual(timestamps["impact"], 950)
        self.assertEqual(timestamps["finish"], 1300)

    def test_parse_gemini_payload_validates_structured_feedback(self):
        payload = parse_gemini_payload(
            json.dumps(
                {
                    "coachScore": 12,
                    "headline": "Good structure with one main contact leak.",
                    "summary": "The swing is balanced but posture changes through impact.",
                    "confidence": "Very sure",
                    "strengths": ["Balanced setup"],
                    "priorityFixes": [
                        {
                            "title": "Posture through impact",
                            "phase": "impact",
                            "observation": "Chest rises before contact.",
                            "whyItMatters": "Low point moves.",
                            "cue": "Keep sternum down.",
                        }
                    ],
                    "practicePlan": [
                        {
                            "title": "Impact pauses",
                            "reps": "3 sets of 8",
                            "focus": "Hold posture at impact.",
                        }
                    ],
                    "limitations": ["Ball flight is not visible."],
                }
            )
        )

        self.assertEqual(payload.coachScore, 10.0)
        self.assertEqual(payload.confidence, "medium")
        self.assertEqual(payload.priorityFixes[0].cue, "Keep sternum down.")

    def test_prepare_gemini_video_clip_applies_trim_and_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            output = Path(directory) / "output.mp4"
            write_test_video(source)

            prepare_gemini_video_clip(
                source_path=source,
                output_path=output,
                client_duration_ms=1200,
                trim_start_ms=200,
                trim_end_ms=1000,
                rotation_degrees=90,
                mirror_horizontal=True,
            )

            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            try:
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                capture.release()

        self.assertGreater(frame_count, 0)
        self.assertLess(frame_count, 12)
        self.assertEqual((width, height), (4, 6))

    def test_wait_for_gemini_file_active_polls_until_file_is_ready(self):
        client = FakeGeminiClient([FakeGeminiFile("ACTIVE")])

        with patch.dict(
            os.environ,
            {
                "GEMINI_FILE_ACTIVE_TIMEOUT_SECONDS": "1",
                "GEMINI_FILE_POLL_INTERVAL_SECONDS": "0",
            },
        ):
            file = wait_for_gemini_file_active(client, FakeGeminiFile("PROCESSING"))

        self.assertEqual(file.state, "ACTIVE")
        self.assertEqual(client.files.get_names, ["files/video-1"])

    def test_wait_for_gemini_file_active_fails_on_failed_state(self):
        with self.assertRaises(GeminiResponseError):
            wait_for_gemini_file_active(
                FakeGeminiClient([]),
                FakeGeminiFile("FAILED", error=type("Error", (), {"message": "bad video"})()),
            )

    def test_wait_for_gemini_file_active_times_out(self):
        with patch.dict(os.environ, {"GEMINI_FILE_ACTIVE_TIMEOUT_SECONDS": "0"}):
            with self.assertRaises(GeminiTimeoutError):
                wait_for_gemini_file_active(FakeGeminiClient([]), FakeGeminiFile("PROCESSING"))


if __name__ == "__main__":
    unittest.main()
