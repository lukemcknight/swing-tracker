from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import cv2
from pydantic import BaseModel, Field, ValidationError

from swing_analyzer import (
    POSITION_NAMES,
    apply_video_edit,
    client_time_to_source_time,
    frame_timestamp_ms,
    normalize_rotation_degrees,
    open_video_capture,
    timeline_scale_for,
)


class GeminiConfigError(RuntimeError):
    pass


class GeminiResponseError(RuntimeError):
    pass


class GeminiTimeoutError(TimeoutError):
    pass


class GeminiPriorityFix(BaseModel):
    title: str = Field(description="Short name for the flaw to fix.")
    phase: str | None = Field(default=None, description="Swing phase where this appears, if clear.")
    observation: str = Field(description="What is visible in the video or supported by metrics.")
    whyItMatters: str = Field(description="Why this changes contact, start line, power, or consistency.")
    cue: str = Field(description="One concise feel or checkpoint the golfer can try.")


class GeminiPracticeDrill(BaseModel):
    title: str = Field(description="Short drill name.")
    reps: str = Field(description="Simple dosage, for example '3 sets of 8 slow reps'.")
    focus: str = Field(description="What the golfer should pay attention to during the drill.")


class GeminiCoachPayload(BaseModel):
    coachScore: float = Field(description="Coach quality score from 0 to 10.")
    headline: str = Field(description="One sentence summary of the swing.")
    summary: str = Field(description="Specific swing feedback in two to four sentences.")
    confidence: str = Field(description="One of high, medium, or low.")
    strengths: list[str] = Field(description="Two to four visible strengths.")
    priorityFixes: list[GeminiPriorityFix] = Field(description="Two to four highest value fixes.")
    practicePlan: list[GeminiPracticeDrill] = Field(description="Two or three drills for the next practice session.")
    limitations: list[str] = Field(description="Important caveats about visibility, angle, confidence, or missing evidence.")


def build_gemini_prompt(swing_id: str, club: str, analysis: dict[str, Any]) -> str:
    digest = {
        "swingId": swing_id,
        "club": club,
        "senseiScore": analysis.get("senseiScore"),
        "rawSenseiScore": analysis.get("rawSenseiScore"),
        "durationMs": analysis.get("durationMs"),
        "analysisVersion": analysis.get("analysisVersion"),
        "phaseTimestampsMs": phase_timestamps(analysis),
        "metrics": analysis.get("metrics"),
        "analysisQuality": analysis.get("analysisQuality"),
        "phaseConfidence": analysis.get("phaseConfidence"),
        "swingThought": analysis.get("swingThought"),
        "existingFeedback": compact_feedback(analysis),
    }

    return f"""
You are an expert golf swing coach reviewing one user-uploaded swing video.

Use the selected club, the visible video, and the app's pose analysis together:
- Selected club: {club}
- Treat the app metrics and phase timestamps as measurement hints, not absolute truth.
- Give evidence-based coaching only. If a point is not visible or not supported by metrics, list it as a limitation instead of guessing.
- Do not infer ball flight, carry distance, launch, spin, or target result unless the ball flight is plainly visible in the video.
- Prefer a small number of high-leverage fixes over a long generic checklist.
- Make the feedback practical enough that the golfer can record another swing and check the same cue.
- For wedges and short irons, value low-point control and face/contact consistency; for woods and hybrids, value posture, sequencing, and path stability.

App analysis digest:
{json.dumps(digest, indent=2, sort_keys=True)}
""".strip()


def phase_timestamps(analysis: dict[str, Any]) -> dict[str, int]:
    frames = analysis.get("frames") or []
    positions = analysis.get("positions") or {}
    timestamps: dict[str, int] = {}

    for phase in POSITION_NAMES:
        index = positions.get(phase)
        if isinstance(index, int) and 0 <= index < len(frames):
            time_ms = frames[index].get("t")
            if isinstance(time_ms, int):
                timestamps[phase] = time_ms

    return timestamps


def compact_feedback(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    items = analysis.get("feedback") or []
    compacted = []

    for item in items:
        compacted.append(
            {
                "position": item.get("position"),
                "label": item.get("label"),
                "status": item.get("status"),
                "detail": item.get("detail"),
            }
        )

    return compacted


async def analyze_swing_with_gemini(
    swing_id: str,
    club: str,
    video_path: Path,
    analysis: dict[str, Any],
    client_duration_ms: int | None = None,
    trim_start_ms: int | None = None,
    trim_end_ms: int | None = None,
    rotation_degrees: int | None = None,
    mirror_horizontal: bool = False,
) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiConfigError("GEMINI_API_KEY is not configured.")

    timeout_seconds = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "180"))

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _analyze_swing_with_gemini_sync,
                api_key,
                swing_id,
                club,
                video_path,
                analysis,
                client_duration_ms,
                trim_start_ms,
                trim_end_ms,
                rotation_degrees,
                mirror_horizontal,
            ),
            timeout=timeout_seconds,
        )
    except GeminiTimeoutError:
        raise
    except TimeoutError as error:
        raise GeminiTimeoutError("Gemini analysis timed out.") from error


def _analyze_swing_with_gemini_sync(
    api_key: str,
    swing_id: str,
    club: str,
    video_path: Path,
    analysis: dict[str, Any],
    client_duration_ms: int | None,
    trim_start_ms: int | None,
    trim_end_ms: int | None,
    rotation_degrees: int | None,
    mirror_horizontal: bool,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError as error:
        raise GeminiConfigError("Install google-genai to enable Gemini AI analysis.") from error

    prepared_video = NamedTemporaryFile(delete=False, suffix=".mp4")
    prepared_path = Path(prepared_video.name)
    prepared_video.close()

    try:
        prepare_gemini_video_clip(
            source_path=video_path,
            output_path=prepared_path,
            client_duration_ms=client_duration_ms,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            rotation_degrees=rotation_degrees,
            mirror_horizontal=mirror_horizontal,
        )

        try:
            client = genai.Client(api_key=api_key)
            uploaded_file = client.files.upload(file=str(prepared_path))
            uploaded_file = wait_for_gemini_file_active(client, uploaded_file)
            prompt = build_gemini_prompt(swing_id=swing_id, club=club, analysis=analysis)
            model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
            config = types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_json_schema=GeminiCoachPayload.model_json_schema(),
                media_resolution=resolve_media_resolution(types),
            )
            response = client.models.generate_content(
                model=model,
                contents=[uploaded_file, prompt],
                config=config,
            )
        except errors.APIError as error:
            raise GeminiResponseError(format_gemini_api_error(error)) from error

        payload = parse_gemini_payload(response.text)
        return build_ai_feedback_response(
            swing_id=swing_id,
            club=club,
            analysis=analysis,
            model=model,
            payload=payload,
        )
    finally:
        prepared_path.unlink(missing_ok=True)


def wait_for_gemini_file_active(client: Any, uploaded_file: Any) -> Any:
    file_name = getattr(uploaded_file, "name", None)
    if not file_name:
        return uploaded_file

    timeout_seconds = float(os.environ.get("GEMINI_FILE_ACTIVE_TIMEOUT_SECONDS", "120"))
    poll_interval_seconds = float(os.environ.get("GEMINI_FILE_POLL_INTERVAL_SECONDS", "2"))
    deadline = time.monotonic() + max(0, timeout_seconds)
    current_file = uploaded_file

    while True:
        state = gemini_file_state_value(getattr(current_file, "state", None))
        if state in {"", "ACTIVE"}:
            return current_file
        if state == "FAILED":
            detail = gemini_file_error_detail(getattr(current_file, "error", None))
            raise GeminiResponseError(f"Gemini file processing failed{detail}.")
        if time.monotonic() >= deadline:
            raise GeminiTimeoutError(
                f"Gemini file {file_name} did not become ACTIVE within {timeout_seconds:g} seconds."
            )

        time.sleep(max(0, poll_interval_seconds))
        current_file = client.files.get(name=file_name)


def gemini_file_state_value(state: Any) -> str:
    if state is None:
        return ""
    return str(getattr(state, "value", state)).upper()


def gemini_file_error_detail(error: Any) -> str:
    if not error:
        return ""

    message = getattr(error, "message", None)
    if message:
        return f": {message}"

    return f": {error}"


def format_gemini_api_error(error: Any) -> str:
    message = getattr(error, "message", None) or str(error)
    code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    if code and status:
        return f"Gemini request failed ({code} {status}): {message}"
    if status:
        return f"Gemini request failed ({status}): {message}"
    return f"Gemini request failed: {message}"


def resolve_media_resolution(types: Any) -> Any:
    raw_value = os.environ.get("GEMINI_MEDIA_RESOLUTION", "MEDIA_RESOLUTION_HIGH").strip().upper()
    if not raw_value.startswith("MEDIA_RESOLUTION_"):
        raw_value = f"MEDIA_RESOLUTION_{raw_value}"
    return getattr(types.MediaResolution, raw_value, types.MediaResolution.MEDIA_RESOLUTION_HIGH)


def parse_gemini_payload(raw_text: str | None) -> GeminiCoachPayload:
    if not raw_text:
        raise GeminiResponseError("Gemini returned an empty response.")

    try:
        payload = GeminiCoachPayload.model_validate_json(raw_text)
    except ValidationError as error:
        raise GeminiResponseError("Gemini returned feedback in an invalid format.") from error

    payload.coachScore = max(0.0, min(10.0, round(float(payload.coachScore), 1)))
    payload.confidence = payload.confidence.lower().strip()
    if payload.confidence not in {"high", "medium", "low"}:
        payload.confidence = "medium"
    return payload


def build_ai_feedback_response(
    swing_id: str,
    club: str,
    analysis: dict[str, Any],
    model: str,
    payload: GeminiCoachPayload,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "id": f"ai-{swing_id}-{uuid4().hex[:10]}",
        "provider": "gemini",
        "model": model,
        "generatedAt": generated_at,
        "club": club,
        "sourceAnalysisVersion": analysis.get("analysisVersion"),
        **payload.model_dump(),
    }


def prepare_gemini_video_clip(
    source_path: Path,
    output_path: Path,
    client_duration_ms: int | None = None,
    trim_start_ms: int | None = None,
    trim_end_ms: int | None = None,
    rotation_degrees: int | None = None,
    mirror_horizontal: bool = False,
) -> None:
    rotation_degrees = normalize_rotation_degrees(rotation_degrees)
    capture = open_video_capture(source_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open video for AI analysis.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_duration_ms = int((total_frames / source_fps) * 1000) if total_frames else 0
    timeline_scale = timeline_scale_for(source_duration_ms, client_duration_ms)
    output_duration_ms = int(round(source_duration_ms * timeline_scale)) if source_duration_ms else int(client_duration_ms or 0)
    manual_start_ms = max(0, int(trim_start_ms or 0))
    manual_end_ms = int(trim_end_ms) if trim_end_ms is not None and trim_end_ms > 0 else output_duration_ms

    if output_duration_ms:
        manual_start_ms = min(manual_start_ms, max(0, output_duration_ms - 800))
        manual_end_ms = min(output_duration_ms, max(manual_end_ms, manual_start_ms + 800))

    if manual_end_ms and manual_end_ms <= manual_start_ms:
        manual_end_ms = output_duration_ms or manual_start_ms + 800

    source_trim_start_ms = client_time_to_source_time(manual_start_ms, timeline_scale)
    source_trim_end_ms = client_time_to_source_time(manual_end_ms, timeline_scale) if manual_end_ms else 0

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    frame_index = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            current_ms = frame_timestamp_ms(frame_index, source_fps)
            frame_index += 1

            if current_ms < source_trim_start_ms:
                continue
            if source_trim_end_ms and current_ms > source_trim_end_ms:
                break

            edited = apply_video_edit(
                frame,
                rotation_degrees=rotation_degrees,
                mirror_horizontal=mirror_horizontal,
            )

            if writer is None:
                height, width = edited.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Could not prepare video for AI analysis.")

            writer.write(edited)
            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError("Could not prepare a trimmed video clip for AI analysis.")
