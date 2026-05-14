from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict

import cv2
import mediapipe as mp
import numpy as np

POSITION_NAMES = ["address", "takeaway", "top", "impact", "finish"]
ANALYSIS_VERSION = 26
PHASE_CONFIDENCE_THRESHOLD = 0.55
QUALITY_SCORE_CAPS = {"ok": 10.0, "warning": 8.0, "poor": 6.0}
RELIABLE_KEYPOINT_SCORE = 0.35
DRAWABLE_KEYPOINT_SCORE = 0.2
INFERRED_KEYPOINT_SCORE_CAP = 0.42
SUPPRESSED_KEYPOINT_SCORE = DRAWABLE_KEYPOINT_SCORE - 0.01
FOOT_REPAIR_GAP_FRAMES = 3
CLUB_REPAIR_GAP_FRAMES = 3
CLUB_MIN_STABLE_RUN_POINTS = 3
CLUB_MIN_RENDERABLE_POINTS = 3
CLUB_MIN_PATH_DISPLACEMENT = 0.035
CLUB_MIN_PRIMARY_SCORE = 0.58
CLUB_MIN_PRIMARY_PHASE_SCORE = 0.25
CLUB_MIN_PRIMARY_THROUGH_IMPACT_RATIO = 0.35
CLUB_AMBIGUOUS_PRIMARY_SCORE_RATIO = 0.86
CLUB_AMBIGUOUS_PHASE_SCORE_GAP = 0.25
CLUB_MAX_FRAME_GAP = 1
CLUB_MAX_STEP_MIN = 0.16
CLUB_MAX_STEP_MAX = 0.34
CLUB_MAX_STEP_BODY_HEIGHT_RATIO = 0.55
CLUB_MIN_GRIP_DISTANCE_BODY_HEIGHT_RATIO = 0.045
CLUB_MAX_GRIP_DISTANCE_BODY_HEIGHT_RATIO = 1.08
FOOT_KEYPOINT_TYPES = ("heel", "foot_index")
FOOT_AXIS_ALIGNMENT_THRESHOLD = 0.55
BODY_QUALITY_KEYPOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_wrist",
    "right_wrist",
]

KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


class Point(TypedDict):
    name: str
    x: float
    y: float
    score: float


class ClubHead(TypedDict):
    x: float
    y: float


class Frame(TypedDict, total=False):
    t: int
    keypoints: list[Point]
    clubHead: ClubHead


class CaptureProfile(TypedDict):
    status: str
    cameraAngle: str
    captureConfidence: float
    scoreConfidence: float
    poseCoverage: float
    bodyVisibility: float
    golferScale: float
    framing: float
    warnings: list[str]


class AnalysisTimeline(TypedDict):
    source_duration_ms: int
    timeline_scale: float
    output_duration_ms: int
    manual_start_ms: int
    manual_end_ms: int
    source_trim_start_ms: int
    source_trim_end_ms: int


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if capture.isOpened():
        orientation_auto_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_auto_property is not None:
            capture.set(orientation_auto_property, 1)
    return capture


def normalize_rotation_degrees(rotation_degrees: int | None) -> int:
    rotation = int(rotation_degrees or 0) % 360
    if rotation not in {0, 90, 180, 270}:
        raise RuntimeError("rotation_degrees must be one of 0, 90, 180, or 270.")
    return rotation


def apply_video_edit(
    frame: np.ndarray,
    rotation_degrees: int | None = None,
    mirror_horizontal: bool = False,
) -> np.ndarray:
    rotation = normalize_rotation_degrees(rotation_degrees)
    edited = frame

    if rotation == 90:
        edited = cv2.rotate(edited, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        edited = cv2.rotate(edited, cv2.ROTATE_180)
    elif rotation == 270:
        edited = cv2.rotate(edited, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if mirror_horizontal:
        edited = cv2.flip(edited, 1)

    return edited


def analyze_video(
    swing_id: str,
    video_path: Path,
    client_duration_ms: int | None = None,
    debug_dir: Path | None = None,
    debug_url_path: str | None = None,
    trim_start_ms: int | None = None,
    trim_end_ms: int | None = None,
    rotation_degrees: int | None = None,
    mirror_horizontal: bool = False,
) -> dict:
    rotation_degrees = normalize_rotation_degrees(rotation_degrees)
    capture = open_video_capture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open video for analysis.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    timeline = resolve_analysis_timeline(
        source_fps=source_fps,
        total_frames=total_frames,
        client_duration_ms=client_duration_ms,
        trim_start_ms=trim_start_ms,
        trim_end_ms=trim_end_ms,
    )
    source_duration_ms = timeline["source_duration_ms"]
    timeline_scale = timeline["timeline_scale"]
    output_duration_ms = timeline["output_duration_ms"]
    manual_start_ms = timeline["manual_start_ms"]
    manual_end_ms = timeline["manual_end_ms"]
    source_trim_start_ms = timeline["source_trim_start_ms"]
    source_trim_end_ms = timeline["source_trim_end_ms"]

    sample_fps, sample_every = analysis_sample_cadence(source_fps)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    frames: list[Frame] = []
    frame_index = 0
    last_club_head: ClubHead | None = None

    while True:
        success, frame = capture.read()
        if not success:
            break

        current_ms = frame_timestamp_ms(frame_index, source_fps)
        if current_ms < source_trim_start_ms:
            frame_index += 1
            continue

        if source_trim_end_ms and current_ms > source_trim_end_ms:
            break

        frame = apply_video_edit(
            frame,
            rotation_degrees=rotation_degrees,
            mirror_horizontal=mirror_horizontal,
        )
        frame_height, frame_width = frame.shape[:2]
        video_width = int(frame_width)
        video_height = int(frame_height)

        if frame_index % sample_every == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            if result.pose_landmarks:
                keypoints = [
                    {
                        "name": KEYPOINT_NAMES[index],
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "score": float(landmark.visibility),
                    }
                    for index, landmark in enumerate(result.pose_landmarks.landmark)
                ]
                keypoints = repair_occluded_arm_keypoints(keypoints)
                analysis_frame: Frame = {"t": current_ms, "keypoints": keypoints}
                club_head = detect_club_head(frame, keypoints, previous=last_club_head)
                if club_head is not None:
                    analysis_frame["clubHead"] = club_head
                    last_club_head = club_head
                frames.append(analysis_frame)

        frame_index += 1
        if len(frames) >= 360:
            break

    capture.release()
    pose.close()

    frames = repair_foot_keypoints(frames)

    if len(frames) < 5:
        raise RuntimeError("Could not detect enough body pose frames. Try a clearer full-body swing video.")

    original_frame_count = len(frames)
    selected_start, selected_end = select_golfer_segment_range(frames)
    selected_frames = frames[selected_start : selected_end + 1]
    if len(selected_frames) < 5:
        raise RuntimeError("Could not isolate a full-body golfer segment. Try a clip framed on the golfer.")

    capture_profile = build_capture_profile(frames, selected_start, selected_end, video_width, video_height)
    local_positions = detect_positions(selected_frames, capture_profile)
    positions = {name: selected_start + index for name, index in local_positions.items()}
    frames, club_path_uncertain = stabilize_club_head_points(frames, positions)
    frames = repair_club_head_points(frames)
    frames = smooth_club_head_points(frames)
    if club_path_uncertain:
        capture_profile["warnings"] = dedupe_warnings([*capture_profile["warnings"], "club_path_uncertain"])
    output_frames = scale_frame_times(frames, timeline_scale)
    metrics = compute_metrics(output_frames, positions)
    phase_confidence = build_phase_confidence(output_frames, positions, capture_profile)
    analysis_quality = finalize_analysis_quality(capture_profile, phase_confidence)
    raw_feedback = build_feedback(metrics)
    feedback = gate_feedback(raw_feedback, phase_confidence)
    swing_thought = build_swing_thought(metrics, feedback, phase_confidence, analysis_quality)
    raw_score = compute_score(raw_feedback)
    score = cap_score_for_quality(raw_score, analysis_quality)
    debug_info = build_debug_artifacts(
        swing_id=swing_id,
        video_path=video_path,
        debug_dir=debug_dir,
        debug_url_path=debug_url_path,
        frames=frames,
        output_frames=output_frames,
        positions=positions,
        timeline_scale=timeline_scale,
        analysis_quality=analysis_quality,
        phase_confidence=phase_confidence,
        rotation_degrees=rotation_degrees,
        mirror_horizontal=mirror_horizontal,
    )
    print(
        "analysis timing",
        {
            "clientDuration": client_duration_ms,
            "outputDuration": output_duration_ms,
            "sourceDuration": source_duration_ms,
            "timelineScale": round(timeline_scale, 4),
            "manualTrim": (manual_start_ms, manual_end_ms or output_frames[-1]["t"]),
            "videoEdit": {
                "rotationDegrees": rotation_degrees,
                "mirrorHorizontal": mirror_horizontal,
            },
            "selectedFrames": (selected_start, selected_end, original_frame_count),
            "positions": {name: output_frames[index]["t"] for name, index in positions.items()},
            "clubHeadFrames": sum(1 for frame in output_frames if "clubHead" in frame),
            "analysisQuality": analysis_quality,
            "phaseConfidence": phase_confidence,
            "debug": debug_info,
        },
        flush=True,
    )

    return {
        "id": swing_id,
        "videoUri": str(video_path),
        "fps": sample_fps,
        "durationMs": output_duration_ms or output_frames[-1]["t"],
        "videoWidth": video_width,
        "videoHeight": video_height,
        "analysisVersion": ANALYSIS_VERSION,
        "trimStartMs": manual_start_ms,
        "trimEndMs": manual_end_ms or output_frames[-1]["t"],
        "frames": output_frames,
        "positions": positions,
        "metrics": metrics,
        "feedback": feedback,
        "swingThought": swing_thought,
        "analysisQuality": analysis_quality,
        "phaseConfidence": phase_confidence,
        "rawSenseiScore": raw_score,
        "senseiScore": score,
        "debug": debug_info,
    }


def timeline_scale_for(source_duration_ms: int, client_duration_ms: int | None) -> float:
    if not client_duration_ms or client_duration_ms <= 0 or source_duration_ms <= 0:
        return 1.0

    ratio = client_duration_ms / source_duration_ms
    # AVAsset/Photos can report the media track duration instead of the playback
    # presentation duration for some phone/social exports. In those cases the
    # client duration is shorter than the decoded source timeline and compressing
    # pose timestamps makes the overlay run early. Only stretch source timestamps
    # when the player duration is longer than the decoded timeline.
    if 1.05 <= ratio <= 3.5:
        return float(ratio)

    return 1.0


def resolve_analysis_timeline(
    source_fps: float,
    total_frames: int,
    client_duration_ms: int | None = None,
    trim_start_ms: int | None = None,
    trim_end_ms: int | None = None,
) -> AnalysisTimeline:
    source_fps = max(1.0, float(source_fps or 30.0))
    total_frames = max(0, int(total_frames or 0))
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

    return {
        "source_duration_ms": source_duration_ms,
        "timeline_scale": timeline_scale,
        "output_duration_ms": output_duration_ms,
        "manual_start_ms": manual_start_ms,
        "manual_end_ms": manual_end_ms,
        "source_trim_start_ms": client_time_to_source_time(manual_start_ms, timeline_scale),
        "source_trim_end_ms": client_time_to_source_time(manual_end_ms, timeline_scale) if manual_end_ms else 0,
    }


def analysis_sample_cadence(source_fps: float) -> tuple[float, int]:
    source_fps = max(1.0, float(source_fps or 30.0))
    target_fps = min(30.0, max(12.0, source_fps))
    sample_every = max(1, round(source_fps / target_fps))
    return source_fps / sample_every, sample_every


def frame_timestamp_ms(frame_index: int, source_fps: float) -> int:
    return int((frame_index / source_fps) * 1000)


def client_time_to_source_time(time_ms: int, timeline_scale: float) -> int:
    return int(round(time_ms / max(0.001, timeline_scale)))


def source_time_to_client_time(time_ms: int, timeline_scale: float) -> int:
    return int(round(time_ms * timeline_scale))


def scale_frame_times(frames: list[Frame], timeline_scale: float) -> list[Frame]:
    if timeline_scale == 1.0:
        return frames

    return [
        {
            **frame,
            "t": source_time_to_client_time(frame["t"], timeline_scale),
        }
        for frame in frames
    ]


def detect_club_head(
    video_frame: np.ndarray,
    keypoints: list[Point],
    previous: ClubHead | None = None,
) -> ClubHead | None:
    pose_frame: Frame = {"t": 0, "keypoints": keypoints}
    wrist_score = max(keypoint_score(pose_frame, "left_wrist"), keypoint_score(pose_frame, "right_wrist"))
    if wrist_score < DRAWABLE_KEYPOINT_SCORE:
        return None

    frame_height, frame_width = video_frame.shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return None

    grip = grip_point(pose_frame)
    grip_px = (grip[0] * frame_width, grip[1] * frame_height)
    box = body_bbox(pose_frame)
    body_height_px = max(80.0, min(float(max(frame_width, frame_height)), max(box["height"], 0.22) * frame_height))
    roi_radius_x = int(max(72.0, min(frame_width, body_height_px * 0.78)))
    roi_radius_y = int(max(72.0, min(frame_height, body_height_px * 0.78)))

    center_x = int(round(grip_px[0]))
    center_y = int(round(grip_px[1]))
    left = max(0, center_x - roi_radius_x)
    right = min(frame_width, center_x + roi_radius_x)
    top = max(0, center_y - roi_radius_y)
    bottom = min(frame_height, center_y + roi_radius_y)
    if right - left < 24 or bottom - top < 24:
        return None

    roi = video_frame[top:bottom, left:right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    min_line_length = int(max(18.0, body_height_px * 0.075))
    max_line_gap = int(max(10.0, body_height_px * 0.055))
    candidate_lines = club_candidate_lines(gray, min_line_length, max_line_gap)
    if not candidate_lines:
        return None

    best: tuple[float, ClubHead] | None = None
    grip_tolerance = max(24.0, body_height_px * 0.16)
    min_head_distance = max(24.0, body_height_px * 0.10)
    max_head_distance = max(min_head_distance, body_height_px * 1.12)

    for x1, y1, x2, y2 in candidate_lines:
        first = (x1 + left, y1 + top)
        second = (x2 + left, y2 + top)
        line_length = math.hypot(second[0] - first[0], second[1] - first[1])
        if line_length < min_line_length:
            continue

        projection = segment_projection_amount(grip_px, first, second)
        grip_distance = point_to_segment_distance(grip_px, first, second)
        if -0.38 <= projection <= 1.38:
            grip_distance = min(grip_distance, point_to_line_distance(grip_px, first, second) * 1.25)
        if grip_distance > grip_tolerance:
            continue

        first_distance = math.hypot(first[0] - grip_px[0], first[1] - grip_px[1])
        second_distance = math.hypot(second[0] - grip_px[0], second[1] - grip_px[1])
        head_px = first if first_distance >= second_distance else second
        head_distance = max(first_distance, second_distance)
        if head_distance < min_head_distance or head_distance > max_head_distance:
            continue

        head: ClubHead = {
            "x": round(clamp_float(head_px[0] / frame_width), 4),
            "y": round(clamp_float(head_px[1] / frame_height), 4),
        }
        line_score = clamp_float(line_length / max(min_line_length, body_height_px * 0.45))
        grip_score = 1.0 - clamp_float(grip_distance / grip_tolerance)
        distance_score = clamp_float(head_distance / max(min_head_distance, body_height_px * 0.55))
        temporal_score = 0.70
        if previous is not None:
            previous_px = (previous["x"] * frame_width, previous["y"] * frame_height)
            temporal_distance = math.hypot(head_px[0] - previous_px[0], head_px[1] - previous_px[1])
            temporal_score = 1.0 - clamp_float(temporal_distance / max(32.0, body_height_px * 0.38))

        inside_body_penalty = 0.14 if point_inside_body_box(head, box, padding=0.04) else 0.0
        score = 0.34 * line_score + 0.33 * grip_score + 0.20 * distance_score + 0.13 * temporal_score
        score -= inside_body_penalty
        if score < 0.38:
            continue

        if best is None or score > best[0]:
            best = (score, head)

    return best[1] if best is not None else None


def club_candidate_lines(
    gray_roi: np.ndarray,
    min_line_length: int,
    max_line_gap: int,
) -> list[tuple[float, float, float, float]]:
    edges = adaptive_canny_edges(gray_roi)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    lines = cv2.HoughLinesP(
        closed_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=12,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    candidates: list[tuple[float, float, float, float]] = []
    if lines is not None:
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [float(value) for value in line]
            candidates.append((x1, y1, x2, y2))

    line_segment_detector = getattr(cv2, "createLineSegmentDetector", None)
    if line_segment_detector is not None:
        try:
            detector = line_segment_detector(0)
            segments = detector.detect(gray_roi)[0]
        except Exception:
            segments = None
        if segments is not None:
            for segment in segments.reshape(-1, 4):
                x1, y1, x2, y2 = [float(value) for value in segment]
                if math.hypot(x2 - x1, y2 - y1) >= min_line_length * 0.62:
                    candidates.append((x1, y1, x2, y2))

    return dedupe_line_candidates(candidates)


def adaptive_canny_edges(gray_roi: np.ndarray) -> np.ndarray:
    equalized = cv2.equalizeHist(gray_roi)
    median_intensity = float(np.median(equalized))
    lower = int(max(18, 0.58 * median_intensity))
    upper = int(min(180, max(lower + 24, 1.55 * median_intensity)))
    adaptive_edges = cv2.Canny(equalized, lower, upper)
    soft_edges = cv2.Canny(equalized, 24, 88)
    return cv2.bitwise_or(adaptive_edges, soft_edges)


def dedupe_line_candidates(
    lines: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    deduped: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for x1, y1, x2, y2 in lines:
        key = tuple(int(round(value / 3.0)) for value in (x1, y1, x2, y2))
        reverse_key = (key[2], key[3], key[0], key[1])
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        deduped.append((x1, y1, x2, y2))
    return deduped


def repair_club_head_points(frames: list[Frame]) -> list[Frame]:
    repaired: list[Frame] = []
    for frame in frames:
        copied: Frame = {**frame, "keypoints": [dict(point) for point in frame["keypoints"]]}
        if "clubHead" in frame:
            copied["clubHead"] = dict(frame["clubHead"])
        repaired.append(copied)

    valid_indices = [index for index, frame in enumerate(repaired) if "clubHead" in frame]
    for first_index, second_index in zip(valid_indices, valid_indices[1:]):
        gap = second_index - first_index - 1
        if gap <= 0 or gap > CLUB_REPAIR_GAP_FRAMES:
            continue

        start = repaired[first_index]["clubHead"]
        end = repaired[second_index]["clubHead"]
        for offset in range(1, gap + 1):
            amount = offset / (gap + 1)
            repaired[first_index + offset]["clubHead"] = {
                "x": round(start["x"] + (end["x"] - start["x"]) * amount, 4),
                "y": round(start["y"] + (end["y"] - start["y"]) * amount, 4),
            }

    return repaired


def stabilize_club_head_points(
    frames: list[Frame],
    positions: dict[str, int] | None = None,
) -> tuple[list[Frame], bool]:
    stabilized: list[Frame] = []
    for frame in frames:
        copied: Frame = {**frame, "keypoints": [dict(point) for point in frame["keypoints"]]}
        if "clubHead" in frame:
            copied["clubHead"] = dict(frame["clubHead"])
        stabilized.append(copied)

    raw_indices = [index for index, frame in enumerate(stabilized) if "clubHead" in frame]
    if not raw_indices:
        return stabilized, False

    plausible_indices = [
        index
        for index in raw_indices
        if is_plausible_club_head_point(stabilized[index])
    ]
    plausible_indices = remove_local_club_head_spikes(stabilized, plausible_indices)
    primary_run = select_primary_club_path_run(stabilized, plausible_indices, positions or {})

    if primary_run is None or len(primary_run) < CLUB_MIN_RENDERABLE_POINTS:
        for frame in stabilized:
            frame.pop("clubHead", None)
        return stabilized, True

    kept_indices = set(primary_run)
    for index, frame in enumerate(stabilized):
        if index not in kept_indices:
            frame.pop("clubHead", None)

    return stabilized, False


def select_primary_club_path_run(
    frames: list[Frame],
    indices: list[int],
    positions: dict[str, int] | None = None,
) -> list[int] | None:
    if not indices:
        return None

    stable_runs = stable_club_head_runs(
        frames,
        indices,
        maximum_frame_gap=max(CLUB_MAX_FRAME_GAP, CLUB_REPAIR_GAP_FRAMES + 1),
    )
    candidates = [
        score_club_path_run(frames, run, positions or {})
        for run in stable_runs
        if len(run) >= CLUB_MIN_STABLE_RUN_POINTS and club_path_displacement(frames, run) >= CLUB_MIN_PATH_DISPLACEMENT
    ]
    if not candidates:
        return None

    required_score = 0.34 if not positions else CLUB_MIN_PRIMARY_SCORE
    candidates = [candidate for candidate in candidates if candidate["score"] >= required_score]
    if positions:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["phaseScore"] >= CLUB_MIN_PRIMARY_PHASE_SCORE
            and candidate["throughImpactScore"] >= CLUB_MIN_PRIMARY_THROUGH_IMPACT_RATIO
        ]
    if not candidates:
        return None

    candidates.sort(key=lambda candidate: (candidate["score"], len(candidate["run"])), reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        if (
            second["score"] >= best["score"] * CLUB_AMBIGUOUS_PRIMARY_SCORE_RATIO
            and best["phaseScore"] - second["phaseScore"] < CLUB_AMBIGUOUS_PHASE_SCORE_GAP
        ):
            return None

    return list(best["run"])


def score_club_path_run(frames: list[Frame], run: list[int], positions: dict[str, int]) -> dict[str, object]:
    length_score = min(1.0, len(run) / 12.0)
    displacement_score = min(1.0, club_path_displacement(frames, run) / 0.22)
    continuity_score = club_path_continuity_score(frames, run)
    phase_score, through_impact_score = club_path_phase_scores(run, positions)
    score = length_score * 0.34 + displacement_score * 0.26 + continuity_score * 0.18 + phase_score * 0.42
    return {"run": run, "score": score, "phaseScore": phase_score, "throughImpactScore": through_impact_score}


def club_path_continuity_score(frames: list[Frame], run: list[int]) -> float:
    if len(run) < 2:
        return 0.0

    ratios = []
    for previous_index, index in zip(run, run[1:]):
        previous = frames[previous_index]["clubHead"]
        current = frames[index]["clubHead"]
        step_limit = club_head_step_limit(frames[previous_index], frames[index])
        gap_multiplier = max(1, index - previous_index)
        ratios.append(min(1.0, normalized_point_distance(previous, current) / max(0.001, step_limit * gap_multiplier)))

    average_ratio = float(np.mean(ratios)) if ratios else 1.0
    return max(0.0, 1.0 - average_ratio)


def club_path_phase_score(run: list[int], positions: dict[str, int]) -> float:
    return club_path_phase_scores(run, positions)[0]


def club_path_phase_scores(run: list[int], positions: dict[str, int]) -> tuple[float, float]:
    if not run:
        return 0.0, 0.0
    address = positions.get("address")
    finish = positions.get("finish")
    if address is None or finish is None or finish < address:
        return 0.0, 0.0

    impact = positions.get("impact", finish)
    top = positions.get("top", impact)
    in_swing = index_overlap_ratio(run, address, finish)
    through_impact = index_anchor_ratio(run, address, max(address, impact))
    backswing_to_top = index_anchor_ratio(run, address, max(address, top))
    outside_penalty = 1.0 - in_swing
    phase_score = clamp_float(
        in_swing * 0.50
        + through_impact * 0.30
        + backswing_to_top * 0.15
        - outside_penalty * 0.45
    )
    return phase_score, through_impact


def index_overlap_ratio(indices: list[int], lower_bound: int, upper_bound: int) -> float:
    if not indices or upper_bound < lower_bound:
        return 0.0

    count = sum(1 for index in indices if lower_bound <= index <= upper_bound)
    return count / len(indices)


def index_anchor_ratio(indices: list[int], lower_bound: int, upper_bound: int) -> float:
    if not indices or upper_bound < lower_bound:
        return 0.0

    count = sum(1 for index in indices if lower_bound <= index <= upper_bound)
    required_anchors = max(2, min(CLUB_MIN_STABLE_RUN_POINTS, upper_bound - lower_bound + 1))
    return min(1.0, count / required_anchors)


def smooth_club_head_points(frames: list[Frame]) -> list[Frame]:
    smoothed_frames: list[Frame] = []
    for frame in frames:
        copied: Frame = {**frame, "keypoints": [dict(point) for point in frame["keypoints"]]}
        if "clubHead" in frame:
            copied["clubHead"] = dict(frame["clubHead"])
        smoothed_frames.append(copied)

    smoothed = [
        {"x": frame["clubHead"]["x"], "y": frame["clubHead"]["y"]} if "clubHead" in frame else None
        for frame in smoothed_frames
    ]
    for index, frame in enumerate(smoothed_frames):
        if "clubHead" not in frame:
            continue
        neighbors = [
            point
            for point in smoothed[max(0, index - 1) : min(len(smoothed), index + 2)]
            if point is not None
        ]
        if len(neighbors) < 2:
            continue
        frame["clubHead"] = {
            "x": round(float(np.mean([point["x"] for point in neighbors])), 4),
            "y": round(float(np.mean([point["y"] for point in neighbors])), 4),
        }

    return smoothed_frames


def is_plausible_club_head_point(frame: Frame) -> bool:
    point = frame.get("clubHead")
    if point is None:
        return False
    if not np.isfinite(point["x"]) or not np.isfinite(point["y"]):
        return False
    if point["x"] < 0 or point["x"] > 1 or point["y"] < 0 or point["y"] > 1:
        return False

    box = body_bbox(frame)
    body_height = club_reference_body_height(frame)
    grip = grip_point(frame)
    grip_distance = normalized_point_distance(point, {"x": grip[0], "y": grip[1]})
    min_grip_distance = max(0.018, body_height * CLUB_MIN_GRIP_DISTANCE_BODY_HEIGHT_RATIO)
    max_grip_distance = min(0.86, max(0.24, body_height * CLUB_MAX_GRIP_DISTANCE_BODY_HEIGHT_RATIO))
    if grip_distance < min_grip_distance or grip_distance > max_grip_distance:
        return False

    return not point_inside_body_box(point, box, padding=-0.02)


def remove_local_club_head_spikes(frames: list[Frame], indices: list[int]) -> list[int]:
    if len(indices) < 3:
        return indices

    removed: set[int] = set()
    for previous_index, index, next_index in zip(indices, indices[1:], indices[2:]):
        if index - previous_index > CLUB_MAX_FRAME_GAP or next_index - index > CLUB_MAX_FRAME_GAP:
            continue

        previous = frames[previous_index]["clubHead"]
        current = frames[index]["clubHead"]
        next_point = frames[next_index]["clubHead"]
        step_limit = club_head_step_limit(frames[previous_index], frames[index])
        previous_to_current = normalized_point_distance(previous, current)
        current_to_next = normalized_point_distance(current, next_point)
        previous_to_next = normalized_point_distance(previous, next_point)
        if (
            previous_to_current > step_limit * 0.75
            and current_to_next > step_limit * 0.75
            and previous_to_next <= step_limit * 0.70
        ):
            removed.add(index)

    return [index for index in indices if index not in removed]


def stable_club_head_runs(
    frames: list[Frame],
    indices: list[int],
    maximum_frame_gap: int = CLUB_MAX_FRAME_GAP,
) -> list[list[int]]:
    runs: list[list[int]] = []
    current_run: list[int] = []

    for index in indices:
        if not current_run:
            current_run = [index]
            continue

        previous_index = current_run[-1]
        frame_gap = index - previous_index
        distance = normalized_point_distance(frames[previous_index]["clubHead"], frames[index]["clubHead"])
        step_limit = club_head_step_limit(frames[previous_index], frames[index])
        if frame_gap <= maximum_frame_gap and distance <= step_limit:
            current_run.append(index)
            continue

        runs.append(current_run)
        current_run = [index]

    if current_run:
        runs.append(current_run)

    return runs


def club_head_step_limit(first_frame: Frame, second_frame: Frame) -> float:
    body_height = max(club_reference_body_height(first_frame), club_reference_body_height(second_frame))
    return max(CLUB_MAX_STEP_MIN, min(CLUB_MAX_STEP_MAX, body_height * CLUB_MAX_STEP_BODY_HEIGHT_RATIO))


def club_reference_body_height(frame: Frame) -> float:
    box = body_bbox(frame)
    return max(0.24, min(0.72, float(box["height"])))


def club_path_displacement(frames: list[Frame], indices: list[int]) -> float:
    if len(indices) < 2:
        return 0.0

    points = [frames[index]["clubHead"] for index in indices]
    min_x = min(point["x"] for point in points)
    max_x = max(point["x"] for point in points)
    min_y = min(point["y"] for point in points)
    max_y = max(point["y"] for point in points)
    return math.hypot(max_x - min_x, max_y - min_y)


def normalized_point_distance(first: dict[str, float], second: dict[str, float]) -> float:
    return math.hypot(first["x"] - second["x"], first["y"] - second["y"])


def fit_swing_plane(
    frames: list[Frame],
    positions: dict[str, int],
    analysis_quality: CaptureProfile,
) -> dict[str, object] | None:
    if analysis_quality.get("cameraAngle") != "down_the_line":
        return None
    if not frames:
        return None

    start = clamp_index(positions.get("takeaway", 0), frames)
    end = clamp_index(positions.get("impact", len(frames) - 1), frames)
    if end < start:
        return None

    points = [
        (frame["clubHead"]["x"], frame["clubHead"]["y"])
        for frame in frames[start : end + 1]
        if "clubHead" in frame
    ]
    if len(points) < 5:
        return None

    values = np.array(points, dtype=np.float64)
    center = values.mean(axis=0)
    centered = values - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projections = centered @ direction
    min_projection = float(projections.min())
    max_projection = float(projections.max())
    if max_projection - min_projection < 0.04:
        return None

    residuals = np.abs(centered[:, 0] * direction[1] - centered[:, 1] * direction[0])
    band_width = max(0.018, min(0.10, float(np.median(residuals) * 2.4)))
    start_point = center + direction * min_projection
    end_point = center + direction * max_projection

    return {
        "start": {"x": round(clamp_float(start_point[0]), 4), "y": round(clamp_float(start_point[1]), 4)},
        "end": {"x": round(clamp_float(end_point[0]), 4), "y": round(clamp_float(end_point[1]), 4)},
        "bandWidth": round(band_width, 4),
        "pointCount": len(points),
    }


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])

    amount = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    amount = max(0.0, min(1.0, amount))
    projected = (start[0] + amount * dx, start[1] + amount * dy)
    return math.hypot(point[0] - projected[0], point[1] - projected[1])


def point_to_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    return abs(dx * (start[1] - point[1]) - (start[0] - point[0]) * dy) / length


def segment_projection_amount(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 0:
        return 0.0
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared


def point_inside_body_box(point: ClubHead, box: dict[str, float], padding: float = 0.0) -> bool:
    if box["width"] <= 0 or box["height"] <= 0:
        return False
    return (
        box["min_x"] - padding <= point["x"] <= box["max_x"] + padding
        and box["min_y"] - padding <= point["y"] <= box["max_y"] + padding
    )


def build_debug_artifacts(
    swing_id: str,
    video_path: Path,
    debug_dir: Path | None,
    debug_url_path: str | None,
    frames: list[Frame],
    output_frames: list[Frame],
    positions: dict[str, int],
    timeline_scale: float,
    analysis_quality: CaptureProfile | None = None,
    phase_confidence: dict[str, float] | None = None,
    rotation_degrees: int | None = None,
    mirror_horizontal: bool = False,
) -> dict:
    if debug_dir is None:
        return {}

    rotation_degrees = normalize_rotation_degrees(rotation_degrees)
    debug_dir.mkdir(exist_ok=True)
    safe_swing_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in swing_id)
    contact_sheet_path = debug_dir / f"{safe_swing_id}_phases_v{ANALYSIS_VERSION}.jpg"
    impact_filmstrip_path = debug_dir / f"{safe_swing_id}_impact_v{ANALYSIS_VERSION}.jpg"

    try:
        create_phase_contact_sheet(
            video_path=video_path,
            frames=frames,
            output_frames=output_frames,
            positions=positions,
            timeline_scale=timeline_scale,
            destination=contact_sheet_path,
            analysis_quality=analysis_quality,
            phase_confidence=phase_confidence,
            rotation_degrees=rotation_degrees,
            mirror_horizontal=mirror_horizontal,
        )
    except Exception as error:  # pragma: no cover - diagnostic best effort.
        print("debug contact sheet failed", {"error": str(error), "path": str(contact_sheet_path)}, flush=True)
        return {"contactSheetError": str(error)}

    debug_info = {
        "contactSheetPath": str(contact_sheet_path),
        "positionTimesMs": {
            name: {
                "analysis": output_frames[index]["t"],
                "source": frames[index]["t"],
            }
            for name, index in positions.items()
        },
    }
    if analysis_quality is not None:
        debug_info["analysisQuality"] = analysis_quality
    if phase_confidence is not None:
        debug_info["phaseConfidence"] = phase_confidence
    try:
        impact_window = create_impact_filmstrip(
            video_path=video_path,
            frames=frames,
            output_frames=output_frames,
            impact_index=positions["impact"],
            destination=impact_filmstrip_path,
            rotation_degrees=rotation_degrees,
            mirror_horizontal=mirror_horizontal,
        )
        debug_info["impactFilmstripPath"] = str(impact_filmstrip_path)
        debug_info["impactWindowTimesMs"] = impact_window
    except Exception as error:  # pragma: no cover - diagnostic best effort.
        print("debug impact filmstrip failed", {"error": str(error), "path": str(impact_filmstrip_path)}, flush=True)
        debug_info["impactFilmstripError"] = str(error)

    if debug_url_path:
        debug_info["contactSheetUrl"] = f"{debug_url_path.rstrip('/')}/{contact_sheet_path.name}"
        if "impactFilmstripPath" in debug_info:
            debug_info["impactFilmstripUrl"] = f"{debug_url_path.rstrip('/')}/{impact_filmstrip_path.name}"

    return debug_info


def create_phase_contact_sheet(
    video_path: Path,
    frames: list[Frame],
    output_frames: list[Frame],
    positions: dict[str, int],
    timeline_scale: float,
    destination: Path,
    analysis_quality: CaptureProfile | None = None,
    phase_confidence: dict[str, float] | None = None,
    rotation_degrees: int | None = None,
    mirror_horizontal: bool = False,
) -> None:
    rotation_degrees = normalize_rotation_degrees(rotation_degrees)
    capture = open_video_capture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open video for debug contact sheet.")

    try:
        tiles = []
        for name in POSITION_NAMES:
            frame_index = clamp_index(positions[name], frames)
            source_time_ms = frames[frame_index]["t"]
            analysis_time_ms = output_frames[frame_index]["t"]
            tile = read_video_frame_at(
                capture,
                source_time_ms,
                rotation_degrees=rotation_degrees,
                mirror_horizontal=mirror_horizontal,
            )
            tile = fit_debug_tile(tile)
            extra_lines = []
            if phase_confidence is not None:
                extra_lines.append(f"confidence {phase_confidence.get(name, 0.0):.2f}")
            if analysis_quality is not None:
                extra_lines.append(
                    f"{analysis_quality['cameraAngle']} {analysis_quality['status']} cap {analysis_quality['scoreConfidence']:.2f}"
                )
            draw_debug_label(
                tile,
                f"{name.upper()}",
                f"analysis {analysis_time_ms} ms",
                f"source {source_time_ms} ms scale {timeline_scale:.4f}",
                extra_lines=extra_lines,
            )
            tiles.append(tile)

        if not tiles:
            raise RuntimeError("No phase tiles were generated.")

        contact_sheet = np.concatenate(tiles, axis=1)
        if not cv2.imwrite(str(destination), contact_sheet):
            raise RuntimeError("OpenCV could not write debug contact sheet.")
    finally:
        capture.release()


def create_impact_filmstrip(
    video_path: Path,
    frames: list[Frame],
    output_frames: list[Frame],
    impact_index: int,
    destination: Path,
    before: int = 6,
    after: int = 6,
    rotation_degrees: int | None = None,
    mirror_horizontal: bool = False,
) -> list[dict]:
    rotation_degrees = normalize_rotation_degrees(rotation_degrees)
    capture = open_video_capture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open video for debug impact filmstrip.")

    try:
        selected = clamp_index(impact_index, frames)
        start = max(0, selected - before)
        end = min(len(frames) - 1, selected + after)
        tiles = []
        window = []

        for frame_index in range(start, end + 1):
            source_time_ms = frames[frame_index]["t"]
            analysis_time_ms = output_frames[frame_index]["t"]
            offset = frame_index - selected
            tile = read_video_frame_at(
                capture,
                source_time_ms,
                rotation_degrees=rotation_degrees,
                mirror_horizontal=mirror_horizontal,
            )
            tile = fit_debug_tile(tile, tile_width=160, tile_height=240)
            draw_debug_label(
                tile,
                "IMPACT" if offset == 0 else f"{offset:+d} FRAME",
                f"analysis {analysis_time_ms} ms",
                f"source {source_time_ms} ms",
            )
            tiles.append(tile)
            window.append(
                {
                    "frame": frame_index,
                    "offset": offset,
                    "analysis": analysis_time_ms,
                    "source": source_time_ms,
                    "selected": offset == 0,
                }
            )

        if not tiles:
            raise RuntimeError("No impact filmstrip tiles were generated.")

        filmstrip = np.concatenate(tiles, axis=1)
        if not cv2.imwrite(str(destination), filmstrip):
            raise RuntimeError("OpenCV could not write debug impact filmstrip.")

        return window
    finally:
        capture.release()


def read_video_frame_at(
    capture: cv2.VideoCapture,
    source_time_ms: int,
    rotation_degrees: int | None = None,
    mirror_horizontal: bool = False,
) -> np.ndarray:
    rotation_degrees = normalize_rotation_degrees(rotation_degrees)
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0, source_time_ms))
    success, frame = capture.read()
    if success:
        return apply_video_edit(
            frame,
            rotation_degrees=rotation_degrees,
            mirror_horizontal=mirror_horizontal,
        )

    frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index - 1))
    success, frame = capture.read()
    if success:
        return apply_video_edit(
            frame,
            rotation_degrees=rotation_degrees,
            mirror_horizontal=mirror_horizontal,
        )

    raise RuntimeError(f"Could not read video frame at {source_time_ms}ms.")


def fit_debug_tile(frame: np.ndarray, tile_width: int = 240, tile_height: int = 360) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((tile_height, tile_width, 3), dtype=np.uint8)

    scale = max(tile_width / width, tile_height / height)
    resized_width = max(tile_width, int(round(width * scale)))
    resized_height = max(tile_height, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    offset_x = max(0, (resized_width - tile_width) // 2)
    offset_y = max(0, (resized_height - tile_height) // 2)
    return resized[offset_y : offset_y + tile_height, offset_x : offset_x + tile_width].copy()


def draw_debug_label(
    tile: np.ndarray,
    title: str,
    analysis_time: str,
    source_time: str,
    extra_lines: list[str] | None = None,
) -> None:
    lines = [analysis_time, source_time, *(extra_lines or [])]
    header_height = min(tile.shape[0], 38 + 22 * len(lines))
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tile.shape[1], header_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, tile, 0.38, 0, tile)
    cv2.putText(tile, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (34, 197, 94), 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        color = (255, 255, 255) if index == 0 else (230, 230, 230)
        cv2.putText(tile, line, (10, 50 + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def select_golfer_segment_range(frames: list[Frame]) -> tuple[int, int]:
    qualities = np.array([golfer_frame_quality(frame) for frame in frames])
    eligible = qualities >= 0.5
    segments: list[tuple[int, int, float]] = []
    start: int | None = None

    for index, is_eligible in enumerate(eligible):
        if is_eligible and start is None:
            start = index
        elif not is_eligible and start is not None:
            if index - start >= 5:
                segments.append((start, index - 1, float(np.mean(qualities[start:index]))))
            start = None

    if start is not None and len(frames) - start >= 5:
        segments.append((start, len(frames) - 1, float(np.mean(qualities[start:]))))

    if not segments:
        return (0, len(frames) - 1)

    velocities = wrist_velocity(frames)

    def segment_score(segment: tuple[int, int, float]) -> float:
        start_index, end_index, quality = segment
        segment_velocity = float(np.sum(velocities[start_index : end_index + 1]))
        length_bonus = min(1.0, (end_index - start_index + 1) / 24)
        early_bonus = 1.0 - (start_index / max(1, len(frames))) * 0.18
        return (segment_velocity + 0.08) * quality * (1 + length_bonus) * early_bonus

    best_start, best_end, _quality = max(segments, key=segment_score)
    pad = 4
    best_start = max(0, best_start - pad)
    best_end = min(len(frames) - 1, best_end + pad)

    return (best_start, best_end)


def golfer_frame_quality(frame: Frame) -> float:
    points = [point for point in frame["keypoints"] if point["name"] in BODY_QUALITY_KEYPOINTS]
    if len(points) < len(BODY_QUALITY_KEYPOINTS):
        return 0

    visibility = float(np.mean([point["score"] for point in points]))
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)

    if bbox_height < 0.22 or bbox_width < 0.08:
        return 0

    if bbox_height > 1.05 or min(xs) < -0.12 or max(xs) > 1.12:
        return 0

    lower_body_visibility = np.mean(
        [
            keypoint_score(frame, "left_knee"),
            keypoint_score(frame, "right_knee"),
            keypoint_score(frame, "left_ankle"),
            keypoint_score(frame, "right_ankle"),
        ]
    )
    full_body_bonus = min(1.0, bbox_height / 0.55)

    return float(visibility * 0.55 + lower_body_visibility * 0.3 + full_body_bonus * 0.15)


def build_capture_profile(
    frames: list[Frame],
    selected_start: int,
    selected_end: int,
    video_width: int = 0,
    video_height: int = 0,
) -> CaptureProfile:
    selected_frames = frames[selected_start : selected_end + 1] if frames else []
    if not selected_frames:
        return {
            "status": "poor",
            "cameraAngle": "unknown",
            "captureConfidence": 0.0,
            "scoreConfidence": 0.0,
            "poseCoverage": 0.0,
            "bodyVisibility": 0.0,
            "golferScale": 0.0,
            "framing": 0.0,
            "warnings": ["no_pose_segment"],
        }

    qualities = np.array([golfer_frame_quality(frame) for frame in selected_frames])
    pose_coverage = float(np.mean(qualities >= 0.5))
    body_visibility = mean_body_visibility(selected_frames)
    golfer_scale = median_body_scale(selected_frames)
    framing = estimate_framing_quality(selected_frames)
    camera_angle, angle_confidence = classify_camera_angle(selected_frames, body_visibility, golfer_scale)

    scale_score = clamp_float((golfer_scale - 0.22) / 0.45)
    capture_confidence = (
        pose_coverage * 0.26
        + body_visibility * 0.24
        + scale_score * 0.18
        + framing * 0.18
        + angle_confidence * 0.14
    )
    capture_confidence = clamp_float(capture_confidence)
    if camera_angle == "unknown":
        capture_confidence = min(capture_confidence, 0.74)
    status = status_from_confidence(capture_confidence)

    warnings: list[str] = []
    if camera_angle == "unknown":
        warnings.append("unusual_camera_angle")
    if pose_coverage < 0.7:
        warnings.append("low_pose_coverage")
    if body_visibility < 0.58:
        warnings.append("low_pose_visibility")
    if golfer_scale < 0.34:
        warnings.append("golfer_too_small")
    if framing < 0.62:
        warnings.append("poor_framing")
    if status != "ok":
        warnings.append("provisional_analysis")

    profile: CaptureProfile = {
        "status": status,
        "cameraAngle": camera_angle,
        "captureConfidence": round(capture_confidence, 2),
        "scoreConfidence": round(capture_confidence, 2),
        "poseCoverage": round(pose_coverage, 2),
        "bodyVisibility": round(body_visibility, 2),
        "golferScale": round(golfer_scale, 2),
        "framing": round(framing, 2),
        "warnings": dedupe_warnings(warnings),
    }
    print(
        "capture profile",
        {
            **profile,
            "selectedFrames": (selected_start, selected_end, len(frames)),
            "videoSize": (video_width, video_height),
        },
        flush=True,
    )
    return profile


def mean_body_visibility(frames: list[Frame]) -> float:
    if not frames:
        return 0.0

    return float(
        np.mean(
            [
                np.mean([keypoint_score(frame, name) for name in BODY_QUALITY_KEYPOINTS])
                for frame in frames
            ]
        )
    )


def median_body_scale(frames: list[Frame]) -> float:
    boxes = [body_bbox(frame) for frame in frames]
    heights = [box["height"] for box in boxes if box["height"] > 0]
    return float(np.median(heights)) if heights else 0.0


def estimate_framing_quality(frames: list[Frame]) -> float:
    boxes = [body_bbox(frame) for frame in frames]
    valid_boxes = [box for box in boxes if box["height"] > 0 and box["width"] > 0]
    if not valid_boxes:
        return 0.0

    min_x = float(np.percentile([box["min_x"] for box in valid_boxes], 5))
    max_x = float(np.percentile([box["max_x"] for box in valid_boxes], 95))
    min_y = float(np.percentile([box["min_y"] for box in valid_boxes], 5))
    max_y = float(np.percentile([box["max_y"] for box in valid_boxes], 95))
    bbox_height = max_y - min_y
    bbox_width = max_x - min_x

    margin_penalty = 0.0
    margin_penalty += max(0.0, 0.04 - min_x) / 0.04
    margin_penalty += max(0.0, max_x - 0.96) / 0.04
    margin_penalty += max(0.0, 0.03 - min_y) / 0.03
    margin_penalty += max(0.0, max_y - 0.99) / 0.04

    scale_penalty = max(0.0, 0.30 - bbox_height) / 0.30
    width_penalty = max(0.0, 0.10 - bbox_width) / 0.10
    return clamp_float(1.0 - min(0.85, margin_penalty * 0.22 + scale_penalty * 0.2 + width_penalty * 0.14))


def classify_camera_angle(frames: list[Frame], body_visibility: float, golfer_scale: float) -> tuple[str, float]:
    if not frames or body_visibility < 0.45 or golfer_scale < 0.24:
        return ("unknown", 0.32)

    shoulder_ratios = []
    hip_ratios = []
    ankle_ratios = []
    side_balances = []
    for frame in frames:
        box = body_bbox(frame)
        body_height = max(0.001, box["height"])
        shoulder_ratios.append(horizontal_pair_ratio(frame, "left_shoulder", "right_shoulder", body_height))
        hip_ratios.append(horizontal_pair_ratio(frame, "left_hip", "right_hip", body_height))
        ankle_ratios.append(horizontal_pair_ratio(frame, "left_ankle", "right_ankle", body_height))
        left_scores = [keypoint_score(frame, name) for name in ["left_shoulder", "left_hip", "left_knee", "left_ankle"]]
        right_scores = [keypoint_score(frame, name) for name in ["right_shoulder", "right_hip", "right_knee", "right_ankle"]]
        side_balances.append(1.0 - min(1.0, abs(float(np.mean(left_scores)) - float(np.mean(right_scores)))))

    shoulder_ratio = float(np.median(shoulder_ratios))
    hip_ratio = float(np.median(hip_ratios))
    ankle_ratio = float(np.median(ankle_ratios))
    side_balance = float(np.median(side_balances))

    face_on_score = (
        clamp_float((shoulder_ratio - 0.16) / 0.16) * 0.36
        + clamp_float((hip_ratio - 0.12) / 0.14) * 0.26
        + clamp_float((ankle_ratio - 0.13) / 0.16) * 0.2
        + side_balance * 0.18
    )
    profile_score = (
        clamp_float((0.24 - shoulder_ratio) / 0.16) * 0.34
        + clamp_float((0.20 - hip_ratio) / 0.14) * 0.24
        + clamp_float((0.19 - ankle_ratio) / 0.15) * 0.22
        + (1.0 - side_balance) * 0.2
    )
    side_profile_score = (
        clamp_float((shoulder_ratio - 0.13) / 0.12) * 0.24
        + clamp_float((0.36 - shoulder_ratio) / 0.16) * 0.20
        + clamp_float((hip_ratio - 0.08) / 0.09) * 0.18
        + clamp_float((0.27 - hip_ratio) / 0.12) * 0.14
        + clamp_float((ankle_ratio - 0.16) / 0.16) * 0.18
        + side_balance * 0.06
    )

    if face_on_score >= 0.68 and face_on_score >= profile_score + 0.08:
        return ("face_on", round(clamp_float(face_on_score), 2))
    if profile_score >= 0.55 and profile_score >= face_on_score - 0.06:
        return ("down_the_line", round(clamp_float(profile_score), 2))
    if (
        side_profile_score >= 0.62
        and 0.13 <= shoulder_ratio <= 0.38
        and 0.08 <= hip_ratio <= 0.28
        and ankle_ratio >= 0.16
    ):
        return ("side_profile", round(clamp_float(side_profile_score), 2))
    return ("unknown", round(max(0.42, min(face_on_score, profile_score)), 2))


def uses_face_on_top_detection(camera_angle: str) -> bool:
    return camera_angle in {"face_on", "side_profile"}


def horizontal_pair_ratio(frame: Frame, first_name: str, second_name: str, body_height: float) -> float:
    first = keypoint(frame, first_name)
    second = keypoint(frame, second_name)
    return abs(first[0] - second[0]) / max(0.001, body_height)


def body_bbox(frame: Frame) -> dict[str, float]:
    points = [point for point in frame["keypoints"] if point["name"] in BODY_QUALITY_KEYPOINTS and point["score"] >= 0.25]
    if not points:
        return {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0, "width": 0.0, "height": 0.0}

    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    min_x = float(min(xs))
    max_x = float(max(xs))
    min_y = float(min(ys))
    max_y = float(max(ys))
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def status_from_confidence(confidence: float) -> str:
    if confidence >= 0.78:
        return "ok"
    if confidence >= 0.55:
        return "warning"
    return "poor"


def dedupe_warnings(warnings: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        deduped.append(warning)
    return deduped


def detect_positions(frames: list[Frame], capture_profile: CaptureProfile | None = None) -> dict[str, int]:
    if len(frames) < 8:
        return fallback_positions(frames, "too_short")

    camera_angle = capture_profile["cameraAngle"] if capture_profile else "unknown"
    xs, ys = hand_center_series(frames)
    xs = smooth_series(xs, 5)
    ys = smooth_series(ys, 5)
    velocities = smooth_series(wrist_velocity(frames), 3)
    sample_fps = estimate_sample_fps(frames)

    burst = detect_motion_burst(frames)
    burst_peak = int(burst["peak"])
    burst_peak_velocity = float(velocities[burst_peak]) if 0 <= burst_peak < len(velocities) else 0.0
    burst_length = int(burst["end"]) - int(burst["start"])

    if burst_peak_velocity < 0.04 or burst_length < max(4, int(sample_fps * 0.4)):
        return _detect_positions_legacy(
            frames, xs, ys, velocities, sample_fps, reason="weak_burst", camera_angle=camera_angle
        )

    pre_burst_cap = max(2, int(burst["start"]) - max(1, int(sample_fps * 0.10)))
    address = detect_address_index(
        frames, ys, velocities, sample_fps, search_end_cap=pre_burst_cap
    )

    address_x = xs[address]
    address_y = ys[address]
    height_gain = address_y - ys
    displacement = np.sqrt((xs - address_x) ** 2 + (ys - address_y) ** 2)

    impact_seed = detect_first_hand_return(height_gain, velocities, address, sample_fps)
    if impact_seed is not None:
        top = detect_backswing_top(height_gain, velocities, address, impact_seed, sample_fps)
        mode = "rise_return"
        correction_threshold = burst_peak + max(1, int(round(sample_fps * 0.15)))
        if top > correction_threshold:
            top = detect_top_near_burst_peak(height_gain, velocities, address, burst_peak, sample_fps)
            impact_seed = detect_impact_after_top(
                height_gain,
                velocities,
                top,
                burst,
                sample_fps,
            )
            mode = "burst_corrected"

        impact = impact_seed if mode == "burst_corrected" else refine_impact(height_gain, velocities, address, top, impact_seed, sample_fps)
        if uses_face_on_top_detection(camera_angle):
            top = detect_face_on_top(height_gain, velocities, address, impact, sample_fps)
            mode = f"{mode}_face_on_top"
        takeaway = detect_takeaway(displacement, height_gain, address, top, sample_fps)
        finish = detect_finish_checkpoint(frames, height_gain, velocities, address, impact, sample_fps)

        positions = {
            "address": address,
            "takeaway": takeaway,
            "top": top,
            "impact": impact,
            "finish": finish,
        }
        positions = validate_phase_separation(positions, frames, sample_fps)
        positions = enforce_position_order(positions, frames)
        log_position_detection(
            frames,
            positions,
            height_gain,
            displacement,
            velocities,
            mode,
            details={
                "burst": burst,
                "burstPeakVelocity": round(burst_peak_velocity, 4),
                "impactSeed": int(impact_seed),
                "preBurstCap": int(pre_burst_cap),
            },
        )
        return positions

    # Detect top FIRST (max height_gain in burst, earliest qualifying local max).
    # Independent of impact, so impact detection can't collapse the top window.
    top = detect_top_in_burst(height_gain, velocities, address, burst, sample_fps)

    # Impact comes strictly AFTER top — first descending return to address height.
    impact = detect_impact_after_top(height_gain, velocities, top, burst, sample_fps)
    if uses_face_on_top_detection(camera_angle):
        top = detect_face_on_top(height_gain, velocities, address, impact, sample_fps)

    takeaway = detect_takeaway(displacement, height_gain, address, top, sample_fps)
    finish = detect_finish_checkpoint(frames, height_gain, velocities, address, impact, sample_fps)

    positions = {
        "address": address,
        "takeaway": takeaway,
        "top": top,
        "impact": impact,
        "finish": finish,
    }
    positions = validate_phase_separation(positions, frames, sample_fps)
    positions = enforce_position_order(positions, frames)

    burst_start = int(burst["start"])
    burst_end = min(len(frames) - 1, int(burst["end"]))
    burst_profile = [
        {
            "f": idx,
            "t": frames[idx]["t"],
            "g": round(float(height_gain[idx]), 4),
            "v": round(float(velocities[idx]), 4),
        }
        for idx in range(burst_start, burst_end + 1)
    ]
    log_position_detection(
        frames,
        positions,
        height_gain,
        displacement,
        velocities,
        "velocity_anchored",
        details={
            "burst": burst,
            "burstPeakVelocity": round(burst_peak_velocity, 4),
            "preBurstCap": int(pre_burst_cap),
            "burstProfile": burst_profile,
        },
    )

    return positions


def _detect_positions_legacy(
    frames: list[Frame],
    xs: np.ndarray,
    ys: np.ndarray,
    velocities: np.ndarray,
    sample_fps: float,
    reason: str,
    camera_angle: str = "unknown",
) -> dict[str, int]:
    """Cascading height-threshold detection — kept as fallback when motion burst is weak."""
    address = detect_address_index(frames, ys, velocities, sample_fps)
    address_x = xs[address]
    address_y = ys[address]
    height_gain = address_y - ys
    displacement = np.sqrt((xs - address_x) ** 2 + (ys - address_y) ** 2)

    impact_seed = detect_first_hand_return(height_gain, velocities, address, sample_fps)
    if impact_seed is None:
        return fallback_positions(frames, "no_hand_return", address)

    top = detect_backswing_top(height_gain, velocities, address, impact_seed, sample_fps)
    impact = refine_impact(height_gain, velocities, address, top, impact_seed, sample_fps)
    if uses_face_on_top_detection(camera_angle):
        top = detect_face_on_top(height_gain, velocities, address, impact, sample_fps)
    takeaway = detect_takeaway(displacement, height_gain, address, top, sample_fps)
    finish = detect_finish_checkpoint(frames, height_gain, velocities, address, impact, sample_fps)

    positions = {
        "address": address,
        "takeaway": takeaway,
        "top": top,
        "impact": impact,
        "finish": finish,
    }
    positions = validate_phase_separation(positions, frames, sample_fps)
    positions = enforce_position_order(positions, frames)
    log_position_detection(
        frames,
        positions,
        height_gain,
        displacement,
        velocities,
        f"legacy_{reason}",
    )
    return positions


def hand_center_series(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray]:
    centers = [grip_point(frame) for frame in frames]
    return (
        np.array([point[0] for point in centers]),
        np.array([point[1] for point in centers]),
    )


def grip_point(frame: Frame) -> tuple[float, float]:
    left = keypoint(frame, "left_wrist")
    right = keypoint(frame, "right_wrist")
    left_score = keypoint_score(frame, "left_wrist")
    right_score = keypoint_score(frame, "right_wrist")

    if left_score >= RELIABLE_KEYPOINT_SCORE and right_score >= RELIABLE_KEYPOINT_SCORE:
        return weighted_midpoint(frame, "left_wrist", "right_wrist")
    if left_score >= RELIABLE_KEYPOINT_SCORE:
        return left
    if right_score >= RELIABLE_KEYPOINT_SCORE:
        return right
    if left_score >= DRAWABLE_KEYPOINT_SCORE and right_score >= DRAWABLE_KEYPOINT_SCORE:
        return weighted_midpoint(frame, "left_wrist", "right_wrist")
    if left_score >= DRAWABLE_KEYPOINT_SCORE:
        return left
    if right_score >= DRAWABLE_KEYPOINT_SCORE:
        return right

    left_elbow_score = keypoint_score(frame, "left_elbow")
    right_elbow_score = keypoint_score(frame, "right_elbow")
    if left_elbow_score >= DRAWABLE_KEYPOINT_SCORE and right_elbow_score >= DRAWABLE_KEYPOINT_SCORE:
        return weighted_midpoint(frame, "left_elbow", "right_elbow")
    if left_elbow_score >= DRAWABLE_KEYPOINT_SCORE:
        return keypoint(frame, "left_elbow")
    if right_elbow_score >= DRAWABLE_KEYPOINT_SCORE:
        return keypoint(frame, "right_elbow")

    return weighted_midpoint(frame, "left_wrist", "right_wrist")


def detect_address_index(
    frames: list[Frame],
    hand_y: np.ndarray,
    velocities: np.ndarray,
    sample_fps: float,
    search_end_cap: int | None = None,
) -> int:
    natural_end = min(len(frames) - 1, max(3, int(sample_fps * 1.15), int(len(frames) * 0.28)))
    if search_end_cap is None:
        search_end = natural_end
    else:
        search_end = max(2, min(natural_end, int(search_end_cap)))
    search = np.arange(0, search_end + 1)
    velocity_window = velocities[search]
    y_window = hand_y[search]
    lower_body_velocity = smooth_series(body_velocity(frames, ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]), 3)
    lower_body_window = lower_body_velocity[search]
    max_velocity = max(float(np.max(velocity_window)), 0.001)
    max_lower_body_velocity = max(float(np.max(lower_body_window)), 0.001)
    y_range = max(float(np.max(y_window) - np.min(y_window)), 0.001)

    best_score = -float("inf")
    best_index = 0
    for index in search:
        low_hands_score = (hand_y[index] - float(np.min(y_window))) / y_range
        stable_score = 1.0 - min(1.0, float(velocities[index]) / max_velocity)
        body_stable_score = 1.0 - min(1.0, float(lower_body_velocity[index]) / max_lower_body_velocity)
        early_score = 1.0 - (index / max(1, search_end))
        score = low_hands_score * 0.5 + stable_score * 0.25 + body_stable_score * 0.15 + early_score * 0.1
        if score > best_score:
            best_score = score
            best_index = int(index)

    return best_index


def detect_first_hand_return(height_gain: np.ndarray, velocities: np.ndarray, address: int, sample_fps: float) -> int | None:
    search_start = min(len(height_gain) - 1, address + 1)
    search_end = min(len(height_gain) - 1, address + max(10, int(sample_fps * 3.4)))
    if search_end <= search_start:
        return None

    gains = height_gain[search_start : search_end + 1]
    max_gain = max(float(np.max(gains)), 0.0)
    if max_gain < 0.055:
        return None

    high_threshold = max(0.055, min(0.14, max_gain * 0.32))
    high_candidates = np.where(gains >= high_threshold)[0]
    if len(high_candidates) == 0:
        return None

    first_high = search_start + int(high_candidates[0])
    return_threshold = max(0.035, min(0.105, max_gain * 0.24))
    return_search_start = min(len(height_gain) - 1, first_high + max(3, int(sample_fps * 0.35)))
    return_search_end = min(search_end, first_high + max(8, int(sample_fps * 2.0)))

    for index in range(return_search_start, return_search_end + 1):
        if height_gain[index] <= return_threshold and is_descending_into_impact(height_gain, index, sample_fps):
            return index

    search = np.arange(return_search_start, return_search_end + 1)
    if len(search):
        closeness = np.abs(height_gain[search])
        max_velocity = max(float(np.max(velocities[search])), 0.001)
        speed = velocities[search] / max_velocity
        scores = (1.0 - np.minimum(1.0, closeness / max(0.14, max_gain * 0.35))) * 0.72 + speed * 0.28
        best = int(search[np.argmax(scores)])
        if height_gain[best] < max(0.14, max_gain * 0.38):
            return best

    return None


def detect_backswing_top(height_gain: np.ndarray, velocities: np.ndarray, address: int, impact_seed: int, sample_fps: float) -> int:
    search_start = min(len(height_gain) - 1, address + 2)
    impact_guard_frames = max(1, int(round(sample_fps * 0.10)))
    search_end = max(search_start, impact_seed - impact_guard_frames)
    if search_end <= search_start:
        return min(len(height_gain) - 1, address + 2)

    search = np.arange(search_start, search_end + 1)
    gain_window = height_gain[search]
    max_gain = max(float(np.max(gain_window)), 0.001)
    high_threshold = max(0.055, max_gain * 0.68)
    high_candidates = search[gain_window >= high_threshold]
    candidate_search = high_candidates if len(high_candidates) else search
    local_velocities = velocities[candidate_search]
    max_velocity = max(float(np.max(local_velocities)), 0.001)
    gain_scores = height_gain[candidate_search] / max_gain
    slow_scores = 1.0 - np.minimum(1.0, local_velocities / max_velocity)
    late_scores = (candidate_search - search_start) / max(1.0, float(search_end - search_start))
    scores = gain_scores * 0.55 + slow_scores * 0.2 + late_scores * 0.25
    return int(candidate_search[np.argmax(scores)])


def detect_face_on_top(height_gain: np.ndarray, velocities: np.ndarray, address: int, impact_seed: int, sample_fps: float) -> int:
    search_start = min(len(height_gain) - 1, address + max(2, int(round(sample_fps * 0.18))))
    search_end = max(search_start, impact_seed - max(1, int(round(sample_fps * 0.18))))
    if search_end <= search_start:
        return min(len(height_gain) - 1, address + 2)

    search = np.arange(search_start, search_end + 1)
    gains = smooth_series(height_gain[search], 3)
    max_gain = max(float(np.max(gains)), 0.001)
    high_threshold = max(0.04, max_gain * 0.56)
    high_candidates = search[gains >= high_threshold]
    candidate_search = high_candidates if len(high_candidates) else search

    local_gains = height_gain[candidate_search]
    local_velocities = velocities[candidate_search]
    max_velocity = max(float(np.max(local_velocities)), 0.001)
    lookahead = max(1, int(round(sample_fps * 0.18)))
    descent_scores = []
    for index in candidate_search:
        future = min(len(height_gain) - 1, int(index) + lookahead)
        drop = max(0.0, float(height_gain[index] - height_gain[future]))
        descent_scores.append(min(1.0, drop / max(0.001, max_gain * 0.32)))

    gain_scores = np.minimum(1.0, local_gains / max_gain)
    slow_scores = 1.0 - np.minimum(1.0, local_velocities / max_velocity)
    late_scores = (candidate_search - search_start) / max(1.0, float(search_end - search_start))
    scores = gain_scores * 0.34 + slow_scores * 0.28 + late_scores * 0.28 + np.array(descent_scores) * 0.10
    return int(candidate_search[np.argmax(scores)])


def detect_top_in_burst(
    height_gain: np.ndarray,
    velocities: np.ndarray,
    address: int,
    burst: dict[str, int],
    sample_fps: float,
) -> int:
    """Top of backswing — earliest local maximum of height_gain inside the burst.

    Picking the *earliest* qualifying local max (not the global argmax) protects
    against high-finish swings where the finish position has greater
    height_gain than the actual top. Top is detected independently of impact
    so impact detection cannot collapse the top window.
    """
    last = len(height_gain) - 1
    burst_start = max(address + 1, int(burst["start"]))
    burst_end = min(last, int(burst["end"]))
    if burst_end <= burst_start:
        return min(last, max(address + 1, burst_start))

    # Restrict to pre-finish portion of burst: stop searching once height_gain
    # has descended past its running max and crossed back near address height
    # (i.e., we've gone through impact and into the follow-through rise).
    search = np.arange(burst_start, burst_end + 1)
    gain_window = height_gain[search]
    if len(gain_window) == 0:
        return min(last, max(address + 1, burst_start))

    # Smooth inside the burst to suppress single-frame noise dips (e.g. a 0.144
    # → 0.142 wobble) that would otherwise create false local maxima.
    smoothed_gain = smooth_series(gain_window, 3)

    global_max = float(np.max(smoothed_gain))
    if global_max < 0.05:
        # No clear backswing rise — fall back to argmax inside burst
        return int(search[int(np.argmax(smoothed_gain))])

    threshold = global_max * 0.65
    radius = max(1, int(round(sample_fps * 0.18)))

    qualifying: list[int] = []
    for offset, frame_idx in enumerate(search):
        if smoothed_gain[offset] < threshold:
            continue
        lo = max(0, offset - radius)
        hi = min(len(smoothed_gain) - 1, offset + radius)
        if smoothed_gain[offset] >= float(np.max(smoothed_gain[lo : hi + 1])) - 1e-6:
            qualifying.append(int(frame_idx))

    if not qualifying:
        return int(search[int(np.argmax(smoothed_gain))])

    # Among qualifying local maxima, prefer the EARLIEST one — that's the
    # top of backswing. Anything later is finish-rise.
    return qualifying[0]


def detect_top_near_burst_peak(
    height_gain: np.ndarray,
    velocities: np.ndarray,
    address: int,
    burst_peak: int,
    sample_fps: float,
) -> int:
    last = len(height_gain) - 1
    search_start = max(address + 2, burst_peak - max(2, int(round(sample_fps * 0.70))))
    search_end = min(last, burst_peak + max(1, int(round(sample_fps * 0.12))))
    if search_end <= search_start:
        return clamp_array_index(burst_peak, height_gain)

    search = np.arange(search_start, search_end + 1)
    gains = smooth_series(height_gain[search], 3)
    max_gain = max(float(np.max(gains)), 0.001)
    high_threshold = max(0.055, max_gain * 0.68)
    radius = max(1, int(round(sample_fps * 0.12)))
    pre_peak_candidates = search[(search <= burst_peak) & (gains >= high_threshold)]
    if len(pre_peak_candidates):
        return int(pre_peak_candidates[0])

    local_maxima: list[int] = []
    for offset, frame_idx in enumerate(search):
        if gains[offset] < high_threshold:
            continue
        lo = max(0, offset - radius)
        hi = min(len(gains) - 1, offset + radius)
        if gains[offset] >= float(np.max(gains[lo : hi + 1])) - 1e-6:
            local_maxima.append(int(frame_idx))

    if local_maxima:
        pre_peak_maxima = [index for index in local_maxima if index <= burst_peak]
        return pre_peak_maxima[-1] if pre_peak_maxima else local_maxima[0]

    return int(search[int(np.argmax(gains))])


def detect_impact_after_top(
    height_gain: np.ndarray,
    velocities: np.ndarray,
    top: int,
    burst: dict[str, int],
    sample_fps: float,
    contact_lead_seconds: float = 0.0,
) -> int:
    """Impact = first descending zero-crossing of height_gain after top.

    In a real swing the hand rises through takeaway, peaks at top, descends
    through the downswing, returns to address height at impact, then rises
    again into finish. Searching strictly after `top` prevents pre-swing low
    height_gain frames from being mistaken for impact.
    """
    last = len(height_gain) - 1
    lo = min(last, top + max(1, int(round(sample_fps * 0.15))))
    burst_end = min(last, int(burst["end"]))
    hi = min(last, max(lo + 1, burst_end + max(2, int(round(sample_fps * 0.30)))))
    hi = min(hi, detect_post_top_descent_end(height_gain, top, hi, sample_fps))
    if hi <= lo:
        return min(last, top + 1)

    search = np.arange(lo, hi + 1)

    # Reference gain magnitude — height the hand fell through during downswing.
    pre_search_max = float(np.max(height_gain[top : lo + 1])) if lo > top else float(height_gain[top])
    reference_gain = max(pre_search_max, 1e-3)

    selected = select_impact_from_window(height_gain, velocities, search, reference_gain, sample_fps)
    return apply_contact_frame_lead(selected, top, lo, sample_fps, contact_lead_seconds)


def detect_post_top_descent_end(height_gain: np.ndarray, top: int, fallback_end: int, sample_fps: float) -> int:
    last = len(height_gain) - 1
    start = min(last, top + 1)
    end = min(last, max(start, fallback_end))
    if end <= start:
        return end

    rise_threshold = max(0.025, abs(float(height_gain[top])) * 0.10)
    min_gain = float(height_gain[start])
    min_index = start

    for index in range(start + 1, end + 1):
        current_gain = float(height_gain[index])
        if current_gain < min_gain:
            min_gain = current_gain
            min_index = index
            continue

        has_descended = min_index > start or min_gain < float(height_gain[start]) - 0.01
        enough_frames = index - start >= max(2, int(round(sample_fps * 0.18)))
        if has_descended and enough_frames and current_gain >= min_gain + rise_threshold:
            return max(start, index - 1)

    return end


def detect_impact_near_burst_peak(
    height_gain: np.ndarray,
    velocities: np.ndarray,
    address: int,
    burst_peak: int,
    sample_fps: float,
) -> int:
    last = len(height_gain) - 1
    search_start = max(address + 2, burst_peak - max(1, int(round(sample_fps * 0.25))))
    search_end = min(last, burst_peak + max(1, int(round(sample_fps * 0.35))))
    if search_end <= search_start:
        return clamp_array_index(burst_peak, height_gain)

    search = np.arange(search_start, search_end + 1)
    reference_gain = max(float(np.max(height_gain[address : min(last, burst_peak) + 1])), 0.001)
    return select_impact_from_window(height_gain, velocities, search, reference_gain, sample_fps)


def validate_phase_separation(
    positions: dict[str, int],
    frames: list[Frame],
    sample_fps: float,
    min_gap_seconds: float = 0.15,
) -> dict[str, int]:
    """Detect clustered phases and redistribute takeaway/top proportionally.

    If any consecutive pair in (address → takeaway → top → impact) is closer
    than min_gap_seconds in time, redistribute takeaway and top at 25% / 60%
    of the address→impact frame span.
    """
    if not frames or sample_fps <= 0:
        return positions

    order = ["address", "takeaway", "top", "impact"]
    times = [frames[positions[name]]["t"] for name in order]
    min_gap_ms = min_gap_seconds * 1000

    clustered = any(times[i + 1] - times[i] < min_gap_ms for i in range(len(order) - 1))
    if not clustered:
        return positions

    address_idx = positions["address"]
    impact_idx = positions["impact"]
    span = impact_idx - address_idx
    if span < 3:
        return positions

    redistributed = dict(positions)
    redistributed["takeaway"] = address_idx + max(1, int(round(span * 0.25)))
    redistributed["top"] = address_idx + max(2, int(round(span * 0.60)))
    return redistributed


def detect_takeaway(displacement: np.ndarray, height_gain: np.ndarray, address: int, top: int, sample_fps: float) -> int:
    search_start = min(len(displacement) - 1, address + 1)
    # Search up through most of the way to top — the original 0.9s cap from address
    # was too tight for clips with long pre-swing setup, where motion may not start
    # until well after the detected address frame.
    search_end = max(search_start, min(top - 1, top - max(1, int(round(sample_fps * 0.15)))))
    if search_end <= search_start:
        return min(len(displacement) - 1, address + 1)

    search = np.arange(search_start, search_end + 1)
    max_displacement = max(float(np.max(displacement[address : top + 1])), 0.001)
    threshold = max(0.028, min(0.075, max_displacement * 0.24))
    candidates = search[(displacement[search] >= threshold) & (height_gain[search] >= -0.015)]

    if len(candidates):
        return int(candidates[0])

    return int(search[np.argmax(displacement[search])])


def refine_impact(
    height_gain: np.ndarray,
    velocities: np.ndarray,
    address: int,
    top: int,
    impact_seed: int,
    sample_fps: float,
) -> int:
    last = len(height_gain) - 1
    search_start = max(top + 1, min(last, impact_seed))
    search_end = min(
        last,
        max(
            impact_seed + max(2, int(round(sample_fps * 0.75))),
            top + max(4, int(round(sample_fps * 1.4))),
        ),
    )
    search = np.arange(search_start, search_end + 1)
    if len(search) == 0:
        return impact_seed

    backswing_gain = max(float(np.max(height_gain[address : max(top + 1, address + 1)])), 0.001)
    selected = select_impact_from_window(height_gain, velocities, search, backswing_gain, sample_fps)
    return apply_contact_frame_lead(selected, top, search_start, sample_fps, 0.0)


def detect_finish_checkpoint(
    frames: list[Frame],
    height_gain: np.ndarray,
    velocities: np.ndarray,
    address: int,
    impact: int,
    sample_fps: float,
) -> int:
    last = len(frames) - 1
    if impact >= last:
        return last

    address_to_impact_ms = max(0, frames[impact]["t"] - frames[address]["t"])
    target_delay_ms = int(round(min(650, max(350, address_to_impact_ms * 0.34))))
    target_time_ms = frames[impact]["t"] + target_delay_ms
    search_start = min(last, impact + 1)
    search_end = min(last, impact + max(2, int(round(sample_fps * 0.9))))
    if search_end < search_start:
        return min(last, impact + 1)

    search = np.arange(search_start, search_end + 1)
    qualities = np.array([golfer_frame_quality(frames[index]) for index in search])
    quality_threshold = max(0.45, float(np.percentile(qualities, 25)))
    high_quality_candidates = search[qualities >= quality_threshold]
    candidates = high_quality_candidates if len(high_quality_candidates) else search
    return int(min(candidates, key=lambda index: abs(frames[int(index)]["t"] - target_time_ms)))


def select_impact_from_window(
    height_gain: np.ndarray,
    velocities: np.ndarray,
    search: np.ndarray,
    reference_gain: float,
    sample_fps: float,
) -> int:
    near_threshold = max(0.045, reference_gain * 0.38)
    descending_mask = np.array(
        [is_descending_into_impact(height_gain, int(index), sample_fps) for index in search],
        dtype=bool,
    )
    near_mask = np.abs(height_gain[search]) <= near_threshold
    candidates = search[descending_mask & near_mask]

    if len(candidates):
        candidates = first_candidate_cluster(candidates, sample_fps)
        candidate_velocities = velocities[candidates]
        candidate_closeness = 1.0 - np.minimum(1.0, np.abs(height_gain[candidates]) / near_threshold)
        max_velocity = max(float(np.max(candidate_velocities)), 0.001)
        scores = (candidate_velocities / max_velocity) * 0.72 + candidate_closeness * 0.28
        return int(candidates[int(np.argmax(scores))])

    descending_candidates = search[descending_mask]
    if len(descending_candidates):
        descending_candidates = first_candidate_cluster(descending_candidates, sample_fps)
        max_velocity = max(float(np.max(velocities[descending_candidates])), 0.001)
        closeness = 1.0 - np.minimum(1.0, np.abs(height_gain[descending_candidates]) / max(near_threshold, reference_gain))
        scores = (velocities[descending_candidates] / max_velocity) * 0.65 + closeness * 0.35
        return int(descending_candidates[int(np.argmax(scores))])

    max_velocity = max(float(np.max(velocities[search])), 0.001)
    closeness = 1.0 - np.minimum(1.0, np.abs(height_gain[search]) / max(near_threshold, reference_gain))
    scores = (velocities[search] / max_velocity) * 0.6 + closeness * 0.4
    return int(search[int(np.argmax(scores))])


def first_candidate_cluster(candidates: np.ndarray, sample_fps: float) -> np.ndarray:
    if len(candidates) <= 1:
        return candidates

    max_gap = max(1, int(round(sample_fps * 0.25)))
    cluster_end = 1
    while cluster_end < len(candidates) and int(candidates[cluster_end] - candidates[cluster_end - 1]) <= max_gap:
        cluster_end += 1

    return candidates[:cluster_end]


def apply_contact_frame_lead(
    selected: int,
    top: int,
    lower_bound: int,
    sample_fps: float,
    lead_seconds: float,
) -> int:
    if lead_seconds <= 0 or sample_fps <= 0:
        return selected

    lead_frames = max(1, int(round(sample_fps * lead_seconds)))
    late_enough_to_lead = selected - top >= max(lead_frames + 1, int(round(sample_fps * 0.75)))
    if not late_enough_to_lead:
        return selected

    return max(lower_bound, selected - lead_frames)


def enforce_position_order(positions: dict[str, int], frames: list[Frame]) -> dict[str, int]:
    last = len(frames) - 1
    ordered = {
        "address": clamp_index(positions["address"], frames),
        "takeaway": clamp_index(positions["takeaway"], frames),
        "top": clamp_index(positions["top"], frames),
        "impact": clamp_index(positions["impact"], frames),
        "finish": clamp_index(positions["finish"], frames),
    }
    ordered["takeaway"] = min(last, max(ordered["takeaway"], ordered["address"] + 1))
    ordered["top"] = min(last, max(ordered["top"], ordered["takeaway"] + 1))
    ordered["impact"] = min(last, max(ordered["impact"], ordered["top"] + 1))
    ordered["finish"] = min(last, max(ordered["finish"], ordered["impact"] + 1))
    return ordered


def is_descending_into_impact(height_gain: np.ndarray, index: int, sample_fps: float) -> bool:
    lookback = max(2, int(sample_fps * 0.18))
    start = max(0, index - lookback)
    if index <= start:
        return True

    return float(height_gain[start]) >= float(height_gain[index])


def body_velocity(frames: list[Frame], names: list[str]) -> np.ndarray:
    if not frames:
        return np.array([])

    series = []
    for name in names:
        points = keypoint_series(frames, name)
        xs = np.array([point[0] for point in points])
        ys = np.array([point[1] for point in points])
        series.append(np.abs(np.gradient(xs)) + np.abs(np.gradient(ys)))

    return np.mean(np.array(series), axis=0)


def fallback_positions(frames: list[Frame], reason: str, address: int = 0) -> dict[str, int]:
    last = max(0, len(frames) - 1)
    address = clamp_index(address, frames)
    remaining = max(1, last - address)
    positions = {
        "address": address,
        "takeaway": clamp_index(address + int(remaining * 0.18), frames),
        "top": clamp_index(address + int(remaining * 0.42), frames),
        "impact": clamp_index(address + int(remaining * 0.62), frames),
        "finish": clamp_index(address + int(remaining * 0.84), frames),
    }
    positions["takeaway"] = min(last, max(positions["takeaway"], positions["address"] + 1))
    positions["top"] = min(last, max(positions["top"], positions["takeaway"] + 1))
    positions["impact"] = min(last, max(positions["impact"], positions["top"] + 1))
    positions["finish"] = min(last, max(positions["finish"], positions["impact"] + 1))
    print(
        "position detection fallback",
        {"reason": reason, "positions": {name: frames[index]["t"] for name, index in positions.items()}},
        flush=True,
    )

    return positions


def log_position_detection(
    frames: list[Frame],
    positions: dict[str, int],
    height_gain: np.ndarray,
    displacement: np.ndarray,
    velocities: np.ndarray,
    mode: str,
    details: dict | None = None,
) -> None:
    payload: dict = {
        "mode": mode,
        "positions": {
            name: {
                "frame": index,
                "t": frames[index]["t"],
                "heightGain": round(float(height_gain[index]), 4),
                "handDisplacement": round(float(displacement[index]), 4),
                "velocity": round(float(velocities[index]), 4),
            }
            for name, index in positions.items()
        },
    }
    if details:
        payload["details"] = details
    print("position detection", payload, flush=True)


def detect_motion_burst(frames: list[Frame]) -> dict[str, int]:
    velocities = smooth_series(wrist_velocity(frames), 3)
    if len(velocities) < 3:
        return {"start": 0, "peak": 0, "end": max(0, len(frames) - 1)}

    peak = int(np.argmax(velocities))
    max_velocity = max(float(velocities[peak]), 0.001)
    threshold = max(0.01, max_velocity * 0.18)

    start = peak
    while start > 0 and velocities[start] > threshold:
        start -= 1

    end = peak
    while end < len(velocities) - 1 and velocities[end] > threshold * 0.55:
        end += 1

    start = max(0, start - 3)
    end = min(len(frames) - 1, end + 4)

    return {"start": start, "peak": peak, "end": end}


def estimate_sample_fps(frames: list[Frame]) -> float:
    if len(frames) < 2:
        return 10

    deltas = [frames[index + 1]["t"] - frames[index]["t"] for index in range(len(frames) - 1)]
    median_delta = max(1, float(np.median(deltas)))
    return 1000 / median_delta


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values

    kernel = np.ones(window) / window
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded_values = np.pad(values, (left_pad, right_pad), mode="edge")
    return np.convolve(padded_values, kernel, mode="valid")


def wrist_velocity(frames: list[Frame]) -> np.ndarray:
    xs, ys = hand_center_series(frames)

    return np.abs(np.gradient(xs)) + np.abs(np.gradient(ys))


def compute_metrics(frames: list[Frame], positions: dict[str, int]) -> dict[str, float]:
    address = frames[positions["address"]]
    takeaway = frames[positions["takeaway"]]
    top = frames[positions["top"]]
    impact = frames[positions["impact"]]
    finish = frames[positions["finish"]]

    head_displacement = distance(keypoint(address, "nose"), keypoint(takeaway, "nose"))
    shoulder_turn = angle_between(keypoint(top, "left_shoulder"), keypoint(top, "right_shoulder"))
    hip_sway = distance(midpoint(address, "left_hip", "right_hip"), midpoint(impact, "left_hip", "right_hip"))
    posture_loss = abs(spine_angle(address) - spine_angle(impact))
    tempo = max(0.1, (frames[positions["impact"]]["t"] - frames[positions["address"]]["t"]) / 1000)

    return {
        "headDisplacement": round(head_displacement, 4),
        "shoulderTurn": round(shoulder_turn, 2),
        "hipSway": round(hip_sway, 4),
        "postureLoss": round(posture_loss, 2),
        "tempo": round(tempo, 2),
    }


def build_phase_confidence(
    frames: list[Frame],
    positions: dict[str, int],
    capture_profile: CaptureProfile,
) -> dict[str, float]:
    if not frames:
        return {name: 0.0 for name in POSITION_NAMES}

    capture_confidence = float(capture_profile["captureConfidence"])
    camera_angle = capture_profile["cameraAngle"]
    sample_fps = estimate_sample_fps(frames)
    confidences: dict[str, float] = {}

    for name in POSITION_NAMES:
        index = clamp_index(positions[name], frames)
        local_quality = golfer_frame_quality(frames[index])
        separation = phase_separation_confidence(name, positions, frames, sample_fps)
        phase_confidence = capture_confidence * 0.68 + local_quality * 0.22 + separation * 0.10

        if name == "top":
            phase_confidence += top_phase_bonus(frames, positions, sample_fps) * capture_confidence
            if camera_angle == "unknown":
                phase_confidence -= 0.16
        elif name == "impact":
            phase_confidence += impact_phase_bonus(frames, positions, sample_fps) * capture_confidence
        elif name == "finish" and positions["finish"] <= positions["impact"]:
            phase_confidence -= 0.18

        confidences[name] = round(clamp_float(phase_confidence), 2)

    return confidences


def phase_separation_confidence(name: str, positions: dict[str, int], frames: list[Frame], sample_fps: float) -> float:
    if name == "address":
        return 0.92

    order = POSITION_NAMES
    index = order.index(name)
    current = positions[name]
    previous = positions[order[index - 1]]
    previous_gap_ms = frames[current]["t"] - frames[previous]["t"]
    min_gap_ms = 1000 * max(0.12, min(0.28, 2.0 / max(1.0, sample_fps)))
    previous_score = clamp_float(previous_gap_ms / min_gap_ms)

    if index >= len(order) - 1:
        return previous_score

    next_phase = positions[order[index + 1]]
    next_gap_ms = frames[next_phase]["t"] - frames[current]["t"]
    next_score = clamp_float(next_gap_ms / min_gap_ms)
    return float(min(previous_score, next_score))


def top_phase_bonus(frames: list[Frame], positions: dict[str, int], sample_fps: float) -> float:
    address = positions["address"]
    top = positions["top"]
    impact = positions["impact"]
    if impact <= address:
        return -0.18

    xs, ys = hand_center_series(frames)
    ys = smooth_series(ys, 5)
    height_gain = ys[address] - ys
    top_gain = float(height_gain[top])
    search_start = max(address, min(top, positions["takeaway"]))
    search_end = max(search_start, min(impact - 1, len(frames) - 1))
    max_gain = max(float(np.max(height_gain[search_start : search_end + 1])), 0.001)
    gain_score = clamp_float(top_gain / max_gain)
    timing_ratio = (top - address) / max(1, impact - address)
    timing_score = 1.0 - min(1.0, abs(timing_ratio - 0.58) / 0.38)
    minimum_gap_frames = max(1, int(round(sample_fps * 0.18)))
    gap_penalty = 0.16 if impact - top < minimum_gap_frames else 0.0
    return gain_score * 0.08 + timing_score * 0.05 - gap_penalty


def impact_phase_bonus(frames: list[Frame], positions: dict[str, int], sample_fps: float) -> float:
    top = positions["top"]
    impact = positions["impact"]
    if impact <= top:
        return -0.22

    velocities = smooth_series(wrist_velocity(frames), 3)
    search_start = top + 1
    search_end = min(len(frames) - 1, impact + max(1, int(round(sample_fps * 0.2))))
    local_max_velocity = max(float(np.max(velocities[search_start : search_end + 1])), 0.001)
    speed_score = min(1.0, float(velocities[impact]) / local_max_velocity)
    gap_ms = frames[impact]["t"] - frames[top]["t"]
    gap_score = clamp_float(gap_ms / 320)
    return speed_score * 0.06 + gap_score * 0.06


def finalize_analysis_quality(
    capture_profile: CaptureProfile,
    phase_confidence: dict[str, float],
) -> CaptureProfile:
    score_confidence = min(
        float(capture_profile["captureConfidence"]),
        float(np.mean([phase_confidence.get(name, 0.0) for name in POSITION_NAMES])),
    )
    capture_status = capture_profile["status"]
    phase_status = status_from_confidence(score_confidence)
    status = capture_status
    if capture_status != "ok" or phase_status == "poor":
        status = worst_quality_status(capture_status, phase_status)
    warnings = list(capture_profile["warnings"])
    low_phase_names = [name for name, confidence in phase_confidence.items() if confidence < PHASE_CONFIDENCE_THRESHOLD]
    if low_phase_names:
        warnings.append("low_phase_confidence")
    if status != "ok" and "provisional_analysis" not in warnings:
        warnings.append("provisional_analysis")

    return {
        **capture_profile,
        "status": status,
        "scoreConfidence": round(clamp_float(score_confidence), 2),
        "warnings": dedupe_warnings(warnings),
    }


def worst_quality_status(first: str, second: str) -> str:
    rank = {"ok": 0, "warning": 1, "poor": 2}
    return first if rank.get(first, 2) >= rank.get(second, 2) else second


def build_feedback(metrics: dict[str, float]) -> list[dict]:
    return [
        feedback_item(
            "takeaway",
            "Head In Zone",
            "good" if metrics["headDisplacement"] < 0.055 else "warning",
            "Keep your head inside a small window during takeaway. Smaller movement usually improves low-point control.",
        ),
        feedback_item(
            "top",
            "Shoulder turn",
            "good" if metrics["shoulderTurn"] > 8 else "warning",
            "A stronger shoulder turn gives you room to shallow the club before impact.",
        ),
        feedback_item(
            "impact",
            "Posture maintained",
            "good" if metrics["postureLoss"] < 8 else "bad",
            "Limit early extension by keeping your trail hip back while the lead shoulder works upward.",
        ),
        feedback_item(
            "impact",
            "Hip sway",
            "good" if metrics["hipSway"] < 0.085 else "warning",
            "Too much lateral hip movement can move the low point behind the ball.",
        ),
        feedback_item(
            "finish",
            "Balanced finish",
            "good" if metrics["tempo"] > 0.6 else "warning",
            "Hold the finish and let speed collect through the ball instead of rushing transition.",
        ),
    ]


def gate_feedback(
    feedback: list[dict],
    phase_confidence: dict[str, float] | None,
    threshold: float = PHASE_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    if phase_confidence is None:
        return feedback

    return [
        item
        for item in feedback
        if phase_confidence.get(str(item.get("position")), 0.0) >= threshold
    ]


def build_swing_thought(
    metrics: dict[str, float],
    feedback: list[dict],
    phase_confidence: dict[str, float] | None = None,
    analysis_quality: CaptureProfile | None = None,
) -> dict:
    issues = [
        {
            "key": "postureLoss",
            "position": "impact",
            "label": "Posture through impact",
            "status": "bad" if metrics["postureLoss"] >= 8 else "good",
            "severity": max(0.0, (metrics["postureLoss"] - 8) / 8),
            "thought": "Keep your chest over the ball as your lead shoulder works up.",
            "reason": "Your spine angle is changing before impact, so one posture cue is the best priority.",
        },
        {
            "key": "hipSway",
            "position": "impact",
            "label": "Lower-body control",
            "status": "warning" if metrics["hipSway"] >= 0.085 else "good",
            "severity": max(0.0, (metrics["hipSway"] - 0.085) / 0.085),
            "thought": "Turn around your trail hip, then post into your lead side.",
            "reason": "Your hips are drifting laterally, which can move the low point behind the ball.",
        },
        {
            "key": "headDisplacement",
            "position": "takeaway",
            "label": "Quiet takeaway",
            "status": "warning" if metrics["headDisplacement"] >= 0.055 else "good",
            "severity": max(0.0, (metrics["headDisplacement"] - 0.055) / 0.055),
            "thought": "Keep your nose quiet while the handle starts back.",
            "reason": "Your head is moving early, so a quiet-start cue should simplify the takeaway.",
        },
        {
            "key": "shoulderTurn",
            "position": "top",
            "label": "Complete the turn",
            "status": "warning" if metrics["shoulderTurn"] <= 8 else "good",
            "severity": max(0.0, (8 - metrics["shoulderTurn"]) / 8),
            "thought": "Turn your chest until your back starts facing the target.",
            "reason": "Your shoulder turn is short, which can make the downswing feel rushed.",
        },
        {
            "key": "tempo",
            "position": "finish",
            "label": "Smooth tempo",
            "status": "warning" if metrics["tempo"] <= 0.6 else "good",
            "severity": max(0.0, (0.6 - metrics["tempo"]) / 0.6),
            "thought": "Start down smooth, then let speed collect through the ball.",
            "reason": "Your address-to-impact window is quick, so tempo is the simplest focus.",
        },
    ]
    priority = {"bad": 3, "warning": 2, "good": 1}
    safe_issues = [
        issue
        for issue in issues
        if phase_confidence is None
        or phase_confidence.get(issue["position"], 0.0) >= PHASE_CONFIDENCE_THRESHOLD
    ]
    if not safe_issues and analysis_quality is not None:
        angle = analysis_quality.get("cameraAngle", "unknown")
        angle_copy = "down-the-line or face-on" if angle == "unknown" else angle.replace("_", "-")
        return {
            "position": "address",
            "label": "Retake checkpoint",
            "status": "warning",
            "thought": f"Retake this from a clearer {angle_copy} angle before changing your swing.",
            "reason": "The capture confidence is too low to choose a body-motion cue reliably.",
            "metricKey": "capture",
        }

    best = max(safe_issues or issues, key=lambda issue: (priority[issue["status"]], issue["severity"]))
    if best["status"] == "good":
        best = {
            "key": "balanced",
            "position": "finish",
            "label": "Keep the pattern",
            "status": "good",
            "thought": "Hold your finish and repeat the same smooth tempo.",
            "reason": "Your measured checkpoints are in good ranges, so the best thought is a repeatable finish cue.",
        }

    return {
        "position": best["position"],
        "label": best["label"],
        "status": best["status"],
        "thought": best["thought"],
        "reason": best["reason"],
        "metricKey": best["key"],
    }


def compute_score(feedback: list[dict]) -> float:
    weights = {"good": 1.0, "warning": 0.55, "bad": 0.1}
    raw_score = sum(weights[item["status"]] for item in feedback) / max(1, len(feedback))
    return round(raw_score * 10, 1)


def cap_score_for_quality(raw_score: float, analysis_quality: CaptureProfile) -> float:
    cap = QUALITY_SCORE_CAPS.get(analysis_quality["status"], 6.0)
    return round(min(raw_score, cap), 1)


def feedback_item(position: str, label: str, status: str, detail: str) -> dict:
    return {
        "position": position,
        "label": label,
        "status": status,
        "detail": detail,
        "drillIds": [],
    }


def repair_occluded_arm_keypoints(keypoints: list[Point]) -> list[Point]:
    repaired = [dict(point) for point in keypoints]
    points_by_name = {point["name"]: point for point in repaired}
    infer_occluded_arm_side(points_by_name, hidden_side="left", visible_side="right")
    infer_occluded_arm_side(points_by_name, hidden_side="right", visible_side="left")
    return repaired


def repair_foot_keypoints(frames: list[Frame]) -> list[Frame]:
    repaired_frames: list[Frame] = [
        {**frame, "keypoints": [dict(point) for point in frame["keypoints"]]}
        for frame in frames
    ]

    for side in ("left", "right"):
        for foot_type in FOOT_KEYPOINT_TYPES:
            name = f"{side}_{foot_type}"
            valid = validate_foot_keypoint_series(repaired_frames, side, name)
            valid = interpolate_foot_keypoint_gaps(repaired_frames, side, name, valid)
            smooth_foot_keypoint_series(repaired_frames, side, name, valid)

    for side in ("left", "right"):
        repair_foot_axis_series(repaired_frames, side)

    return repaired_frames


def validate_foot_keypoint_series(frames: list[Frame], side: str, name: str) -> list[bool]:
    valid = []
    for frame in frames:
        point = frame_keypoint(frame, name)
        is_valid = point is not None and is_plausible_foot_keypoint(frame, side, point)
        valid.append(is_valid)
        if point is not None and not is_valid:
            suppress_keypoint(point)

    return valid


def interpolate_foot_keypoint_gaps(
    frames: list[Frame],
    side: str,
    name: str,
    valid: list[bool],
) -> list[bool]:
    repaired_valid = list(valid)
    index = 0
    while index < len(frames):
        if valid[index]:
            index += 1
            continue

        start = index
        while index < len(frames) and not valid[index]:
            index += 1

        end = index - 1
        gap_length = end - start + 1
        previous_index = start - 1
        next_index = index
        if (
            gap_length > FOOT_REPAIR_GAP_FRAMES
            or previous_index < 0
            or next_index >= len(frames)
            or not valid[previous_index]
            or not valid[next_index]
        ):
            continue

        for repair_index in range(start, next_index):
            repaired_valid[repair_index] = infer_foot_keypoint_between(
                frames,
                side,
                name,
                repair_index,
                previous_index,
                next_index,
            )

    return repaired_valid


def infer_foot_keypoint_between(
    frames: list[Frame],
    side: str,
    name: str,
    repair_index: int,
    previous_index: int,
    next_index: int,
) -> bool:
    previous_frame = frames[previous_index]
    next_frame = frames[next_index]
    frame = frames[repair_index]
    previous_point = frame_keypoint(previous_frame, name)
    next_point = frame_keypoint(next_frame, name)
    point = frame_keypoint(frame, name)
    previous_ankle = frame_keypoint(previous_frame, f"{side}_ankle")
    next_ankle = frame_keypoint(next_frame, f"{side}_ankle")
    ankle = frame_keypoint(frame, f"{side}_ankle")
    if any(
        value is None
        for value in (previous_point, next_point, point, previous_ankle, next_ankle, ankle)
    ):
        return False
    if ankle["score"] < DRAWABLE_KEYPOINT_SCORE:
        return False

    amount = (repair_index - previous_index) / max(1, next_index - previous_index)
    previous_offset = (
        previous_point["x"] - previous_ankle["x"],
        previous_point["y"] - previous_ankle["y"],
    )
    next_offset = (
        next_point["x"] - next_ankle["x"],
        next_point["y"] - next_ankle["y"],
    )
    interpolated_offset = (
        previous_offset[0] + (next_offset[0] - previous_offset[0]) * amount,
        previous_offset[1] + (next_offset[1] - previous_offset[1]) * amount,
    )

    point["x"] = ankle["x"] + interpolated_offset[0]
    point["y"] = ankle["y"] + interpolated_offset[1]
    point["score"] = min(
        INFERRED_KEYPOINT_SCORE_CAP,
        max(
            DRAWABLE_KEYPOINT_SCORE + 0.03,
            min(previous_point["score"], next_point["score"], ankle["score"]) * 0.62,
        ),
    )
    if is_plausible_foot_keypoint(frame, side, point):
        return True

    suppress_keypoint(point)
    return False


def smooth_foot_keypoint_series(
    frames: list[Frame],
    side: str,
    name: str,
    valid: list[bool],
) -> None:
    if len(frames) < 3:
        return

    updates: list[tuple[int, float, float]] = []
    for index, is_valid in enumerate(valid):
        if not is_valid:
            continue

        neighbor_indices = [
            neighbor_index
            for neighbor_index in range(max(0, index - 1), min(len(frames), index + 2))
            if valid[neighbor_index]
        ]
        if len(neighbor_indices) < 2:
            continue

        frame = frames[index]
        point = frame_keypoint(frame, name)
        ankle = frame_keypoint(frame, f"{side}_ankle")
        if point is None or ankle is None or ankle["score"] < DRAWABLE_KEYPOINT_SCORE:
            continue

        total_weight = 0.0
        offset_x = 0.0
        offset_y = 0.0
        for neighbor_index in neighbor_indices:
            neighbor_frame = frames[neighbor_index]
            neighbor_point = frame_keypoint(neighbor_frame, name)
            neighbor_ankle = frame_keypoint(neighbor_frame, f"{side}_ankle")
            if neighbor_point is None or neighbor_ankle is None:
                continue

            weight = max(0.05, neighbor_point["score"])
            total_weight += weight
            offset_x += (neighbor_point["x"] - neighbor_ankle["x"]) * weight
            offset_y += (neighbor_point["y"] - neighbor_ankle["y"]) * weight

        if total_weight <= 0:
            continue

        candidate_x = ankle["x"] + offset_x / total_weight
        candidate_y = ankle["y"] + offset_y / total_weight
        blended = {
            **point,
            "x": point["x"] * 0.55 + candidate_x * 0.45,
            "y": point["y"] * 0.55 + candidate_y * 0.45,
        }
        if is_plausible_foot_keypoint(frame, side, blended):
            updates.append((index, blended["x"], blended["y"]))

    for index, x, y in updates:
        point = frame_keypoint(frames[index], name)
        if point is not None:
            point["x"] = x
            point["y"] = y


def repair_foot_axis_series(frames: list[Frame], side: str) -> None:
    for frame in frames:
        repair_foot_axis(frame, side)


def repair_foot_axis(frame: Frame, side: str) -> None:
    heel = frame_keypoint(frame, f"{side}_heel")
    toe = frame_keypoint(frame, f"{side}_foot_index")
    ankle = frame_keypoint(frame, f"{side}_ankle")
    if heel is None or toe is None or ankle is None:
        return
    if ankle["score"] < DRAWABLE_KEYPOINT_SCORE:
        suppress_keypoint(heel)
        suppress_keypoint(toe)
        return
    if not is_plausible_foot_keypoint(frame, side, heel) or not is_plausible_foot_keypoint(frame, side, toe):
        suppress_keypoint(heel)
        suppress_keypoint(toe)
        return

    foot_axis = normalized_vector(heel, toe)
    if foot_axis is None:
        synthesize_stance_aligned_foot(frame, side, heel, toe)
        return

    stance_axis = stance_axis_for_frame(frame)
    alignment = abs(dot_product(foot_axis, stance_axis))
    if alignment >= FOOT_AXIS_ALIGNMENT_THRESHOLD:
        return

    synthesize_stance_aligned_foot(frame, side, heel, toe)


def synthesize_stance_aligned_foot(frame: Frame, side: str, heel: Point, toe: Point) -> None:
    stance_axis = stance_axis_for_frame(frame)
    raw_axis = normalized_vector(heel, toe)
    if raw_axis is not None and dot_product(raw_axis, stance_axis) < 0:
        stance_axis = (-stance_axis[0], -stance_axis[1])

    body_height = foot_reference_body_height(frame)
    raw_length = math.hypot(toe["x"] - heel["x"], toe["y"] - heel["y"])
    min_length = max(0.035, body_height * 0.07)
    max_length = max(min_length, min(0.16, body_height * 0.22))
    foot_length = max(min_length, min(max_length, raw_length))
    center = ((heel["x"] + toe["x"]) / 2, (heel["y"] + toe["y"]) / 2)
    half_length = foot_length / 2

    candidate_heel = {
        **heel,
        "x": center[0] - stance_axis[0] * half_length,
        "y": center[1] - stance_axis[1] * half_length,
    }
    candidate_toe = {
        **toe,
        "x": center[0] + stance_axis[0] * half_length,
        "y": center[1] + stance_axis[1] * half_length,
    }
    if not (
        is_plausible_foot_keypoint(frame, side, candidate_heel)
        and is_plausible_foot_keypoint(frame, side, candidate_toe)
    ):
        suppress_keypoint(heel)
        suppress_keypoint(toe)
        return

    inferred_score = min(
        INFERRED_KEYPOINT_SCORE_CAP,
        max(DRAWABLE_KEYPOINT_SCORE + 0.03, min(heel["score"], toe["score"]) * 0.62),
    )
    heel["x"] = candidate_heel["x"]
    heel["y"] = candidate_heel["y"]
    heel["score"] = inferred_score
    toe["x"] = candidate_toe["x"]
    toe["y"] = candidate_toe["y"]
    toe["score"] = inferred_score


def stance_axis_for_frame(frame: Frame) -> tuple[float, float]:
    ankle_axis = normalized_pair_axis(frame, "left_ankle", "right_ankle")
    if ankle_axis is not None:
        return ankle_axis

    hip_axis = normalized_pair_axis(frame, "left_hip", "right_hip")
    if hip_axis is not None:
        return hip_axis

    return (1.0, 0.0)


def normalized_pair_axis(frame: Frame, first_name: str, second_name: str) -> tuple[float, float] | None:
    first = frame_keypoint(frame, first_name)
    second = frame_keypoint(frame, second_name)
    if first is None or second is None:
        return None
    if first["score"] < DRAWABLE_KEYPOINT_SCORE or second["score"] < DRAWABLE_KEYPOINT_SCORE:
        return None

    axis = normalize_delta(second["x"] - first["x"], second["y"] - first["y"])
    if axis is None:
        return None
    if axis[0] < 0:
        return (-axis[0], -axis[1])
    return axis


def normalized_vector(first: Point, second: Point) -> tuple[float, float] | None:
    return normalize_delta(second["x"] - first["x"], second["y"] - first["y"])


def normalize_delta(dx: float, dy: float) -> tuple[float, float] | None:
    length = math.hypot(dx, dy)
    if length < 0.01:
        return None
    return (dx / length, dy / length)


def dot_product(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1]


def is_plausible_foot_keypoint(frame: Frame, side: str, point: Point) -> bool:
    ankle = frame_keypoint(frame, f"{side}_ankle")
    if ankle is None:
        return False
    if point["score"] < DRAWABLE_KEYPOINT_SCORE or ankle["score"] < DRAWABLE_KEYPOINT_SCORE:
        return False
    if point["x"] < -0.15 or point["x"] > 1.15 or point["y"] < -0.1 or point["y"] > 1.15:
        return False

    body_height = foot_reference_body_height(frame)
    max_distance = max(0.055, min(0.22, body_height * 0.30))
    max_vertical_delta = max(0.055, min(0.18, body_height * 0.24))
    distance_from_ankle = math.hypot(point["x"] - ankle["x"], point["y"] - ankle["y"])

    return distance_from_ankle <= max_distance and abs(point["y"] - ankle["y"]) <= max_vertical_delta


def foot_reference_body_height(frame: Frame) -> float:
    box = body_bbox(frame)
    if box["height"] > 0:
        return max(0.22, min(1.05, box["height"]))
    return 0.55


def suppress_keypoint(point: Point) -> None:
    point["score"] = min(point["score"], SUPPRESSED_KEYPOINT_SCORE)


def frame_keypoint(frame: Frame, name: str) -> Point | None:
    return next((item for item in frame["keypoints"] if item["name"] == name), None)


def infer_occluded_arm_side(points_by_name: dict[str, Point], hidden_side: str, visible_side: str) -> None:
    hidden_shoulder = points_by_name.get(f"{hidden_side}_shoulder")
    visible_wrist = points_by_name.get(f"{visible_side}_wrist")
    if hidden_shoulder is None or visible_wrist is None:
        return
    if hidden_shoulder["score"] < RELIABLE_KEYPOINT_SCORE or visible_wrist["score"] < RELIABLE_KEYPOINT_SCORE:
        return

    shoulder_to_grip = math.hypot(
        visible_wrist["x"] - hidden_shoulder["x"],
        visible_wrist["y"] - hidden_shoulder["y"],
    )
    if shoulder_to_grip < 0.04 or shoulder_to_grip > 0.85:
        return

    inferred_score = min(
        INFERRED_KEYPOINT_SCORE_CAP,
        max(DRAWABLE_KEYPOINT_SCORE + 0.03, min(hidden_shoulder["score"], visible_wrist["score"]) * 0.62),
    )

    hidden_wrist = points_by_name.get(f"{hidden_side}_wrist")
    if hidden_wrist is not None and hidden_wrist["score"] < RELIABLE_KEYPOINT_SCORE:
        hidden_wrist["x"] = visible_wrist["x"]
        hidden_wrist["y"] = visible_wrist["y"]
        hidden_wrist["score"] = inferred_score

    hidden_elbow = points_by_name.get(f"{hidden_side}_elbow")
    if hidden_elbow is not None and hidden_elbow["score"] < RELIABLE_KEYPOINT_SCORE:
        hidden_elbow["x"] = hidden_shoulder["x"] + (visible_wrist["x"] - hidden_shoulder["x"]) * 0.58
        hidden_elbow["y"] = hidden_shoulder["y"] + (visible_wrist["y"] - hidden_shoulder["y"]) * 0.58
        hidden_elbow["score"] = inferred_score


def keypoint(frame: Frame, name: str) -> tuple[float, float]:
    point = next((item for item in frame["keypoints"] if item["name"] == name), None)
    if point is None:
        return (0.5, 0.5)
    return (point["x"], point["y"])


def keypoint_score(frame: Frame, name: str) -> float:
    point = next((item for item in frame["keypoints"] if item["name"] == name), None)
    return float(point["score"]) if point is not None else 0.0


def keypoint_series(frames: list[Frame], name: str) -> list[tuple[float, float]]:
    return [keypoint(frame, name) for frame in frames]


def average_series(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(first, second)]


def midpoint(frame: Frame, first_name: str, second_name: str) -> tuple[float, float]:
    first = keypoint(frame, first_name)
    second = keypoint(frame, second_name)
    return ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)


def weighted_midpoint(frame: Frame, first_name: str, second_name: str) -> tuple[float, float]:
    first = keypoint(frame, first_name)
    second = keypoint(frame, second_name)
    first_weight = max(0.05, keypoint_score(frame, first_name))
    second_weight = max(0.05, keypoint_score(frame, second_name))
    total_weight = first_weight + second_weight

    return (
        (first[0] * first_weight + second[0] * second_weight) / total_weight,
        (first[1] * first_weight + second[1] * second_weight) / total_weight,
    )


def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


def angle_between(first: tuple[float, float], second: tuple[float, float]) -> float:
    return abs(math.degrees(math.atan2(second[1] - first[1], second[0] - first[0])))


def spine_angle(frame: Frame) -> float:
    shoulders = midpoint(frame, "left_shoulder", "right_shoulder")
    hips = midpoint(frame, "left_hip", "right_hip")
    return math.degrees(math.atan2(hips[1] - shoulders[1], hips[0] - shoulders[0]))


def clamp_index(index: int, frames: list[Frame]) -> int:
    return max(0, min(len(frames) - 1, index))


def clamp_array_index(index: int, values: np.ndarray) -> int:
    return max(0, min(len(values) - 1, int(index)))


def clamp_float(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
