import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swing_analyzer import (
    KEYPOINT_NAMES,
    analysis_sample_cadence,
    apply_video_edit,
    apply_contact_frame_lead,
    build_capture_profile,
    build_feedback,
    build_phase_confidence,
    build_swing_thought,
    cap_score_for_quality,
    detect_club_head,
    detect_backswing_top,
    detect_face_on_top,
    detect_positions,
    finalize_analysis_quality,
    fit_swing_plane,
    gate_feedback,
    hand_center_series,
    refine_impact,
    repair_club_head_points,
    repair_foot_keypoints,
    repair_occluded_arm_keypoints,
    scale_frame_times,
    stabilize_club_head_points,
    timeline_scale_for,
    uses_face_on_top_detection,
)


def make_frame(index: int, hand_x: float, hand_y: float, score: float = 0.95) -> dict:
    keypoints = []
    defaults = {
        "nose": (0.5, 0.24),
        "left_shoulder": (0.43, 0.38),
        "right_shoulder": (0.57, 0.38),
        "left_elbow": (hand_x - 0.04, (hand_y + 0.48) / 2),
        "right_elbow": (hand_x + 0.04, (hand_y + 0.48) / 2),
        "left_wrist": (hand_x - 0.015, hand_y),
        "right_wrist": (hand_x + 0.015, hand_y),
        "left_hip": (0.44, 0.58),
        "right_hip": (0.56, 0.58),
        "left_knee": (0.45, 0.76),
        "right_knee": (0.55, 0.76),
        "left_ankle": (0.45, 0.94),
        "right_ankle": (0.55, 0.94),
        "left_heel": (0.435, 0.95),
        "right_heel": (0.565, 0.95),
        "left_foot_index": (0.415, 0.955),
        "right_foot_index": (0.585, 0.955),
    }
    for name in KEYPOINT_NAMES:
        x, y = defaults.get(name, (0.5, 0.5))
        point_score = score if name in {"left_wrist", "right_wrist"} else 0.95
        keypoints.append({"name": name, "x": x, "y": y, "score": point_score})

    return {"t": index * 100, "keypoints": keypoints}


def set_keypoint(frame: dict, name: str, x: float, y: float, score: float) -> None:
    point = next(item for item in frame["keypoints"] if item["name"] == name)
    point["x"] = x
    point["y"] = y
    point["score"] = score


def interpolate_path(points: list[tuple[int, float, float]], score: float = 0.95) -> list[dict]:
    frames = []
    for start, end in zip(points, points[1:]):
        start_index, start_x, start_y = start
        end_index, end_x, end_y = end
        span = end_index - start_index
        for index in range(start_index, end_index):
            amount = (index - start_index) / max(1, span)
            x = start_x + (end_x - start_x) * amount
            y = start_y + (end_y - start_y) * amount
            frames.append(make_frame(index, x, y, score))

    last_index, last_x, last_y = points[-1]
    frames.append(make_frame(last_index, last_x, last_y, score))
    return frames


def make_capture_frame(
    index: int,
    shoulder_width: float,
    hip_width: float,
    ankle_width: float,
    body_height: float = 0.56,
    visibility: float = 0.95,
    center_x: float = 0.5,
) -> dict:
    frame = make_frame(index, center_x, 0.70, score=visibility)
    top_y = 0.22
    shoulder_y = top_y + body_height * 0.24
    hip_y = top_y + body_height * 0.58
    knee_y = top_y + body_height * 0.78
    ankle_y = top_y + body_height
    wrist_width = max(0.08, shoulder_width * 0.6)
    overrides = {
        "nose": (center_x, top_y, visibility),
        "left_shoulder": (center_x - shoulder_width / 2, shoulder_y, visibility),
        "right_shoulder": (center_x + shoulder_width / 2, shoulder_y, visibility),
        "left_elbow": (center_x - wrist_width / 2, (shoulder_y + hip_y) / 2, visibility),
        "right_elbow": (center_x + wrist_width / 2, (shoulder_y + hip_y) / 2, visibility),
        "left_wrist": (center_x - wrist_width / 2, hip_y + body_height * 0.10, visibility),
        "right_wrist": (center_x + wrist_width / 2, hip_y + body_height * 0.10, visibility),
        "left_hip": (center_x - hip_width / 2, hip_y, visibility),
        "right_hip": (center_x + hip_width / 2, hip_y, visibility),
        "left_knee": (center_x - hip_width / 2, knee_y, visibility),
        "right_knee": (center_x + hip_width / 2, knee_y, visibility),
        "left_ankle": (center_x - ankle_width / 2, ankle_y, visibility),
        "right_ankle": (center_x + ankle_width / 2, ankle_y, visibility),
    }
    for point in frame["keypoints"]:
        if point["name"] in overrides:
            point["x"], point["y"], point["score"] = overrides[point["name"]]

    return frame


def make_capture_sequence(**kwargs) -> list[dict]:
    return [make_capture_frame(index, **kwargs) for index in range(12)]


def quality_profile(status: str = "ok") -> dict:
    confidence = {"ok": 0.9, "warning": 0.68, "poor": 0.42}[status]
    return {
        "status": status,
        "cameraAngle": "down_the_line",
        "captureConfidence": confidence,
        "scoreConfidence": confidence,
        "poseCoverage": confidence,
        "bodyVisibility": confidence,
        "golferScale": 0.5,
        "framing": confidence,
        "warnings": [] if status == "ok" else ["provisional_analysis"],
    }


class PositionDetectionTests(unittest.TestCase):
    def test_video_edit_identity_leaves_frame_unchanged(self):
        frame = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))

        edited = apply_video_edit(frame, rotation_degrees=0, mirror_horizontal=False)

        np.testing.assert_array_equal(edited, frame)

    def test_video_edit_rotation_swaps_dimensions(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)

        rotated_90 = apply_video_edit(frame, rotation_degrees=90)
        rotated_270 = apply_video_edit(frame, rotation_degrees=270)

        self.assertEqual(rotated_90.shape, (3, 2, 3))
        self.assertEqual(rotated_270.shape, (3, 2, 3))

    def test_video_edit_horizontal_mirror_flips_columns(self):
        frame = np.array([[[1, 0, 0], [2, 0, 0], [3, 0, 0]]], dtype=np.uint8)

        edited = apply_video_edit(frame, mirror_horizontal=True)

        self.assertEqual([int(pixel[0]) for pixel in edited[0]], [3, 2, 1])

    def test_video_edit_rotates_then_mirrors_final_frame(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[:, :, 0] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

        edited = apply_video_edit(frame, rotation_degrees=90, mirror_horizontal=True)

        np.testing.assert_array_equal(
            edited[:, :, 0],
            np.array([[1, 4], [2, 5], [3, 6]], dtype=np.uint8),
        )

    def test_detect_club_head_finds_synthetic_shaft_endpoint(self):
        image = np.zeros((640, 360, 3), dtype=np.uint8)
        frame = make_frame(0, 0.50, 0.70)
        cv2.line(image, (180, 448), (264, 558), (255, 255, 255), 4)

        club_head = detect_club_head(image, frame["keypoints"])

        self.assertIsNotNone(club_head)
        self.assertAlmostEqual(club_head["x"], 264 / 360, delta=0.04)
        self.assertAlmostEqual(club_head["y"], 558 / 640, delta=0.04)

    def test_detect_club_head_accepts_visible_shaft_segment_near_hands(self):
        image = np.zeros((640, 360, 3), dtype=np.uint8)
        frame = make_frame(0, 0.50, 0.70)
        cv2.line(image, (205, 480), (264, 558), (255, 255, 255), 4)

        club_head = detect_club_head(image, frame["keypoints"])

        self.assertIsNotNone(club_head)
        self.assertAlmostEqual(club_head["x"], 264 / 360, delta=0.04)
        self.assertAlmostEqual(club_head["y"], 558 / 640, delta=0.04)

    def test_detect_club_head_accepts_thin_dark_range_shaft(self):
        image = np.zeros((2048, 944, 3), dtype=np.uint8)
        image[:, :] = (96, 142, 82)
        image[:940, :] = (205, 170, 112)
        frame = make_frame(0, 0.62, 0.62)
        cv2.line(image, (610, 1320), (805, 1670), (34, 34, 34), 2)

        club_head = detect_club_head(image, frame["keypoints"])

        self.assertIsNotNone(club_head)
        self.assertAlmostEqual(club_head["x"], 805 / 944, delta=0.05)
        self.assertAlmostEqual(club_head["y"], 1670 / 2048, delta=0.05)

    def test_detect_club_head_omits_blank_frame(self):
        image = np.zeros((640, 360, 3), dtype=np.uint8)
        frame = make_frame(0, 0.50, 0.70)

        self.assertIsNone(detect_club_head(image, frame["keypoints"]))

    def test_repair_club_head_points_interpolates_short_gaps_only(self):
        short_gap = [make_frame(index, 0.50, 0.70) for index in range(4)]
        short_gap[0]["clubHead"] = {"x": 0.30, "y": 0.40}
        short_gap[3]["clubHead"] = {"x": 0.60, "y": 0.70}

        repaired_short_gap = repair_club_head_points(short_gap)

        self.assertIn("clubHead", repaired_short_gap[1])
        self.assertIn("clubHead", repaired_short_gap[2])

        long_gap = [make_frame(index, 0.50, 0.70) for index in range(6)]
        long_gap[0]["clubHead"] = {"x": 0.30, "y": 0.40}
        long_gap[5]["clubHead"] = {"x": 0.80, "y": 0.90}

        repaired_long_gap = repair_club_head_points(long_gap)

        self.assertNotIn("clubHead", repaired_long_gap[2])

    def test_stabilize_club_head_points_preserves_smooth_sequence(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(6)]
        for index, frame in enumerate(frames):
            frame["clubHead"] = {"x": 0.62 + index * 0.025, "y": 0.82 - index * 0.015}

        stabilized, uncertain = stabilize_club_head_points(frames)

        self.assertFalse(uncertain)
        self.assertEqual(sum(1 for frame in stabilized if "clubHead" in frame), 6)

    def test_stabilize_club_head_points_removes_single_frame_spike(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(9)]
        path = [
            (0.62, 0.82),
            (0.64, 0.81),
            (0.66, 0.80),
            (0.68, 0.79),
            (0.98, 0.52),
            (0.70, 0.78),
            (0.72, 0.77),
            (0.74, 0.76),
            (0.76, 0.75),
        ]
        for frame, point in zip(frames, path):
            frame["clubHead"] = {"x": point[0], "y": point[1]}

        stabilized, uncertain = stabilize_club_head_points(frames)

        self.assertFalse(uncertain)
        self.assertNotIn("clubHead", stabilized[4])
        self.assertEqual(sum(1 for frame in stabilized if "clubHead" in frame), 8)

    def test_stabilize_club_head_points_removes_fully_unstable_track(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(6)]
        path = [
            (0.62, 0.82),
            (0.98, 0.54),
            (0.20, 0.86),
            (0.95, 0.50),
            (0.22, 0.88),
            (0.90, 0.48),
        ]
        for frame, point in zip(frames, path):
            frame["clubHead"] = {"x": point[0], "y": point[1]}

        stabilized, uncertain = stabilize_club_head_points(frames)

        self.assertTrue(uncertain)
        self.assertEqual(sum(1 for frame in stabilized if "clubHead" in frame), 0)

    def test_stabilize_club_head_points_keeps_short_repaired_gap_inside_stable_run(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(6)]
        frames[0]["clubHead"] = {"x": 0.62, "y": 0.82}
        frames[3]["clubHead"] = {"x": 0.68, "y": 0.79}
        frames[4]["clubHead"] = {"x": 0.70, "y": 0.78}
        frames[5]["clubHead"] = {"x": 0.72, "y": 0.77}

        repaired = repair_club_head_points(frames)
        stabilized, uncertain = stabilize_club_head_points(repaired)

        self.assertFalse(uncertain)
        self.assertEqual(sum(1 for frame in stabilized if "clubHead" in frame), 6)

    def test_stabilize_club_head_points_keeps_three_point_range_track(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(5)]
        for index, point in enumerate([(0.62, 0.82), (0.65, 0.80), (0.69, 0.77)]):
            frames[index]["clubHead"] = {"x": point[0], "y": point[1]}

        stabilized, uncertain = stabilize_club_head_points(
            frames,
            {"address": 0, "takeaway": 1, "top": 2, "impact": 4, "finish": 4},
        )

        self.assertFalse(uncertain)
        self.assertEqual([index for index, frame in enumerate(stabilized) if "clubHead" in frame], [0, 1, 2])

    def test_stabilize_club_head_points_keeps_phase_aligned_primary_run(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(10)]
        false_path = [
            (0.18, 0.64),
            (0.20, 0.64),
            (0.22, 0.65),
            (0.24, 0.65),
        ]
        primary_path = [
            (0.62, 0.82),
            (0.64, 0.81),
            (0.66, 0.80),
            (0.68, 0.79),
            (0.70, 0.78),
        ]
        for index, point in enumerate(false_path):
            frames[index]["clubHead"] = {"x": point[0], "y": point[1]}
        for offset, point in enumerate(primary_path, start=5):
            frames[offset]["clubHead"] = {"x": point[0], "y": point[1]}

        stabilized, uncertain = stabilize_club_head_points(
            frames,
            {"address": 5, "takeaway": 6, "top": 7, "impact": 9, "finish": 9},
        )

        self.assertFalse(uncertain)
        self.assertEqual([index for index, frame in enumerate(stabilized) if "clubHead" in frame], [5, 6, 7, 8, 9])

    def test_stabilize_club_head_points_keeps_full_swing_with_long_followthrough(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(61)]
        for index, frame in enumerate(frames):
            amount = index / max(1, len(frames) - 1)
            frame["clubHead"] = {
                "x": round(0.62 + float(np.sin(amount * np.pi * 0.85)) * 0.20, 4),
                "y": round(0.82 - float(np.sin(amount * np.pi)) * 0.52, 4),
            }

        stabilized, uncertain = stabilize_club_head_points(
            frames,
            {"address": 0, "takeaway": 5, "top": 10, "impact": 14, "finish": 60},
        )

        self.assertFalse(uncertain)
        self.assertEqual([index for index, frame in enumerate(stabilized) if "clubHead" in frame], list(range(61)))

    def test_stabilize_club_head_points_rejects_ambiguous_competing_runs(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(9)]
        path = [
            (0.62, 0.82),
            (0.64, 0.81),
            (0.66, 0.80),
            (0.68, 0.79),
            None,
            (0.28, 0.72),
            (0.30, 0.71),
            (0.32, 0.70),
            (0.34, 0.69),
        ]
        for frame, point in zip(frames, path):
            if point is not None:
                frame["clubHead"] = {"x": point[0], "y": point[1]}

        stabilized, uncertain = stabilize_club_head_points(
            frames,
            {"address": 0, "takeaway": 1, "top": 4, "impact": 8, "finish": 8},
        )

        self.assertTrue(uncertain)
        self.assertEqual(sum(1 for frame in stabilized if "clubHead" in frame), 0)

    def test_stabilize_club_head_points_rejects_run_outside_swing_window(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(11)]
        late_false_path = [
            (0.15, 0.03),
            (0.15, 0.02),
            (0.14, 0.01),
            (0.14, 0.00),
            (0.13, 0.00),
            (0.13, 0.00),
        ]
        for offset, point in enumerate(late_false_path, start=5):
            frames[offset]["clubHead"] = {"x": point[0], "y": point[1]}

        stabilized, uncertain = stabilize_club_head_points(
            frames,
            {"address": 0, "takeaway": 1, "top": 2, "impact": 3, "finish": 4},
        )

        self.assertTrue(uncertain)
        self.assertEqual(sum(1 for frame in stabilized if "clubHead" in frame), 0)

    def test_stabilize_club_head_points_rejects_post_impact_only_run(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(12)]
        post_impact_path = [
            (0.72, 0.82),
            (0.73, 0.75),
            (0.74, 0.68),
            (0.75, 0.61),
            (0.76, 0.54),
        ]
        for offset, point in enumerate(post_impact_path, start=7):
            frames[offset]["clubHead"] = {"x": point[0], "y": point[1]}

        stabilized, uncertain = stabilize_club_head_points(
            frames,
            {"address": 0, "takeaway": 2, "top": 4, "impact": 6, "finish": 11},
        )

        self.assertTrue(uncertain)
        self.assertEqual(sum(1 for frame in stabilized if "clubHead" in frame), 0)

    def test_primary_club_path_repair_only_bridges_selected_run(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(11)]
        for index, point in enumerate([(0.18, 0.64), (0.20, 0.64), (0.22, 0.65), (0.24, 0.65)]):
            frames[index]["clubHead"] = {"x": point[0], "y": point[1]}
        frames[5]["clubHead"] = {"x": 0.62, "y": 0.82}
        frames[8]["clubHead"] = {"x": 0.68, "y": 0.79}
        frames[9]["clubHead"] = {"x": 0.70, "y": 0.78}
        frames[10]["clubHead"] = {"x": 0.72, "y": 0.77}

        stabilized, uncertain = stabilize_club_head_points(
            frames,
            {"address": 5, "takeaway": 6, "top": 8, "impact": 10, "finish": 10},
        )
        repaired = repair_club_head_points(stabilized)

        self.assertFalse(uncertain)
        self.assertEqual([index for index, frame in enumerate(repaired) if "clubHead" in frame], [5, 6, 7, 8, 9, 10])

    def test_scale_frame_times_preserves_club_head(self):
        frames = [make_frame(1, 0.50, 0.70)]
        frames[0]["clubHead"] = {"x": 0.65, "y": 0.80}

        scaled = scale_frame_times(frames, 2.0)

        self.assertEqual(scaled[0]["t"], 200)
        self.assertEqual(scaled[0]["clubHead"], {"x": 0.65, "y": 0.80})

    def test_swing_plane_fit_rejects_face_on_or_insufficient_data(self):
        frames = [make_frame(index, 0.50, 0.70) for index in range(6)]
        for index, frame in enumerate(frames):
            frame["clubHead"] = {"x": 0.30 + index * 0.05, "y": 0.82 - index * 0.04}
        positions = {"takeaway": 1, "impact": 5}

        self.assertIsNone(fit_swing_plane(frames, positions, {**quality_profile("ok"), "cameraAngle": "face_on"}))
        self.assertIsNone(fit_swing_plane(frames[:4], {"takeaway": 0, "impact": 3}, quality_profile("ok")))
        self.assertIsNotNone(fit_swing_plane(frames, positions, quality_profile("ok")))

    def test_capture_profile_classifies_down_the_line(self):
        frames = make_capture_sequence(shoulder_width=0.045, hip_width=0.04, ankle_width=0.05)

        profile = build_capture_profile(frames, 0, len(frames) - 1, 1080, 1920)

        self.assertEqual(profile["cameraAngle"], "down_the_line")
        self.assertEqual(profile["status"], "ok")
        self.assertGreaterEqual(profile["captureConfidence"], 0.78)

    def test_capture_profile_classifies_face_on(self):
        frames = make_capture_sequence(shoulder_width=0.24, hip_width=0.2, ankle_width=0.25)

        profile = build_capture_profile(frames, 0, len(frames) - 1, 1080, 1920)

        self.assertEqual(profile["cameraAngle"], "face_on")
        self.assertEqual(profile["status"], "ok")
        self.assertGreaterEqual(profile["captureConfidence"], 0.78)

    def test_capture_profile_classifies_good_side_profile_range_view(self):
        frames = make_capture_sequence(shoulder_width=0.10, hip_width=0.07, ankle_width=0.16)

        profile = build_capture_profile(frames, 0, len(frames) - 1, 1080, 1920)

        self.assertEqual(profile["cameraAngle"], "side_profile")
        self.assertEqual(profile["status"], "ok")
        self.assertGreaterEqual(profile["captureConfidence"], 0.78)
        self.assertNotIn("unusual_camera_angle", profile["warnings"])

    def test_capture_profile_marks_low_unknown_angle_as_provisional(self):
        frames = make_capture_sequence(
            shoulder_width=0.04,
            hip_width=0.035,
            ankle_width=0.04,
            body_height=0.18,
            visibility=0.45,
        )

        profile = build_capture_profile(frames, 0, len(frames) - 1, 1080, 1920)

        self.assertEqual(profile["cameraAngle"], "unknown")
        self.assertEqual(profile["status"], "poor")
        self.assertIn("golfer_too_small", profile["warnings"])
        self.assertIn("provisional_analysis", profile["warnings"])

    def test_capture_profile_warns_on_partial_body_visibility(self):
        frames = make_capture_sequence(
            shoulder_width=0.2,
            hip_width=0.17,
            ankle_width=0.2,
            visibility=0.42,
        )

        profile = build_capture_profile(frames, 0, len(frames) - 1, 1080, 1920)

        self.assertIn(profile["status"], {"warning", "poor"})
        self.assertIn("low_pose_visibility", profile["warnings"])

    def test_detects_normal_swing_sequence(self):
        frames = interpolate_path(
            [
                (0, 0.50, 0.72),
                (6, 0.58, 0.66),
                (16, 0.68, 0.30),
                (27, 0.50, 0.72),
                (40, 0.36, 0.40),
            ]
        )

        positions = detect_positions(frames)

        self.assertLess(positions["address"], positions["takeaway"])
        self.assertLess(positions["takeaway"], positions["top"])
        self.assertLess(positions["top"], positions["impact"])
        self.assertLess(positions["impact"], positions["finish"])
        self.assertLessEqual(abs(positions["top"] - 16), 3)
        self.assertLessEqual(abs(positions["impact"] - 27), 3)

    def test_repair_occluded_lead_arm_uses_visible_grip(self):
        frame = make_frame(0, 0.52, 0.70)
        set_keypoint(frame, "left_wrist", 0.12, 0.18, 0.28)
        set_keypoint(frame, "left_elbow", 0.10, 0.20, 0.30)

        repaired = repair_occluded_arm_keypoints(frame["keypoints"])
        points = {point["name"]: point for point in repaired}

        self.assertGreater(points["left_wrist"]["score"], 0.2)
        self.assertGreater(points["left_elbow"]["score"], 0.2)
        self.assertAlmostEqual(points["left_wrist"]["x"], points["right_wrist"]["x"], places=4)
        self.assertAlmostEqual(points["left_wrist"]["y"], points["right_wrist"]["y"], places=4)
        self.assertGreater(points["left_elbow"]["x"], points["left_shoulder"]["x"])
        self.assertLess(points["left_elbow"]["y"], points["left_wrist"]["y"])

    def test_repair_foot_keypoints_preserves_reliable_points(self):
        frame = make_frame(0, 0.52, 0.70)
        set_keypoint(frame, "left_heel", 0.432, 0.952, 0.94)
        set_keypoint(frame, "left_foot_index", 0.412, 0.956, 0.93)

        repaired = repair_foot_keypoints([frame])
        points = {point["name"]: point for point in repaired[0]["keypoints"]}

        self.assertAlmostEqual(points["left_ankle"]["x"], 0.45, places=4)
        self.assertAlmostEqual(points["left_ankle"]["y"], 0.94, places=4)
        self.assertAlmostEqual(points["left_heel"]["x"], 0.432, places=4)
        self.assertAlmostEqual(points["left_heel"]["y"], 0.952, places=4)
        self.assertAlmostEqual(points["left_foot_index"]["x"], 0.412, places=4)
        self.assertAlmostEqual(points["left_foot_index"]["y"], 0.956, places=4)
        self.assertGreater(points["left_heel"]["score"], 0.9)
        self.assertGreater(points["left_foot_index"]["score"], 0.9)

    def test_repair_foot_keypoints_interpolates_short_low_confidence_gap(self):
        frames = [make_frame(index, 0.52, 0.70) for index in range(5)]
        set_keypoint(frames[1], "left_foot_index", 0.415, 0.955, 0.94)
        set_keypoint(frames[2], "left_foot_index", 0.83, 0.42, 0.05)
        set_keypoint(frames[3], "left_foot_index", 0.435, 0.965, 0.92)

        repaired = repair_foot_keypoints(frames)
        points = {point["name"]: point for point in repaired[2]["keypoints"]}

        self.assertGreater(points["left_foot_index"]["score"], 0.2)
        self.assertLess(points["left_foot_index"]["score"], 0.43)
        self.assertAlmostEqual(points["left_foot_index"]["x"], 0.425, places=3)
        self.assertAlmostEqual(points["left_foot_index"]["y"], 0.960, places=3)

    def test_repair_foot_keypoints_replaces_vertical_axis_with_stance_axis(self):
        frame = make_frame(0, 0.52, 0.70)
        set_keypoint(frame, "left_heel", 0.445, 0.94, 0.95)
        set_keypoint(frame, "left_foot_index", 0.445, 0.99, 0.95)

        repaired = repair_foot_keypoints([frame])
        points = {point["name"]: point for point in repaired[0]["keypoints"]}
        heel = points["left_heel"]
        toe = points["left_foot_index"]
        dx = toe["x"] - heel["x"]
        dy = toe["y"] - heel["y"]

        self.assertAlmostEqual(points["left_ankle"]["x"], 0.45, places=4)
        self.assertAlmostEqual(points["left_ankle"]["y"], 0.94, places=4)
        self.assertGreater(abs(dx), abs(dy) * 2)
        self.assertGreater(abs(dx), 0.035)
        self.assertLess(heel["score"], 0.43)
        self.assertLess(toe["score"], 0.43)
        self.assertGreater(heel["score"], 0.2)
        self.assertGreater(toe["score"], 0.2)

    def test_repair_foot_keypoints_suppresses_implausible_unrepairable_jump(self):
        frame = make_frame(0, 0.52, 0.70)
        set_keypoint(frame, "right_heel", 0.93, 0.46, 0.95)

        repaired = repair_foot_keypoints([frame])
        points = {point["name"]: point for point in repaired[0]["keypoints"]}

        self.assertLess(points["right_heel"]["score"], 0.2)
        self.assertAlmostEqual(points["right_ankle"]["x"], 0.55, places=4)
        self.assertAlmostEqual(points["right_ankle"]["y"], 0.94, places=4)

    def test_hand_center_ignores_unreliable_hidden_wrist(self):
        frames = interpolate_path(
            [
                (0, 0.50, 0.72),
                (6, 0.58, 0.66),
                (16, 0.68, 0.30),
                (27, 0.50, 0.72),
                (40, 0.36, 0.40),
            ]
        )
        for index, frame in enumerate(frames):
            set_keypoint(frame, "left_wrist", 0.10 + index * 0.01, 0.16, 0.28)
            set_keypoint(frame, "left_elbow", 0.12 + index * 0.01, 0.24, 0.30)

        xs, ys = hand_center_series(frames)
        positions = detect_positions(frames)

        self.assertAlmostEqual(xs[0], 0.515, places=4)
        self.assertAlmostEqual(ys[0], 0.72, places=4)
        self.assertLessEqual(abs(positions["top"] - 16), 3)
        self.assertLessEqual(abs(positions["impact"] - 27), 3)

    def test_follow_through_high_hands_does_not_replace_top(self):
        frames = interpolate_path(
            [
                (0, 0.50, 0.72),
                (7, 0.58, 0.65),
                (16, 0.67, 0.32),
                (27, 0.50, 0.72),
                (38, 0.30, 0.22),
                (48, 0.36, 0.36),
            ]
        )

        positions = detect_positions(frames)

        self.assertLess(positions["top"], positions["impact"])
        self.assertLess(positions["top"], 24)
        self.assertLessEqual(abs(positions["top"] - 16), 4)

    def test_impact_prefers_fast_near_address_return_over_early_crossing(self):
        frames = interpolate_path(
            [
                (0, 0.50, 0.72),
                (8, 0.58, 0.62),
                (14, 0.68, 0.30),
                (20, 0.56, 0.70),
                (22, 0.56, 0.70),
                (25, 0.30, 0.72),
                (39, 0.36, 0.40),
            ]
        )

        positions = detect_positions(frames)

        self.assertGreaterEqual(positions["impact"], 22)
        self.assertLessEqual(abs(positions["impact"] - 25), 3)

    def test_finish_tracks_early_followthrough_checkpoint_after_impact(self):
        frames = interpolate_path(
            [
                (0, 0.50, 0.72),
                (8, 0.60, 0.60),
                (15, 0.68, 0.30),
                (24, 0.50, 0.72),
                (29, 0.50, 0.72),
                (37, 0.42, 0.48),
                (46, 0.36, 0.40),
            ]
        )

        positions = detect_positions(frames)

        self.assertLessEqual(abs((positions["finish"] - positions["impact"]) - 5), 2)
        self.assertLess(positions["impact"], positions["finish"])

    def test_follow_through_return_does_not_move_top_after_speed_burst(self):
        frames = interpolate_path(
            [
                (0, 0.50, 0.72),
                (7, 0.60, 0.58),
                (11, 0.68, 0.30),
                (16, 0.46, 0.74),
                (22, 0.32, 0.24),
                (28, 0.42, 0.72),
                (40, 0.36, 0.40),
            ]
        )

        positions = detect_positions(frames)

        self.assertLess(positions["top"], 18)
        self.assertLessEqual(positions["impact"], 18)
        self.assertLess(positions["impact"], positions["finish"])

    def test_burst_correction_treats_speed_peak_as_top_not_impact(self):
        frames = interpolate_path(
            [
                (0, 0.50, 0.72),
                (7, 0.59, 0.60),
                (10, 0.68, 0.30),
                (13, 0.50, 0.72),
                (17, 0.32, 0.24),
                (23, 0.42, 0.72),
                (40, 0.36, 0.40),
            ]
        )

        positions = detect_positions(frames)

        self.assertLessEqual(abs(positions["top"] - 10), 2)
        self.assertGreaterEqual(positions["impact"], 12)
        self.assertLessEqual(positions["impact"], 15)
        self.assertLess(positions["top"], positions["impact"])

    def test_top_prefers_late_transition_frame_before_impact_on_high_hands_plateau(self):
        height_gain = np.array([0.0, 0.02, 0.06, 0.12, 0.22, 0.30, 0.32, 0.33, 0.31, 0.18, 0.04, -0.01])
        velocities = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.04, 0.035, 0.025, 0.08, 0.10, 0.09, 0.04])

        top = detect_backswing_top(height_gain, velocities, address=0, impact_seed=10, sample_fps=6.0)

        self.assertEqual(top, 7)

    def test_top_allows_fast_transition_close_to_impact(self):
        height_gain = np.array([
            0.0,
            0.01,
            0.03,
            0.06,
            0.10,
            0.15,
            0.20,
            0.24,
            0.27,
            0.29,
            0.31,
            0.32,
            0.33,
            0.33,
            0.32,
            0.34,
            0.06,
            -0.01,
        ])
        velocities = np.array([
            0.01,
            0.01,
            0.02,
            0.03,
            0.04,
            0.04,
            0.04,
            0.035,
            0.03,
            0.03,
            0.028,
            0.026,
            0.024,
            0.025,
            0.04,
            0.018,
            0.11,
            0.08,
        ])

        top = detect_backswing_top(height_gain, velocities, address=0, impact_seed=16, sample_fps=6.0)

        self.assertEqual(top, 15)

    def test_face_on_top_prefers_late_high_hands_transition(self):
        height_gain = np.array([0.0, 0.02, 0.06, 0.12, 0.22, 0.28, 0.31, 0.32, 0.30, 0.18, 0.04, -0.01])
        velocities = np.array([0.01, 0.02, 0.03, 0.05, 0.05, 0.04, 0.03, 0.02, 0.08, 0.10, 0.09, 0.04])

        top = detect_face_on_top(height_gain, velocities, address=0, impact_seed=10, sample_fps=6.0)

        self.assertEqual(top, 7)

    def test_side_profile_uses_face_on_top_detection(self):
        self.assertTrue(uses_face_on_top_detection("side_profile"))
        self.assertTrue(uses_face_on_top_detection("face_on"))
        self.assertFalse(uses_face_on_top_detection("down_the_line"))

    def test_late_burst_corrected_impact_gets_contact_lead(self):
        self.assertEqual(apply_contact_frame_lead(18, 10, 11, 25.0, 0.0), 18)
        self.assertEqual(apply_contact_frame_lead(12, 8, 9, 10.0, 0.15), 12)
        self.assertEqual(apply_contact_frame_lead(18, 10, 11, 6.0, 0.85), 13)
        self.assertEqual(apply_contact_frame_lead(14, 10, 11, 6.0, 0.85), 14)

    def test_refined_impact_does_not_shift_before_selected_contact_frame(self):
        height_gain = np.array([
            0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30,
            0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02,
            0.01, 0.005, -0.002, 0.0, 0.03, 0.08, 0.12, 0.16, 0.18, 0.19,
        ])
        velocities = np.full(len(height_gain), 0.05)
        velocities[31] = 0.4
        velocities[32] = 0.55
        velocities[33] = 0.7
        velocities[34] = 1.0
        velocities[35] = 0.4

        impact = refine_impact(height_gain, velocities, address=0, top=10, impact_seed=31, sample_fps=25.0)

        self.assertEqual(impact, 34)

    def test_analysis_sample_cadence_keeps_normal_video_frame_level(self):
        self.assertEqual(analysis_sample_cadence(25.0), (25.0, 1))
        self.assertEqual(analysis_sample_cadence(30.0), (30.0, 1))
        self.assertEqual(analysis_sample_cadence(60.0), (30.0, 2))
        self.assertEqual(analysis_sample_cadence(240.0), (30.0, 8))

    def test_trimmed_clip_starting_at_address_keeps_timestamps_ordered(self):
        frames = interpolate_path(
            [
                (70, 0.50, 0.72),
                (76, 0.58, 0.66),
                (86, 0.68, 0.30),
                (97, 0.50, 0.72),
                (110, 0.36, 0.40),
            ]
        )

        positions = detect_positions(frames)
        times = [frames[positions[name]]["t"] for name in ["address", "takeaway", "top", "impact", "finish"]]

        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(times[0], 7000)
        self.assertLessEqual(abs(times[2] - 8600), 400)
        self.assertLessEqual(abs(times[3] - 9700), 400)

    def test_low_confidence_flat_motion_falls_back_in_order(self):
        frames = [make_frame(index, 0.5 + index * 0.001, 0.72, score=0.1) for index in range(20)]

        positions = detect_positions(frames)

        self.assertEqual(list(positions), ["address", "takeaway", "top", "impact", "finish"])
        self.assertLess(positions["address"], positions["takeaway"])
        self.assertLess(positions["takeaway"], positions["top"])
        self.assertLess(positions["top"], positions["impact"])
        self.assertLess(positions["impact"], positions["finish"])

    def test_timeline_scale_does_not_compress_to_short_client_duration(self):
        self.assertEqual(timeline_scale_for(2600, 1293), 1.0)

    def test_timeline_scale_stretches_to_longer_client_duration(self):
        self.assertAlmostEqual(timeline_scale_for(1293, 2600), 2.0108, places=4)

    def test_swing_thought_prioritizes_posture_loss(self):
        thought = build_swing_thought(
            {"headDisplacement": 0.02, "shoulderTurn": 20, "hipSway": 0.03, "postureLoss": 12, "tempo": 1.1},
            [],
        )

        self.assertEqual(thought["metricKey"], "postureLoss")
        self.assertEqual(thought["position"], "impact")
        self.assertIn("chest", thought["thought"])

    def test_swing_thought_uses_repeatable_cue_when_all_metrics_pass(self):
        thought = build_swing_thought(
            {"headDisplacement": 0.02, "shoulderTurn": 20, "hipSway": 0.03, "postureLoss": 2, "tempo": 1.1},
            [],
        )

        self.assertEqual(thought["metricKey"], "balanced")
        self.assertEqual(thought["status"], "good")

    def test_low_confidence_top_skips_shoulder_turn_feedback(self):
        metrics = {"headDisplacement": 0.02, "shoulderTurn": 2, "hipSway": 0.03, "postureLoss": 2, "tempo": 1.1}
        feedback = gate_feedback(
            build_feedback(metrics),
            {"address": 0.9, "takeaway": 0.9, "top": 0.42, "impact": 0.9, "finish": 0.9},
        )
        thought = build_swing_thought(
            metrics,
            feedback,
            {"address": 0.9, "takeaway": 0.9, "top": 0.42, "impact": 0.9, "finish": 0.9},
            quality_profile("warning"),
        )

        self.assertNotIn("Shoulder turn", [item["label"] for item in feedback])
        self.assertNotEqual(thought["metricKey"], "shoulderTurn")

    def test_low_confidence_impact_skips_posture_and_hip_feedback(self):
        metrics = {"headDisplacement": 0.02, "shoulderTurn": 20, "hipSway": 0.2, "postureLoss": 14, "tempo": 1.1}
        feedback = gate_feedback(
            build_feedback(metrics),
            {"address": 0.9, "takeaway": 0.9, "top": 0.9, "impact": 0.41, "finish": 0.9},
        )
        thought = build_swing_thought(
            metrics,
            feedback,
            {"address": 0.9, "takeaway": 0.9, "top": 0.9, "impact": 0.41, "finish": 0.9},
            quality_profile("warning"),
        )

        self.assertNotIn("Posture maintained", [item["label"] for item in feedback])
        self.assertNotIn("Hip sway", [item["label"] for item in feedback])
        self.assertNotIn(thought["metricKey"], {"postureLoss", "hipSway"})

    def test_score_caps_when_capture_quality_is_warning_or_poor(self):
        self.assertEqual(cap_score_for_quality(9.6, quality_profile("warning")), 8.0)
        self.assertEqual(cap_score_for_quality(9.6, quality_profile("poor")), 6.0)

    def test_no_safe_coaching_returns_capture_swing_thought(self):
        thought = build_swing_thought(
            {"headDisplacement": 0.2, "shoulderTurn": 2, "hipSway": 0.2, "postureLoss": 14, "tempo": 0.2},
            [],
            {"address": 0.3, "takeaway": 0.3, "top": 0.3, "impact": 0.3, "finish": 0.3},
            quality_profile("poor"),
        )

        self.assertEqual(thought["metricKey"], "capture")
        self.assertIn("Retake", thought["thought"])

    def test_low_confidence_side_profile_retake_copy_names_side_profile(self):
        thought = build_swing_thought(
            {"headDisplacement": 0.2, "shoulderTurn": 2, "hipSway": 0.2, "postureLoss": 14, "tempo": 0.2},
            [],
            {"address": 0.3, "takeaway": 0.3, "top": 0.3, "impact": 0.3, "finish": 0.3},
            {**quality_profile("poor"), "cameraAngle": "side_profile"},
        )

        self.assertEqual(thought["metricKey"], "capture")
        self.assertIn("side-profile", thought["thought"])

    def test_ok_side_profile_capture_with_warning_phase_is_not_provisional(self):
        quality = {**quality_profile("ok"), "cameraAngle": "side_profile"}
        finalized = finalize_analysis_quality(
            quality,
            {"address": 0.9, "takeaway": 0.9, "top": 0.2, "impact": 0.9, "finish": 0.9},
        )

        self.assertEqual(finalized["status"], "ok")
        self.assertIn("low_phase_confidence", finalized["warnings"])
        self.assertNotIn("provisional_analysis", finalized["warnings"])

    def test_ok_capture_with_poor_phase_confidence_stays_provisional(self):
        finalized = finalize_analysis_quality(
            quality_profile("ok"),
            {"address": 0.48, "takeaway": 0.48, "top": 0.42, "impact": 0.48, "finish": 0.48},
        )

        self.assertEqual(finalized["status"], "poor")
        self.assertIn("provisional_analysis", finalized["warnings"])

    def test_phase_confidence_flags_weak_capture(self):
        frames = interpolate_path(
            [
                (0, 0.50, 0.72),
                (6, 0.58, 0.66),
                (16, 0.68, 0.30),
                (27, 0.50, 0.72),
                (40, 0.36, 0.40),
            ],
            score=0.4,
        )
        for frame in frames:
            for point in frame["keypoints"]:
                point["score"] = 0.4
        positions = detect_positions(frames)
        confidence = build_phase_confidence(frames, positions, quality_profile("poor"))

        self.assertLess(confidence["top"], 0.55)
        self.assertLess(confidence["impact"], 0.55)


if __name__ == "__main__":
    unittest.main()
