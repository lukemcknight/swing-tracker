import math

import pytest

from chdet.geometry import Box
from chdet.metrics import (
    center_error_pct, percentile, detection_rate, phantom_rate, path_jitter,
)


def test_center_error_pct_zero_for_identical_centers():
    b = Box(0.5, 0.5, 0.1, 0.1)
    assert center_error_pct(b, b, aspect=16 / 9) == 0.0


def test_center_error_pct_pure_horizontal_offset():
    pred = Box(0.52, 0.5, 0.1, 0.1)
    gt = Box(0.50, 0.5, 0.1, 0.1)
    # dx = 0.02 -> 2% of frame width.
    assert math.isclose(center_error_pct(pred, gt, aspect=16 / 9), 2.0, abs_tol=1e-9)


def test_center_error_pct_rescales_vertical_offset_by_aspect():
    pred = Box(0.5, 0.6, 0.1, 0.1)
    gt = Box(0.5, 0.5, 0.1, 0.1)
    # dy = 0.1 of height; aspect 2.0 -> 0.05 of width -> 5%.
    assert math.isclose(center_error_pct(pred, gt, aspect=2.0), 5.0, abs_tol=1e-9)


def test_percentile_median_and_p90():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 50) == 3.0
    assert math.isclose(percentile(values, 90), 4.6, abs_tol=1e-9)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 150)
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], -1)


def test_percentile_two_element_interpolation():
    assert math.isclose(percentile([1.0, 3.0], 50), 2.0, abs_tol=1e-9)


def test_detection_rate_counts_only_clubhead_present_frames():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    hit = Box(0.5, 0.5, 0.1, 0.1)
    miss = Box(0.9, 0.9, 0.1, 0.1)
    frames = [
        (hit, gt),      # hit
        (miss, gt),     # miss (IoU below threshold)
        (None, gt),     # miss (no prediction)
        (hit, None),    # negative frame — excluded from denominator
    ]
    assert math.isclose(detection_rate(frames, min_iou=0.3), 1 / 3, rel_tol=1e-9)


def test_detection_rate_no_present_frames_raises():
    with pytest.raises(ValueError):
        detection_rate([(None, None)], min_iou=0.3)


def test_phantom_rate_fraction_of_negatives_with_a_detection():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    frames = [
        (gt, None),     # phantom
        (None, None),   # correct (no detection on a negative frame)
        (gt, gt),       # positive frame — excluded
    ]
    assert phantom_rate(frames) == 0.5


def test_phantom_rate_no_negatives_is_zero():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    assert phantom_rate([(gt, gt)]) == 0.0


def test_path_jitter_zero_for_straight_constant_motion():
    # Constant velocity -> second difference is zero -> no jitter.
    centers = [(0.1, 0.5), (0.2, 0.5), (0.3, 0.5), (0.4, 0.5)]
    assert math.isclose(path_jitter(centers, aspect=1.0), 0.0, abs_tol=1e-9)


def test_path_jitter_detects_a_kink():
    # A single back-and-forth kink in an otherwise still path.
    centers = [(0.5, 0.5), (0.6, 0.5), (0.5, 0.5)]
    # second diff dx = 0.5 - 2*0.6 + 0.5 = -0.2 -> 20% of width.
    assert math.isclose(path_jitter(centers, aspect=1.0), 20.0, abs_tol=1e-9)


def test_path_jitter_skips_triples_with_a_missing_frame():
    centers = [(0.5, 0.5), None, (0.5, 0.5), (0.6, 0.5), (0.5, 0.5)]
    # Only the last triple (indices 2,3,4) is fully present.
    assert math.isclose(path_jitter(centers, aspect=1.0), 20.0, abs_tol=1e-9)
