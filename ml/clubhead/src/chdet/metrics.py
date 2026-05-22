"""Evaluation metrics for clubhead detection.

A "frame" is a (prediction, ground_truth) pair; each side is `Box | None`.
`aspect` is always frame_width / frame_height; vertical offsets (fractions of
height) are divided by it so every distance is expressed in fractions of width.
"""
from __future__ import annotations

import math

from .geometry import Box, iou


def center_error_pct(pred: Box, gt: Box, aspect: float) -> float:
    """Euclidean center distance as a percentage of frame width."""
    dx = pred.cx - gt.cx
    dy = (pred.cy - gt.cy) / aspect
    return 100.0 * math.hypot(dx, dy)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile; `p` in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return s[lo] * (1 - (rank - lo)) + s[hi] * (rank - lo)


def detection_rate(
    frames: list[tuple[Box | None, Box | None]], min_iou: float = 0.3
) -> float:
    """Fraction of clubhead-present frames where the model found the clubhead.

    Only frames with a ground-truth box count toward the denominator. A frame
    is a hit when a prediction exists and overlaps the ground truth at or above
    `min_iou`.
    """
    present = [(p, g) for p, g in frames if g is not None]
    if not present:
        raise ValueError("no clubhead-present frames")
    hits = sum(1 for p, g in present if p is not None and iou(p, g) >= min_iou)
    return hits / len(present)


def phantom_rate(frames: list[tuple[Box | None, Box | None]]) -> float:
    """Fraction of no-clubhead frames where the model produced a detection."""
    negatives = [p for p, g in frames if g is None]
    if not negatives:
        return 0.0
    return sum(1 for p in negatives if p is not None) / len(negatives)


def path_jitter(centers: list[tuple[float, float] | None], aspect: float) -> float:
    """Mean second-difference magnitude of the center path, in % of frame width.

    A smooth arc has a small second difference; frame-to-frame jitter inflates
    it. Only triples of three consecutive present frames contribute.
    """
    diffs: list[float] = []
    for i in range(1, len(centers) - 1):
        a, b, c = centers[i - 1], centers[i], centers[i + 1]
        if a is None or b is None or c is None:
            continue
        ddx = a[0] - 2 * b[0] + c[0]
        ddy = (a[1] - 2 * b[1] + c[1]) / aspect
        diffs.append(100.0 * math.hypot(ddx, ddy))
    return sum(diffs) / len(diffs) if diffs else 0.0
