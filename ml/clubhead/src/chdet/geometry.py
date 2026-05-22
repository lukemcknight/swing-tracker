"""Box geometry and coordinate conversions for the clubhead pipeline.

Canonical internal form: `Box` in normalized [0, 1] coordinates, center form
(cx, cy, w, h), origin top-left. Every other format converts to/from this.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    w: float
    h: float

    def corners(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2): top-left and bottom-right corners."""
        return (
            self.cx - self.w / 2,
            self.cy - self.h / 2,
            self.cx + self.w / 2,
            self.cy + self.h / 2,
        )


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a.corners()
    bx1, by1, bx2, by2 = b.corners()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def from_createml(coords: dict, img_w: int, img_h: int) -> Box:
    """Convert a CreateML annotation (pixel center form) to a normalized Box."""
    return Box(
        coords["x"] / img_w,
        coords["y"] / img_h,
        coords["width"] / img_w,
        coords["height"] / img_h,
    )


def from_label_studio(value: dict) -> Box:
    """Convert a Label Studio rectanglelabels value (percent, top-left) to a Box."""
    w = value["width"] / 100.0
    h = value["height"] / 100.0
    return Box(value["x"] / 100.0 + w / 2, value["y"] / 100.0 + h / 2, w, h)


def from_yolo_line(line: str) -> Box:
    """Parse a YOLO label line `class cx cy w h` (normalized) into a Box."""
    _, cx, cy, w, h = line.split()[:5]
    return Box(float(cx), float(cy), float(w), float(h))


def to_yolo_line(box: Box, class_id: int = 0) -> str:
    """Render a Box as a YOLO label line."""
    return f"{class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}"


def unletterbox_box(
    box: Box,
    scale: float,
    pad_x: float,
    pad_y: float,
    target_w: int,
    target_h: int,
    orig_w: int,
    orig_h: int,
) -> Box:
    """Map a Box detected on a letterboxed canvas back to original-image coords.

    `scale`, `pad_x`, `pad_y` are the values returned by `detector.letterbox`.
    """
    cx_px = box.cx * target_w - pad_x
    cy_px = box.cy * target_h - pad_y
    return Box(
        cx_px / scale / orig_w,
        cy_px / scale / orig_h,
        box.w * target_w / scale / orig_w,
        box.h * target_h / scale / orig_h,
    )
