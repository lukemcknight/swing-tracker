"""Detector interface and the Core ML adapter for the current CreateML model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .geometry import Box, unletterbox_box


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float


class Detector(Protocol):
    """Anything that turns an image into at most one clubhead Detection."""

    def detect(self, image: Image.Image) -> Detection | None:
        ...


def decode_createml_outputs(
    coordinates: np.ndarray, confidence: np.ndarray, threshold: float
) -> Detection | None:
    """Decode CreateML object-detector outputs into the single best Detection.

    `coordinates` is (N, 4) normalized [cx, cy, w, h]; `confidence` is (N, C).
    Returns the highest-confidence detection at or above `threshold`, or None.
    """
    best: Detection | None = None
    for i in range(coordinates.shape[0]):
        conf = float(confidence[i].max())
        if conf < threshold:
            continue
        if best is None or conf > best.confidence:
            cx, cy, w, h = (float(v) for v in coordinates[i])
            best = Detection(Box(cx, cy, w, h), conf)
    return best


def letterbox(
    image: Image.Image, target_w: int, target_h: int
) -> tuple[Image.Image, float, int, int]:
    """Resize `image` into a target canvas, preserving aspect ratio with padding.

    Returns (padded_image, scale, pad_x, pad_y). This matches the aspect-correct
    scaling Vision uses with `.scaleFit`, so harness measurements are faithful.
    """
    iw, ih = image.size
    scale = min(target_w / iw, target_h / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    resized = image.resize((nw, nh))
    padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    pad_x, pad_y = (target_w - nw) // 2, (target_h - nh) // 2
    padded.paste(resized, (pad_x, pad_y))
    return padded, scale, pad_x, pad_y


class CoreMLClubheadDetector:
    """Runs a CreateML object-detector `.mlmodel` over images via coremltools."""

    def __init__(self, model_path: Path, threshold: float = 0.25) -> None:
        import coremltools as ct

        self.threshold = threshold
        self.model = ct.models.MLModel(str(model_path))
        image_input = self.model.get_spec().description.input[0]
        if not image_input.type.HasField("imageType"):
            raise ValueError(
                "Expected an image input, got "
                f"{image_input.type.WhichOneof('Type')}"
            )
        self.input_name = image_input.name
        self.input_w = image_input.type.imageType.width
        self.input_h = image_input.type.imageType.height

    def detect(self, image: Image.Image) -> Detection | None:
        padded, scale, pad_x, pad_y = letterbox(
            image.convert("RGB"), self.input_w, self.input_h
        )
        out = self.model.predict({self.input_name: padded})
        det = decode_createml_outputs(
            np.asarray(out["coordinates"]),
            np.asarray(out["confidence"]),
            self.threshold,
        )
        if det is None:
            return None
        orig_w, orig_h = image.size
        return Detection(
            unletterbox_box(det.box, scale, pad_x, pad_y,
                            self.input_w, self.input_h, orig_w, orig_h),
            det.confidence,
        )
