import numpy as np
from PIL import Image

from chdet.detector import Detection, decode_createml_outputs, letterbox
from chdet.geometry import Box


def test_decode_picks_highest_confidence_above_threshold():
    coords = np.array([[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.05, 0.05]])
    confidence = np.array([[0.40], [0.90]])
    det = decode_createml_outputs(coords, confidence, threshold=0.25)
    assert det == Detection(Box(0.2, 0.2, 0.05, 0.05), 0.90)


def test_decode_returns_none_when_all_below_threshold():
    coords = np.array([[0.5, 0.5, 0.1, 0.1]])
    confidence = np.array([[0.10]])
    assert decode_createml_outputs(coords, confidence, threshold=0.25) is None


def test_decode_returns_none_for_no_rows():
    assert decode_createml_outputs(
        np.zeros((0, 4)), np.zeros((0, 1)), threshold=0.25
    ) is None


def test_letterbox_preserves_aspect_and_centers():
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    padded, scale, pad_x, pad_y = letterbox(img, 320, 320)
    assert padded.size == (320, 320)
    assert scale == 0.8          # min(320/400, 320/200)
    assert pad_x == 0
    assert pad_y == 80           # (320 - 160) / 2
