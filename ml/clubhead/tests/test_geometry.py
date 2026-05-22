import math

from chdet.geometry import (
    Box, iou, from_createml, from_label_studio, from_yolo_line, to_yolo_line,
    unletterbox_box,
)


def test_box_corners():
    assert Box(0.5, 0.5, 0.2, 0.4).corners() == (0.4, 0.3, 0.6, 0.7)


def test_iou_identical_boxes_is_one():
    b = Box(0.5, 0.5, 0.2, 0.2)
    assert math.isclose(iou(b, b), 1.0, rel_tol=1e-9)


def test_iou_disjoint_boxes_is_zero():
    assert iou(Box(0.1, 0.1, 0.1, 0.1), Box(0.9, 0.9, 0.1, 0.1)) == 0.0


def test_iou_half_overlap():
    a = Box(0.25, 0.5, 0.5, 0.5)   # x in [0.0, 0.5]
    b = Box(0.50, 0.5, 0.5, 0.5)   # x in [0.25, 0.75]
    # intersection 0.25*0.5=0.125, union 0.25+0.25-0.125=0.375
    assert math.isclose(iou(a, b), 0.125 / 0.375, rel_tol=1e-9)


def test_from_createml_pixel_center_form():
    box = from_createml({"x": 200.0, "y": 100.0, "width": 40.0, "height": 20.0},
                        img_w=400, img_h=200)
    assert box == Box(0.5, 0.5, 0.1, 0.1)


def test_from_label_studio_percent_topleft_form():
    # Label Studio stores top-left x/y and size as percentages.
    box = from_label_studio({"x": 40.0, "y": 30.0, "width": 20.0, "height": 40.0})
    assert math.isclose(box.cx, 0.5, rel_tol=1e-9)
    assert math.isclose(box.cy, 0.5, rel_tol=1e-9)
    assert math.isclose(box.w, 0.2, rel_tol=1e-9)
    assert math.isclose(box.h, 0.4, rel_tol=1e-9)


def test_yolo_line_roundtrip():
    box = Box(0.123456, 0.654321, 0.1, 0.2)
    parsed = from_yolo_line(to_yolo_line(box, class_id=0))
    assert math.isclose(parsed.cx, box.cx, abs_tol=1e-6)
    assert math.isclose(parsed.cy, box.cy, abs_tol=1e-6)


def test_unletterbox_box_inverts_centered_padding():
    # A 400x200 image letterboxed into a 320x320 square:
    # scale = min(320/400, 320/200) = 0.8 -> resized 320x160, pad_y = 80, pad_x = 0.
    # A box at the resized-image center maps back to the original center.
    box_in_canvas = Box(160 / 320, 160 / 320, 80 / 320, 40 / 320)
    box = unletterbox_box(box_in_canvas, scale=0.8, pad_x=0, pad_y=80,
                          target_w=320, target_h=320, orig_w=400, orig_h=200)
    assert math.isclose(box.cx, 0.5, abs_tol=1e-9)
    assert math.isclose(box.cy, 0.5, abs_tol=1e-9)
    assert math.isclose(box.w, 80 / 0.8 / 400, abs_tol=1e-9)
    assert math.isclose(box.h, 40 / 0.8 / 200, abs_tol=1e-9)
