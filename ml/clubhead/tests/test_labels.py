import json
import math

from chdet.geometry import Box
from chdet.labels import load_yolo_label, load_label_studio_export


def test_load_yolo_label_reads_a_box(tmp_path):
    f = tmp_path / "frame-0001.txt"
    f.write_text("0 0.5 0.5 0.1 0.2\n")
    box = load_yolo_label(f)
    assert box == Box(0.5, 0.5, 0.1, 0.2)


def test_load_yolo_label_empty_file_is_negative_frame(tmp_path):
    f = tmp_path / "frame-0002.txt"
    f.write_text("")
    assert load_yolo_label(f) is None


def test_load_label_studio_export_parses_annotated_task(tmp_path):
    export = [
        {
            "data": {"image": "/data/local-files/?d=images/frame-0001.jpg"},
            "annotations": [
                {"result": [
                    {"value": {"x": 40.0, "y": 30.0, "width": 20.0,
                               "height": 40.0, "rectanglelabels": ["clubhead"]}}
                ]}
            ],
        }
    ]
    f = tmp_path / "export.json"
    f.write_text(json.dumps(export))
    result = load_label_studio_export(f)
    box = result["frame-0001.jpg"]
    assert box is not None
    assert math.isclose(box.cx, 0.5, abs_tol=1e-9)
    assert math.isclose(box.cy, 0.5, abs_tol=1e-9)


def test_load_label_studio_export_task_with_no_annotation_is_negative(tmp_path):
    export = [
        {"data": {"image": "/data/local-files/?d=images/frame-0009.jpg"},
         "annotations": [{"result": []}]}
    ]
    f = tmp_path / "export.json"
    f.write_text(json.dumps(export))
    result = load_label_studio_export(f)
    assert result == {"frame-0009.jpg": None}
