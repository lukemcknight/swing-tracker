import json

from chdet.dataset import import_label_studio_export


def _write_jpg(path):
    # Minimal valid file content; the importer copies bytes, it does not decode.
    path.write_bytes(b"\xff\xd8\xff\xd9")


def test_import_writes_images_and_labels_under_source(tmp_path):
    src_images = tmp_path / "src"
    src_images.mkdir()
    _write_jpg(src_images / "frame-0001.jpg")
    _write_jpg(src_images / "frame-0002.jpg")

    export = [
        {"data": {"image": "/data/local-files/?d=images/frame-0001.jpg"},
         "annotations": [{"result": [
             {"value": {"x": 40.0, "y": 30.0, "width": 20.0, "height": 40.0,
                        "rectanglelabels": ["clubhead"]}}]}]},
        {"data": {"image": "/data/local-files/?d=images/frame-0002.jpg"},
         "annotations": [{"result": []}]},
    ]
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export))

    dest = tmp_path / "dataset" / "test"
    count = import_label_studio_export(export_path, src_images, "swingX", dest)

    assert count == 2
    assert (dest / "images" / "swingX" / "frame-0001.jpg").exists()
    label = (dest / "labels" / "swingX" / "frame-0001.txt").read_text().strip()
    assert label.startswith("0 0.5")            # annotated frame
    negative = (dest / "labels" / "swingX" / "frame-0002.txt").read_text()
    assert negative == ""                       # negative frame -> empty label


def test_import_skips_export_entry_with_missing_image(tmp_path):
    src_images = tmp_path / "src"
    src_images.mkdir()
    export = [
        {"data": {"image": "/data/local-files/?d=images/missing.jpg"},
         "annotations": [{"result": []}]},
    ]
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export))
    dest = tmp_path / "dataset" / "test"
    count = import_label_studio_export(export_path, src_images, "swingX", dest)
    assert count == 0
