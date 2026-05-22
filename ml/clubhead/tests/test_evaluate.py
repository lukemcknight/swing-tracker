import math

from chdet.detector import Detection
from chdet.evaluate import (
    FrameResult, compute_report, render_markdown, collect_frames,
)
from chdet.geometry import Box


def _hit(cx=0.5, cy=0.5):
    return Detection(Box(cx, cy, 0.1, 0.1), 0.9)


def test_compute_report_overall_detection_rate():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    results = [
        FrameResult("swingA", 0, _hit(), gt),
        FrameResult("swingA", 1, None, gt),       # miss
        FrameResult("swingA", 2, _hit(), gt),
    ]
    report = compute_report("fake", results, aspect=1.0)
    assert report.total_frames == 3
    assert report.detection_rate == 2 / 3


def test_compute_report_phantom_rate_uses_negative_frames():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    results = [
        FrameResult("swingA", 0, _hit(), gt),     # positive — excluded
        FrameResult("swingA", 1, _hit(), None),   # phantom
        FrameResult("swingA", 2, None, None),     # correct
    ]
    report = compute_report("fake", results, aspect=1.0)
    assert report.phantom_rate == 0.5


def test_compute_report_groups_per_source():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    results = [
        FrameResult("swingA", 0, _hit(), gt),
        FrameResult("swingB", 0, None, gt),
    ]
    report = compute_report("fake", results, aspect=1.0)
    sources = {s.source: s for s in report.per_source}
    assert sources["swingA"].detection_rate == 1.0
    assert sources["swingB"].detection_rate == 0.0


def test_render_markdown_contains_headline_numbers():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    report = compute_report("createml-v6",
                            [FrameResult("swingA", 0, _hit(), gt)], aspect=1.0)
    md = render_markdown(report)
    assert "createml-v6" in md
    assert "Detection rate" in md
    assert "swingA" in md


def test_compute_report_path_jitter_is_frame_weighted():
    gt = Box(0.5, 0.5, 0.1, 0.1)
    # swingA: 3 frames with a kink -> per-source jitter 20.0
    kink = [_hit(0.5, 0.5), _hit(0.6, 0.5), _hit(0.5, 0.5)]
    # swingB: 9 frames on a constant point -> per-source jitter 0.0
    straight = [_hit(0.3, 0.5) for _ in range(9)]
    results = (
        [FrameResult("swingA", i, p, gt) for i, p in enumerate(kink)]
        + [FrameResult("swingB", i, p, gt) for i, p in enumerate(straight)]
    )
    report = compute_report("fake", results, aspect=1.0)
    # frame-weighted: (3*20 + 9*0) / 12 = 5.0  (an unweighted mean would be 10.0)
    assert math.isclose(report.path_jitter, 5.0, abs_tol=1e-9)


def test_collect_frames_missing_images_dir_returns_empty(tmp_path):
    # A test-set root with no images/ subdir yields no frames, not a crash.
    assert collect_frames(tmp_path) == []


def test_collect_frames_pairs_images_with_labels(tmp_path):
    img_dir = tmp_path / "images" / "swingA"
    lbl_dir = tmp_path / "labels" / "swingA"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in (1, 2):
        (img_dir / f"frame-{i:04d}.jpg").write_bytes(b"")
        (lbl_dir / f"frame-{i:04d}.txt").write_text("")
    frames = collect_frames(tmp_path)
    assert [f.source for f in frames] == ["swingA", "swingA"]
    assert [f.index for f in frames] == [0, 1]
