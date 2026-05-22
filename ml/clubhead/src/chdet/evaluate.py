"""Eval harness: score a clubhead detector on a held-out test set.

Test-set layout:
    <root>/images/<source>/<frame>.jpg
    <root>/labels/<source>/<frame>.txt   (empty file = negative frame)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from . import metrics
from .detector import CoreMLClubheadDetector, Detection, Detector
from .geometry import Box
from .labels import load_yolo_label


@dataclass
class FrameSpec:
    source: str
    index: int
    image_path: Path
    label_path: Path


@dataclass
class FrameResult:
    source: str
    index: int
    prediction: Detection | None
    ground_truth: Box | None


@dataclass
class SourceMetrics:
    source: str
    frames: int
    detection_rate: float
    median_center_error: float
    p90_center_error: float
    path_jitter: float


@dataclass
class Report:
    model_name: str
    total_frames: int
    detection_rate: float
    median_center_error: float
    p90_center_error: float
    phantom_rate: float
    path_jitter: float
    per_source: list[SourceMetrics]


def collect_frames(root: Path) -> list[FrameSpec]:
    """Discover every image in the test set, paired with its label file."""
    specs: list[FrameSpec] = []
    images_root = root / "images"
    if not images_root.is_dir():
        return specs
    for source_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        images = sorted(source_dir.glob("*.jpg"))
        for index, image_path in enumerate(images):
            label_path = root / "labels" / source_dir.name / f"{image_path.stem}.txt"
            specs.append(FrameSpec(source_dir.name, index, image_path, label_path))
    return specs


def run_detector(detector: Detector, specs: list[FrameSpec]) -> list[FrameResult]:
    """Run the detector over every frame and pair it with ground truth."""
    results: list[FrameResult] = []
    for spec in specs:
        with Image.open(spec.image_path) as img:
            prediction = detector.detect(img)
        ground_truth = (
            load_yolo_label(spec.label_path) if spec.label_path.exists() else None
        )
        results.append(
            FrameResult(spec.source, spec.index, prediction, ground_truth)
        )
    return results


def compute_report(
    model_name: str, results: list[FrameResult], aspect: float
) -> Report:
    """Aggregate per-frame results into per-source and overall metrics."""
    by_source: dict[str, list[FrameResult]] = {}
    for r in results:
        by_source.setdefault(r.source, []).append(r)

    per_source: list[SourceMetrics] = []
    all_pairs: list[tuple[Box | None, Box | None]] = []
    all_errors: list[float] = []

    for source, frs in sorted(by_source.items()):
        frs = sorted(frs, key=lambda r: r.index)
        pairs = [
            (r.prediction.box if r.prediction else None, r.ground_truth)
            for r in frs
        ]
        all_pairs.extend(pairs)
        errors = [
            metrics.center_error_pct(p, g, aspect)
            for p, g in pairs
            if p is not None and g is not None
        ]
        all_errors.extend(errors)
        centers = [
            (r.prediction.box.cx, r.prediction.box.cy) if r.prediction else None
            for r in frs
        ]
        has_present = any(g is not None for _, g in pairs)
        per_source.append(
            SourceMetrics(
                source=source,
                frames=len(frs),
                detection_rate=metrics.detection_rate(pairs) if has_present else 0.0,
                median_center_error=metrics.percentile(errors, 50) if errors else 0.0,
                p90_center_error=metrics.percentile(errors, 90) if errors else 0.0,
                path_jitter=metrics.path_jitter(centers, aspect),
            )
        )

    has_present = any(g is not None for _, g in all_pairs)
    return Report(
        model_name=model_name,
        total_frames=len(all_pairs),
        detection_rate=metrics.detection_rate(all_pairs) if has_present else 0.0,
        median_center_error=metrics.percentile(all_errors, 50) if all_errors else 0.0,
        p90_center_error=metrics.percentile(all_errors, 90) if all_errors else 0.0,
        phantom_rate=metrics.phantom_rate(all_pairs),
        # Frame-weighted so an even mix of small and large sources is not
        # biased toward whichever clips happen to be short.
        path_jitter=(
            sum(s.frames * s.path_jitter for s in per_source)
            / sum(s.frames for s in per_source)
            if per_source else 0.0
        ),
        per_source=per_source,
    )


def render_markdown(report: Report) -> str:
    """Render a Report as a human-readable Markdown document."""
    lines = [
        f"# Clubhead detector eval — {report.model_name}",
        "",
        f"- Total frames: {report.total_frames}",
        f"- **Detection rate:** {report.detection_rate:.1%}",
        f"- **Median center error:** {report.median_center_error:.2f}% of frame width",
        f"- **p90 center error:** {report.p90_center_error:.2f}% of frame width",
        f"- **Phantom rate:** {report.phantom_rate:.1%}",
        f"- **Mean path jitter:** {report.path_jitter:.3f}% of frame width",
        "",
        "## Per source",
        "",
        "| Source | Frames | Detection rate | Median err % | p90 err % | Jitter % |",
        "|---|---|---|---|---|---|",
    ]
    for s in report.per_source:
        lines.append(
            f"| {s.source} | {s.frames} | {s.detection_rate:.1%} "
            f"| {s.median_center_error:.2f} | {s.p90_center_error:.2f} "
            f"| {s.path_jitter:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a clubhead detector.")
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to a Core ML .mlmodel / .mlpackage")
    parser.add_argument("--test-set", required=True, type=Path,
                        help="Test-set root (contains images/ and labels/)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Markdown report output path (.json written alongside)")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args(argv)

    detector = CoreMLClubheadDetector(args.model, threshold=args.threshold)
    specs = collect_frames(args.test_set)
    if not specs:
        raise SystemExit(f"no frames found under {args.test_set}/images")

    # Assumes every test frame shares one aspect ratio (read from the first).
    with Image.open(specs[0].image_path) as first:
        aspect = first.width / first.height

    results = run_detector(detector, specs)
    report = compute_report(args.model.name, results, aspect)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report))
    args.out.with_suffix(".json").write_text(json.dumps(asdict(report), indent=2))
    print(f"wrote {args.out}")
    print(f"detection rate {report.detection_rate:.1%}, "
          f"median center error {report.median_center_error:.2f}%")


if __name__ == "__main__":
    main()
