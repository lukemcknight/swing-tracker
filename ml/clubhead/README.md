# chdet — SwingSensei clubhead detector pipeline

Python pipeline for training and evaluating the on-device clubhead detector.
See `docs/superpowers/specs/2026-05-21-clubhead-detector-v2-design.md`.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Test

```bash
source ../.venv/bin/activate && pytest
```

## Phase 0: eval harness

```bash
python -m chdet.evaluate \
  --model "../../swing-tracker-ml.mlproj/Models/swing-tracker-ml 6.mlmodel" \
  --test-set dataset/test \
  --out eval/reports/createml-v6.md
```
