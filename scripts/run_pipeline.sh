#!/usr/bin/env bash
# End-to-end pipeline: download (optional) → train → evaluate
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="data/raw/chest_xray"
BACKBONE="${BACKBONE:-resnet50}"
EPOCHS="${EPOCHS:-15}"

if [ ! -d "$DATA_ROOT/train" ]; then
  echo "Dataset not found. Running download..."
  python scripts/download_dataset.py
fi

echo "=== Training ($BACKBONE, $EPOCHS epochs) ==="
python -m src.train --data-root "$DATA_ROOT" --backbone "$BACKBONE" --epochs "$EPOCHS"

echo "=== Evaluation + CAM generation ==="
python -m src.evaluate --checkpoint "models/best_${BACKBONE}.pth" --data-root "$DATA_ROOT"

echo "=== Done ==="
echo "Figures: reports/figures/"
echo "Metrics: reports/figures/test_metrics.json"
