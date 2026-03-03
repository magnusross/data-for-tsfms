#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-50000}"

echo "=== Training transport_only (${STEPS} steps) ==="
uv run tsfms train --config configs/train/transport_only.yaml --training-steps "${STEPS}"

echo "=== Training energy_only (${STEPS} steps) ==="
uv run tsfms train --config configs/train/energy_only.yaml --training-steps "${STEPS}"

echo "=== Training joint (${STEPS} steps) ==="
uv run tsfms train --config configs/train/joint.yaml --training-steps "${STEPS}"

echo "Experiment complete. Start MLflow UI with: uv run mlflow ui"
