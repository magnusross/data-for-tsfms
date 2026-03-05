#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-50000}"

echo "=== Training cloudops_only (${STEPS} steps) ==="
uv run tsfms train --config configs/train/cloudops_only.yaml --training-steps "${STEPS}"

echo "=== Training energy_only (${STEPS} steps) ==="
uv run tsfms train --config configs/train/energy_only.yaml --training-steps "${STEPS}"

echo "=== Training cloudops_vs_energy (${STEPS} steps) ==="
uv run tsfms train --config configs/train/cloudops_vs_energy.yaml --training-steps "${STEPS}"

echo "Experiment complete. Start MLflow UI with: uv run mlflow ui"
