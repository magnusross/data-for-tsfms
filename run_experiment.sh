#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-50000}"

uv run python data/prepare_data.py --config configs/prepare_energy.yaml
uv run python data/prepare_data.py --config configs/prepare_transport.yaml

echo "=== Training transport_only (${STEPS} steps) ==="
uv run python train.py --config configs/train_transport_only.yaml --training-steps "${STEPS}"
echo "=== Evaluating transport_only ==="
uv run python evaluate.py --config configs/evaluate.yaml --checkpoint checkpoints/transport_only/final

echo "=== Training energy_only (${STEPS} steps) ==="
uv run python train.py --config configs/train_energy_only.yaml --training-steps "${STEPS}"
echo "=== Evaluating energy_only ==="
uv run python evaluate.py --config configs/evaluate.yaml --checkpoint checkpoints/energy_only/final

echo "=== Training joint (${STEPS} steps) ==="
uv run python train.py --config configs/train_joint.yaml --training-steps "${STEPS}"
echo "=== Evaluating joint ==="
uv run python evaluate.py --config configs/evaluate.yaml --checkpoint checkpoints/joint/final

echo "Experiment complete. Start MLflow UI with: uv run mlflow ui"
