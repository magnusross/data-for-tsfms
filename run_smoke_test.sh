#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-5}"
BATCH_SIZE="${2:-8}"

required_files=(
  "data/transport_train.arrow"
  "data/transport_test.arrow"
  "data/energy_test.arrow"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing required data file: ${file}" >&2
    echo "Run data prep first:" >&2
    echo "  uv run tsfms prepare --config configs/prepare/energy.yaml" >&2
    echo "  uv run tsfms prepare --config configs/prepare/transport.yaml" >&2
    exit 1
  fi
done

uv run python -m compileall src >/dev/null

echo "=== Smoke: training transport_only (${STEPS} steps, batch=${BATCH_SIZE}) ==="
run_id=$(uv run tsfms train \
  --config configs/train/transport_only.yaml \
  --training-steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --save-steps "${STEPS}" \
  --eval-batch-size 64 \
  --eval-max-windows 64 \
  | tee /dev/stderr \
  | awk '/MLflow run_id:/ {print $3}' \
  | tail -n1)

if [[ -z "${run_id}" ]]; then
  echo "Failed to capture MLflow run_id from training output" >&2
  exit 1
fi

echo "=== Smoke: evaluate from MLflow artifact checkpoint (run_id=${run_id}) ==="
uv run tsfms evaluate \
  --config configs/evaluate/transport_only.yaml \
  --mlflow-run-id "${run_id}" \
  --max-windows 64 \
  --plot-samples-per-domain 2 \
  --plot-context-points 64

echo "Smoke test passed. Run ID: ${run_id}"
