#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-50000}"

uv run tsfms prepare --config configs/prepare/energy.yaml
uv run tsfms prepare --config configs/prepare/transport.yaml

run_and_capture_id() {
	local train_config="$1"
	local run_id
	run_id=$(uv run tsfms train --config "$train_config" --training-steps "${STEPS}" | tee /dev/stderr | awk '/MLflow run_id:/ {print $3}' | tail -n1)
	if [[ -z "${run_id}" ]]; then
		echo "Failed to capture MLflow run_id from training output" >&2
		exit 1
	fi
	echo "${run_id}"
}

echo "=== Training transport_only (${STEPS} steps) ==="
transport_run_id="$(run_and_capture_id configs/train/transport_only.yaml)"
echo "=== Evaluating transport_only ==="
uv run tsfms evaluate --config configs/evaluate/default.yaml --mlflow-run-id "${transport_run_id}"

echo "=== Training energy_only (${STEPS} steps) ==="
energy_run_id="$(run_and_capture_id configs/train/energy_only.yaml)"
echo "=== Evaluating energy_only ==="
uv run tsfms evaluate --config configs/evaluate/default.yaml --mlflow-run-id "${energy_run_id}"

echo "=== Training joint (${STEPS} steps) ==="
joint_run_id="$(run_and_capture_id configs/train/joint.yaml)"
echo "=== Evaluating joint ==="
uv run tsfms evaluate --config configs/evaluate/default.yaml --mlflow-run-id "${joint_run_id}"

echo "Experiment complete. Start MLflow UI with: uv run mlflow ui"
