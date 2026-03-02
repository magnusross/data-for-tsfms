# Negative Transfer Experiment: Chronos-2 Small

This project runs a controlled negative-transfer experiment with three Chronos-2 small runs from random initialization:

- `transport_only`
- `energy_only`
- `joint` (50/50 interleaving)

## Requirements

- Python `3.12` (pinned in `.python-version`)
- `uv`

Dependencies are managed with `uv` and already recorded in `pyproject.toml`.

CLI configuration is handled with `typer-config` and YAML files in `configs/`.

## Data preparation

Prepare Arrow train/test files with strict temporal holdout and no leakage:

```bash
uv run python data/prepare_data.py --config configs/prepare_energy.yaml
uv run python data/prepare_data.py --config configs/prepare_transport.yaml
```

Outputs:

- `data/energy_train.arrow`, `data/energy_test.arrow`
- `data/transport_train.arrow`, `data/transport_test.arrow`

### Temporal split parameters

- `num_rolling_windows` = number of consecutive evaluation windows extracted from each series tail.
- Held-out tail length is `H = num_rolling_windows * prediction_length`.
- With defaults (`num_rolling_windows=5`, `prediction_length=64`), `H=320` timesteps are held out.
- Increasing `num_rolling_windows` gives more test windows (more stable metrics) but reduces train-prefix length.

## Train one run

```bash
uv run python train.py --config configs/train_transport_only.yaml --training-steps 500
```

Run names: `transport_only`, `energy_only`, `joint`.

Training logs:

- Per-step: `train_loss`
- Post-training: `transport/crps`, `transport/mae`, `transport/rmse`, `energy/crps`, `energy/mae`, `energy/rmse`

Experiment name defaults to `negative_transfer_chronos2`.

## Evaluate checkpoint

```bash
uv run python evaluate.py --config configs/evaluate.yaml --checkpoint checkpoints/transport_only/final
```

## Full experiment script

```bash
chmod +x run_experiment.sh
./run_experiment.sh 50000
```

The script runs:

1. data prep for both domains
2. `transport_only` training + evaluation
3. `energy_only` training + evaluation
4. `joint` training + evaluation

## Quick verification

```bash
uv run python -c "from chronos.chronos2 import Chronos2Model; print('ok')"
uv run python train.py --config configs/train_transport_only.yaml --training-steps 500 --batch-size 8
uv run mlflow ui
```

## Smoke test (config + override)

Run a minimal training smoke test that proves YAML config loading works and that CLI args override YAML values:

```bash
uv run python train.py --config configs/train_transport_only.yaml --training-steps 50 --batch-size 8
```

In this command, `--training-steps 50` and `--batch-size 8` override the values in `configs/train_transport_only.yaml`.
