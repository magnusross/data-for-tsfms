# Negative Transfer Experiment: Chronos-2 Small

This project runs a controlled negative-transfer experiment with three Chronos-2 small runs from random initialization:

- `transport_only`
- `energy_only`
- `joint` (50/50 interleaving)

## Requirements

- Python `3.12` (pinned in `.python-version`)
- `uv`

Dependencies are managed with `uv` and already recorded in `pyproject.toml`.

CLI configuration is handled with `typer-config` and YAML files in:

- `configs/prepare/`
- `configs/train/`
- `configs/evaluate/`
- `configs/evaluate_fev/`

Project layout:

- Source code: `src/data_for_tsfms/`
- Data artifacts only: `data/*.arrow`

Install package and expose CLI endpoints:

```bash
uv sync
```

Available commands:

- `uv run tsfms prepare`
- `uv run tsfms train`
- `uv run tsfms evaluate`
- `uv run tsfms evaluate-fev`

## Data preparation

Prepare Arrow train/test files with strict temporal holdout and no leakage:

```bash
uv run tsfms prepare --config configs/prepare/energy.yaml
uv run tsfms prepare --config configs/prepare/transport.yaml
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
uv run tsfms train --config configs/train/transport_only.yaml --training-steps 500
```

Run names: `transport_only`, `energy_only`, `joint`.

Training logs:

- Per-step: `train_loss`
- Post-training: `transport/crps`, `transport/mae`, `transport/rmse`, `energy/crps`, `energy/mae`, `energy/rmse`

Experiment name defaults to `negative_transfer_chronos2`.

## Evaluate checkpoint from MLflow artifacts

```bash
uv run tsfms evaluate --config configs/evaluate/transport_only.yaml --mlflow-run-id <RUN_ID>
```

`tsfms train` logs all checkpoints as MLflow artifacts under `checkpoints/`, and logs the final checkpoint URI as run param `final_checkpoint_uri`.

## Evaluate on FEV benchmark tasks

Run Chronos-2 evaluation on FEV tasks with task definitions in `configs/evaluate_fev/`:

```bash
uv run tsfms evaluate-fev --config configs/evaluate_fev/default.yaml --mlflow-run-id <RUN_ID>
```

Task presets included:

- `configs/evaluate_fev/transport.yaml` (`LOOP_SEATTLE_1H`)
- `configs/evaluate_fev/energy.yaml` (`entsoe_1H`)
- `configs/evaluate_fev/joint.yaml` (both tasks)

To switch benchmark tasks while keeping other defaults:

```bash
uv run tsfms evaluate-fev --config configs/evaluate_fev/default.yaml --mlflow-run-id <RUN_ID> --task-config configs/evaluate_fev/transport.yaml
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
uv run tsfms train --config configs/train/transport_only.yaml --training-steps 500 --batch-size 8
uv run mlflow ui
```

## Smoke test (config + override)

Run a dedicated smoke script that verifies:

- config loading
- minimal training
- checkpoint logging to MLflow artifacts
- evaluation loading from artifact checkpoint
- evaluation plot artifact logging

```bash
chmod +x run_smoke_test.sh
./run_smoke_test.sh 5 8
```

Arguments are optional:

- first arg = training steps (default `5`)
- second arg = batch size (default `8`)

The script expects prepared Arrow files in `data/`. If missing, it prints the prep commands.

You can still run a direct override check with:

```bash
uv run tsfms train --config configs/train/transport_only.yaml --training-steps 50 --batch-size 8
```

In this command, `--training-steps 50` and `--batch-size 8` override the values in `configs/train/transport_only.yaml`.
