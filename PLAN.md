# Negative Transfer Experiment: Chronos-2 Small

## Context
Investigate whether joint pre-training on heterogeneous domains (transport + energy) causes
negative transfer in Chronos-2 small. Three models are trained from **random initialisation**
for the same number of steps with identical hyperparameters:
1. `transport_only` – trained on transport data only
2. `energy_only` – trained on energy data only
3. `joint` – trained on both domains interleaved

Per-domain evaluation of all three models reveals whether the joint model is worse (negative
transfer), equal, or better than the domain-specialist models.

---

## Architecture: Chronos-2 Small (exact published config)

Source: `autogluon/chronos-2-small` on HuggingFace (28M params, Apache 2.0).

```
d_model=512, d_ff=2048, num_heads=8, num_layers=6, d_kv=64
dropout=0.1, initializer_factor=0.05, RoPE (rope_theta=10000)
input_patch_size=16, input_patch_stride=16, max_output_patches=64
quantiles=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
context_length=2048 (Stage 1 from paper), use_arcsinh=True, use_reg_token=True
```

All three runs share this config and are initialised randomly (no pretrained weights loaded).

---

## Datasets

### Energy (confirmed available in `autogluon/chronos_datasets`)
- `electricity_15min` – 370 smart meter series, 15-min intervals
- `solar_1h` – solar power, hourly
- `wind_farms_hourly` – wind farm output, hourly

### Transport (confirmed available in `autogluon/chronos_datasets`)
- `monash_traffic` – 862 road-traffic occupancy series from SF Bay Area PEMS sensors, hourly
- `taxi_30min` – 2,428 NYC taxi demand series, 30-min intervals
- `uber_tlc_hourly` – 262 Uber TLC pickup-count series by NYC region, hourly

All three configs are loaded with:
```python
from datasets import load_dataset
ds = load_dataset("autogluon/chronos_datasets", "<config_name>", split="train")
# Each row has fields: "start" (pd.Timestamp), "target" (np.ndarray)
```
The HuggingFace `datasets` library caches downloads under `~/.cache/huggingface/datasets`;
no manual registration is required.

---

## Data Pipeline

All datasets are converted to **GluonTS Arrow format** (what `Chronos2Dataset` consumes).

```python
# Reuse the convert_to_arrow pattern from chronos training scripts
from gluonts.dataset.arrow import ArrowWriter

def convert_to_arrow(path, time_series, compression="lz4"):
    dataset = [{"start": start, "target": ts} for ts in time_series]
    ArrowWriter(compression=compression).write_to_file(dataset, path=path)
```

Outputs: `data/transport_train.arrow`, `data/transport_test.arrow`,
         `data/energy_train.arrow`, `data/energy_test.arrow`

### Temporal Holdout — No Leakage Guarantee

All `autogluon/chronos_datasets` configs expose the **full** historical series in a single
`train` split; there are no pre-defined test splits. To prevent any temporal leakage the
train/test split must be done **by time**, never by random index.

For every series of length $T$, define:
- $H$ = `num_rolling_windows × prediction_length` — total held-out tail length
  (default: `num_rolling_windows=5`, `prediction_length=64`, so $H=320$ steps)
- **Training prefix**: timesteps $[0, T-H)$ — written into `*_train.arrow`
- **Test windows**: `num_rolling_windows` consecutive windows of length
  `context_length + prediction_length` taken from timesteps $[T-H-context\_length, T)$
  — written into `*_test.arrow`

This is implemented in `data/split_utils.py` and called by `prepare_data.py`:

```python
def temporal_split(ts: np.ndarray, start: pd.Timestamp, freq: str,
                   context_length: int = 2048, prediction_length: int = 64,
                   num_rolling_windows: int = 5):
    """Return (train_row, test_rows) with no temporal overlap."""
    H = num_rolling_windows * prediction_length
    if len(ts) < context_length + H:
        return None, []   # series too short — skip entirely
    train_ts = ts[:len(ts) - H]
    test_rows = []
    for i in range(num_rolling_windows):
        offset = len(ts) - H - context_length + i * prediction_length
        ctx = ts[offset : offset + context_length]
        label = ts[offset + context_length : offset + context_length + prediction_length]
        test_rows.append({"context": ctx, "label": label, "start": start, "freq": freq})
    train_row = {"start": start, "target": train_ts}
    return train_row, test_rows
```

Key invariant: the last time-step of any training series is strictly earlier than the first
time-step of the corresponding test window. Series shorter than
`context_length + num_rolling_windows * prediction_length` are **dropped** from both splits.

---

## Training Infrastructure

The `chronos` package ships `src/chronos/chronos2/trainer.py`. **Investigate this first** —
use it rather than writing a custom loop. If it is sufficient, call it directly. If it only
supports the full training pipeline (multi-GPU, S3, etc.), fall back to **HuggingFace Trainer**:

```python
from chronos.chronos2 import Chronos2CoreConfig, Chronos2ForecastingConfig, Chronos2Model, Chronos2Dataset

# Initialise from config (random weights)
core_cfg = Chronos2CoreConfig(d_model=512, d_ff=2048, num_heads=8, num_layers=6, ...)
fcst_cfg = Chronos2ForecastingConfig(context_length=2048, ...)
model = Chronos2Model(core_cfg, fcst_cfg)  # random init — no from_pretrained

# Dataset
dataset = Chronos2Dataset(inputs=time_series_dicts, context_length=2048,
                          prediction_length=64, batch_size=32,
                          output_patch_size=16, mode=TRAIN)

# Loss: pinball loss across all 13 quantiles
# Trainer: HuggingFace Trainer or custom loop
```

MLFlow is integrated directly in the training loop via `mlflow.log_metric` at each step.

---

## Hyperparameters (from paper / Chronos training defaults)

| Param | Value |
|---|---|
| Learning rate | 1e-3 |
| LR schedule | Linear warmup then linear decay |
| Warmup ratio | 0.06 |
| Optimizer | AdamW (weight_decay=0.01) |
| Batch size | 32 per GPU (grad accumulation to effective 256 if needed) |
| Training steps | Configurable via CLI arg (default: 50,000; smoke test: 500) |
| Steps per domain (joint) | Same total steps S; the joint dataloader interleaves 50/50 per batch |
| Random seed | 42 (identical for all 3 runs) |

### Constant-Compute Guarantee

All three models perform exactly **S gradient updates** (same batch size, same steps).
Total FLOPs are therefore identical across runs — the only variable is which data fills
the batches:

| Run | Batches drawn from | Domain samples per step |
|---|---|---|
| `transport_only` | transport pool only | 32 transport |
| `energy_only` | energy pool only | 32 energy |
| `joint` | 50 % transport + 50 % energy | ~16 transport + ~16 energy |

Because the joint model sees each domain at half the rate of the corresponding
specialist, any CRPS improvement in the joint model is a genuine benefit from
cross-domain transfer, not from seeing more data. The 50/50 interleaving is
enforced by constructing a single `ConcatDataset` with equal-length epochs
(re-sampling the smaller domain if necessary) so neither domain dominates.

| Param | Value |
|---|---|
| Context length | 2048 (Stage 1 per paper) |
| Prediction length | 64 |

---

## MLFlow Logging

```python
mlflow.set_experiment("negative_transfer_chronos2")
with mlflow.start_run(run_name=run_name):  # "transport_only" | "energy_only" | "joint"
    mlflow.log_params({...model_config, training_steps, seed, datasets})
    # per step:
    mlflow.log_metric("train_loss", loss, step=step)
    # post-training eval:
    mlflow.log_metrics({"transport/crps": ..., "transport/mae": ...,
                        "energy/crps": ..., "energy/mae": ...})
```

---

## File Structure

```
data-for-tsfms/
├── .python-version          # CHANGE: 3.14.2 → 3.12 (3.14 not supported by PyTorch)
├── pyproject.toml           # add dependencies below
├── data/
│   ├── split_utils.py       # temporal_split() — shared train/test split logic
│   └── prepare_data.py      # --domain {energy,transport} → downloads & writes Arrow files
├── train.py                 # training script (model init, Chronos2Dataset, loss, MLFlow)
├── evaluate.py              # load checkpoints, compute CRPS/MAE per domain
└── run_experiment.sh        # sequential: transport_only → energy_only → joint
```

---

## Dependencies (add via `uv add`)

```
chronos-forecasting   # Chronos2Model, Chronos2Dataset, pipeline classes
mlflow
gluonts[arrow]        # ArrowWriter for data conversion
datasets              # HuggingFace datasets (energy domain download)
torch
accelerate
pyarrow
```

Note: Python version must be set to 3.12 in `.python-version` (PyTorch does not support 3.14).

---

## Evaluation Protocol

After each training run, `evaluate.py` runs the trained checkpoint in inference mode on
held-out test series from **both** domains.

Primary metric: **CRPS** (Continuous Ranked Probability Score) — proper scoring rule for
probabilistic quantile forecasts. Also log MAE and RMSE.

Negative transfer signal:
```
CRPS(joint, transport_domain) > CRPS(transport_only, transport_domain)  → negative transfer on transport
CRPS(joint, energy_domain)    > CRPS(energy_only,    energy_domain)     → negative transfer on energy
```

Evaluation uses the pre-split `*_test.arrow` files produced by the data-prep scripts.
Each test record contains a `context` array (length `context_length`) and a `label`
array (length `prediction_length`). The model receives only the context; the label is
used solely to score the forecast. Because the test split is a temporal tail of each
series, **no future information is accessible to the model under evaluation**.

---

## Verification Steps

1. `uv run python -c "from chronos.chronos2 import Chronos2Model; print('ok')"` — confirm install
2. Inspect `chronos2/trainer.py` — reuse if it exposes a simple training API
3. Run `prepare_data.py --domain energy` and `prepare_data.py --domain transport` — confirm all datasets download cleanly
   and that `*_train.arrow` / `*_test.arrow` files are written with non-overlapping time ranges
4. 500-step smoke test on CPU (tiny batch) — verify forward/backward pass and MLFlow logging
5. Run full experiment on GPU server (`run_experiment.sh`)
6. Open MLFlow UI (`mlflow ui`) and compare CRPS across the 3 runs

