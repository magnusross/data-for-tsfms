from __future__ import annotations

import random
import shutil
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
import typer
from typer_config import use_config
from data_for_tsfms.hf_utils import load_target_series_cached
from data_for_tsfms.config_utils import yaml_conf_callback
from transformers import TrainerCallback, TrainingArguments

from chronos.chronos2 import (
    Chronos2CoreConfig,
    Chronos2Dataset,
    Chronos2ForecastingConfig,
    Chronos2Model,
)
from chronos.chronos2.dataset import DatasetMode
from chronos.chronos2.trainer import Chronos2Trainer
from data_for_tsfms.cli.evaluate import evaluate_model_all_domains

QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
RUNS = ("transport_only", "energy_only", "joint")


class MlflowTrainLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        mlflow.log_metric(
            "train_loss", float(logs["loss"]), step=int(state.global_step)
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_hf_train_inputs(
    hf_repo: str,
    dataset_names: list[str],
    prediction_length: int,
    num_rolling_windows: int,
    cache_dir: Path,
) -> list[dict[str, np.ndarray]]:
    heldout = num_rolling_windows * prediction_length
    rows = []
    for config_name in dataset_names:
        for ts in load_target_series_cached(hf_repo, config_name, cache_dir):
            if ts.shape[-1] <= heldout:
                continue
            rows.append({"target": ts[..., :-heldout]})
    return rows


def _balance_two_lists(
    a: list[dict], b: list[dict], seed: int
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    target_len = max(len(a), len(b))

    def _resample(xs: list[dict]) -> list[dict]:
        if len(xs) == target_len:
            return list(xs)
        idx = rng.integers(low=0, high=len(xs), size=target_len)
        return [xs[int(i)] for i in idx]

    return _resample(a), _resample(b)


def _build_inputs(
    run_name: str,
    hf_repo: str,
    transport_datasets: list[str],
    energy_datasets: list[str],
    prediction_length: int,
    num_rolling_windows: int,
    cache_dir: Path,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    transport = _load_hf_train_inputs(
        hf_repo, transport_datasets, prediction_length, num_rolling_windows, cache_dir
    )
    energy = _load_hf_train_inputs(
        hf_repo, energy_datasets, prediction_length, num_rolling_windows, cache_dir
    )

    if run_name == "transport_only":
        return transport
    if run_name == "energy_only":
        return energy

    transport_bal, energy_bal = _balance_two_lists(transport, energy, seed)
    joint: list[dict[str, np.ndarray]] = []
    for t_row, e_row in zip(transport_bal, energy_bal):
        joint.append(t_row)
        joint.append(e_row)
    return joint


def _build_model(
    d_model: int,
    d_ff: int,
    num_heads: int,
    num_layers: int,
    d_kv: int,
    dropout_rate: float,
    initializer_factor: float,
    rope_theta: float,
    context_length: int,
    output_patch_size: int,
    input_patch_size: int,
    input_patch_stride: int,
    max_output_patches: int,
    time_encoding_scale: int,
    use_arcsinh: bool,
    use_reg_token: bool,
    quantiles: list[float],
) -> Chronos2Model:
    fcst_cfg = Chronos2ForecastingConfig(
        context_length=context_length,
        output_patch_size=output_patch_size,
        input_patch_size=input_patch_size,
        input_patch_stride=input_patch_stride,
        max_output_patches=max_output_patches,
        time_encoding_scale=time_encoding_scale,
        quantiles=quantiles,
        use_arcsinh=use_arcsinh,
        use_reg_token=use_reg_token,
    )
    core_cfg = Chronos2CoreConfig(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        d_kv=d_kv,
        dropout_rate=dropout_rate,
        initializer_factor=initializer_factor,
        rope_theta=rope_theta,
        chronos_config=asdict(fcst_cfg),
    )
    return Chronos2Model(core_cfg)


@use_config(yaml_conf_callback)
def main(
    run_name: str = typer.Option(..., "--run-name"),
    hf_repo: str = typer.Option("autogluon/chronos_datasets", "--hf-repo"),
    transport_datasets: list[str] = typer.Option(
        ["monash_traffic", "taxi_30min", "uber_tlc_hourly"], "--transport-datasets"
    ),
    energy_datasets: list[str] = typer.Option(
        ["electricity_15min", "solar_1h", "wind_farms_hourly"], "--energy-datasets"
    ),
    num_rolling_windows: int = typer.Option(5, "--num-rolling-windows"),
    data_cache_dir: Path = typer.Option(Path(".hf_cache"), "--data-cache-dir"),
    local_checkpoint_tmp_root: Path = typer.Option(
        Path(".tmp_checkpoints"), "--local-checkpoint-tmp-root"
    ),
    seed: int = typer.Option(42, "--seed"),
    device: str | None = typer.Option(None, "--device"),
    training_steps: int = typer.Option(50000, "--training-steps"),
    batch_size: int = typer.Option(32, "--batch-size"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate"),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    warmup_ratio: float = typer.Option(0.06, "--warmup-ratio"),
    logging_steps: int = typer.Option(1, "--logging-steps"),
    save_steps: int = typer.Option(1_000_000, "--save-steps"),
    dataloader_workers: int = typer.Option(0, "--dataloader-workers"),
    eval_batch_size: int = typer.Option(128, "--eval-batch-size"),
    eval_max_windows: int | None = typer.Option(
        None, "--eval-max-windows", help="Limit evaluated windows per domain."
    ),
    plot_samples_per_domain: int = typer.Option(10, "--plot-samples-per-domain"),
    plot_context_points: int = typer.Option(128, "--plot-context-points"),
    plot_artifact_dir: str = typer.Option("evaluation_plots", "--plot-artifact-dir"),
    mlflow_experiment: str = typer.Option(
        "negative_transfer_chronos2", "--mlflow-experiment"
    ),
    checkpoint_artifact_dir: str = typer.Option(
        "checkpoints", "--checkpoint-artifact-dir"
    ),
    keep_local_checkpoints: bool = typer.Option(
        False, "--keep-local-checkpoints/--no-keep-local-checkpoints"
    ),
    d_model: int = typer.Option(512, "--d-model"),
    d_ff: int = typer.Option(2048, "--d-ff"),
    num_heads: int = typer.Option(8, "--num-heads"),
    num_layers: int = typer.Option(6, "--num-layers"),
    d_kv: int = typer.Option(64, "--d-kv"),
    dropout_rate: float = typer.Option(0.1, "--dropout-rate"),
    initializer_factor: float = typer.Option(0.05, "--initializer-factor"),
    rope_theta: float = typer.Option(10000.0, "--rope-theta"),
    context_length: int = typer.Option(2048, "--context-length"),
    prediction_length: int = typer.Option(64, "--prediction-length"),
    output_patch_size: int = typer.Option(16, "--output-patch-size"),
    input_patch_size: int = typer.Option(16, "--input-patch-size"),
    input_patch_stride: int = typer.Option(16, "--input-patch-stride"),
    max_output_patches: int = typer.Option(64, "--max-output-patches"),
    time_encoding_scale: int | None = typer.Option(None, "--time-encoding-scale"),
    use_arcsinh: bool = typer.Option(True, "--use-arcsinh/--no-use-arcsinh"),
    use_reg_token: bool = typer.Option(True, "--use-reg-token/--no-use-reg-token"),
    quantiles: list[float] = typer.Option(QUANTILES, "--quantiles"),
) -> None:
    if run_name not in RUNS:
        raise ValueError(f"run_name must be one of {RUNS}")

    _set_seed(seed)
    train_inputs = _build_inputs(
        run_name=run_name,
        hf_repo=hf_repo,
        transport_datasets=list(transport_datasets),
        energy_datasets=list(energy_datasets),
        prediction_length=prediction_length,
        num_rolling_windows=num_rolling_windows,
        cache_dir=data_cache_dir,
        seed=seed,
    )

    train_dataset = Chronos2Dataset(
        inputs=train_inputs,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        output_patch_size=output_patch_size,
        mode=DatasetMode.TRAIN,
    )

    model = _build_model(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        d_kv=d_kv,
        dropout_rate=dropout_rate,
        initializer_factor=initializer_factor,
        rope_theta=rope_theta,
        context_length=context_length,
        output_patch_size=output_patch_size,
        input_patch_size=input_patch_size,
        input_patch_stride=input_patch_stride,
        max_output_patches=max_output_patches,
        time_encoding_scale=time_encoding_scale or context_length,
        use_arcsinh=use_arcsinh,
        use_reg_token=use_reg_token,
        quantiles=[float(q) for q in quantiles],
    )

    output_dir = local_checkpoint_tmp_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=training_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy="steps",
        dataloader_num_workers=dataloader_workers,
        remove_unused_columns=False,
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        seed=seed,
    )

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "run_name": run_name,
                "seed": seed,
                "training_steps": training_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "context_length": context_length,
                "prediction_length": prediction_length,
                "output_patch_size": output_patch_size,
                "datasets": run_name,
            }
        )

        trainer = Chronos2Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            callbacks=[MlflowTrainLossCallback()],
        )

        trainer.train()

        final_dir = output_dir / "final"
        trainer.save_model(str(final_dir))

        mlflow.log_artifacts(str(output_dir), artifact_path=checkpoint_artifact_dir)
        final_checkpoint_uri = (
            f"runs:/{run.info.run_id}/{checkpoint_artifact_dir}/final"
        )
        mlflow.log_param("final_checkpoint_uri", final_checkpoint_uri)

        model.eval()
        eval_device = next(model.parameters()).device
        eval_metrics = evaluate_model_all_domains(
            model=model,
            hf_repo=hf_repo,
            transport_datasets=list(transport_datasets),
            energy_datasets=list(energy_datasets),
            context_length=context_length,
            prediction_length=prediction_length,
            num_rolling_windows=num_rolling_windows,
            cache_dir=data_cache_dir,
            batch_size=eval_batch_size,
            device=eval_device,
            max_windows=eval_max_windows,
            log_plots=True,
            plot_artifact_dir=plot_artifact_dir,
            samples_per_domain=plot_samples_per_domain,
            context_points=plot_context_points,
        )
        mlflow.log_metrics(eval_metrics)

        print(f"Checkpoint artifact URI: {final_checkpoint_uri}")
        print("Evaluation:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.6f}")
        print(f"MLflow run_id: {run.info.run_id}")

    if not keep_local_checkpoints and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
