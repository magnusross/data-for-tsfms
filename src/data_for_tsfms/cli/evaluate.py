from __future__ import annotations

import math
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import typer
from datasets import load_dataset
from typer_config import use_config
from data_for_tsfms.cli.config_utils import yaml_conf_callback

from chronos.chronos2 import Chronos2Model

DOMAINS = ("transport", "energy")
_TARGET_KEYS = ("target", "consumption_kW", "power_mw")


def _device_from_arg(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_target(row: dict) -> np.ndarray:
    for key in _TARGET_KEYS:
        if key in row:
            return np.asarray(row[key], dtype=np.float32)
    raise KeyError(
        f"Could not find target field in row keys={sorted(row.keys())}. "
        f"Expected one of {_TARGET_KEYS}."
    )


def _compute_metrics(
    quantile_preds: np.ndarray, labels: np.ndarray, quantiles: np.ndarray
) -> dict[str, float]:
    labels_expanded = labels[:, None, :]
    pinball2 = 2.0 * np.abs(
        (labels_expanded - quantile_preds)
        * ((labels_expanded <= quantile_preds) - quantiles[None, :, None])
    )
    crps = float(pinball2.mean())

    median_idx = int(np.argmin(np.abs(quantiles - 0.5)))
    point_forecast = quantile_preds[:, median_idx, :]
    errors = point_forecast - labels

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    return {"crps": crps, "mae": mae, "rmse": rmse}


def _resolve_checkpoint(
    checkpoint: Path | None,
    mlflow_run_id: str | None,
    checkpoint_artifact_path: str,
) -> Path:
    if checkpoint is not None:
        return checkpoint
    if mlflow_run_id is None:
        raise ValueError(
            "Either --checkpoint or --mlflow-run-id must be provided for evaluation."
        )
    uri = f"runs:/{mlflow_run_id}/{checkpoint_artifact_path}"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
    return Path(local_path)


def _predict_on_domain(
    model: Chronos2Model,
    hf_repo: str,
    dataset_names: list[str],
    context_length: int,
    prediction_length: int,
    num_rolling_windows: int,
    batch_size: int,
    device: torch.device,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    heldout = num_rolling_windows * prediction_length
    all_contexts: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for config_name in dataset_names:
        split = load_dataset(hf_repo, config_name, split="train")
        for row in split:
            ts = _get_target(row)
            if len(ts) < context_length + heldout:
                continue
            for window_idx in range(num_rolling_windows):
                offset = (
                    len(ts) - heldout - context_length + window_idx * prediction_length
                )
                all_contexts.append(ts[offset : offset + context_length])
                all_labels.append(
                    ts[offset + context_length : offset + context_length + prediction_length]
                )

    if not all_contexts:
        raise ValueError(
            f"No evaluation windows found for datasets {dataset_names}. "
            "Series may be too short."
        )

    contexts = np.stack(all_contexts)
    labels = np.stack(all_labels)

    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be > 0 when provided")
        contexts = contexts[:max_windows]
        labels = labels[:max_windows]

    pred_horizon = labels.shape[1]
    output_patch_size = model.chronos_config.output_patch_size
    num_output_patches = math.ceil(pred_horizon / output_patch_size)

    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            batch_context = torch.from_numpy(contexts[start : start + batch_size]).to(
                device
            )
            out = model(context=batch_context, num_output_patches=num_output_patches)
            pred = out.quantile_preds[..., :pred_horizon].detach().cpu().numpy()
            preds.append(pred)

    quantile_preds = np.concatenate(preds, axis=0)
    quantiles = np.asarray(model.chronos_config.quantiles, dtype=np.float32)
    return contexts, labels, quantile_preds, quantiles


def _log_forecast_plots(
    *,
    domain: str,
    contexts: np.ndarray,
    labels: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: np.ndarray,
    mlflow_artifact_dir: str,
    samples_per_domain: int,
    context_points: int,
) -> None:
    if len(contexts) == 0:
        return

    median_idx = int(np.argmin(np.abs(quantiles - 0.5)))
    lower_idx = int(np.argmin(np.abs(quantiles - 0.1)))
    upper_idx = int(np.argmin(np.abs(quantiles - 0.9)))

    count = min(samples_per_domain, len(contexts))
    sample_indices = np.linspace(0, len(contexts) - 1, num=count, dtype=int)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for plot_num, sample_idx in enumerate(sample_indices, start=1):
            context = contexts[int(sample_idx)]
            label = labels[int(sample_idx)]
            pred_median = quantile_preds[int(sample_idx), median_idx, :]
            pred_low = quantile_preds[int(sample_idx), lower_idx, :]
            pred_high = quantile_preds[int(sample_idx), upper_idx, :]

            context_tail = context[-context_points:]
            context_x = np.arange(len(context_tail))
            forecast_x = np.arange(len(context_tail), len(context_tail) + len(label))

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(context_x, context_tail, label="context", color="tab:blue")
            ax.plot(forecast_x, label, label="ground truth", color="tab:green")
            ax.plot(forecast_x, pred_median, label="forecast p50", color="tab:orange")
            ax.fill_between(
                forecast_x,
                pred_low,
                pred_high,
                color="tab:orange",
                alpha=0.2,
                label="forecast p10-p90",
            )
            ax.axvline(
                x=len(context_tail) - 1, color="black", linestyle="--", alpha=0.5
            )
            ax.set_title(f"{domain} sample {plot_num}")
            ax.set_xlabel("time index")
            ax.set_ylabel("value")
            ax.legend(loc="best")
            fig.tight_layout()

            out_file = tmp_path / f"{domain}_sample_{plot_num}.png"
            fig.savefig(out_file, dpi=120)
            plt.close(fig)

        mlflow.log_artifacts(
            str(tmp_path), artifact_path=f"{mlflow_artifact_dir}/{domain}"
        )


def evaluate_model_all_domains(
    model: Chronos2Model,
    hf_repo: str,
    transport_datasets: list[str],
    energy_datasets: list[str],
    context_length: int,
    prediction_length: int,
    num_rolling_windows: int,
    batch_size: int,
    device: torch.device,
    max_windows: int | None = None,
    log_plots: bool = False,
    plot_artifact_dir: str = "evaluation_plots",
    samples_per_domain: int = 10,
    context_points: int = 128,
) -> dict[str, float]:
    domain_datasets = {"transport": transport_datasets, "energy": energy_datasets}
    results: dict[str, float] = {}
    for domain in DOMAINS:
        contexts, labels, quantile_preds, quantiles = _predict_on_domain(
            model=model,
            hf_repo=hf_repo,
            dataset_names=domain_datasets[domain],
            context_length=context_length,
            prediction_length=prediction_length,
            num_rolling_windows=num_rolling_windows,
            batch_size=batch_size,
            device=device,
            max_windows=max_windows,
        )
        metrics = _compute_metrics(
            quantile_preds=quantile_preds, labels=labels, quantiles=quantiles
        )
        metrics["num_windows"] = float(labels.shape[0])
        for key, value in metrics.items():
            results[f"{domain}/{key}"] = value
        if log_plots:
            _log_forecast_plots(
                domain=domain,
                contexts=contexts,
                labels=labels,
                quantile_preds=quantile_preds,
                quantiles=quantiles,
                mlflow_artifact_dir=plot_artifact_dir,
                samples_per_domain=samples_per_domain,
                context_points=context_points,
            )
    return results


def evaluate_checkpoint_all_domains(
    checkpoint: Path,
    hf_repo: str,
    transport_datasets: list[str],
    energy_datasets: list[str],
    context_length: int,
    prediction_length: int,
    num_rolling_windows: int,
    batch_size: int,
    device: torch.device | None = None,
    max_windows: int | None = None,
) -> dict[str, float]:
    device = device or _device_from_arg(None)
    model = Chronos2Model.from_pretrained(checkpoint)
    model.eval()
    model.to(device)
    return evaluate_model_all_domains(
        model=model,
        hf_repo=hf_repo,
        transport_datasets=transport_datasets,
        energy_datasets=energy_datasets,
        context_length=context_length,
        prediction_length=prediction_length,
        num_rolling_windows=num_rolling_windows,
        batch_size=batch_size,
        device=device,
        max_windows=max_windows,
    )


@use_config(yaml_conf_callback)
def main(
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", help="Path to model checkpoint directory."
    ),
    checkpoint_artifact_path: str = typer.Option(
        "checkpoints/final",
        "--checkpoint-artifact-path",
        help="Artifact path inside MLflow run used when --checkpoint is omitted.",
    ),
    domain: str = typer.Option("both", "--domain", help="transport | energy | both"),
    hf_repo: str = typer.Option(
        "autogluon/chronos_datasets", "--hf-repo"
    ),
    transport_datasets: list[str] = typer.Option(
        ["monash_traffic", "taxi_30min", "uber_tlc_hourly"], "--transport-datasets"
    ),
    energy_datasets: list[str] = typer.Option(
        ["electricity_15min", "solar_1h", "wind_farms_hourly"], "--energy-datasets"
    ),
    num_rolling_windows: int = typer.Option(5, "--num-rolling-windows"),
    context_length: int = typer.Option(1024, "--context-length"),
    prediction_length: int = typer.Option(64, "--prediction-length"),
    batch_size: int = typer.Option(128, "--batch-size"),
    device: str | None = typer.Option(None, "--device"),
    mlflow_run_id: str | None = typer.Option(None, "--mlflow-run-id"),
    max_windows: int | None = typer.Option(
        None, "--max-windows", help="Limit evaluated windows per domain."
    ),
    plot_samples_per_domain: int = typer.Option(3, "--plot-samples-per-domain"),
    plot_context_points: int = typer.Option(128, "--plot-context-points"),
    plot_artifact_dir: str = typer.Option("evaluation_plots", "--plot-artifact-dir"),
) -> None:
    if domain not in {"transport", "energy", "both"}:
        raise ValueError("domain must be one of: transport, energy, both")

    device_obj = _device_from_arg(device)
    resolved_checkpoint = _resolve_checkpoint(
        checkpoint=checkpoint,
        mlflow_run_id=mlflow_run_id,
        checkpoint_artifact_path=checkpoint_artifact_path,
    )

    model = Chronos2Model.from_pretrained(resolved_checkpoint)
    model.eval()
    model.to(device_obj)

    domains = DOMAINS if domain == "both" else (domain,)
    domain_datasets = {"transport": transport_datasets, "energy": energy_datasets}
    all_metrics: dict[str, float] = {}

    active_run = mlflow.start_run(run_id=mlflow_run_id) if mlflow_run_id else None
    try:
        for current_domain in domains:
            contexts, labels, quantile_preds, quantiles = _predict_on_domain(
                model=model,
                hf_repo=hf_repo,
                dataset_names=domain_datasets[current_domain],
                context_length=context_length,
                prediction_length=prediction_length,
                num_rolling_windows=num_rolling_windows,
                batch_size=batch_size,
                device=device_obj,
                max_windows=max_windows,
            )
            metrics = _compute_metrics(
                quantile_preds=quantile_preds,
                labels=labels,
                quantiles=quantiles,
            )
            metrics["num_windows"] = float(labels.shape[0])

            print(
                f"{current_domain} -> "
                + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
            )
            for key, value in metrics.items():
                all_metrics[f"{current_domain}/{key}"] = value

            if mlflow_run_id:
                _log_forecast_plots(
                    domain=current_domain,
                    contexts=contexts,
                    labels=labels,
                    quantile_preds=quantile_preds,
                    quantiles=quantiles,
                    mlflow_artifact_dir=plot_artifact_dir,
                    samples_per_domain=plot_samples_per_domain,
                    context_points=plot_context_points,
                )

        if mlflow_run_id:
            mlflow.log_metrics(all_metrics)
    finally:
        if active_run is not None:
            mlflow.end_run()
