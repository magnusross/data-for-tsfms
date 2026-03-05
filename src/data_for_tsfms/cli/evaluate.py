from __future__ import annotations

import math
from pathlib import Path

import mlflow
import numpy as np
import torch
import typer
from typer_config import use_config
from data_for_tsfms.config_utils import yaml_conf_callback, get_loaded_config
from data_for_tsfms.hf_utils import DomainConfig, load_domain_series_cached
from data_for_tsfms.evaluation_utils import (
    device_from_arg,
    log_forecast_plots,
    resolve_checkpoint,
)

from chronos.chronos2 import Chronos2Model


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


def _predict_on_domain(
    model: Chronos2Model,
    domain_cfg: DomainConfig,
    context_length: int,
    prediction_length: int,
    num_rolling_windows: int,
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    heldout = num_rolling_windows * prediction_length
    all_contexts: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for ts in load_domain_series_cached(domain_cfg, cache_dir):
        ts_len = ts.shape[-1]
        if ts_len < context_length + heldout:
            continue
        for window_idx in range(num_rolling_windows):
            offset = (
                ts_len - heldout - context_length + window_idx * prediction_length
            )
            all_contexts.append(ts[..., offset : offset + context_length])
            all_labels.append(
                ts[
                    ...,
                    offset + context_length : offset + context_length + prediction_length,
                ]
            )

    if not all_contexts:
        raise ValueError(
            f"No evaluation windows found for datasets {domain_cfg.datasets}. "
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


def evaluate_model_all_domains(
    model: Chronos2Model,
    active_domains: list[str],
    domain_configs: dict[str, DomainConfig],
    context_length: int,
    prediction_length: int,
    num_rolling_windows: int,
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
    max_windows: int | None = None,
    log_plots: bool = False,
    plot_artifact_dir: str = "evaluation_plots",
    samples_per_domain: int = 10,
    context_points: int = 128,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for domain in active_domains:
        contexts, labels, quantile_preds, quantiles = _predict_on_domain(
            model=model,
            domain_cfg=domain_configs[domain],
            context_length=context_length,
            prediction_length=prediction_length,
            num_rolling_windows=num_rolling_windows,
            cache_dir=cache_dir,
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
            log_forecast_plots(
                label=domain,
                contexts=contexts,
                labels=labels,
                quantile_preds=quantile_preds,
                quantiles=quantiles,
                mlflow_artifact_dir=plot_artifact_dir,
                samples=samples_per_domain,
                context_points=context_points,
            )
    return results


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
    active_domains: list[str] = typer.Option([], "--active-domains"),
    num_rolling_windows: int = typer.Option(5, "--num-rolling-windows"),
    data_cache_dir: Path = typer.Option(Path(".hf_cache"), "--data-cache-dir"),
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
    cfg = get_loaded_config()
    all_domain_cfgs = {
        name: DomainConfig.from_dict(d)
        for name, d in cfg.get("domains", {}).items()
    }

    domains_to_eval = list(active_domains) if active_domains else list(all_domain_cfgs.keys())
    for d in domains_to_eval:
        if d not in all_domain_cfgs:
            raise ValueError(f"Domain '{d}' not in config. Available: {list(all_domain_cfgs.keys())}")
    domain_configs = {d: all_domain_cfgs[d] for d in domains_to_eval}

    device_obj = device_from_arg(device)
    resolved_checkpoint = resolve_checkpoint(
        checkpoint=checkpoint,
        mlflow_run_id=mlflow_run_id,
        checkpoint_artifact_path=checkpoint_artifact_path,
    )

    model = Chronos2Model.from_pretrained(resolved_checkpoint)
    model.eval()
    model.to(device_obj)

    all_metrics: dict[str, float] = {}
    active_run = mlflow.start_run(run_id=mlflow_run_id) if mlflow_run_id else None
    try:
        for domain in domains_to_eval:
            contexts, labels, quantile_preds, quantiles = _predict_on_domain(
                model=model,
                domain_cfg=domain_configs[domain],
                context_length=context_length,
                prediction_length=prediction_length,
                num_rolling_windows=num_rolling_windows,
                cache_dir=data_cache_dir,
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
                f"{domain} -> "
                + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
            )
            for key, value in metrics.items():
                all_metrics[f"{domain}/{key}"] = value

            if mlflow_run_id:
                log_forecast_plots(
                    label=domain,
                    contexts=contexts,
                    labels=labels,
                    quantile_preds=quantile_preds,
                    quantiles=quantiles,
                    mlflow_artifact_dir=plot_artifact_dir,
                    samples=plot_samples_per_domain,
                    context_points=plot_context_points,
                )

        if mlflow_run_id:
            mlflow.log_metrics(all_metrics)
    finally:
        if active_run is not None:
            mlflow.end_run()
