from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import typer
import yaml
from typer_config import use_config

from chronos.chronos2 import Chronos2Pipeline
from data_for_tsfms.config_utils import yaml_conf_callback
from data_for_tsfms.evaluation_utils import (
    device_from_arg,
    log_forecast_plots,
    resolve_checkpoint,
)


def _load_task_configs(task_config: Path) -> list[dict[str, Any]]:
    if not task_config.exists():
        raise FileNotFoundError(f"Task config not found: {task_config}")
    with task_config.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            f"Task config {task_config} must contain a non-empty 'tasks' list"
        )
    return [dict(task) for task in tasks]


def _task_name(task) -> str:
    return task.task_name or task.dataset_config


def _extract_plot_arrays(
    task, predictions_per_window
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not predictions_per_window:
        return None

    first_window = task.get_window(0)
    predictions = predictions_per_window[0]
    target_columns = list(task.target_columns)
    if not target_columns:
        return None
    target_column = target_columns[0]

    if target_column not in predictions:
        return None

    past_data, _ = first_window.get_input_data()
    ground_truth = first_window.get_ground_truth()
    pred_ds = predictions[target_column]

    contexts = np.stack(
        [
            np.asarray(past_data[i][target_column], dtype=np.float32)
            for i in range(len(past_data))
        ]
    )
    labels = np.stack(
        [
            np.asarray(ground_truth[i][target_column], dtype=np.float32)
            for i in range(len(ground_truth))
        ]
    )

    quantiles = np.asarray(task.quantile_levels, dtype=np.float32)
    quantile_preds = np.stack(
        [
            np.stack(
                [
                    np.asarray(pred_ds[i][str(q)], dtype=np.float32)
                    for i in range(len(pred_ds))
                ],
                axis=0,
            )
            for q in quantiles
        ],
        axis=1,
    )
    return contexts, labels, quantile_preds, quantiles


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
    task_config: Path = typer.Option(
        Path("configs/evaluate_fev/joint.yaml"),
        "--task-config",
        help="Path to YAML file containing FEV task definitions under key 'tasks'.",
    ),
    model_name: str = typer.Option("chronos2-finetuned", "--model-name"),
    batch_size: int = typer.Option(128, "--batch-size"),
    device: str | None = typer.Option(None, "--device"),
    mlflow_run_id: str | None = typer.Option(None, "--mlflow-run-id"),
    max_windows: int | None = typer.Option(
        None,
        "--max-windows",
        help="Optional upper bound for the number of windows evaluated per task.",
    ),
    as_univariate: bool = typer.Option(False, "--as-univariate/--no-as-univariate"),
    cross_learning: bool = typer.Option(False, "--cross-learning/--no-cross-learning"),
    plot_samples_per_task: int = typer.Option(3, "--plot-samples-per-task"),
    plot_context_points: int = typer.Option(128, "--plot-context-points"),
    plot_artifact_dir: str = typer.Option(
        "evaluation_plots_fev", "--plot-artifact-dir"
    ),
) -> None:
    try:
        import fev
    except ImportError as exc:
        raise ImportError(
            "fev is required. Install project dependencies with `uv sync`."
        ) from exc

    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be > 0 when provided")

    task_definitions = _load_task_configs(task_config)
    device_obj = device_from_arg(device)
    resolved_checkpoint = resolve_checkpoint(
        checkpoint=checkpoint,
        mlflow_run_id=mlflow_run_id,
        checkpoint_artifact_path=checkpoint_artifact_path,
    )

    pipeline = Chronos2Pipeline.from_pretrained(resolved_checkpoint)
    pipeline.model.eval()
    pipeline.model.to(device_obj)

    all_metrics: dict[str, float] = {}
    active_run = mlflow.start_run(run_id=mlflow_run_id) if mlflow_run_id else None
    try:
        for task_kwargs in task_definitions:
            if max_windows is not None:
                current_windows = int(task_kwargs.get("num_windows", 1))
                task_kwargs["num_windows"] = min(current_windows, max_windows)

            task = fev.Task(**task_kwargs)
            task_name = _task_name(task)

            predictions_per_window, inference_time_s = pipeline.predict_fev(
                task=task,
                batch_size=batch_size,
                as_univariate=as_univariate,
                cross_learning=cross_learning,
            )

            summary = task.evaluation_summary(
                predictions_per_window,
                model_name=model_name,
                inference_time_s=inference_time_s,
            )

            print(f"{task_name} -> test_error={summary['test_error']:.6f}")
            for key, value in summary.items():
                if isinstance(value, (int, float, np.floating)):
                    all_metrics[f"{task_name}/{key}"] = float(value)

            if mlflow_run_id:
                plot_arrays = _extract_plot_arrays(task, predictions_per_window)
                if plot_arrays is not None:
                    contexts, labels, quantile_preds, quantiles = plot_arrays
                    log_forecast_plots(
                        label=task_name,
                        contexts=contexts,
                        labels=labels,
                        quantile_preds=quantile_preds,
                        quantiles=quantiles,
                        mlflow_artifact_dir=plot_artifact_dir,
                        samples=plot_samples_per_task,
                        context_points=plot_context_points,
                    )

        if mlflow_run_id:
            mlflow.log_metrics(all_metrics)
    finally:
        if active_run is not None:
            mlflow.end_run()
