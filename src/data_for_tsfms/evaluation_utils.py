from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch


def device_from_arg(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(
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


def log_forecast_plots(
    *,
    label: str,
    contexts: np.ndarray,
    labels: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: np.ndarray,
    mlflow_artifact_dir: str,
    samples: int,
    context_points: int,
) -> None:
    if len(contexts) == 0:
        return

    median_idx = int(np.argmin(np.abs(quantiles - 0.5)))
    lower_idx = int(np.argmin(np.abs(quantiles - 0.1)))
    upper_idx = int(np.argmin(np.abs(quantiles - 0.9)))

    count = min(samples, len(contexts))
    sample_indices = np.linspace(0, len(contexts) - 1, num=count, dtype=int)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        safe_label = label.replace("/", "_")
        for plot_num, sample_idx in enumerate(sample_indices, start=1):
            context = contexts[int(sample_idx)]
            label_values = labels[int(sample_idx)]
            pred_median = quantile_preds[int(sample_idx), median_idx, :]
            pred_low = quantile_preds[int(sample_idx), lower_idx, :]
            pred_high = quantile_preds[int(sample_idx), upper_idx, :]

            context_tail = context[-context_points:]
            context_x = np.arange(len(context_tail))
            forecast_x = np.arange(
                len(context_tail), len(context_tail) + len(label_values)
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(context_x, context_tail, label="context", color="tab:blue")
            ax.plot(forecast_x, label_values, label="ground truth", color="tab:green")
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
            ax.set_title(f"{label} sample {plot_num}")
            ax.set_xlabel("time index")
            ax.set_ylabel("value")
            ax.legend(loc="best")
            fig.tight_layout()

            out_file = tmp_path / f"{safe_label}_sample_{plot_num}.png"
            fig.savefig(out_file, dpi=120)
            plt.close(fig)

        mlflow.log_artifacts(
            str(tmp_path), artifact_path=f"{mlflow_artifact_dir}/{safe_label}"
        )
