from __future__ import annotations

import math
from pathlib import Path

import mlflow
import numpy as np
import torch
import typer
from gluonts.dataset.arrow import File
from typer_config import use_yaml_config

from chronos.chronos2 import Chronos2Model

DOMAINS = ("transport", "energy")
app = typer.Typer(
    add_completion=False,
    help="Evaluate Chronos2 checkpoints on held-out Arrow test files.",
)


def _device_from_arg(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_arrow_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Arrow file not found: {path}")
    return list(File.infer(path))


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


def evaluate_checkpoint_on_domain(
    checkpoint: Path,
    data_dir: Path,
    domain: str,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or _device_from_arg(None)
    model = Chronos2Model.from_pretrained(checkpoint)
    model.eval()
    model.to(device)

    rows = _load_arrow_rows(data_dir / f"{domain}_test.arrow")
    if not rows:
        raise ValueError(f"No rows in {domain}_test.arrow")

    contexts = np.stack([np.asarray(row["context"], dtype=np.float32) for row in rows])
    labels = np.stack([np.asarray(row["label"], dtype=np.float32) for row in rows])

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

    metrics = _compute_metrics(
        quantile_preds=quantile_preds, labels=labels, quantiles=quantiles
    )
    metrics["num_windows"] = float(len(rows))
    return metrics


def evaluate_checkpoint_all_domains(
    checkpoint: Path,
    data_dir: Path,
    batch_size: int,
    device: torch.device | None = None,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for domain in DOMAINS:
        metrics = evaluate_checkpoint_on_domain(
            checkpoint=checkpoint,
            data_dir=data_dir,
            domain=domain,
            batch_size=batch_size,
            device=device,
        )
        for key, value in metrics.items():
            results[f"{domain}/{key}"] = value
    return results


@app.command()
@use_yaml_config()
def main(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", help="Path to model checkpoint directory."
    ),
    domain: str = typer.Option("both", "--domain", help="transport | energy | both"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    batch_size: int = typer.Option(128, "--batch-size"),
    device: str | None = typer.Option(None, "--device"),
    mlflow_run_id: str | None = typer.Option(None, "--mlflow-run-id"),
) -> None:
    if domain not in {"transport", "energy", "both"}:
        raise ValueError("domain must be one of: transport, energy, both")

    device_obj = _device_from_arg(device)

    domains = DOMAINS if domain == "both" else (domain,)
    all_metrics: dict[str, float] = {}
    for domain in domains:
        metrics = evaluate_checkpoint_on_domain(
            checkpoint=checkpoint,
            data_dir=data_dir,
            domain=domain,
            batch_size=batch_size,
            device=device_obj,
        )
        print(f"{domain} -> " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
        for key, value in metrics.items():
            all_metrics[f"{domain}/{key}"] = value

    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metrics(all_metrics)


if __name__ == "__main__":
    app()
