from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from datasets import load_dataset
from gluonts.dataset.arrow import ArrowWriter
from typer_config import use_config
from data_for_tsfms.cli.config_utils import yaml_conf_callback

from data_for_tsfms.split_utils import temporal_split


def _to_timestamp(raw_start: Any) -> pd.Timestamp:
    if isinstance(raw_start, pd.Timestamp):
        return raw_start
    return pd.Timestamp(raw_start)


def _extract_target_and_start(row: dict[str, Any]) -> tuple[np.ndarray, pd.Timestamp]:
    target_keys = ("target", "consumption_kW", "power_mw")
    target_key = next((key for key in target_keys if key in row), None)
    if target_key is None:
        raise KeyError(
            f"Could not find target field in row keys={sorted(row.keys())}. "
            f"Expected one of {target_keys}."
        )

    timestamp = row.get("start")
    if timestamp is None:
        timestamp = row.get("timestamp")
    if timestamp is None:
        raise KeyError(
            f"Could not find start/timestamp field in row keys={sorted(row.keys())}."
        )

    if isinstance(timestamp, (list, tuple, np.ndarray)):
        if len(timestamp) == 0:
            raise ValueError("timestamp sequence is empty")
        timestamp = timestamp[0]

    target = np.asarray(row[target_key], dtype=np.float32)
    if target.ndim != 1:
        raise ValueError(
            f"Expected 1D target for key '{target_key}', got shape={target.shape}"
        )

    return target, _to_timestamp(timestamp)


def convert_to_arrow(
    path: Path, rows: list[dict[str, Any]], compression: str = "lz4"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ArrowWriter(compression=compression).write_to_file(rows, path=path)


@use_config(yaml_conf_callback)
def main(
    domain: str = typer.Option(
        ..., "--domain", help="Domain name used for output file names."
    ),
    hf_repo: str = typer.Option(
        "autogluon/chronos_datasets",
        "--hf-repo",
        help="Hugging Face dataset repository.",
    ),
    datasets: list[str] = typer.Option(
        ..., "--datasets", help="Dataset config names to download."
    ),
    freqs: list[str] = typer.Option(
        ..., "--freqs", help="Frequency strings matching datasets."
    ),
    context_length: int = typer.Option(2048, "--context-length"),
    prediction_length: int = typer.Option(64, "--prediction-length"),
    num_rolling_windows: int = typer.Option(5, "--num-rolling-windows"),
    output_dir: Path = typer.Option(Path("data"), "--output-dir"),
    compression: str = typer.Option("lz4", "--compression"),
) -> None:
    if domain not in {"energy", "transport"}:
        raise ValueError("domain must be one of: energy, transport")
    if len(datasets) != len(freqs):
        raise ValueError("datasets and freqs must have the same length")

    dataset_map = dict(zip(datasets, freqs))

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"kept": 0, "dropped": 0})

    for config_name, freq in dataset_map.items():
        split = load_dataset(hf_repo, str(config_name), split="train")
        for row in split:
            ts, start = _extract_target_and_start(row)
            train_row, windows = temporal_split(
                ts=ts,
                start=start,
                freq=str(freq),
                context_length=context_length,
                prediction_length=prediction_length,
                num_rolling_windows=num_rolling_windows,
            )

            if train_row is None:
                stats[str(config_name)]["dropped"] += 1
                continue

            train_row["dataset"] = str(config_name)
            train_rows.append(train_row)
            for w in windows:
                w["dataset"] = str(config_name)
                test_rows.append(w)
            stats[str(config_name)]["kept"] += 1

    train_path = output_dir / f"{domain}_train.arrow"
    test_path = output_dir / f"{domain}_test.arrow"
    convert_to_arrow(train_path, train_rows, compression=compression)
    convert_to_arrow(test_path, test_rows, compression=compression)

    print(f"Wrote {len(train_rows)} train rows -> {train_path}")
    print(f"Wrote {len(test_rows)} test rows  -> {test_path}")
    for ds_name, ds_stats in stats.items():
        print(f"{ds_name}: kept={ds_stats['kept']} dropped={ds_stats['dropped']}")
