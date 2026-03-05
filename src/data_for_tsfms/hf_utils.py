from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset, Sequence, Value, load_dataset

_NUMERIC_DTYPES = frozenset({"float32", "float64", "double", "int8", "int16", "int32", "int64"})

GIFT_EVAL_REPO = "Salesforce/GiftEvalPretrain"


@dataclass
class DomainConfig:
    hf_repo: str
    datasets: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "DomainConfig":
        return cls(
            hf_repo=d["hf_repo"],
            datasets=list(d["datasets"]),
        )


def get_target_columns(features) -> list[str]:
    """Return names of all numeric Sequence columns (target time series per the dataset schema)."""
    return [
        name for name, feat in features.items()
        if isinstance(feat, Sequence)
        and isinstance(feat.feature, Value)
        and feat.feature.dtype in _NUMERIC_DTYPES
    ]


def load_target_series(split, target_cols: list[str]) -> list[np.ndarray]:
    """Load all target series from a dataset split using column-wise access.

    Prefetches each column in one Arrow batch read rather than deserializing
    all columns row-by-row.

    Returns a list of 1D arrays (univariate) or 2D (n_targets, T) arrays (multivariate).
    """
    columns = [split[col] for col in target_cols]
    if len(columns) == 1:
        return [np.asarray(ts, dtype=np.float32) for ts in columns[0]]
    return [
        np.stack([np.asarray(columns[j][i], dtype=np.float32) for j in range(len(target_cols))])
        for i in range(len(split))
    ]


def load_target_series_cached(
    hf_repo: str,
    config_name: str,
    cache_dir: Path,
) -> list[np.ndarray]:
    """Load all target series for a dataset config, with transparent disk caching."""
    cache_path = cache_dir / hf_repo.replace("/", "_") / f"{config_name}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    split = load_dataset(hf_repo, config_name, split="train")
    target_cols = get_target_columns(split.features)
    series = load_target_series(split, target_cols)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(series, f)

    return series


def load_gift_eval_series_cached(
    hf_repo: str,
    dataset_name: str,
    cache_dir: Path,
) -> list[np.ndarray]:
    """Load target series from a GiftEvalPretrain dataset.

    Downloads only the first arrow shard to avoid fetching the full dataset.
    Multivariate series (target shape (N, T)) are split into N univariate series.
    """
    cache_path = cache_dir / hf_repo.replace("/", "_") / f"{dataset_name}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from huggingface_hub import HfApi, hf_hub_download

    tree = HfApi().list_repo_tree(repo_id=hf_repo, path_in_repo=dataset_name, repo_type="dataset")
    first_shard = min((item.path for item in tree if item.path.endswith(".arrow")), default=None)
    if first_shard is None:
        raise FileNotFoundError(f"No arrow files found for {hf_repo}/{dataset_name}")

    local_path = hf_hub_download(repo_id=hf_repo, filename=first_shard, repo_type="dataset")
    shard = Dataset.from_file(local_path)

    series: list[np.ndarray] = []
    for t in shard["target"]:
        arr = np.asarray(t, dtype=np.float32)
        if arr.ndim == 1:
            series.append(arr)
        else:
            # Multivariate: split each channel into a separate univariate series
            for i in range(arr.shape[0]):
                series.append(arr[i])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(series, f)

    return series


def load_domain_series_cached(domain_cfg: DomainConfig, cache_dir: Path) -> list[np.ndarray]:
    """Load all series for a domain, dispatching to the appropriate loader based on hf_repo."""
    if domain_cfg.hf_repo == GIFT_EVAL_REPO:
        series: list[np.ndarray] = []
        for ds_name in domain_cfg.datasets:
            series.extend(load_gift_eval_series_cached(domain_cfg.hf_repo, ds_name, cache_dir))
        return series

    # Standard HF datasets (e.g. autogluon/chronos_datasets)
    series = []
    for ds_name in domain_cfg.datasets:
        series.extend(load_target_series_cached(domain_cfg.hf_repo, ds_name, cache_dir))
    return series
