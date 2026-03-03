from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from datasets import Sequence, Value, load_dataset

_NUMERIC_DTYPES = frozenset({"float32", "float64", "double", "int8", "int16", "int32", "int64"})


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
    """Load all target series for a dataset config, with transparent disk caching.

    On first call the data is fetched from HuggingFace and written to
    ``cache_dir/{hf_repo}/{config_name}.pkl``. Subsequent calls load
    directly from that file and return instantly.

    Delete the cache file (or the whole cache_dir) to force a refresh.
    """
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
