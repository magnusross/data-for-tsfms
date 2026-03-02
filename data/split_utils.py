from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplitConfig:
    context_length: int = 2048
    prediction_length: int = 64
    num_rolling_windows: int = 5

    @property
    def heldout_horizon(self) -> int:
        return self.num_rolling_windows * self.prediction_length


def temporal_split(
    ts: np.ndarray,
    start: pd.Timestamp,
    freq: str,
    context_length: int = 2048,
    prediction_length: int = 64,
    num_rolling_windows: int = 5,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (train_row, test_rows) with strict temporal separation.

    The train target is the historical prefix up to T-H, where
    H = num_rolling_windows * prediction_length. Test rows are rolling windows
    of length context_length + prediction_length carved from the final H tail.
    """
    ts = np.asarray(ts, dtype=np.float32)
    heldout_horizon = num_rolling_windows * prediction_length

    if ts.ndim != 1:
        raise ValueError(f"Expected univariate 1D array, got shape={ts.shape}")

    if len(ts) < context_length + heldout_horizon:
        return None, []

    train_target = ts[: len(ts) - heldout_horizon]
    train_row = {"start": start, "target": train_target}

    test_rows: list[dict[str, Any]] = []
    for window_idx in range(num_rolling_windows):
        offset = (
            len(ts) - heldout_horizon - context_length + window_idx * prediction_length
        )
        context = ts[offset : offset + context_length]
        label = ts[
            offset + context_length : offset + context_length + prediction_length
        ]
        test_rows.append(
            {
                "start": start,
                "freq": freq,
                "context": context,
                "label": label,
                "window_idx": window_idx,
            }
        )

    return train_row, test_rows
