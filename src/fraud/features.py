"""Feature engineering shared by training and serving (single code path -> no train/serve skew)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import V_COLS

MODEL_FEATURES = [*V_COLS, "log_amount", "hour_sin", "hour_cos"]

# Engineered columns that should be explained as one human-readable signal.
FEATURE_GROUPS = {"log_amount": "Amount", "hour_sin": "hour_of_day", "hour_cos": "hour_of_day"}


def hour_of_day(time_seconds: pd.Series | np.ndarray) -> np.ndarray:
    return (np.asarray(time_seconds, dtype=float) % 86_400) / 3_600


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw transaction columns (Time, V1..V28, Amount) -> model feature matrix.

    Built in numpy and wrapped once: column-by-column pandas ops cost ~2ms on a 1-row frame."""
    angle = 2 * np.pi * hour_of_day(raw["Time"].to_numpy()) / 24
    amount = np.clip(raw["Amount"].to_numpy(dtype=float), 0, None)
    arr = np.column_stack([raw[V_COLS].to_numpy(dtype=float), np.log1p(amount), np.sin(angle), np.cos(angle)])
    return pd.DataFrame(arr, columns=MODEL_FEATURES, index=raw.index)
