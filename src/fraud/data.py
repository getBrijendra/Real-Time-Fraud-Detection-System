"""Dataset download, loading and chronological splitting.

Dataset: "Credit Card Fraud Detection" (ULB Machine Learning Group), Kaggle `mlg-ulb/creditcardfraud`.
284,807 European card transactions over two days in Sept 2013, 492 frauds (0.172%).
Features V1..V28 are PCA components (anonymised); `Time` is seconds since the first transaction.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import load_config, resolve

log = logging.getLogger(__name__)

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", *V_COLS, "Amount"]
TARGET = "Class"
EXPECTED_ROWS = 284_807


def _kaggle_available() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or (
        Path.home() / ".kaggle" / "kaggle.json"
    ).exists()


def download_dataset(force: bool = False) -> Path:
    """Download creditcard.csv: Kaggle API if a token is configured, otherwise the public mirror."""
    cfg = load_config()
    dest = resolve(cfg.data.raw_path)
    if dest.exists() and not force:
        log.info("Dataset already present at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _kaggle_available():
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            with tempfile.TemporaryDirectory() as tmp:
                log.info("Downloading %s from Kaggle ...", cfg.data.kaggle_dataset)
                api.dataset_download_files(cfg.data.kaggle_dataset, path=tmp, unzip=False)
                archive = next(Path(tmp).glob("*.zip"))
                with zipfile.ZipFile(archive) as zf:
                    zf.extract("creditcard.csv", tmp)
                shutil.move(str(Path(tmp) / "creditcard.csv"), dest)
            return _validate(dest)
        except Exception as exc:  # noqa: BLE001 - fall back to mirror
            log.warning("Kaggle download failed (%s); falling back to mirror", exc)

    log.info("Downloading dataset from mirror %s ...", cfg.data.mirror_url)
    tmp_path = dest.with_suffix(".part")
    urllib.request.urlretrieve(cfg.data.mirror_url, tmp_path)  # noqa: S310 - fixed https URL from config
    tmp_path.replace(dest)
    return _validate(dest)


def _validate(path: Path) -> Path:
    df = pd.read_csv(path, nrows=5)
    missing = set(RAW_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    log.info("Dataset ready at %s", path)
    return path


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    path = Path(path) if path else resolve(cfg.data.raw_path)
    if not path.exists():
        download_dataset()
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(int)
    if len(df) != EXPECTED_ROWS:
        log.warning("Expected %d rows, got %d", EXPECTED_ROWS, len(df))
    return df


def chronological_split(
    df: pd.DataFrame, train_frac: float | None = None, val_frac: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time so that validation/test are strictly "in the future" of training data.

    A random split would leak future fraud patterns into training and overstate performance.
    """
    cfg = load_config()
    train_frac = cfg.data.train_frac if train_frac is None else train_frac
    val_frac = cfg.data.val_frac if val_frac is None else val_frac
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    i_train, i_val = int(n * train_frac), int(n * (train_frac + val_frac))
    train, val, test = df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]
    for name, part in (("train", train), ("val", val), ("test", test)):
        log.info(
            "%-5s rows=%-7d frauds=%-4d fraud_rate=%.4f%% time=[%.0f, %.0f]s",
            name, len(part), part[TARGET].sum(), 100 * part[TARGET].mean(), part.Time.min(), part.Time.max(),
        )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def write_stream_source(test: pd.DataFrame) -> Path:
    """Persist the held-out future window that the Kafka producer replays as live traffic."""
    path = resolve(load_config().data.stream_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = test.copy()
    out.insert(0, "transaction_id", [f"tx-{int(t):06d}-{i:06d}" for i, t in zip(out.index, out.Time)])
    out.to_parquet(path, index=False)
    log.info("Stream source written to %s (%d transactions)", path, len(out))
    return path


if __name__ == "__main__":
    from .config import setup_logging

    setup_logging()
    download_dataset()
