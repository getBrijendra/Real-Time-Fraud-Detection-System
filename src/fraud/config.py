"""Configuration loading and path resolution."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("FRAUD_HOME", Path(__file__).resolve().parents[2]))
CONFIG_PATH = Path(os.environ.get("FRAUD_CONFIG", PROJECT_ROOT / "configs" / "config.yaml"))


class Config(dict):
    """dict with attribute access (nested dicts are wrapped on access)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value


@lru_cache(maxsize=1)
def load_config() -> Config:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Environment overrides (used by docker-compose).
    if url := os.environ.get("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = url
    if servers := os.environ.get("KAFKA_BOOTSTRAP_SERVERS"):
        cfg["kafka"]["bootstrap_servers"] = servers
    if api := os.environ.get("API_URL"):
        cfg["serving"]["api_url"] = api
    return Config(cfg)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def mlflow_tracking_uri(cfg: Config | None = None) -> str:
    cfg = cfg or load_config()
    uri = cfg.mlflow.tracking_uri
    if uri.startswith("sqlite:///") and not Path(uri[len("sqlite:///"):]).is_absolute():
        uri = "sqlite:///" + resolve(uri[len("sqlite:///"):]).as_posix()
    return uri


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("mlflow", "alembic", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
