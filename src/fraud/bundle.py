"""The deployable unit: model + decision threshold + explainer + drift reference + metadata."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import load_config, resolve
from .drift import ReferenceProfile
from .explain import Explainer, summarize
from .features import build_features
from .metrics import CostModel
from .models import _single_threaded_inference, predict_scores


@dataclass
class ModelBundle:
    model: Any
    model_name: str
    version: str
    threshold: float
    explainer: Explainer
    reference: ReferenceProfile
    cost_model: CostModel
    metrics: dict = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    mlflow_run_id: str | None = None

    # ---------------------------------------------------------------- inference
    def score_features(self, X: pd.DataFrame) -> np.ndarray:
        return predict_scores(self.model, X)

    def score(self, raw: pd.DataFrame) -> np.ndarray:
        return self.score_features(build_features(raw))

    def risk_level(self, score: float) -> str:
        medium = float(load_config().serving.risk_bands.medium)
        if score >= self.threshold:
            return "high"
        return "medium" if score >= medium * self.threshold else "low"

    def predict(self, raw: pd.DataFrame, explain: str = "flagged", top_k: int = 5) -> list[dict]:
        """explain: 'all' | 'flagged' | 'none'."""
        X = build_features(raw)
        scores = self.score_features(X)
        flagged = scores >= self.threshold
        explanations: dict[int, list[dict]] = {}
        rows = np.arange(len(X)) if explain == "all" else np.flatnonzero(flagged) if explain == "flagged" else []
        if len(rows):
            exp = self.explainer.explain(raw.iloc[rows], X.iloc[rows], top_k=top_k)
            explanations = dict(zip(rows.tolist(), exp))
        out = []
        for i, s in enumerate(scores):
            factors = explanations.get(i)
            out.append({
                "fraud_score": float(s),
                "is_fraud": bool(flagged[i]),
                "risk_level": self.risk_level(float(s)),
                "threshold": self.threshold,
                "explanation": None if factors is None else {
                    "summary": summarize(float(s), self.threshold, factors),
                    "method": self.explainer.method,
                    "top_factors": factors,
                },
            })
        return out

    # ---------------------------------------------------------------- persistence
    def metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "threshold": self.threshold,
            "trained_at": self.trained_at,
            "mlflow_run_id": self.mlflow_run_id,
            "explainer": self.explainer.method,
            "cost_model": self.cost_model.as_dict(),
            "metrics": self.metrics,
        }

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "bundle.joblib", compress=3)
        (directory / "metadata.json").write_text(json.dumps(self.metadata(), indent=2, default=float))
        return directory

    @staticmethod
    def load(directory: str | Path | None = None) -> "ModelBundle":
        directory = Path(directory) if directory else resolve(load_config().artifacts.champion_dir)
        bundle = joblib.load(directory / "bundle.joblib")
        _single_threaded_inference(bundle.model)  # runtime thread settings are not persisted
        return bundle


def promote(bundle: ModelBundle) -> Path:
    """Make `bundle` the serving champion; archive the previous champion."""
    cfg = load_config()
    champion, archive = resolve(cfg.artifacts.champion_dir), resolve(cfg.artifacts.archive_dir)
    if (champion / "metadata.json").exists():
        old = json.loads((champion / "metadata.json").read_text())
        dest = archive / str(old.get("version", "unknown"))
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(champion, dest)
    return bundle.save(champion)
