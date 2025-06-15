"""Data / prediction drift detection against a training reference profile.

- Population Stability Index (PSI) per feature and for the model score, on reference-decile bins.
  Rule of thumb: PSI < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
- Two-sample Kolmogorov-Smirnov test on the score distribution (reported alongside PSI).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-4


def _edges(values: np.ndarray, bins: int) -> np.ndarray:
    qs = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(qs)


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return np.clip(counts / max(len(values), 1), EPS, None)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


@dataclass
class ReferenceProfile:
    features: list[str]
    edges: dict[str, np.ndarray]
    expected: dict[str, np.ndarray]
    score_edges: np.ndarray
    score_expected: np.ndarray
    score_sample: np.ndarray
    warn: float = 0.10
    alert: float = 0.25
    min_features: int = 1
    meta: dict = field(default_factory=dict)

    @classmethod
    def build(cls, X: pd.DataFrame, scores: np.ndarray, bins: int = 10, warn: float = 0.10,
              alert: float = 0.25, min_features: int = 1, exclude: list[str] | None = None,
              seed: int = 42) -> "ReferenceProfile":
        X = X.drop(columns=[c for c in (exclude or []) if c in X.columns])
        edges, expected = {}, {}
        for c in X.columns:
            v = X[c].to_numpy()
            edges[c] = _edges(v, bins)
            expected[c] = _proportions(v, edges[c])
        # Scores are extremely skewed (most near 0); quantile bins still give equally populated buckets.
        s_edges = _edges(scores, bins)
        rng = np.random.default_rng(seed)
        sample = rng.choice(scores, size=min(len(scores), 5000), replace=False)
        return cls(list(X.columns), edges, expected, s_edges, _proportions(scores, s_edges), sample,
                   warn, alert, min_features, {"n_reference": int(len(X))})

    def _status(self, value: float) -> str:
        return "alert" if value >= self.alert else "warn" if value >= self.warn else "ok"

    def report(self, X: pd.DataFrame, scores: np.ndarray | None = None) -> dict:
        per_feature = {}
        for c in self.features:
            v = psi(self.expected[c], _proportions(X[c].to_numpy(), self.edges[c]))
            per_feature[c] = {"psi": round(v, 4), "status": self._status(v)}
        n_alert = sum(1 for r in per_feature.values() if r["status"] == "alert")
        out: dict = {
            "n_samples": int(len(X)),
            "features": per_feature,
            "n_features_drifted": n_alert,
            "share_features_drifted": n_alert / len(self.features),
            "top_drifted": sorted(per_feature, key=lambda c: -per_feature[c]["psi"])[:5],
        }
        score_drift = False
        if scores is not None and len(scores):
            s_psi = psi(self.score_expected, _proportions(np.asarray(scores), self.score_edges))
            ks = ks_2samp(self.score_sample, scores)
            out["score"] = {"psi": round(s_psi, 4), "status": self._status(s_psi),
                            "ks_statistic": round(float(ks.statistic), 4), "ks_pvalue": float(ks.pvalue)}
            score_drift = s_psi >= self.alert
        out["dataset_drift"] = bool(n_alert >= self.min_features or score_drift)
        return out
