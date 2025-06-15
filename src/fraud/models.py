"""Model factory: logistic regression, random forest, XGBoost and an Isolation-Forest anomaly baseline,
each combined with a class-imbalance strategy (none / class weights / SMOTE / undersampling / scale_pos_weight).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import Config, load_config

log = logging.getLogger(__name__)


class AnomalyDetector(ClassifierMixin, BaseEstimator):
    """Unsupervised baseline: Isolation Forest fitted on legitimate transactions only (novelty detection).

    Exposes `predict_proba` so it plugs into the same evaluation/serving code. The "probability" is a
    monotonic squashing of the anomaly score centred on the 99.9th percentile of legitimate traffic,
    so it ranks transactions but is not a calibrated fraud probability.
    """

    def __init__(self, n_estimators: int = 300, max_samples: int = 512, random_state: int = 42, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        normal = X if y is None else X[np.asarray(y) == 0]
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators, max_samples=self.max_samples,
            random_state=self.random_state, n_jobs=self.n_jobs,
        ).fit(normal)
        raw = -self.forest_.score_samples(normal)  # higher = more anomalous
        self.center_ = float(np.quantile(raw, 0.999))
        self.scale_ = float(raw.std() or 1.0)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        raw = -self.forest_.score_samples(np.asarray(X, dtype=float))
        p = expit((raw - self.center_) / self.scale_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _logreg(imbalance: str, params: dict, mp: Config, seed: int):
    lr = LogisticRegression(
        C=params["C"], max_iter=params["max_iter"],
        class_weight="balanced" if imbalance == "class_weight" else None,
    )
    if imbalance == "smote":
        return ImbPipeline([("scale", StandardScaler()), ("smote", SMOTE(random_state=seed, **mp.smote)), ("clf", lr)])
    if imbalance == "undersample":
        return ImbPipeline([
            ("scale", StandardScaler()), ("under", RandomUnderSampler(random_state=seed, **mp.undersample)), ("clf", lr),
        ])
    return Pipeline([("scale", StandardScaler()), ("clf", lr)])


def build_model(name: str, spec: dict, y_train: np.ndarray, cfg: Config | None = None) -> Any:
    cfg = cfg or load_config()
    mp, seed = cfg.model_params, cfg.seed
    kind, imbalance = spec["type"], spec.get("imbalance", "none")

    if kind == "logreg":
        return _logreg(imbalance, mp.logreg, mp, seed)
    if kind == "random_forest":
        return RandomForestClassifier(
            **mp.random_forest, n_jobs=-1, random_state=seed,
            class_weight="balanced_subsample" if imbalance == "class_weight" else None,
        )
    if kind == "xgboost":
        p = dict(mp.xgboost)
        power = p.pop("spw_power", 1.0)
        neg, pos = np.sum(y_train == 0), np.sum(y_train == 1)
        spw = float((neg / pos) ** power) if imbalance == "scale_pos_weight" else 1.0
        p.setdefault("eval_metric", "logloss")
        return XGBClassifier(**p, scale_pos_weight=spw, tree_method="hist", random_state=seed, n_jobs=-1)
    if kind == "isolation_forest":
        return AnomalyDetector(**mp.isolation_forest, random_state=seed)
    raise ValueError(f"Unknown model type {kind!r} for {name}")


def fit_model(model: Any, X: pd.DataFrame, y: np.ndarray) -> Any:
    """Fit a model. XGBoost uses early stopping on the latest 15% of the training window
    (chronological, so the validation set stays untouched), then refits on all training data."""
    if isinstance(model, XGBClassifier) and model.get_params().get("early_stopping_rounds"):
        cut = int(len(X) * 0.85)
        model.fit(X.iloc[:cut], y[:cut], eval_set=[(X.iloc[cut:], y[cut:])], verbose=False)
        best = int(model.best_iteration) + 1
        log.info("XGBoost early stopping: best_iteration=%d", best)
        model.set_params(n_estimators=max(best, 50), early_stopping_rounds=None)
    model.fit(X, y)
    _single_threaded_inference(model)
    return model


def _single_threaded_inference(model: Any) -> None:
    """Parallel training is fast, but n_jobs=-1 at inference spins up a thread pool on every call
    (~10-25 ms for a single transaction). Real-time serving scores 1..few hundred rows -> use 1 thread."""
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    for est in (final, getattr(final, "forest_", None)):
        if est is not None and hasattr(est, "n_jobs"):
            est.n_jobs = 1
    if isinstance(final, XGBClassifier):
        final.get_booster().set_param({"nthread": 1})


def predict_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Fraud probability. XGBoost goes through `inplace_predict` on a numpy array, which skips the
    DataFrame/DMatrix conversion of `predict_proba` (~6ms -> <1ms for a single transaction)."""
    if isinstance(model, XGBClassifier):
        return model.get_booster().inplace_predict(np.ascontiguousarray(X.to_numpy(dtype=np.float32)),
                                                   validate_features=False)
    return model.predict_proba(X)[:, 1]
