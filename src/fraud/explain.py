"""Per-transaction explanations: why was this transaction scored as risky?

- XGBoost: exact TreeSHAP contributions computed natively by XGBoost (`pred_contribs=True`), log-odds units.
- Logistic regression: coefficient x standardised feature value, log-odds units.
- Other models (random forest, isolation forest): standardised deviation from legitimate traffic
  weighted by global feature importance (a fast, model-agnostic approximation).

Engineered columns are grouped back into business concepts (log_amount -> Amount, hour_sin/cos -> hour_of_day)
and every factor gets a plain-English reason that compares the value to the typical legitimate range.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .features import FEATURE_GROUPS, MODEL_FEATURES, hour_of_day


def _final_estimator(model: Any) -> Any:
    return model.steps[-1][1] if hasattr(model, "steps") else model


class Explainer:
    def __init__(self, model: Any, X_train: pd.DataFrame, raw_train: pd.DataFrame, y_train: np.ndarray):
        self.model = model
        legit_feats = X_train[np.asarray(y_train) == 0]
        self.mean_ = legit_feats.mean().to_numpy()
        self.std_ = legit_feats.std().replace(0, 1).to_numpy()

        legit_raw = raw_train[np.asarray(y_train) == 0]
        ranges = {c: (float(legit_feats[c].quantile(0.05)), float(legit_feats[c].quantile(0.95))) for c in MODEL_FEATURES}
        ranges["Amount"] = (float(legit_raw.Amount.quantile(0.05)), float(legit_raw.Amount.quantile(0.95)))
        self.typical_range_ = ranges

        final = _final_estimator(model)
        if isinstance(final, XGBClassifier):
            self.method = "tree_shap"
        elif isinstance(final, LogisticRegression):
            self.method = "linear"
        else:
            self.method = "deviation"
            imp = getattr(final, "feature_importances_", None)
            self.importance_ = np.asarray(imp) if imp is not None else np.ones(len(MODEL_FEATURES))
            self.importance_ = self.importance_ / self.importance_.max()

    # ------------------------------------------------------------------ contributions
    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """(n, n_features) signed contributions; positive = pushes towards fraud."""
        if self.method == "tree_shap":
            booster = _final_estimator(self.model).get_booster()
            dm = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=list(X.columns), nthread=1)
            contrib = booster.predict(dm, pred_contribs=True)
            return contrib[:, :-1]  # drop bias column
        if self.method == "linear":
            pre = self.model[:-1].transform(X) if hasattr(self.model, "steps") else X.to_numpy()
            return np.asarray(pre) * _final_estimator(self.model).coef_[0]
        z = (X.to_numpy() - self.mean_) / self.std_
        return np.abs(z) * self.importance_  # deviation magnitude, always "towards anomalous"

    # ------------------------------------------------------------------ public API
    def explain(self, raw: pd.DataFrame, X: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
        contrib = self.contributions(X)
        grouped_names: list[str] = []
        for f in MODEL_FEATURES:
            g = FEATURE_GROUPS.get(f, f)
            if g not in grouped_names:
                grouped_names.append(g)
        idx = {g: i for i, g in enumerate(grouped_names)}
        grouped = np.zeros((len(X), len(grouped_names)))
        for j, f in enumerate(MODEL_FEATURES):
            grouped[:, idx[FEATURE_GROUPS.get(f, f)]] += contrib[:, j]

        hours = hour_of_day(raw["Time"].to_numpy())
        amounts = raw["Amount"].to_numpy(dtype=float)
        values = X.to_numpy()
        col = {f: j for j, f in enumerate(X.columns)}
        results = []
        for r in range(len(X)):
            order = np.argsort(-np.abs(grouped[r]))[:top_k]
            factors = []
            for c in order:
                name, val = grouped_names[c], float(grouped[r, c])
                if name == "Amount":
                    value = float(amounts[r])
                elif name == "hour_of_day":
                    value = round(float(hours[r]), 2)
                else:
                    value = float(values[r, col[name]])
                factors.append({
                    "feature": name,
                    "value": round(value, 4),
                    "contribution": round(val, 4),
                    "direction": "increases_risk" if val > 0 else "decreases_risk",
                    "typical_range": [round(v, 3) for v in self.typical_range_.get(name, (np.nan, np.nan))]
                    if name in self.typical_range_ else None,
                    "reason": self._reason(name, value, val),
                })
            results.append(factors)
        return results

    def _reason(self, name: str, value: float, contribution: float) -> str:
        effect = "raises" if contribution > 0 else "lowers"
        if name == "Amount":
            lo, hi = self.typical_range_["Amount"]
            where = "above" if value > hi else "below" if value < lo else "within"
            return f"Amount ${value:,.2f} is {where} the typical legitimate range (${lo:,.2f}-${hi:,.2f}); {effect} risk"
        if name == "hour_of_day":
            night = value < 6
            return f"Transaction at hour {value:.1f}{' (night-time, when fraud is relatively more common)' if night else ''}; {effect} risk"
        lo, hi = self.typical_range_[name]
        if value < lo:
            where = f"unusually low ({value:.2f} vs typical {lo:.2f}..{hi:.2f})"
        elif value > hi:
            where = f"unusually high ({value:.2f} vs typical {lo:.2f}..{hi:.2f})"
        else:
            where = f"in the normal range ({value:.2f})"
        return f"Behavioural signal {name} (anonymised PCA component) is {where}; {effect} risk"


def summarize(score: float, threshold: float, factors: list[dict]) -> str:
    drivers = [f["feature"] for f in factors if f["direction"] == "increases_risk"][:3]
    if score >= threshold:
        head = f"Flagged: fraud score {score:.3f} >= alert threshold {threshold:.3f}."
    else:
        head = f"Not flagged: fraud score {score:.3f} < alert threshold {threshold:.3f}."
    return head + (f" Main risk drivers: {', '.join(drivers)}." if drivers else " No strong risk drivers.")
