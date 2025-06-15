"""Live monitoring: Prometheus metrics, rolling drift detection and label-based performance/cost."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, Info

from .drift import ReferenceProfile
from .metrics import CostModel

LATENCY_BUCKETS = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)

REQUESTS = Counter("fraud_api_requests_total", "API requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("fraud_api_request_latency_seconds", "End-to-end handler latency", ["endpoint"],
                            buckets=LATENCY_BUCKETS)
INFERENCE_LATENCY = Histogram("fraud_inference_latency_seconds", "Model scoring + explanation latency per call",
                              buckets=LATENCY_BUCKETS)
PREDICTIONS = Counter("fraud_predictions_total", "Scored transactions", ["decision"])
SCORES = Histogram("fraud_score", "Distribution of fraud scores",
                   buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0))
FEATURE_PSI = Gauge("fraud_feature_psi", "PSI of a feature vs training reference", ["feature"])
SCORE_PSI = Gauge("fraud_score_psi", "PSI of the score distribution vs reference")
DRIFT_DETECTED = Gauge("fraud_drift_detected", "1 if dataset drift is detected in the live window")
DRIFTED_SHARE = Gauge("fraud_drifted_feature_share", "Share of features at PSI alert level")
MODEL_INFO = Info("fraud_model", "Serving model")
LABELS = Counter("fraud_feedback_labels_total", "Delayed labels received", ["outcome"])  # tp/fp/fn/tn
BUSINESS_COST = Counter("fraud_business_cost_usd_total", "Realised business cost from labeled outcomes", ["outcome"])
LIVE_PRECISION = Gauge("fraud_live_precision", "Precision over labeled feedback since start")
LIVE_RECALL = Gauge("fraud_live_recall", "Recall over labeled feedback since start")


class DriftMonitor:
    """Keeps a rolling window of live feature vectors + scores and periodically compares it to the reference."""

    def __init__(self, reference: ReferenceProfile, window: int = 2000, min_samples: int = 500, check_every: int = 250):
        self.reference = reference
        self.window, self.min_samples, self.check_every = window, min_samples, check_every
        self._X: deque = deque(maxlen=window)
        self._s: deque = deque(maxlen=window)
        self._since_check = 0
        self._lock = threading.Lock()
        self.last_report: dict | None = None

    def add(self, X: pd.DataFrame, scores: np.ndarray) -> None:
        with self._lock:
            self._X.extend(X[self.reference.features].to_numpy())
            self._s.extend(np.asarray(scores).tolist())
            self._since_check += len(X)
            due = self._since_check >= self.check_every and len(self._X) >= self.min_samples
        if due:
            self.check()

    def check(self) -> dict:
        with self._lock:
            n = len(self._X)
            if n < self.min_samples:
                return {"status": "insufficient_data", "n_samples": n, "min_samples": self.min_samples}
            X = pd.DataFrame(np.asarray(self._X), columns=self.reference.features)
            s = np.asarray(self._s)
            self._since_check = 0
        rep = self.reference.report(X, s)
        for f, r in rep["features"].items():
            FEATURE_PSI.labels(feature=f).set(r["psi"])
        SCORE_PSI.set(rep.get("score", {}).get("psi", 0.0))
        DRIFT_DETECTED.set(int(rep["dataset_drift"]))
        DRIFTED_SHARE.set(rep["share_features_drifted"])
        rep["status"] = "drift" if rep["dataset_drift"] else "ok"
        self.last_report = rep
        return rep

    def reset(self, reference: ReferenceProfile | None = None) -> None:
        with self._lock:
            if reference is not None:
                self.reference = reference
            self._X.clear()
            self._s.clear()
            self._since_check = 0
            self.last_report = None


class PerformanceTracker:
    """Confusion counts and realised business cost from delayed labels."""

    def __init__(self, cost: CostModel):
        self.cost = cost
        self.counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.cost_by_outcome = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        self._lock = threading.Lock()

    def update(self, matched: list[dict]) -> None:
        with self._lock:
            for m in matched:
                flagged, label = bool(m["flagged"]), bool(m["label"])
                outcome = ("tp" if label else "fp") if flagged else ("fn" if label else "tn")
                c = {"tp": self.cost.review_cost, "fp": self.cost.fp_cost,
                     "fn": m["amount"] + self.cost.chargeback_fee, "tn": 0.0}[outcome]
                self.counts[outcome] += 1
                self.cost_by_outcome[outcome] += c
                LABELS.labels(outcome=outcome).inc()
                BUSINESS_COST.labels(outcome=outcome).inc(c)
            tp, fp, fn = self.counts["tp"], self.counts["fp"], self.counts["fn"]
            LIVE_PRECISION.set(tp / (tp + fp) if tp + fp else 0)
            LIVE_RECALL.set(tp / (tp + fn) if tp + fn else 0)

    def summary(self) -> dict:
        with self._lock:
            tp, fp, fn, tn = (self.counts[k] for k in ("tp", "fp", "fn", "tn"))
            n = tp + fp + fn + tn
            return {
                "labeled": n,
                "confusion": dict(self.counts),
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                "cost_usd": {k: round(v, 2) for k, v in self.cost_by_outcome.items()},
                "total_cost_usd": round(sum(self.cost_by_outcome.values()), 2),
                "false_positive_cost_share": (self.cost_by_outcome["fp"] / sum(self.cost_by_outcome.values()))
                if sum(self.cost_by_outcome.values()) else None,
            }
