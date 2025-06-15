"""Evaluation metrics: PR-AUC, recall at fixed FPR, and the business cost of decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .config import Config, load_config


@dataclass(frozen=True)
class CostModel:
    """Per-transaction costs (USD).

    - every alert is reviewed by an analyst                -> review_cost
    - a false alert also blocks/annoys a good customer      -> + fp_friction_cost
    - a missed fraud loses the amount plus a chargeback fee -> amount + chargeback_fee
    """

    review_cost: float = 5.0
    fp_friction_cost: float = 10.0
    chargeback_fee: float = 15.0

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "CostModel":
        b = (cfg or load_config()).business
        return cls(float(b.review_cost), float(b.fp_friction_cost), float(b.chargeback_fee))

    @property
    def fp_cost(self) -> float:
        return self.review_cost + self.fp_friction_cost

    def breakdown(self, y: np.ndarray, flagged: np.ndarray, amounts: np.ndarray) -> dict:
        y, flagged, amounts = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool), np.asarray(amounts, float)
        tp, fp = flagged & y, flagged & ~y
        fn = ~flagged & y
        cost_tp = tp.sum() * self.review_cost
        cost_fp = fp.sum() * self.fp_cost
        cost_fn = (amounts[fn] + self.chargeback_fee).sum()
        total = cost_tp + cost_fp + cost_fn
        no_model = (amounts[y] + self.chargeback_fee).sum()  # every fraud goes through
        return {
            "cost_true_positives": float(cost_tp),
            "cost_false_positives": float(cost_fp),
            "cost_false_negatives": float(cost_fn),
            "total_cost": float(total),
            "cost_without_model": float(no_model),
            "savings_vs_no_model": float(no_model - total),
            "fraud_amount_caught": float(amounts[tp].sum()),
            "fraud_amount_missed": float(amounts[fn].sum()),
        }

    def as_dict(self) -> dict:
        return asdict(self)


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Highest recall achievable with false-positive rate <= target. Returns (recall, threshold)."""
    fpr, tpr, thr = roc_curve(y, scores)
    idx = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
    idx = max(idx, 0)
    return float(tpr[idx]), float(min(thr[idx], 1e12))


def cost_curve(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, cost: CostModel) -> tuple[np.ndarray, np.ndarray]:
    """Total cost for every distinct threshold, vectorised. Returns (thresholds, costs).

    Rule: flag when score >= threshold. thresholds[0] = +inf (flag nothing).
    """
    y = np.asarray(y).astype(float)
    order = np.argsort(-np.asarray(scores), kind="stable")
    s, ys, am = np.asarray(scores)[order], y[order], np.asarray(amounts, float)[order]
    fraud_loss = ys * (am + cost.chargeback_fee)
    k = np.arange(1, len(s) + 1)
    fp_k = np.cumsum(1 - ys)
    caught_loss_k = np.cumsum(fraud_loss)
    total_loss = fraud_loss.sum()
    costs_k = cost.review_cost * k + cost.fp_friction_cost * fp_k + (total_loss - caught_loss_k)
    # only cut between distinct scores so ties are flagged together
    boundary = np.r_[s[1:] != s[:-1], True]
    thresholds = np.r_[np.inf, s[boundary]]
    costs = np.r_[total_loss, costs_k[boundary]]
    return thresholds, costs


def optimal_threshold(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, cost: CostModel) -> float:
    thresholds, costs = cost_curve(y, scores, amounts, cost)
    best = int(np.argmin(costs))
    return float(thresholds[best]) if np.isfinite(thresholds[best]) else 1.0 + 1e-9


def evaluate(
    y: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost: CostModel,
    target_fprs: list[float] | None = None,
) -> dict:
    y = np.asarray(y).astype(int)
    target_fprs = target_fprs if target_fprs is not None else list(load_config().metrics.target_fprs)
    flagged = np.asarray(scores) >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alerts_per_10k": 1e4 * (tp + fp) / len(y),
        "fraud_rate": float(y.mean()),
    }
    for t in target_fprs:
        out[f"recall_at_fpr_{t:g}"] = recall_at_fpr(y, scores, t)[0]
    out.update(cost.breakdown(y, flagged, amounts))
    return out
