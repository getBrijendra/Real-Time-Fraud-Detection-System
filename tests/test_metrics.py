import numpy as np

from fraud.metrics import CostModel, cost_curve, evaluate, optimal_threshold, recall_at_fpr


def test_recall_at_fpr_perfect_separation():
    y = np.array([0] * 1000 + [1] * 10)
    s = np.r_[np.linspace(0, 0.5, 1000), np.linspace(0.9, 1, 10)]
    recall, thr = recall_at_fpr(y, s, 0.001)
    assert recall == 1.0 and thr >= 0.9


def test_recall_at_fpr_respects_budget():
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.01).astype(int)
    s = rng.random(20_000) + y * 0.5
    for target in (0.001, 0.01, 0.05):
        recall, thr = recall_at_fpr(y, s, target)
        fpr = ((s >= thr) & (y == 0)).sum() / (y == 0).sum()
        assert fpr <= target + 1e-12


def test_cost_curve_matches_brute_force():
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.05).astype(int)
    s = np.round(rng.random(500) + y * 0.3, 2)  # ties on purpose
    amounts = rng.exponential(100, 500)
    cost = CostModel(5, 10, 15)
    thresholds, costs = cost_curve(y, s, amounts, cost)
    for t, c in zip(thresholds[1:], costs[1:]):
        assert np.isclose(c, cost.breakdown(y, s >= t, amounts)["total_cost"])
    assert np.isclose(costs[0], cost.breakdown(y, np.zeros_like(y, bool), amounts)["total_cost"])
    best = optimal_threshold(y, s, amounts, cost)
    brute = min(cost.breakdown(y, s >= t, amounts)["total_cost"] for t in np.r_[np.unique(s), 2.0])
    assert np.isclose(cost.breakdown(y, s >= best, amounts)["total_cost"], brute)


def test_evaluate_fields():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.6, 0.4, 0.9]), np.array([10, 10, 100, 50]), 0.5, CostModel(5, 10, 15),
                 target_fprs=[0.5])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["cost_false_positives"] == 15 and m["cost_false_negatives"] == 115
    assert "recall_at_fpr_0.5" in m
