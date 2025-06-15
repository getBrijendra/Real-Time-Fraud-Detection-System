"""Business-cost analysis of the champion on the test window.

1. Operating points: what each threshold choice (cost-optimal, fixed FPR targets, 0.5) means in alerts,
   caught fraud, false positives and dollars.
2. Sensitivity: how the cost-optimal threshold and total cost change as a false positive gets more expensive.
   (Thresholds are re-optimised on validation for each cost assumption, then applied to test.)

    python scripts/business_cost.py
"""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from fraud.bundle import ModelBundle
from fraud.config import load_config, resolve
from fraud.data import TARGET, chronological_split, load_raw
from fraud.metrics import evaluate, optimal_threshold, recall_at_fpr

PER = 100_000  # normalise counts/costs per 100k transactions


def main() -> None:
    cfg = load_config()
    bundle = ModelBundle.load()
    cost = bundle.cost_model
    _, val, test = chronological_split(load_raw())
    s_va, s_te = bundle.score(val), bundle.score(test)
    y_va, y_te, a_va, a_te = val[TARGET].to_numpy(), test[TARGET].to_numpy(), val.Amount.to_numpy(), test.Amount.to_numpy()
    k = PER / len(test)

    points = {"cost-optimal (deployed)": bundle.threshold, "naive 0.5": 0.5}
    for fpr in cfg.metrics.target_fprs:
        points[f"FPR <= {fpr:.1%} (set on val)"] = recall_at_fpr(y_va, s_va, fpr)[1]
    op_rows = []
    for name, thr in points.items():
        m = evaluate(y_te, s_te, a_te, thr, cost)
        op_rows.append({"operating point": name, "threshold": thr, "recall": m["recall"], "precision": m["precision"],
                        "FPR": m["fpr"], "alerts/100k": (m["tp"] + m["fp"]) * k, "false pos/100k": m["fp"] * k,
                        "FP cost/100k $": m["cost_false_positives"] * k, "fraud missed/100k $": m["cost_false_negatives"] * k,
                        "total cost/100k $": m["total_cost"] * k, "savings/100k $": m["savings_vs_no_model"] * k})

    sens_rows = []
    for friction in (0, 5, 10, 25, 50, 100, 250):
        c = replace(cost, fp_friction_cost=float(friction))
        thr = optimal_threshold(y_va, s_va, a_va, c)
        m = evaluate(y_te, s_te, a_te, thr, c)
        sens_rows.append({"cost per false positive $": c.fp_cost, "optimal threshold": thr, "recall": m["recall"],
                          "precision": m["precision"], "false pos/100k": m["fp"] * k,
                          "FP cost/100k $": m["cost_false_positives"] * k, "total cost/100k $": m["total_cost"] * k,
                          "FP share of cost": m["cost_false_positives"] / m["total_cost"] if m["total_cost"] else 0})

    op, sens = pd.DataFrame(op_rows), pd.DataFrame(sens_rows)
    pd.set_option("display.width", 250)
    fmt = lambda v: f"{v:,.4f}" if abs(v) < 1 else f"{v:,.1f}"  # noqa: E731
    print(f"Champion {bundle.model_name} {bundle.version} - test window, normalised per {PER:,} transactions\n")
    print("Operating points\n" + op.to_string(index=False, float_format=fmt))
    print("\nSensitivity to the cost of a false positive\n" + sens.to_string(index=False, float_format=fmt))

    out = resolve(cfg.artifacts.reports_dir)
    (out / "business_cost.json").write_text(json.dumps({"operating_points": op_rows, "sensitivity": sens_rows},
                                                       indent=2, default=float))
    md = [f"# Business cost analysis - {bundle.model_name}\n",
          f"Test window, per {PER:,} transactions. Base cost model: {cost.as_dict()}.\n", "## Operating points\n",
          _markdown(op, fmt), "\n## Sensitivity to the cost of a false positive\n", _markdown(sens, fmt)]
    (out / "business_cost.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")


def _markdown(df: pd.DataFrame, fmt) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
