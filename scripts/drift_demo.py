"""Offline drift-detection demo on the held-out stream.

Compares several live windows with the training reference:
  * clean windows from the future test period (natural temporal drift),
  * the same windows with injected covariate drift of increasing strength,
and shows PSI-based detection next to the actual effect on model quality (labels are known here).

    python scripts/drift_demo.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.bundle import ModelBundle
from fraud.config import load_config, resolve
from fraud.data import RAW_FEATURES, TARGET
from fraud.features import build_features
from fraud.metrics import evaluate
from fraud.streaming.producer import inject_drift

WINDOW = 10_000


def main() -> None:
    cfg = load_config()
    bundle = ModelBundle.load()
    stream = pd.read_parquet(resolve(cfg.data.stream_path))
    scenarios = []
    windows = [stream.iloc[i: i + WINDOW] for i in range(0, len(stream) - WINDOW + 1, WINDOW)]
    for i, w in enumerate(windows[:3]):
        scenarios.append((f"clean window {i + 1}", w[RAW_FEATURES], w[TARGET].to_numpy()))
    base = pd.concat(windows[:3])
    for kind in ("amount", "pca", "both"):
        for strength in (0.25, 0.5, 1.0):
            scenarios.append((f"{kind} drift x{strength}", inject_drift(base[RAW_FEATURES], kind, strength),
                              base[TARGET].to_numpy()))

    rows = []
    for name, raw, y in scenarios:
        X = build_features(raw)
        s = bundle.score_features(X)
        rep = bundle.reference.report(X, s)
        m = evaluate(y, s, raw.Amount.to_numpy(), bundle.threshold, bundle.cost_model)
        rows.append({
            "scenario": name, "n": len(raw), "dataset_drift": rep["dataset_drift"],
            "share_features_drifted": rep["share_features_drifted"],
            "max_feature_psi": max(r["psi"] for r in rep["features"].values()),
            "top_drifted": ", ".join(f"{f}={rep['features'][f]['psi']:.2f}" for f in rep["top_drifted"][:3]),
            "score_psi": rep["score"]["psi"], "score_ks": rep["score"]["ks_statistic"],
            "pr_auc": average_precision_score(y, s) if y.sum() else np.nan,
            "recall": m["recall"], "precision": m["precision"], "alerts_per_10k": m["alerts_per_10k"],
            "fp_cost": m["cost_false_positives"], "total_cost": m["total_cost"],
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out_dir = resolve(cfg.artifacts.reports_dir)
    (out_dir / "drift_report.json").write_text(json.dumps(rows, indent=2, default=float))
    lines = ["# Drift detection demo\n",
             f"Reference = validation window (most recent pre-deployment data). PSI warn >= {bundle.reference.warn}, "
             f"alert >= {bundle.reference.alert}; dataset drift = >= {bundle.reference.min_features} feature(s) at alert "
             f"level or score PSI alert. Calendar features excluded. Clean windows = untouched future traffic.\n",
             "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    (out_dir / "drift_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {out_dir / 'drift_report.md'}")


if __name__ == "__main__":
    main()
