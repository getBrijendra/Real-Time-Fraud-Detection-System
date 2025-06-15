"""Train and compare all baselines, select a champion, and publish it for serving.

Protocol
  1. Chronological split: train (first 60%) -> validation (next 20%) -> test (last 20%).
  2. Every experiment (model family x imbalance strategy) is fitted on train.
  3. Per model, the alert threshold minimising *business cost* is chosen on validation.
  4. Champion = highest validation PR-AUC. Everything is reported on the untouched test window.
  5. Each run is tracked in MLflow; the champion is registered (alias @champion) and saved to
     artifacts/champion for the API. The test window is written out as the live stream source.

    python -m fraud.train [--models xgboost logreg_balanced ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .bundle import ModelBundle, promote  # noqa: E402
from .config import load_config, resolve, setup_logging  # noqa: E402
from .data import TARGET, chronological_split, load_raw, write_stream_source  # noqa: E402
from .drift import ReferenceProfile  # noqa: E402
from .explain import Explainer  # noqa: E402
from .features import build_features  # noqa: E402
from .metrics import CostModel, cost_curve, evaluate, optimal_threshold  # noqa: E402
from .models import build_model, fit_model, predict_scores  # noqa: E402
from .tracking import init_mlflow, log_sklearn_model, register_champion  # noqa: E402

log = logging.getLogger("fraud.train")


def _latency_ms(model, X: pd.DataFrame, n: int = 300) -> dict:
    """Single-transaction in-process inference latency (same scoring path as the API)."""
    rows = [X.iloc[[i]] for i in np.random.default_rng(0).integers(0, len(X), n)]
    predict_scores(model, rows[0])  # warm-up
    t = []
    for r in rows:
        t0 = time.perf_counter()
        predict_scores(model, r)
        t.append((time.perf_counter() - t0) * 1e3)
    return {"latency_p50_ms": float(np.percentile(t, 50)), "latency_p99_ms": float(np.percentile(t, 99))}


def build_bundle(name, model, X_tr, raw_tr, y_tr, X_ref, ref_scores, threshold, cost, metrics, version) -> ModelBundle:
    """X_ref/ref_scores: the most recent pre-deployment window (validation), used as the drift reference."""
    cfg = load_config().monitoring
    reference = ReferenceProfile.build(
        X_ref, ref_scores, bins=cfg.psi_bins, warn=cfg.psi_warn, alert=cfg.psi_alert,
        min_features=cfg.drift_min_features, exclude=list(cfg.exclude_features),
    )
    return ModelBundle(
        model=model, model_name=name, version=version, threshold=threshold,
        explainer=Explainer(model, X_tr, raw_tr, y_tr), reference=reference, cost_model=cost, metrics=metrics,
    )


def plot_pr_curves(curves: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y, s, ap) in sorted(curves.items(), key=lambda kv: -kv[1][2]):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{name} (AP={ap:.3f})", lw=1.6)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test window)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cost_curve(y, s, amounts, cost: CostModel, threshold: float, title: str, path) -> None:
    """Total cost and its components (false-positive cost vs missed-fraud cost) across thresholds."""
    _, costs = cost_curve(y, s, amounts, cost)
    grid = np.unique(np.r_[np.geomspace(1e-4, 1, 120), threshold])
    parts = [cost.breakdown(y, s >= t, amounts) for t in grid]
    total = np.array([p["total_cost"] for p in parts])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, total, lw=2, label="total cost")
    ax.plot(grid, [p["cost_false_positives"] for p in parts], lw=1.3, label="false-positive cost (review + friction)")
    ax.plot(grid, [p["cost_false_negatives"] for p in parts], lw=1.3, label="missed-fraud cost (amount + fee)")
    ax.axhline(costs[0], ls=":", c="grey", label=f"no model: ${costs[0]:,.0f}")
    ax.axvline(threshold, ls="--", c="crimson",
               label=f"chosen threshold {threshold:.3f}: ${cost.breakdown(y, s >= threshold, amounts)['total_cost']:,.0f}")
    ax.set(xscale="log", yscale="log", xlabel="alert threshold (log)", ylabel="cost on test window, USD (log)",
           title=title, ylim=(10, None))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_markdown_report(results: list[dict], champion: str, cost: CostModel, path) -> None:
    fprs = load_config().metrics.target_fprs
    cols = ["model", "imbalance", "val_pr_auc", "pr_auc", "roc_auc"] + [f"recall_at_fpr_{t:g}" for t in fprs] + [
        "threshold", "precision", "recall", "fp", "fn", "cost_false_positives", "total_cost", "savings_vs_no_model",
        "latency_p50_ms", "train_seconds"]
    df = pd.DataFrame(results)[cols].sort_values("val_pr_auc", ascending=False)
    lines = [
        "# Model comparison (test window = last 20% of time, never seen in training or threshold tuning)\n",
        f"Champion (best validation PR-AUC within the {load_config().selection.max_latency_p50_ms} ms p50 "
        f"single-transaction latency budget): **{champion}**\n",
        f"Cost model: review ${cost.review_cost:.0f}/alert, false-positive friction ${cost.fp_friction_cost:.0f}, "
        f"missed fraud = amount + ${cost.chargeback_fee:.0f} chargeback fee. Thresholds minimise validation cost.\n",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "threshold":
                cells.append(f"{v:.6g}")
            elif isinstance(v, float):
                cells.append(f"${v:,.0f}" if "cost" in c or "savings" in c else f"{v:.4f}" if abs(v) < 10 else f"{v:,.1f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(selected: list[str] | None = None, register: bool = True) -> dict:
    setup_logging()
    cfg = load_config()
    reports = resolve(cfg.artifacts.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    cost = CostModel.from_config(cfg)

    raw = load_raw()
    train, val, test = chronological_split(raw)
    write_stream_source(test)
    X_tr, X_va, X_te = build_features(train), build_features(val), build_features(test)
    y_tr, y_va, y_te = train[TARGET].to_numpy(), val[TARGET].to_numpy(), test[TARGET].to_numpy()

    init_mlflow()
    specs = {k: v for k, v in cfg.models.items() if not selected or k in selected}
    results, curves, fitted = [], {}, {}

    for name, spec in specs.items():
        log.info("=== %s (%s, imbalance=%s) ===", name, spec["type"], spec.get("imbalance"))
        with mlflow.start_run(run_name=name) as run:
            model = build_model(name, spec, y_tr, cfg)
            t0 = time.perf_counter()
            fit_model(model, X_tr, y_tr)
            train_s = time.perf_counter() - t0

            s_va, s_te = predict_scores(model, X_va), predict_scores(model, X_te)
            thr = optimal_threshold(y_va, s_va, val.Amount.to_numpy(), cost)
            val_m = evaluate(y_va, s_va, val.Amount.to_numpy(), thr, cost)
            test_m = evaluate(y_te, s_te, test.Amount.to_numpy(), thr, cost)
            lat = _latency_ms(model, X_te)

            mlflow.set_tags({"model_type": spec["type"], "imbalance": spec.get("imbalance", "none")})
            mlflow.log_params({"model_type": spec["type"], "imbalance": spec.get("imbalance", "none"),
                               **{f"cost_{k}": v for k, v in cost.as_dict().items()},
                               **{k: v for k, v in model.get_params().items()
                                  if isinstance(v, (int, float, str, bool)) and len(str(v)) < 250}})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float))})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items() if isinstance(v, (int, float))})
            mlflow.log_metrics({**lat, "train_seconds": train_s})

            row = {"model": name, "imbalance": spec.get("imbalance", "none"), "val_pr_auc": val_m["pr_auc"],
                   **test_m, **lat, "train_seconds": train_s, "run_id": run.info.run_id}
            results.append(row)
            curves[name] = (y_te, s_te, test_m["pr_auc"])
            fitted[name] = (model, s_va, thr, val_m, test_m, run.info.run_id)
            log.info("%s: val PR-AUC=%.4f | test PR-AUC=%.4f recall@0.1%%FPR=%.3f cost=$%.0f (no model $%.0f) %.1fs",
                     name, val_m["pr_auc"], test_m["pr_auc"], test_m["recall_at_fpr_0.001"],
                     test_m["total_cost"], test_m["cost_without_model"], train_s)

    budget = float(cfg.selection.max_latency_p50_ms)
    eligible = [r for r in results if r["latency_p50_ms"] <= budget] or results
    for r in results:
        if r not in eligible:
            log.info("%s excluded from champion selection: p50 latency %.1fms > budget %.1fms",
                     r["model"], r["latency_p50_ms"], budget)
    champion = max(eligible, key=lambda r: r[cfg.selection.metric])["model"]
    model, s_va, thr, val_m, test_m, run_id = fitted[champion]
    log.info("Champion: %s (best %s within latency budget)", champion, cfg.selection.metric)

    # ---------------- reports
    plot_pr_curves(curves, reports / "pr_curves.png")
    plot_cost_curve(y_te, curves[champion][1], test.Amount.to_numpy(), cost, thr,
                    f"Business cost vs threshold - {champion} (test)", reports / "cost_curve.png")
    write_markdown_report(results, champion, cost, reports / "model_comparison.md")
    (reports / "metrics.json").write_text(json.dumps({"champion": champion, "results": results}, indent=2, default=float))

    # ---------------- champion bundle + registry
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    bundle = build_bundle(champion, model, X_tr, train, y_tr, X_va, s_va, thr, cost,
                          {"validation": val_m, "test": test_m}, version)
    with mlflow.start_run(run_id=run_id):
        for f in ("pr_curves.png", "cost_curve.png", "model_comparison.md"):
            mlflow.log_artifact(str(reports / f), artifact_path="reports")
        if register:
            model_uri = log_sklearn_model(model, X_te.head(200))
            reg_version = register_champion(model_uri, {"threshold": thr, "bundle_version": version})
            bundle.version = f"{version}-v{reg_version}"
        bundle.mlflow_run_id = run_id
        bundle_dir = promote(bundle)
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
    log.info("Champion bundle %s written to %s", bundle.version, bundle_dir)
    print((reports / "model_comparison.md").read_text(encoding="utf-8"))
    return {"champion": champion, "version": bundle.version, "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="subset of experiment names from config.yaml")
    ap.add_argument("--no-register", action="store_true", help="skip MLflow model registry")
    args = ap.parse_args()
    main(args.models, register=not args.no_register)
