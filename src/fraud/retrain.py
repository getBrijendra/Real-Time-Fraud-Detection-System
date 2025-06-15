"""Champion/challenger retraining driven by drift and delayed labels.

Triggers (any of): --force | dataset drift in the most recent live window | enough new labeled feedback.

    python -m fraud.retrain --check            # only evaluate the triggers
    python -m fraud.retrain                    # retrain if triggered, promote if the challenger wins
    python -m fraud.retrain --force --reload-url http://localhost:8000

Procedure
  1. Labeled feedback from the prediction store is split chronologically: older part -> extra training
     data, most recent `holdout_frac` -> holdout that represents "current" traffic.
  2. Challenger = same model family as the champion, refit on original train+validation + feedback-train.
  3. Champion and challenger are scored on the same holdout (PR-AUC primary, business cost secondary).
  4. If the challenger is better by >= min_pr_auc_gain it is registered in MLflow (@champion),
     written to artifacts/champion (old one archived) and the API is asked to hot-reload.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
import httpx

from .bundle import ModelBundle, promote
from .config import load_config, resolve, setup_logging
from .data import TARGET, chronological_split, load_raw
from .features import build_features
from .metrics import CostModel, evaluate, optimal_threshold
from .models import build_model, fit_model, predict_scores
from .store import PredictionStore
from .tracking import init_mlflow, log_sklearn_model, register_champion
from .train import build_bundle

log = logging.getLogger("fraud.retrain")


def _state_path():
    return resolve(load_config().artifacts.champion_dir).parent / "retrain_state.json"


def _load_state() -> dict:
    p = _state_path()
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(state: dict) -> None:
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, indent=2))


def check_triggers(bundle: ModelBundle, store: PredictionStore) -> dict:
    cfg = load_config()
    since = _load_state().get("last_retrain_at")
    counts = store.counts(since=since)
    recent = store.recent(cfg.monitoring.window_size)
    drift = None
    if len(recent) >= cfg.monitoring.min_samples:
        raw = pd.DataFrame([json.loads(p) for p in recent.payload])
        X = build_features(raw)
        # re-score: stored scores may come from a previous model version
        drift = bundle.reference.report(X, bundle.score_features(X))
    # only labels that arrived after the last retraining count as "new"
    enough_labels = (counts["new_labels"] >= cfg.retraining.min_new_labels
                     and counts["new_label_frauds"] >= cfg.retraining.min_new_frauds)
    return {
        "store": counts,
        "drift_detected": bool(drift and drift["dataset_drift"]),
        "drift_top_features": drift and {f: drift["features"][f]["psi"] for f in drift["top_drifted"]},
        "score_psi": drift and drift.get("score", {}).get("psi"),
        "enough_labels": enough_labels,
        "should_retrain": bool((drift and drift["dataset_drift"]) or enough_labels),
    }


def retrain(force: bool = False, reload_url: str | None = None, min_gain: float | None = None) -> dict:
    setup_logging()
    cfg = load_config()
    started = time.time()
    champion = ModelBundle.load()
    store = PredictionStore()
    triggers = check_triggers(champion, store)
    log.info("Triggers: %s", json.dumps(triggers, default=str))
    if not (force or triggers["should_retrain"]):
        log.info("No retraining needed.")
        return {"retrained": False, "triggers": triggers}

    cost = CostModel.from_config(cfg)
    raw = load_raw()
    train, val, test = chronological_split(raw)
    base = pd.concat([train, val], ignore_index=True)

    feedback = store.labeled_transactions().drop(columns="transaction_id")
    n_hold = int(len(feedback) * cfg.retraining.holdout_frac)
    fb_train, holdout = feedback.iloc[: len(feedback) - n_hold], feedback.iloc[len(feedback) - n_hold:]
    if holdout[TARGET].sum() < 5:
        # not enough recent labeled fraud to compare fairly -> fall back to the original test window
        log.warning("Only %d frauds in feedback holdout; using the original test window as holdout",
                    int(holdout[TARGET].sum()))
        holdout = test
        fb_train = feedback[~feedback.Time.isin(test.Time)] if len(feedback) else feedback
    train_all = pd.concat([base, fb_train], ignore_index=True).sort_values("Time", kind="stable")
    log.info("Challenger training rows=%d (feedback=%d) | holdout rows=%d frauds=%d",
             len(train_all), len(fb_train), len(holdout), int(holdout[TARGET].sum()))

    X_tr, y_tr = build_features(train_all), train_all[TARGET].to_numpy()
    X_ho, y_ho, a_ho = build_features(holdout), holdout[TARGET].to_numpy(), holdout.Amount.to_numpy()

    spec = dict(cfg.models[champion.model_name])
    init_mlflow()
    with mlflow.start_run(run_name=f"retrain-{champion.model_name}") as run:
        challenger = fit_model(build_model(champion.model_name, spec, y_tr, cfg), X_tr, y_tr)
        s_ch, s_cp = predict_scores(challenger, X_ho), champion.score_features(X_ho)
        # Threshold for the challenger re-optimised on the holdout; champion keeps its own threshold.
        thr = optimal_threshold(y_ho, s_ch, a_ho, cost)
        m_ch, m_cp = evaluate(y_ho, s_ch, a_ho, thr, cost), evaluate(y_ho, s_cp, a_ho, champion.threshold, cost)
        gain = m_ch["pr_auc"] - m_cp["pr_auc"]
        promoted = gain >= (cfg.retraining.min_pr_auc_gain if min_gain is None else min_gain)
        mlflow.set_tags({"stage": "retrain", "model_type": spec["type"], "promoted": promoted,
                         "trigger_drift": triggers["drift_detected"], "forced": force})
        mlflow.log_metrics({**{f"challenger_{k}": v for k, v in m_ch.items() if isinstance(v, (int, float))},
                            **{f"champion_{k}": v for k, v in m_cp.items() if isinstance(v, (int, float))},
                            "pr_auc_gain": gain, "feedback_rows": len(fb_train)})
        log.info("Champion PR-AUC=%.4f cost=$%.0f | Challenger PR-AUC=%.4f cost=$%.0f | gain=%+.4f -> %s",
                 m_cp["pr_auc"], m_cp["total_cost"], m_ch["pr_auc"], m_ch["total_cost"], gain,
                 "PROMOTE" if promoted else "keep champion")

        result = {"retrained": True, "promoted": promoted, "pr_auc_gain": gain, "triggers": triggers,
                  "champion": m_cp, "challenger": m_ch, "mlflow_run_id": run.info.run_id}
        _save_state({"last_retrain_at": started, "last_result": "promoted" if promoted else "kept_champion"})
        if promoted:
            version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            # new drift reference = the holdout, i.e. the most recent traffic the challenger was validated on
            bundle = build_bundle(champion.model_name, challenger, X_tr, train_all, y_tr, X_ho, s_ch, thr, cost,
                                  {"holdout": m_ch, "previous_champion_holdout": m_cp}, version)
            reg = register_champion(log_sklearn_model(challenger, X_ho.head(200)),
                                    {"threshold": thr, "bundle_version": version, "source": "retrain"})
            bundle.version, bundle.mlflow_run_id = f"{version}-v{reg}", run.info.run_id
            promote(bundle)
            result["new_version"] = bundle.version
            if reload_url:
                r = httpx.post(f"{reload_url.rstrip('/')}/admin/reload", timeout=30)
                log.info("API reload: %s", r.json())
    return result


def rollback(version: str, reload_url: str | None = None) -> dict:
    """Restore an archived champion bundle and point the MLflow @champion alias back at it."""
    setup_logging()
    cfg = load_config()
    src = resolve(cfg.artifacts.archive_dir) / version
    if not (src / "bundle.joblib").exists():
        available = sorted(p.name for p in resolve(cfg.artifacts.archive_dir).glob("*") if p.is_dir())
        raise SystemExit(f"No archived bundle {version!r}. Available: {available}")
    bundle = ModelBundle.load(src)
    promote(bundle)
    if "-v" in bundle.version:
        from mlflow.tracking import MlflowClient

        init_mlflow()
        MlflowClient().set_registered_model_alias(cfg.mlflow.registered_model, "champion",
                                                  bundle.version.rsplit("-v", 1)[1])
    log.info("Rolled back to %s", bundle.version)
    if reload_url:
        log.info("API reload: %s", httpx.post(f"{reload_url.rstrip('/')}/admin/reload", timeout=30).json())
    return {"rolled_back_to": bundle.version}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only print trigger status")
    ap.add_argument("--force", action="store_true", help="retrain even if no trigger fired")
    ap.add_argument("--reload-url", default=None, help="API base URL to hot-reload after promotion")
    ap.add_argument("--min-gain", type=float, default=None, help="override retraining.min_pr_auc_gain")
    ap.add_argument("--rollback", metavar="VERSION", help="restore an archived champion (artifacts/archive/VERSION)")
    args = ap.parse_args()
    if args.check:
        setup_logging()
        print(json.dumps(check_triggers(ModelBundle.load(), PredictionStore()), indent=2, default=str))
    elif args.rollback:
        print(json.dumps(rollback(args.rollback, args.reload_url), indent=2))
    else:
        out = retrain(force=args.force, reload_url=args.reload_url, min_gain=args.min_gain)
        print(json.dumps({k: v for k, v in out.items() if k not in ("champion", "challenger")}, indent=2, default=str))
