"""FastAPI scoring service.

    uvicorn fraud.api.app:app --host 0.0.0.0 --port 8000

Endpoints
  POST /predict             score + explain one transaction
  POST /predict/batch       score a micro-batch (used by the Kafka consumer)
  POST /feedback            delayed labels (chargebacks / analyst verdicts) -> live precision/recall/cost
  GET  /monitoring/drift    PSI drift report of the live window vs the training reference
  GET  /monitoring/performance  label-based performance + realised business cost
  GET  /model               champion metadata (threshold, offline metrics)
  POST /admin/reload        hot-swap to the current champion bundle (after retraining)
  GET  /metrics             Prometheus metrics
  GET  /health
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from .. import monitoring as mon
from ..bundle import ModelBundle
from ..config import load_config, resolve, setup_logging
from ..data import RAW_FEATURES
from ..features import build_features
from ..store import PredictionStore
from .schemas import BatchRequest, BatchResponse, FeedbackRequest, Prediction, Transaction

log = logging.getLogger("fraud.api")


class ServiceState:
    def __init__(self) -> None:
        self.bundle: ModelBundle | None = None
        self.drift: mon.DriftMonitor | None = None
        self.perf: mon.PerformanceTracker | None = None
        self.store: PredictionStore | None = None
        self.started = time.time()
        self._lock = threading.Lock()

    def load(self) -> None:
        cfg = load_config()
        model_dir = os.environ.get("FRAUD_MODEL_DIR") or resolve(cfg.artifacts.champion_dir)
        bundle = ModelBundle.load(model_dir)
        with self._lock:
            self.bundle = bundle
            m = cfg.monitoring
            if self.drift is None:
                self.drift = mon.DriftMonitor(bundle.reference, m.window_size, m.min_samples, m.check_every)
            else:
                self.drift.reset(bundle.reference)
            if self.perf is None:
                self.perf = mon.PerformanceTracker(bundle.cost_model)
            if self.store is None:
                self.store = PredictionStore(os.environ.get("FRAUD_STORE_PATH") or None)
        mon.MODEL_INFO.info({"name": bundle.model_name, "version": bundle.version, "threshold": f"{bundle.threshold:.6f}"})
        log.info("Loaded model %s version %s (threshold %.4f)", bundle.model_name, bundle.version, bundle.threshold)


state = ServiceState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    state.load()
    yield


app = FastAPI(title="Real-time Fraud Detection API", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


def _score(txs: list[Transaction], explain: str, background: BackgroundTasks,
           as_dicts: bool = False) -> tuple[list, float]:
    bundle = state.bundle
    if bundle is None:
        raise HTTPException(503, "model not loaded")
    t0 = time.perf_counter()
    records = [t.model_dump() for t in txs]
    ids = [r.pop("transaction_id") or str(uuid.uuid4()) for r in records]
    raw = pd.DataFrame.from_records(records, columns=RAW_FEATURES)
    results = bundle.predict(raw, explain=explain, top_k=int(load_config().serving.explain_top_k))
    elapsed = time.perf_counter() - t0
    mon.INFERENCE_LATENCY.observe(elapsed)

    scores = [r["fraud_score"] for r in results]
    flagged = [r["is_fraud"] for r in results]
    for s, f in zip(scores, flagged):
        mon.SCORES.observe(s)
        mon.PREDICTIONS.labels(decision="fraud" if f else "legit").inc()

    # The decision is persisted before responding (audit trail; and delayed labels arriving right after the
    # response must find their prediction). Drift bookkeeping runs after the response is sent.
    if late := state.store.log_predictions(raw, ids, scores, flagged, bundle.version):
        state.perf.update(late)  # labels that arrived before their transaction was scored
    background.add_task(state.drift.add, build_features(raw), scores)
    if as_dicts:
        return [{"transaction_id": i, "model_version": bundle.version, **r} for i, r in zip(ids, results)], elapsed * 1e3
    preds = [Prediction(transaction_id=i, model_version=bundle.version, **r) for i, r in zip(ids, results)]
    return preds, elapsed * 1e3


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction, background: BackgroundTasks,
            explain: bool = Query(True, description="include the explanation even when not flagged")):
    t0 = time.perf_counter()
    preds, _ = _score([tx], "all" if explain else "flagged", background)
    mon.REQUEST_LATENCY.labels(endpoint="/predict").observe(time.perf_counter() - t0)
    mon.REQUESTS.labels(endpoint="/predict", status="200").inc()
    return preds[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest, background: BackgroundTasks):
    t0 = time.perf_counter()
    preds, ms = _score(req.transactions, req.explain, background, as_dicts=True)
    mon.REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(time.perf_counter() - t0)
    mon.REQUESTS.labels(endpoint="/predict/batch", status="200").inc()
    # Hot path for the stream consumer: the payload is built by our own code, so skip FastAPI's
    # response-model re-validation (it costs more than the model inference for large batches).
    return JSONResponse({"predictions": preds, "n_flagged": sum(p["is_fraud"] for p in preds), "latency_ms": ms},
                        background=background)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    matched = state.store.add_labels([(lbl.transaction_id, int(lbl.is_fraud)) for lbl in req.labels])
    state.perf.update(matched)
    mon.REQUESTS.labels(endpoint="/feedback", status="200").inc()
    return {"received": len(req.labels), "matched": len(matched)}


@app.get("/monitoring/drift")
def drift_report(refresh: bool = True):
    return state.drift.check() if refresh or state.drift.last_report is None else state.drift.last_report


@app.get("/monitoring/performance")
def performance():
    return {"since_start": state.perf.summary(), "store": state.store.counts(),
            "threshold": state.bundle.threshold, "model_version": state.bundle.version}


@app.get("/model")
def model_info():
    return state.bundle.metadata()


@app.post("/admin/reload")
def reload_model():
    old = state.bundle.version if state.bundle else None
    state.load()
    return {"previous_version": old, "current_version": state.bundle.version}


@app.get("/health")
def health():
    return {"status": "ok" if state.bundle else "loading",
            "model_version": state.bundle.version if state.bundle else None,
            "uptime_s": round(time.time() - state.started, 1)}
