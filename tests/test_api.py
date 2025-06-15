import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud.data import RAW_FEATURES, TARGET
from fraud.features import build_features
from fraud.metrics import CostModel, optimal_threshold
from fraud.models import build_model, fit_model


@pytest.fixture(scope="module")
def client(synthetic_raw, tmp_path_factory, monkeypatch_module):
    from fraud.train import build_bundle

    d = tmp_path_factory.mktemp("bundle")
    df = synthetic_raw
    tr, va = df.iloc[:4000], df.iloc[4000:]
    X_tr, X_va = build_features(tr), build_features(va)
    y_tr = tr[TARGET].to_numpy()
    model = fit_model(build_model("xgb", {"type": "xgboost", "imbalance": "scale_pos_weight"}, y_tr), X_tr, y_tr)
    s_va = model.predict_proba(X_va)[:, 1]
    cost = CostModel()
    thr = optimal_threshold(va[TARGET].to_numpy(), s_va, va.Amount.to_numpy(), cost)
    build_bundle("xgboost", model, X_tr, tr, y_tr, X_va, s_va, thr, cost, {}, "test-1").save(d)

    monkeypatch_module.setenv("FRAUD_MODEL_DIR", str(d))
    monkeypatch_module.setenv("FRAUD_STORE_PATH", str(d / "store.db"))
    from fraud.api.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _tx(row, tid):
    return {"transaction_id": tid, **{c: float(row[c]) for c in RAW_FEATURES}}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_version"] == "test-1"


def test_predict_fraud_explained(client, synthetic_raw):
    fraud_row = synthetic_raw[synthetic_raw[TARGET] == 1].iloc[-1]
    r = client.post("/predict", json=_tx(fraud_row, "f-1"))
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1
    exp = body["explanation"]
    assert exp["method"] == "tree_shap" and len(exp["top_factors"]) == 5
    assert {f["feature"] for f in exp["top_factors"]} & {"V14", "V17", "Amount"}
    assert exp["summary"]


def test_predict_validation_error(client):
    assert client.post("/predict", json={"Time": 1, "Amount": -5}).status_code == 422


def test_batch_feedback_and_monitoring(client, synthetic_raw):
    rows = synthetic_raw.iloc[4000:5500]
    txs = [_tx(r, f"b-{i}") for i, (_, r) in enumerate(rows.iterrows())]
    r = client.post("/predict/batch", json={"transactions": txs, "explain": "flagged"})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == len(txs)
    assert all((p["explanation"] is not None) == p["is_fraud"] for p in preds)

    labels = [{"transaction_id": f"b-{i}", "is_fraud": bool(y)} for i, y in enumerate(rows[TARGET])]
    fb = client.post("/feedback", json={"labels": labels}).json()
    assert fb["matched"] == len(labels)

    perf = client.get("/monitoring/performance").json()
    assert perf["since_start"]["labeled"] == len(labels)
    assert perf["since_start"]["recall"] is not None

    drift = client.get("/monitoring/drift").json()
    assert drift["n_samples"] >= 500 and "dataset_drift" in drift

    metrics = client.get("/metrics/").text
    assert "fraud_predictions_total" in metrics and "fraud_feature_psi" in metrics


def test_model_metadata(client):
    meta = client.get("/model").json()
    assert meta["model_name"] == "xgboost" and 0 < meta["threshold"] <= 1.0 + 1e-6
    assert np.isfinite(meta["threshold"])
