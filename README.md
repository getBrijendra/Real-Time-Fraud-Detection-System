# Real-Time Fraud Detection System

Detects suspicious card transactions in real time, scores fraud risk, and explains **why** a transaction was flagged.

**Dataset:** [Credit Card Fraud Detection (mlg-ulb, Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains 284,807 European card transactions over 48 hours, of which 492 are frauds (0.172%). Features `V1..V28` are anonymised PCA components, plus `Time` and `Amount`.

```
                    ┌──────────────┐  transactions    ┌──────────────┐  POST /predict/batch  ┌─────────────────────────┐
 replay test window │   producer   │ ───────────────▶ │   consumer   │ ────────────────────▶ │  FastAPI scoring service │
 (+ drift injection)│   (Kafka)    │  transaction-    │ micro-batch  │  POST /feedback       │  XGBoost + TreeSHAP      │
                    └──────────────┘  labels (delayed)└──────┬───────┘ ────────────────────▶ │  cost-optimal threshold  │
                                                             │ scored-transactions          │  drift monitor (PSI/KS)  │
                                                             │ fraud-alerts (+ explanation) │  SQLite prediction store │
                                                             ▼                              └───────────┬─────────────┘
                                                        Kafka topics                         /metrics   │  labels + drift
                                                                                  Prometheus ◀──────────┤
                                                                                  + Grafana             ▼
                                         MLflow tracking + registry ◀──── train.py / retrain.py (champion/challenger)
```

## What's inside

| Requirement | Where |
|---|---|
| Logistic regression, random forest, XGBoost and an Isolation-Forest anomaly baseline | [src/fraud/models.py](src/fraud/models.py), [configs/config.yaml](configs/config.yaml) |
| Class imbalance: none / class weights / SMOTE / random undersampling / `scale_pos_weight` | `models:` in config ([8 experiments](reports/model_comparison.md)) |
| Leak-free chronological train / validation / test split | [src/fraud/data.py](src/fraud/data.py) |
| Cost-optimal alert threshold (business cost model) | [src/fraud/metrics.py](src/fraud/metrics.py) |
| Explanations: exact TreeSHAP (XGBoost), coefficient × value (LR), deviation × importance (others) | [src/fraud/explain.py](src/fraud/explain.py) |
| FastAPI serving with Prometheus metrics | [src/fraud/api/app.py](src/fraud/api/app.py) |
| Kafka producer (replay + drift injection + delayed labels) and consumer | [src/fraud/streaming/](src/fraud/streaming/) |
| MLflow tracking, model registry, `@champion` alias | [src/fraud/tracking.py](src/fraud/tracking.py), [src/fraud/train.py](src/fraud/train.py) |
| Drift detection (PSI per feature + score PSI/KS) | [src/fraud/drift.py](src/fraud/drift.py), [src/fraud/monitoring.py](src/fraud/monitoring.py) |
| Retraining (drift/label triggered, champion vs challenger, hot reload, rollback) | [src/fraud/retrain.py](src/fraud/retrain.py) |
| Grafana dashboard + Prometheus alert rules | [monitoring/](monitoring/) |

## Quick start (local, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python -m fraud.data                 # download dataset (Kaggle API if ~/.kaggle/kaggle.json exists, else public mirror)
python -m fraud.train                # 8 experiments -> MLflow, reports/, artifacts/champion
pytest                               # 13 tests

uvicorn fraud.api.app:app --port 8000          # http://localhost:8000/docs
```

Streaming. Kafka runs in Docker; the producer and consumer can run from the venv:

```powershell
docker compose up -d kafka
python -m fraud.streaming.consumer
python -m fraud.streaming.producer --rate 500 --label-delay 3 --drift-after 15000 --drift-type both --drift-strength 0.5
# without Kafka:  python -m fraud.streaming.producer --sink http
```

Demonstrations and operations:

```powershell
python scripts/benchmark_latency.py --api-url http://127.0.0.1:8000   # latency (in-process + HTTP)
python scripts/business_cost.py                                      # operating points + FP-cost sensitivity
python scripts/drift_demo.py                                         # drift detection vs injected drift
python -m fraud.retrain --check                                      # trigger status (drift / new labels)
python -m fraud.retrain --reload-url http://127.0.0.1:8000           # retrain -> compare -> promote -> hot reload
python -m fraud.retrain --rollback <version> --reload-url http://127.0.0.1:8000
mlflow ui --backend-store-uri sqlite:///mlflow.db                    # http://localhost:5000
```

### Full stack in Docker

```powershell
docker compose up -d kafka mlflow
docker compose run --rm trainer                     # trains inside the image, logs to the MLflow server
docker compose up -d api consumer prometheus grafana
docker compose run --rm producer                    # 300 tx/s, delayed labels, drift after 30k events
docker compose run --rm retrainer
```

The stack serves API docs at :8000/docs, MLflow at :5000, Prometheus at :9090 and Grafana at :3000 (dashboard "Fraud Detection").

To scale the scorers, run `docker compose up -d --scale consumer=3`; the topics have 3 partitions. On the development laptop the Docker VM had 4 CPUs and was shared with other containers. There, one consumer sustained about 430 tx/s, and three consumers held 500 tx/s at about 0.4 s p50 end to end. Running natively, one consumer handles 1,000 tx/s.

## API

`POST /predict` with a transaction (`Time`, `V1..V28`, `Amount`, optional `transaction_id`) returns:

```json
{
  "transaction_id": "tx-146344-002631",
  "fraud_score": 0.9956,
  "is_fraud": true,
  "risk_level": "high",
  "threshold": 0.8095,
  "model_version": "20261123085741-v1",
  "explanation": {
    "summary": "Flagged: fraud score 0.996 >= alert threshold 0.810. Main risk drivers: V14, V17, V10.",
    "method": "tree_shap",
    "top_factors": [
      {"feature": "V14", "value": -7.04, "contribution": 3.93, "direction": "increases_risk",
       "typical_range": [-1.221, 1.579],
       "reason": "Behavioural signal V14 (anonymised PCA component) is unusually low (-7.04 vs typical -1.22..1.58); raises risk"}
    ]
  }
}
```

The other endpoints are:

- `POST /predict/batch`: used by the consumer. It explains `flagged`, `all` or `none`.
- `POST /feedback`: receives delayed labels.
- `GET /monitoring/drift` and `GET /monitoring/performance`: drift report and live label-based performance.
- `GET /model`: champion metadata.
- `POST /admin/reload`: hot-swaps in the current champion.
- `GET /metrics`: Prometheus metrics.
- `GET /health`: liveness check.

Contributions are in log-odds; positive values push towards fraud. `Amount` and `hour_of_day` are reported in business units, not as the engineered `log_amount` / `hour_sin` / `hour_cos`.

## Results

Evaluation protocol:

- The split is chronological: train = first 60% of time, validation = next 20%, test = last 20%.
- Every model's threshold is the one that minimises business cost on **validation**.
- All numbers below come from the **test** window, which was never used for fitting, early stopping, threshold choice or model selection. It contains 56,962 transactions and 75 frauds.

### Model comparison: PR-AUC and recall at a fixed false-positive rate

| model | imbalance | val PR-AUC | **test PR-AUC** | ROC-AUC | recall @ FPR 0.1% | recall @ FPR 0.5% | recall @ FPR 1% | p50 latency |
|---|---|---|---|---|---|---|---|---|
| **xgboost (champion)** | scale_pos_weight | **0.778** | **0.812** | 0.983 | 0.800 | 0.880 | 0.880 | 0.37 ms |
| xgboost_unweighted | none | 0.772 | 0.800 | 0.986 | 0.800 | 0.840 | 0.880 | 0.39 ms |
| random_forest | class weights | 0.762 | 0.822 | 0.966 | 0.827 | 0.880 | 0.880 | 28.6 ms ✗ |
| logreg_balanced | class weights | 0.756 | 0.747 | 0.982 | 0.800 | 0.853 | 0.880 | 1.3 ms |
| logreg_smote | SMOTE 10% | 0.706 | 0.740 | 0.978 | 0.760 | 0.853 | 0.867 | 1.3 ms |
| logreg_undersample | undersample 5% | 0.705 | 0.708 | 0.984 | 0.747 | 0.867 | 0.867 | 1.3 ms |
| logreg_unweighted | none | 0.608 | 0.698 | 0.974 | 0.760 | 0.867 | 0.867 | 1.3 ms |
| isolation_forest | unsupervised | 0.021 | 0.038 | 0.948 | 0.000 | 0.133 | 0.467 | 30.9 ms ✗ |

What the comparison shows:

- **Champion selection:** the champion is the model with the best validation PR-AUC among models within the 10 ms single-transaction latency budget. The random forest is about as accurate as XGBoost but takes about 29 ms per transaction (300 sequential trees), so it is excluded (✗).
- **Imbalance handling matters most for linear models.** Class weighting lifts logistic regression's validation PR-AUC from 0.61 to 0.76. SMOTE and undersampling help less, because they distort the operating region at very low FPR. XGBoost gains a little from `scale_pos_weight = sqrt(neg/pos)`.
- **The unsupervised baseline fails here.** Isolation Forest reaches a decent ROC-AUC (0.95) but its PR-AUC is useless (0.04). Many legitimate transactions are just as "unusual" as fraud; labels carry the signal.
- **ROC-AUC is misleading under 0.17% prevalence.** All supervised models score 0.97–0.99 on ROC-AUC, while PR-AUC separates them clearly.

Plots: [reports/pr_curves.png](reports/pr_curves.png) and [reports/cost_curve.png](reports/cost_curve.png).

### Business cost of false positives

The cost model, configurable in `business:` in the config, is:

- Every alert costs **$5** of analyst review.
- A false alert costs an extra **$10** of customer friction, so **$15** in total.
- A missed fraud costs the **amount plus a $15 chargeback fee**.

Champion on the test window, scaled per 100k transactions (from `scripts/business_cost.py`):

| operating point | threshold | recall | precision | false pos / 100k | FP cost | missed-fraud cost | **total** | vs no model |
|---|---|---|---|---|---|---|---|---|
| **cost-optimal (deployed)** | 0.810 | 0.747 | 0.889 | 12 | $184 | $5,133 | **$5,809** | saves $9,735 |
| naive 0.5 | 0.500 | 0.760 | 0.877 | 14 | $211 | $5,106 | $5,817 | saves $9,728 |
| FPR ≤ 0.1% | 0.178 | 0.773 | 0.652 | 54 | $816 | $4,663 | $5,988 | saves $9,556 |
| FPR ≤ 0.5% | 0.018 | 0.867 | 0.221 | 402 | $6,030 | $4,389 | $10,990 | saves $4,555 |
| FPR ≤ 1% | 0.009 | 0.880 | 0.126 | 801 | $12,008 | $3,249 | $15,836 | **loses $292** |

Chasing recall gets expensive fast. Raising recall from 75% to 88% multiplies false positives by about 65 and turns the model into a net loss. The deployed threshold keeps false positives at about 2–3% of total cost. Sensitivity analysis shows the cost-optimal threshold stays at 0.81 until a false positive costs more than about $105. At $255 per false positive it rises to 0.94, trading 4 points of recall for precision 0.96.

### Prediction latency

Measured on a Windows laptop with a single uvicorn worker (`scripts/benchmark_latency.py`, 1,000 requests):

| path | p50 | p95 | p99 |
|---|---|---|---|
| model only, 1 transaction (in-process) | 2.0 ms | 3.5 ms | 4.0 ms |
| model + TreeSHAP explanation, 1 transaction | 5.4 ms | 9.5 ms | 11.6 ms |
| HTTP `POST /predict` with explanation | 13 ms | 22 ms | 38 ms |
| HTTP `POST /predict/batch` (100 tx) | 25 ms | 46 ms | 292 ms* (~3,000 tx/s) |
| Kafka → consumer → API → Kafka, end to end at 1,000 tx/s | ~140 ms | ~250 ms | ~300 ms |

\*The laptop was also running Docker, Kafka and other containers during this run. An earlier run on a quieter machine measured batch p99 at 66 ms, so the tail is noise from a shared machine.

The end-to-end figure includes the consumer's 50 ms micro-batching window and Kafka linger. The model itself takes 0.37 ms per transaction (XGBoost `inplace_predict`, 349 trees); the rest is feature building, explanation, validation and HTTP. Prometheus exposes `fraud_inference_latency_seconds` and `fraud_api_request_latency_seconds` histograms.

Latency fixes made during development:

- XGBoost `inplace_predict` on numpy instead of `predict_proba` on a DataFrame: 6 ms → 0.4 ms.
- Single-threaded inference, which avoids a thread pool per call.
- One SQLite transaction per batch instead of autocommit per row: 100 ms → 16 ms per 500 rows.
- The batch endpoint skips FastAPI's response re-validation.
- Clients use `httpx`. With `requests` on Windows, large POSTs stall for 200+ ms because of Nagle's algorithm and delayed ACKs.

### Drift detection

Drift is measured as PSI per feature (10 quantile bins) plus PSI and KS on the score. **The reference is the validation window**, the most recent pre-deployment data. The dataset has genuine day-over-day shift (V1 has a PSI of about 1.5 between the day-1 training data and the day-2 evening), so a training-window reference raised false alarms on normal traffic. Calendar features are excluded, because a live window spans hours, not a full day.

Results from `scripts/drift_demo.py`:

| scenario | drift? | max feature PSI | top drifted | PR-AUC |
|---|---|---|---|---|
| clean future windows 1–3 | no | 0.008–0.045 | – | 0.81–0.94 |
| amount drift ×0.25 / ×0.5 / ×1 | yes | 0.47 / 0.79 / 1.39 | log_amount | 0.86–0.85 |
| PCA drift ×0.25 / ×0.5 / ×1 | yes | 1.34 / 2.13 / 1.89 | V1, V2, V7 | 0.86–0.85 |
| both ×0.25 / ×0.5 / ×1 | yes | 1.34 / 2.13 / 1.89 | V1, log_amount, V2 | 0.86–0.85 |

There were no false alarms on clean traffic, and every injected shift was caught and attributed to the right features. On the live stream with drift injected after 15k events, `/monitoring/drift` reported `V1, log_amount, V4, V7, V5`. `fraud_drift_detected` was set to 1, which fires the `FraudDataDrift` Prometheus alert.

### Retraining loop

1. Delayed labels (simulated chargebacks) arrive on `transaction-labels`. They go to `/feedback`, then into the prediction store, and feed live precision, recall and realised cost (`fraud_business_cost_usd_total{outcome}`).
2. `python -m fraud.retrain` fires when there is drift, or when there are ≥ 2,000 new labels including ≥ 10 frauds. It trains a challenger on train + validation + older feedback.
3. The challenger and the champion are compared on the **most recent 30% of feedback**.
4. The challenger is promoted only if its PR-AUC gain is at least `min_pr_auc_gain`. Promotion means:
   - registering it in MLflow and moving `@champion` to it;
   - archiving the old bundle;
   - hot-reloading the API.

Example run on the drifted stream: drift triggered retraining. The champion scored PR-AUC 0.799 and the challenger 0.780 on the newest 9k transactions (18 frauds), so the champion was **kept**. A forced promotion, hot reload (v1 → v2) and `--rollback` were also verified.

## Design notes and limitations

- **Small fraud counts.** The test window holds 75 frauds, and the retraining holdout holds about 18. Differences of about 0.01 PR-AUC between models are within noise; treat the ranking at the top as a tie (XGBoost vs random forest).
- **Anonymised features.** `V1..V28` are PCA components, so explanations can say *which* behavioural signal is abnormal and by how much, but not what it means in business terms. On real data, map them to named features such as merchant category, velocity or geo-distance.
- **No card identifier.** Per-card features such as velocity, time since last transaction and amount vs the card's history aren't possible with this dataset. They are usually the strongest fraud signals.
- **Delivery semantics.** Offsets are committed after scoring and publishing, so delivery is at-least-once. Kafka gives no ordering across the `transactions` and `transaction-labels` topics: a consumer catching up on a backlog can read a label before its transaction. Labels are therefore kept in their own table and matched whichever side arrives first; `tests/test_store.py` covers this.
- **Single worker.** The drift window and the live performance counters are in-process. With several uvicorn workers or replicas, move them to Redis or compute them from the prediction store.
- **Pinned versions.** The model bundle is a joblib pickle, so `requirements.txt` pins exact versions and the Docker image must match the training environment. MLflow also stores the model in the native XGBoost format.
