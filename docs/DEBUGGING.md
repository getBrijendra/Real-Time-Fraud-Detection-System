# Debugging guide: flows, launch configs and breakpoints

VS Code stores breakpoints in your local workspace state, not in a file the repo can ship. So this guide
lists **where to put them** (each link opens the line) and **what to inspect when they hit**. Line numbers
match the code at the time of writing. If a line moves, search for the function name shown next to it.

## Setup (once)

1. Install the **Python** and **Python Debugger** extensions (`ms-python.python`, `ms-python.debugpy`).
2. `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`. This is already the default in `.vscode/settings.json`.
3. Open **Run and Debug** (`Ctrl+Shift+D`) and pick a configuration from the dropdown.

## The sandbox: why debug runs don't touch your real models

Every configuration (except the tests) first runs the task **`debug: prepare sandbox`**
([scripts/debug_sandbox.py](../scripts/debug_sandbox.py)). That task:

- writes `.debug/config.yaml`, a copy of `configs/config.yaml` with all outputs redirected into `.debug/`
  (champion and archive, `mlflow.db`, `mlartifacts`, `reports`, prediction store, replay stream);
- on first use, copies the current champion, its archive and the replay stream into `.debug/`.

The launch configs set `FRAUD_CONFIG=.debug/config.yaml`. Training, promoting or rolling back in a debug
session therefore never changes `artifacts/`, `mlflow.db`, `mlartifacts/` or `reports/`.

| Task (`Ctrl+Shift+P` → *Tasks: Run Task*) | Use |
|---|---|
| `debug: reset sandbox` | delete `.debug/` and start clean |
| `debug: MLflow UI (sandbox)` | view debug runs and registry at http://127.0.0.1:5001 |

## Launch configurations

| Config | Flow | Needs |
|---|---|---|
| **1a. Train: fast** | load → split → features → fit 2 models → threshold → MLflow → champion bundle | `data/raw/creditcard.csv` |
| 1b. Train: all 8 | same, all experiments (~4 min) | same |
| 1c. Data | dataset download / validation | internet if the CSV is missing |
| **2. API** | FastAPI service on :8000; try http://127.0.0.1:8000/docs | a champion (seeded) |
| **3a. Producer → API over HTTP** | replay stream → `/predict/batch` + delayed `/feedback`, drift injected after 3000 events | API running |
| 3b / 3c. Producer / Consumer via Kafka | Kafka → consumer → API → Kafka | Docker (the task starts the `kafka` container) |
| **4. Retrain: check** | drift + label triggers from the prediction store | run 3a first, so the store has traffic |
| 5a. Retrain: force | challenger vs champion, normal promotion rule | — |
| **5b. Retrain: always promote + reload** | register → archive → promote → `POST /admin/reload` | API running |
| 5c. Rollback | restore an archived bundle and move `@champion` | asks for the version folder name |
| Script: current file | run any `scripts/*.py` in the sandbox | — |
| Tests: current file / all | pytest (synthetic data, no sandbox needed) | — |

**Compounds** start several processes in one debug session (`stopAll` stops them all together):
- **Live flow: API + Producer over HTTP**: the recommended way to step through real-time scoring without Kafka.
- **Live flow: API + Consumer + Producer via Kafka**: the full streaming path.

With a compound running, the **Call Stack** panel lists each process. Pick one to see its threads.
Configs that need the API wait for `/health` before they start, so start order doesn't matter.

**Suggested learning path:** `1a` → `2` → `3a` (or the compound) → `4` → `5b` → `5c`.

---

## Breakpoint map

### Flow 1: Offline training ([src/fraud/train.py](../src/fraud/train.py), config `1a`)

| # | Location | What to inspect |
|---|---|---|
| T1 | [data.py:99](../src/fraud/data.py#L99) `chronological_split` | `df` is sorted by `Time`. Step to line 102 and compare `len(train)`, `len(val)`, `len(test)`. Watch `part[TARGET].mean()`: about 0.17% fraud. |
| T2 | [data.py:116](../src/fraud/data.py#L116) `write_stream_source` | the test window gets synthetic `transaction_id`s. This file is what the producer replays as "live" traffic. |
| T3 | [features.py:23](../src/fraud/features.py#L23) `build_features` | raw `Time, V1..V28, Amount` → 31 model features (`log1p(Amount)`, `hour_sin`, `hour_cos`). The API runs this same function. |
| T4 | [train.py:152](../src/fraud/train.py#L152) model loop | `name`, `spec`. Add a **conditional breakpoint** `name == "xgboost"` to skip the others. |
| T5 | [models.py:92](../src/fraud/models.py#L92) `build_model` | `neg/pos` ≈ 580, so `spw` = sqrt ≈ 24: how class imbalance is handled for XGBoost. |
| T6 | [models.py:106](../src/fraud/models.py#L106) `fit_model` | early stopping on the latest 15% of *train* (val stays untouched). Step over, then check `best`. |
| T7 | [metrics.py:88](../src/fraud/metrics.py#L88) `optimal_threshold` | **key business logic.** Step into `cost_curve`. Inspect `costs.min()`, `thresholds[np.argmin(costs)]`, and `costs[0]` (the cost of having no model). |
| T8 | [train.py:163](../src/fraud/train.py#L163) | `val_m` vs `test_m` dicts: everything that gets logged to MLflow (see the MLflow notes in the README). |
| T9 | [train.py:187](../src/fraud/train.py#L187) champion selection | `results`, `eligible` (latency budget 10 ms), `budget`. |
| T10 | [train.py:58](../src/fraud/train.py#L58) `build_bundle` | `ReferenceProfile.build` stores the validation-window histograms that live drift is compared against. |
| T11 | [train.py:207](../src/fraud/train.py#L207) `register_champion` | step into [tracking.py:54](../src/fraud/tracking.py#L54): registry version + `@champion` alias. |
| T12 | [bundle.py:107](../src/fraud/bundle.py#L107) `promote` | old champion copied to `archive/<version>`, new bundle saved to `champion/`. |

### Flow 2: Real-time scoring ([src/fraud/api/app.py](../src/fraud/api/app.py), config `2` + `3a`)

| # | Location | What to inspect |
|---|---|---|
| A1 | [app.py:52](../src/fraud/api/app.py#L52) `ServiceState.load` | runs at startup and on `/admin/reload`: bundle → drift monitor → perf tracker → store. |
| A2 | [app.py:89](../src/fraud/api/app.py#L89) `_score` | `txs` (pydantic) → `records` → `raw` DataFrame. Common to `/predict` and `/predict/batch`. |
| A3 | [bundle.py:51](../src/fraud/bundle.py#L51) `ModelBundle.predict` | `X` → `scores` → `flagged = scores >= self.threshold`. Conditional breakpoint `flagged.any()` on line 55 stops **only on batches that contain an alert**. |
| A4 | [models.py:130](../src/fraud/models.py#L130) `predict_scores` | the fast XGBoost path (`inplace_predict` on numpy instead of `predict_proba`). |
| A5 | [explain.py:67](../src/fraud/explain.py#L67) `Explainer.explain` | `contrib` = per-feature TreeSHAP values (`pred_contribs=True`), grouped and ranked into `top_factors`. |
| A6 | [bundle.py:45](../src/fraud/bundle.py#L45) `risk_level` | the high / medium / low bands relative to the threshold. |
| A7 | [app.py:104](../src/fraud/api/app.py#L104) | the prediction is persisted **before** the response (audit trail). `late` = labels that arrived before their prediction. |
| A8 | [app.py:106](../src/fraud/api/app.py#L106) | drift bookkeeping is scheduled as a background task, so it runs after the response is sent. |

> Tip: use **Logpoints** (right-click the gutter → *Add Logpoint*) on hot paths to avoid stopping on every
> request. For example, on A3: `batch={len(X)} flagged={int(flagged.sum())} max={scores.max():.3f}`.

To fire one request by hand: open http://127.0.0.1:8000/docs → `POST /predict` → *Try it out*.

### Flow 3: Streaming (config `3a` HTTP, or `3b`/`3c` Kafka)

| # | Location | What to inspect |
|---|---|---|
| S1 | [producer.py:68](../src/fraud/streaming/producer.py#L68) | `inject_drift` shifts features from event `--drift-after` on. Compare `df.loc[tail, "V1"].describe()` before and after the step. |
| S2 | [producer.py:88](../src/fraud/streaming/producer.py#L88) | delayed labels: a heap of `(due_time, id, label)`, which simulates chargebacks arriving later. |
| S3 | [producer.py:164](../src/fraud/streaming/producer.py#L164) `HttpSink._flush_lbl` | transactions are always flushed before their labels. |
| S4 | [consumer.py:69](../src/fraud/streaming/consumer.py#L69) | a Kafka micro-batch from **both** topics, split into `txs` and `labels`. |
| S5 | [consumer.py:86](../src/fraud/streaming/consumer.py#L86) | one HTTP call scores the whole batch. Then results go to `scored-transactions`, and flagged ones to `fraud-alerts`. |
| S6 | [consumer.py:103](../src/fraud/streaming/consumer.py#L103) | the offset is committed only **after** scoring and publishing, which gives at-least-once delivery. |
| S7 | [app.py:137](../src/fraud/api/app.py#L137) `/feedback` → [store.py:106](../src/fraud/store.py#L106) | labels are attached to stored predictions. `matched` feeds the live precision/recall/cost numbers. |

### Flow 4: Drift monitoring (happens inside the API during `3a`)

| # | Location | What to inspect |
|---|---|---|
| D1 | [monitoring.py:52](../src/fraud/monitoring.py#L52) `DriftMonitor.add` | `due` becomes True every 250 predictions once 500 samples are in the window. |
| D2 | [drift.py:70](../src/fraud/drift.py#L70) `ReferenceProfile.report` | per-feature PSI against the training-time histograms. With drift injected, V1–V5 go to `alert` (PSI > 0.25). |
| D3 | [drift.py:87](../src/fraud/drift.py#L87) | `dataset_drift` = at least 1 feature in alert, **or** the score distribution in alert. |

Check the result in a browser: http://127.0.0.1:8000/monitoring/drift and `/monitoring/performance`.

### Flow 5: Retraining lifecycle ([src/fraud/retrain.py](../src/fraud/retrain.py), configs `4`, `5a`, `5b`, `5c`)

| # | Location | What to inspect |
|---|---|---|
| R1 | [retrain.py:67](../src/fraud/retrain.py#L67) `check_triggers` | stored traffic is **re-scored** with the current champion, then checked for drift. Also look at `counts["new_labels"]`. |
| R2 | [retrain.py:101](../src/fraud/retrain.py#L101) | if the feedback holdout has fewer than 5 frauds, it falls back to the original test window. It usually does with the 6k-event replay. |
| R3 | [retrain.py:107](../src/fraud/retrain.py#L107) | the challenger trains on train + val + older feedback (`len(fb_train)`). |
| R4 | [retrain.py:122](../src/fraud/retrain.py#L122) | **the champion/challenger decision**: `m_ch["pr_auc"]`, `m_cp["pr_auc"]`, `gain`, `promoted`. |
| R5 | [retrain.py:141](../src/fraud/retrain.py#L141) | promotion: a new registry version, `@champion` moved, the old bundle archived. |
| R6 | [retrain.py:147](../src/fraud/retrain.py#L147) → A1 | the hot reload. With the API also under the debugger, **A1 hits inside the API process.** |
| R7 | [retrain.py:166](../src/fraud/retrain.py#L166) `rollback` | the alias is moved back to the version parsed from the folder name (`...-v1` → `1`). |

---

## Debugger techniques worth using here

- **Debug Console** (while paused): run any expression in the current frame, for example
  `pd.Series(scores).describe()`, `X.iloc[0]`, `self.threshold`, or
  `sorted(rep["features"].items(), key=lambda kv: -kv[1]["psi"])[:5]`.
- **Conditional breakpoints** (right-click a breakpoint → *Edit Condition*): `name == "xgboost"`,
  `flagged.any()`, `promoted`, `len(txs) > 100`.
- **Hit count**: stop on the Nth hit only. Useful in the producer loop (`S2`) and in `DriftMonitor.add`.
- **Exception breakpoints**: in the BREAKPOINTS panel, tick *Raised Exceptions* to stop where an error
  starts rather than where it is caught.
- **Step into libraries** (XGBoost, sklearn, MLflow): set `"justMyCode": false` in the config you're using.
- **Changing config while debugging**: edit `configs/config.yaml`, not `.debug/config.yaml`. The sandbox copy
  is regenerated on every launch. `load_config()` is cached, so restart the session after an edit.
- **Watch panel** suggestions: `bundle.version`, `state.bundle.threshold`, `len(self._X)` (drift window), `stats`
  (consumer).
