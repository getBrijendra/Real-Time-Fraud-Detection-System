"""Prediction latency benchmark: in-process model and the HTTP API.

    python scripts/benchmark_latency.py                       # in-process only
    python scripts/benchmark_latency.py --api-url http://localhost:8000 -n 1000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from fraud.bundle import ModelBundle
from fraud.config import load_config, resolve
from fraud.data import RAW_FEATURES


def _stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"n": len(a), "mean_ms": a.mean(), "p50_ms": np.percentile(a, 50), "p95_ms": np.percentile(a, 95),
            "p99_ms": np.percentile(a, 99), "max_ms": a.max()}


def bench(fn, n: int, warmup: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return _stats(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=None)
    ap.add_argument("-n", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config()
    bundle = ModelBundle.load()
    stream = pd.read_parquet(resolve(cfg.data.stream_path))
    rng = np.random.default_rng(0)
    rows = [stream.iloc[[i]][RAW_FEATURES] for i in rng.integers(0, len(stream), 64)]
    batch = stream.sample(100, random_state=0)[RAW_FEATURES]
    it = iter(range(10**9))

    results = {
        "model": f"{bundle.model_name} {bundle.version}",
        "in_process_single_score_only": bench(lambda: bundle.predict(rows[next(it) % 64], explain="none"), args.n),
        "in_process_single_with_explanation": bench(lambda: bundle.predict(rows[next(it) % 64], explain="all"), args.n),
        "in_process_batch100_explain_flagged": bench(lambda: bundle.predict(batch, explain="flagged"), max(args.n // 5, 50)),
    }

    if args.api_url:
        import httpx

        s = httpx.Client(timeout=10)  # requests on Windows hits Nagle/delayed-ACK stalls (~200ms+) on large POSTs
        url = args.api_url.rstrip("/")
        payloads = [{**r.iloc[0].to_dict(), "transaction_id": f"bench-{i}"} for i, r in enumerate(rows)]
        results["http_single_predict_with_explanation"] = bench(
            lambda: s.post(f"{url}/predict", json=payloads[next(it) % 64], timeout=10).raise_for_status(), args.n)
        body = {"transactions": [{**r, "transaction_id": f"bb-{i}"} for i, r in enumerate(batch.to_dict("records"))],
                "explain": "flagged"}
        results["http_batch100"] = bench(lambda: s.post(f"{url}/predict/batch", json=body, timeout=10).raise_for_status(),
                                         max(args.n // 5, 50))
        stats = results["http_batch100"]
        results["http_batch100"]["throughput_tx_per_s"] = 100 / (stats["mean_ms"] / 1e3)

    out = resolve(cfg.artifacts.reports_dir) / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nLatency benchmark - {results['model']}")
    print(f"{'scenario':42s} {'p50':>8s} {'p95':>8s} {'p99':>8s}  (ms)")
    for k, v in results.items():
        if isinstance(v, dict):
            extra = f"  ~{v['throughput_tx_per_s']:,.0f} tx/s" if "throughput_tx_per_s" in v else ""
            print(f"{k:42s} {v['p50_ms']:8.2f} {v['p95_ms']:8.2f} {v['p99_ms']:8.2f}{extra}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
