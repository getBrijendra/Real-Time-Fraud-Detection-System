"""Simulated card-transaction stream.

Replays the held-out (future) test window in time order to Kafka topic `transactions`, optionally:
  * injecting covariate drift after N events (to exercise drift detection and retraining),
  * publishing delayed ground-truth labels to `transaction-labels` (simulated chargebacks).

    python -m fraud.streaming.producer --rate 200 --limit 20000 --label-delay 5 --drift-after 10000
    python -m fraud.streaming.producer --sink http        # no Kafka: post micro-batches straight to the API
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
import signal
import time

import numpy as np
import pandas as pd

from ..config import load_config, resolve, setup_logging
from ..data import RAW_FEATURES, TARGET

log = logging.getLogger("fraud.producer")

DRIFT_TYPES = ("amount", "pca", "both")


def inject_drift(df: pd.DataFrame, kind: str, strength: float = 1.0, seed: int = 7) -> pd.DataFrame:
    """Covariate shift: bigger tickets (e.g. a new merchant segment) and/or shifted behaviour components."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    if kind in ("amount", "both"):
        out["Amount"] = out["Amount"] * (1 + 2.0 * strength) + rng.exponential(40 * strength, len(out))
    if kind in ("pca", "both"):
        for i, c in enumerate(("V1", "V2", "V3", "V4", "V5", "V7")):
            sd = df[c].std()
            out[c] = out[c] + strength * (1.0 if i % 2 == 0 else -1.0) * sd
    return out


class _Stop:
    flag = False


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=200.0, help="transactions per second (0 = as fast as possible)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N transactions (0 = whole stream)")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N transactions")
    ap.add_argument("--drift-after", type=int, default=-1, help="inject drift from event N on (-1 = never)")
    ap.add_argument("--drift-type", choices=DRIFT_TYPES, default="both")
    ap.add_argument("--drift-strength", type=float, default=1.0)
    ap.add_argument("--label-delay", type=float, default=-1, help="seconds until labels are published (-1 = never)")
    ap.add_argument("--sink", choices=("kafka", "http"), default="kafka")
    ap.add_argument("--batch", type=int, default=100, help="micro-batch size for --sink http")
    args = ap.parse_args()
    setup_logging()

    df = pd.read_parquet(resolve(cfg.data.stream_path)).iloc[args.offset:]
    if args.limit:
        df = df.iloc[: args.limit]
    df = df.reset_index(drop=True)
    if args.drift_after >= 0:
        tail = df.index >= args.drift_after
        df.loc[tail, RAW_FEATURES] = inject_drift(df.loc[tail, RAW_FEATURES], args.drift_type, args.drift_strength)
        log.info("Drift (%s, strength %.1f) injected from event %d", args.drift_type, args.drift_strength, args.drift_after)

    signal.signal(signal.SIGINT, lambda *_: setattr(_Stop, "flag", True))
    sink = KafkaSink(cfg) if args.sink == "kafka" else HttpSink(cfg, args.batch)
    pending_labels: list[tuple[float, str, int]] = []  # (due_time, id, label)
    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    t_start = time.perf_counter()
    sent = frauds = 0

    for i, row in enumerate(df.itertuples(index=False)):
        if _Stop.flag:
            break
        rec = {"transaction_id": row.transaction_id, **{c: float(getattr(row, c)) for c in RAW_FEATURES},
               "produced_at": time.time()}
        sink.send_transaction(rec)
        sent += 1
        label = int(getattr(row, TARGET))
        frauds += label
        if args.label_delay >= 0:
            heapq.heappush(pending_labels, (time.time() + args.label_delay, row.transaction_id, label))
        while pending_labels and pending_labels[0][0] <= time.time():
            _, tid, lbl = heapq.heappop(pending_labels)
            sink.send_label(tid, lbl)
        if sent % 2000 == 0:
            el = time.perf_counter() - t_start
            log.info("sent=%d frauds=%d rate=%.0f/s", sent, frauds, sent / el)
        if interval:
            target = t_start + (i + 1) * interval
            if (sleep := target - time.perf_counter()) > 0:
                time.sleep(sleep)

    # flush remaining labels (waiting out their delay)
    while pending_labels and not _Stop.flag:
        due, tid, lbl = heapq.heappop(pending_labels)
        if (wait := due - time.time()) > 0:
            time.sleep(wait)
        sink.send_label(tid, lbl)
    sink.close()
    log.info("Done: %d transactions (%d frauds) in %.1fs", sent, frauds, time.perf_counter() - t_start)


class KafkaSink:
    def __init__(self, cfg):
        from confluent_kafka import Producer

        from .topics import ensure_topics

        k = cfg.kafka
        self.topic_tx, self.topic_lbl = k.transactions_topic, k.labels_topic
        ensure_topics(cfg)
        self.p = Producer({"bootstrap.servers": k.bootstrap_servers, "linger.ms": 5, "acks": "1",
                           "compression.type": "lz4"})

    def send_transaction(self, rec: dict) -> None:
        # key by transaction id -> spread across partitions; per-card keys would preserve per-card order
        self.p.produce(self.topic_tx, key=rec["transaction_id"], value=json.dumps(rec))
        self.p.poll(0)

    def send_label(self, tid: str, label: int) -> None:
        self.p.produce(self.topic_lbl, key=tid, value=json.dumps({"transaction_id": tid, "is_fraud": bool(label)}))
        self.p.poll(0)

    def close(self) -> None:
        self.p.flush(30)


class HttpSink:
    """Kafka-free fallback: micro-batches straight to the API."""

    def __init__(self, cfg, batch: int):
        import httpx

        self.s, self.url, self.batch = httpx.Client(timeout=30), cfg.serving.api_url.rstrip("/"), batch
        self.buf: list[dict] = []
        self.lbl: list[dict] = []

    def send_transaction(self, rec: dict) -> None:
        rec = {k: v for k, v in rec.items() if k != "produced_at"}
        self.buf.append(rec)
        if len(self.buf) >= self.batch:
            self._flush_tx()

    def send_label(self, tid: str, label: int) -> None:
        self.lbl.append({"transaction_id": tid, "is_fraud": bool(label)})
        if len(self.lbl) >= self.batch:
            self._flush_lbl()

    def _flush_tx(self) -> None:
        if self.buf:
            self.s.post(f"{self.url}/predict/batch", json={"transactions": self.buf, "explain": "flagged"},
                        timeout=30).raise_for_status()
            self.buf = []

    def _flush_lbl(self) -> None:
        if self.lbl:
            self._flush_tx()  # labels must follow their predictions
            self.s.post(f"{self.url}/feedback", json={"labels": self.lbl}, timeout=30).raise_for_status()
            self.lbl = []

    def close(self) -> None:
        self._flush_tx()
        self._flush_lbl()


if __name__ == "__main__":
    main()
