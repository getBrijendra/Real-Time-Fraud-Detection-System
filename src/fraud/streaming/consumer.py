"""Streaming scorer: Kafka -> FastAPI -> Kafka.

Consumes `transactions` in micro-batches, scores them through the API (/predict/batch), publishes every
result to `scored-transactions` and flagged ones (with explanations) to `fraud-alerts`.
Delayed labels from `transaction-labels` are forwarded to /feedback (live precision/recall/cost).

    python -m fraud.streaming.consumer [--max-batch 200] [--max-wait-ms 50]
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import time

import numpy as np
import httpx
from confluent_kafka import Consumer, Producer

from ..config import load_config, setup_logging
from .topics import ensure_topics

log = logging.getLogger("fraud.consumer")


class _Stop:
    flag = False


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-batch", type=int, default=500)
    ap.add_argument("--max-wait-ms", type=int, default=50, help="max time to accumulate a micro-batch")
    ap.add_argument("--from-latest", action="store_true",
                    help="for a new consumer group, skip the backlog (default: score every transaction)")
    ap.add_argument("--idle-exit", type=float, default=0, help="exit after N idle seconds (0 = run forever)")
    args = ap.parse_args()
    setup_logging()
    # SIGTERM = `docker stop`: leave the group cleanly so partitions are reassigned at once, not after timeout
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: setattr(_Stop, "flag", True))

    k, api = cfg.kafka, cfg.serving.api_url.rstrip("/")
    ensure_topics(cfg)
    # Offsets are committed after each micro-batch is scored and published -> at-least-once scoring.
    consumer = Consumer({
        "bootstrap.servers": k.bootstrap_servers, "group.id": k.consumer_group,
        "auto.offset.reset": "latest" if args.from_latest else "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([k.transactions_topic, k.labels_topic])
    producer = Producer({"bootstrap.servers": k.bootstrap_servers, "linger.ms": 5})
    session = httpx.Client(timeout=30)
    for _ in range(60):  # wait for the API
        try:
            session.get(f"{api}/health", timeout=2).raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(2)

    e2e_ms: list[float] = []
    api_ms: list[float] = []
    stats = {"scored": 0, "alerts": 0, "labels": 0}
    last_log = last_msg = time.time()

    while not _Stop.flag:
        msgs = consumer.consume(num_messages=args.max_batch, timeout=args.max_wait_ms / 1000)
        if not msgs:
            if args.idle_exit and time.time() - last_msg > args.idle_exit:
                break
            continue
        last_msg = time.time()
        txs, labels = [], []
        for m in msgs:
            if m.error():
                log.warning("kafka error: %s", m.error())
                continue
            v = json.loads(m.value())
            (labels if m.topic() == k.labels_topic else txs).append(v)

        if txs:
            produced_at = {t["transaction_id"]: t.pop("produced_at", None) for t in txs}
            t0 = time.perf_counter()
            r = session.post(f"{api}/predict/batch", json={"transactions": txs, "explain": "flagged"}, timeout=30)
            r.raise_for_status()
            api_ms.append((time.perf_counter() - t0) * 1e3)
            now = time.time()
            for p in r.json()["predictions"]:
                value = json.dumps(p)
                producer.produce(k.scored_topic, key=p["transaction_id"], value=value)
                if p["is_fraud"]:
                    producer.produce(k.alerts_topic, key=p["transaction_id"], value=value)
                    stats["alerts"] += 1
                if (ts := produced_at.get(p["transaction_id"])) is not None:
                    e2e_ms.append((now - ts) * 1e3)
            stats["scored"] += len(txs)
            producer.poll(0)
        if labels:
            session.post(f"{api}/feedback", json={"labels": labels}, timeout=30).raise_for_status()
            stats["labels"] += len(labels)
        consumer.commit(asynchronous=True)

        if time.time() - last_log > 5 and e2e_ms:
            log.info("scored=%d alerts=%d labels=%d | batch API p50=%.1fms | end-to-end p50=%.0fms p95=%.0fms p99=%.0fms",
                     stats["scored"], stats["alerts"], stats["labels"], np.median(api_ms),
                     *np.percentile(e2e_ms, [50, 95, 99]))
            e2e_ms, api_ms, last_log = e2e_ms[-5000:], api_ms[-500:], time.time()

    producer.flush(10)
    consumer.close()
    log.info("Stopped. %s", stats)


if __name__ == "__main__":
    main()
