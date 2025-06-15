"""Kafka topic bootstrap shared by producer and consumer (idempotent)."""
from __future__ import annotations

import logging

from confluent_kafka.admin import AdminClient, NewTopic

log = logging.getLogger(__name__)


def ensure_topics(cfg, partitions: int = 3) -> None:
    k = cfg.kafka
    admin = AdminClient({"bootstrap.servers": k.bootstrap_servers})
    wanted = (k.transactions_topic, k.labels_topic, k.scored_topic, k.alerts_topic)
    existing = set(admin.list_topics(timeout=10).topics)
    missing = [NewTopic(t, num_partitions=partitions, replication_factor=1) for t in wanted if t not in existing]
    if not missing:
        return
    for topic, fut in admin.create_topics(missing).items():
        try:
            fut.result()
            log.info("created topic %s", topic)
        except Exception as exc:  # noqa: BLE001 - created concurrently by another client
            if "TOPIC_ALREADY_EXISTS" not in str(exc):
                raise
