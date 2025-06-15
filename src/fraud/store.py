"""SQLite store for scored transactions and the delayed labels (chargebacks / analyst decisions)
that arrive later. It closes the loop for live performance monitoring and retraining."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .config import load_config, resolve
from .data import RAW_FEATURES

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    transaction_id TEXT PRIMARY KEY,
    scored_at      REAL NOT NULL,
    amount         REAL NOT NULL,
    payload        TEXT NOT NULL,     -- raw transaction fields as JSON (for retraining)
    score          REAL NOT NULL,
    flagged        INTEGER NOT NULL,
    model_version  TEXT NOT NULL,
    label          INTEGER,           -- NULL until feedback arrives
    labeled_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_pred_scored ON predictions(scored_at);
CREATE INDEX IF NOT EXISTS idx_pred_labeled ON predictions(labeled_at);
-- Every label received, whether or not its prediction exists yet. Kafka gives no ordering across topics:
-- a consumer catching up on a backlog can read a label before the transaction it refers to.
CREATE TABLE IF NOT EXISTS labels (
    transaction_id TEXT PRIMARY KEY,
    label          INTEGER NOT NULL,
    received_at    REAL NOT NULL
);
"""
_MATCH_COLS = ("transaction_id", "amount", "score", "flagged", "label")


class PredictionStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else resolve(load_config().serving.store_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)

    def log_predictions(self, raw: pd.DataFrame, ids: list[str], scores, flagged, model_version: str) -> list[dict]:
        """Persist decisions. Returns predictions that matched labels which had arrived earlier."""
        now = time.time()
        # pandas' C JSON writer is ~10x faster than json.dumps per row
        payloads = raw[RAW_FEATURES].to_json(orient="records", lines=True, double_precision=15).splitlines()
        amounts = raw["Amount"].to_numpy(dtype=float)
        rows = [
            (tid, now, float(a), p, float(s), int(f), model_version)
            for tid, a, p, s, f in zip(ids, amounts, payloads, scores, flagged)
        ]
        with self._lock:
            with self._tx():
                self._conn.executemany(
                    "INSERT OR REPLACE INTO predictions "
                    "(transaction_id, scored_at, amount, payload, score, flagged, model_version) VALUES (?,?,?,?,?,?,?)",
                    rows,
                )
                early = [r for part in self._chunks(ids) for r in self._conn.execute(
                    f"SELECT transaction_id, label FROM labels WHERE transaction_id IN ({','.join('?' * len(part))})", part)]
                if early:
                    self._conn.executemany("UPDATE predictions SET label=?, labeled_at=? WHERE transaction_id=?",
                                           [(lbl, now, tid) for tid, lbl in early])
            return self._matched([tid for tid, _ in early]) if early else []

    @staticmethod
    def _chunks(ids: list[str], size: int = 500):
        for i in range(0, len(ids), size):
            yield ids[i:i + size]

    def _matched(self, ids: list[str]) -> list[dict]:
        out = []
        for part in self._chunks(ids):
            q = (f"SELECT {', '.join(_MATCH_COLS)} FROM predictions "
                 f"WHERE label IS NOT NULL AND transaction_id IN ({','.join('?' * len(part))})")
            out += [dict(zip(_MATCH_COLS, r)) for r in self._conn.execute(q, part)]
        return out

    @contextmanager
    def _tx(self):
        """One transaction per batch. In autocommit mode executemany commits every row (~0.2ms each)."""
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    def add_labels(self, labels: list[tuple[str, int]]) -> list[dict]:
        """Record labels and attach them to existing predictions. Returns the predictions matched now
        (for live cost/performance metrics); labels without a prediction yet are matched in log_predictions."""
        now = time.time()
        with self._lock:
            with self._tx():
                self._conn.executemany("INSERT OR REPLACE INTO labels (transaction_id, label, received_at) VALUES (?,?,?)",
                                       [(tid, int(lbl), now) for tid, lbl in labels])
                self._conn.executemany(
                    "UPDATE predictions SET label=?, labeled_at=? WHERE transaction_id=?",
                    [(int(lbl), now, tid) for tid, lbl in labels],
                )
            return self._matched([tid for tid, _ in labels])

    def recent(self, limit: int, labeled_only: bool = False) -> pd.DataFrame:
        where = "WHERE label IS NOT NULL" if labeled_only else ""
        order = "labeled_at" if labeled_only else "scored_at"
        with self._lock:
            df = pd.read_sql_query(
                f"SELECT * FROM predictions {where} ORDER BY {order} DESC LIMIT ?", self._conn, params=(limit,)
            )
        return df.iloc[::-1].reset_index(drop=True)

    def labeled_transactions(self) -> pd.DataFrame:
        """All labeled feedback as a raw-transaction frame (+ Class), oldest first."""
        with self._lock:
            df = pd.read_sql_query(
                "SELECT transaction_id, payload, label FROM predictions WHERE label IS NOT NULL ORDER BY scored_at",
                self._conn,
            )
        if df.empty:
            return pd.DataFrame(columns=["transaction_id", *RAW_FEATURES, "Class"])
        raw = pd.DataFrame([json.loads(p) for p in df.payload])
        raw.insert(0, "transaction_id", df.transaction_id.values)
        raw["Class"] = df.label.astype(int).values
        return raw

    def counts(self, since: float | None = None) -> dict:
        """Totals; `since` (epoch seconds) additionally counts labels that arrived after that time."""
        with self._lock:
            total, labeled, frauds, flagged = self._conn.execute(
                "SELECT COUNT(*), COUNT(label), COALESCE(SUM(label),0), COALESCE(SUM(flagged),0) FROM predictions"
            ).fetchone()
            new, new_frauds = self._conn.execute(
                "SELECT COUNT(label), COALESCE(SUM(label),0) FROM predictions WHERE labeled_at > ?", (since or 0,)
            ).fetchone()
        return {"predictions": total, "labeled": labeled, "labeled_frauds": frauds, "flagged": flagged,
                "new_labels": new, "new_label_frauds": new_frauds}

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM predictions")
            self._conn.execute("DELETE FROM labels")
