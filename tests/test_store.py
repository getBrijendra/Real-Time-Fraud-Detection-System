import numpy as np

from fraud.data import RAW_FEATURES
from fraud.store import PredictionStore


def test_labels_match_in_either_order(tmp_path, synthetic_raw):
    store = PredictionStore(tmp_path / "s.db")
    raw = synthetic_raw[RAW_FEATURES].iloc[:4]
    ids = ["a", "b", "c", "d"]

    # label for "c" arrives before its transaction is scored (consumer catching up on a backlog)
    assert store.add_labels([("c", 1)]) == []

    late = store.log_predictions(raw, ids, np.array([0.1, 0.2, 0.9, 0.3]), np.array([0, 0, 1, 0]), "v1")
    assert [(m["transaction_id"], m["label"], m["flagged"]) for m in late] == [("c", 1, 1)]

    # normal order: prediction first, label later
    matched = store.add_labels([("a", 0), ("zzz", 1)])
    assert [m["transaction_id"] for m in matched] == ["a"]

    counts = store.counts()
    assert counts["labeled"] == 2 and counts["labeled_frauds"] == 1

    back = store.labeled_transactions()
    assert set(back.transaction_id) == {"a", "c"}
    np.testing.assert_allclose(back.set_index("transaction_id").loc["c", RAW_FEATURES].to_numpy(dtype=float),
                               raw.iloc[2].to_numpy(dtype=float))
