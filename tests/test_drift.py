import numpy as np
import pandas as pd

from fraud.drift import ReferenceProfile, psi
from fraud.features import build_features
from fraud.streaming.producer import inject_drift


def test_psi_identical_is_zero():
    p = np.full(10, 0.1)
    assert psi(p, p) == 0


def test_reference_detects_shift(synthetic_raw):
    X = build_features(synthetic_raw)
    half = len(X) // 2
    rng = np.random.default_rng(0)
    ref = ReferenceProfile.build(X.iloc[:half], rng.random(half), exclude=["hour_sin", "hour_cos"])
    assert "hour_sin" not in ref.features

    same = ref.report(X.iloc[half:], rng.random(len(X) - half))
    assert not same["dataset_drift"]
    assert max(r["psi"] for r in same["features"].values()) < 0.1

    drifted_raw = inject_drift(synthetic_raw.iloc[half:], "both", 1.0)
    shifted = ref.report(build_features(drifted_raw), rng.random(len(X) - half) ** 3)
    assert shifted["dataset_drift"]
    assert "log_amount" in shifted["top_drifted"]
    assert shifted["score"]["psi"] > 0.25


def test_report_handles_constant_feature():
    X = pd.DataFrame({"a": np.zeros(1000), "b": np.arange(1000.0)})
    ref = ReferenceProfile.build(X, np.linspace(0, 1, 1000))
    rep = ref.report(X, np.linspace(0, 1, 1000))
    assert rep["features"]["a"]["psi"] == 0
