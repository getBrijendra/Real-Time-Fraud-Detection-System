import numpy as np
import pandas as pd
import pytest

from fraud.data import RAW_FEATURES, TARGET, V_COLS


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    """Small dataset with the real schema; fraud shifts V14/V17 and Amount."""
    rng = np.random.default_rng(0)
    n = 6000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLS)
    df["Time"] = np.sort(rng.uniform(0, 172_800, n))
    df["Amount"] = rng.exponential(60, n)
    y = (rng.random(n) < 0.03).astype(int)
    df.loc[y == 1, "V14"] -= 4
    df.loc[y == 1, "V17"] -= 3
    df.loc[y == 1, "Amount"] *= 3
    df[TARGET] = y
    return df[[*RAW_FEATURES, TARGET]]
