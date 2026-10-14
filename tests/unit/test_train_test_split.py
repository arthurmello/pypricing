from __future__ import annotations

import pandas as pd
import pytest

from pypricing import LogLogDemandModel


def _panel(periods) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": ["a", "b"] * len(periods),
            "period": list(periods) * 2,
        }
    ).sample(frac=1.0, random_state=0)


@pytest.mark.parametrize(
    "periods",
    [
        range(10),
        pd.date_range("2024-01-01", periods=10, freq="W"),
        pd.date_range("2024-01-01", periods=10, freq="W", tz="UTC"),
    ],
)
def test_split_holds_out_the_latest_periods(periods):
    df = _panel(periods)
    train, test = LogLogDemandModel().train_test_split(df, test_size=0.2)

    held_out = list(periods)[-2:]
    assert len(test) == 4
    assert len(train) == 16
    assert set(pd.Index(test["period"])) == set(pd.Index(held_out))
