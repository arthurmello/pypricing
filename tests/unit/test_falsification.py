from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pypricing.data import PanelColumns
from pypricing.diagnostics.falsification import LEAD_COL, add_lead_log_price


def _panel(prices: dict[str, list[float]], periods: list[int]) -> pd.DataFrame:
    rows = [
        {"sku": sku, "period": t, "price": p, "quantity": 10.0}
        for sku, ps in prices.items()
        for t, p in zip(periods, ps)
    ]
    return pd.DataFrame(rows)


def test_lead_is_next_period_price_centered_within_sku():
    df = _panel({"a": [1.0, 2.0, 4.0], "b": [10.0, 10.0, 10.0]}, [1, 2, 3])
    out = add_lead_log_price(df, panel_columns=PanelColumns())

    assert out[out["sku"] == "a"]["period"].tolist() == [1, 2]
    mean_a = np.mean(np.log([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(
        out[out["sku"] == "a"][LEAD_COL], [np.log(2.0) - mean_a, np.log(4.0) - mean_a]
    )
    np.testing.assert_allclose(out[out["sku"] == "b"][LEAD_COL], [0.0, 0.0])


def test_lead_drops_rows_before_a_gap():
    df = _panel({"a": [1.0, 2.0, 3.0]}, [1, 2, 4])
    df = pd.concat([df, _panel({"b": [1.0, 1.0, 1.0, 1.0]}, [1, 2, 3, 4])])
    out = add_lead_log_price(df, panel_columns=PanelColumns())
    assert out[out["sku"] == "a"]["period"].tolist() == [1]
    assert out[out["sku"] == "b"]["period"].tolist() == [1, 2, 3]


def test_lead_is_taken_within_region():
    df = pd.DataFrame(
        {
            "sku": ["a"] * 4,
            "region": ["n", "n", "s", "s"],
            "period": [1, 2, 1, 2],
            "price": [1.0, 2.0, 5.0, 8.0],
            "quantity": 10.0,
        }
    )
    out = add_lead_log_price(df, panel_columns=PanelColumns(region_col="region"))
    mean_a = np.mean(np.log([1.0, 2.0, 5.0, 8.0]))
    assert out["region"].tolist() == ["n", "s"]
    np.testing.assert_allclose(
        out[LEAD_COL], [np.log(2.0) - mean_a, np.log(8.0) - mean_a]
    )


def test_lead_rejects_existing_column_and_missing_period():
    df = _panel({"a": [1.0, 2.0]}, [1, 2])
    with pytest.raises(ValueError, match="already exists"):
        add_lead_log_price(df.assign(**{LEAD_COL: 0.0}), panel_columns=PanelColumns())
    with pytest.raises(ValueError, match="period"):
        add_lead_log_price(df.drop(columns="period"), panel_columns=PanelColumns())
