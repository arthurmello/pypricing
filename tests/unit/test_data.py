import numpy as np
import pandas as pd
import pytest

from pypricing.data import PanelColumns, PricePanelData, parse_period_index


def test_from_frame_basic():
    df = pd.DataFrame(
        {
            "sku": ["a", "a", "b"],
            "price": [10.0, np.e, 20.0],
            "quantity": [100.0, 100.0, 400.0],
        }
    )
    d = PricePanelData.from_frame(df)
    assert d.n_obs == 3
    assert d.n_skus == 2
    assert np.allclose(d.log_price[1], 1.0)
    assert np.allclose(d.log_quantity[0], np.log(100.0))
    assert d.control_matrix.shape == (3, 0)
    assert np.array_equal(d.obs_sku_idx, [0, 0, 1])


def test_zero_quantity_floor():
    df = pd.DataFrame(
        {
            "sku": ["a"],
            "price": [5.0],
            "quantity": [0.0],
        }
    )
    d = PricePanelData.from_frame(df, panel_columns=PanelColumns(quantity_floor=0.01))
    assert d.log_quantity[0] == np.log(0.01)


def test_auto_controls():
    df = pd.DataFrame(
        {
            "sku": ["x"] * 4,
            "price": [2.0, 2.0, 2.0, 2.0],
            "quantity": [10.0, 10.0, 10.0, 10.0],
            "control_1": [0.0, 0.1, 0.2, 0.3],
        }
    )
    d = PricePanelData.from_frame(df)
    assert d.control_names == ("control_1",)
    assert d.control_matrix.shape == (4, 1)


def test_explicit_controls_subset():
    df = pd.DataFrame(
        {
            "sku": ["x", "x"],
            "price": [1.0, 1.0],
            "quantity": [2.0, 2.0],
            "control_1": [0.0, 0.1],
            "control_2": [1.0, 2.0],
        }
    )
    d = PricePanelData.from_frame(
        df, panel_columns=PanelColumns(control_columns=("control_2",))
    )
    assert d.control_names == ("control_2",)


def test_nan_rejected():
    df = pd.DataFrame(
        {
            "sku": ["a", "a"],
            "price": [1.0, np.nan],
            "quantity": [2.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="NaN"):
        PricePanelData.from_frame(df)


def test_non_positive_price_rejected():
    df = pd.DataFrame(
        {
            "sku": ["a"],
            "price": [0.0],
            "quantity": [1.0],
        }
    )
    with pytest.raises(ValueError, match="strictly positive"):
        PricePanelData.from_frame(df)


def test_group_columns_coords_and_indices():
    df = pd.DataFrame(
        {
            "sku": ["a", "a", "b", "b"],
            "price": [10.0, 10.0, 20.0, 20.0],
            "quantity": [1.0, 1.0, 2.0, 2.0],
            "category_1": ["c0", "c0", "c1", "c1"],
        }
    )
    d = PricePanelData.from_frame(
        df, panel_columns=PanelColumns(group_columns=("category_1",))
    )
    assert d.hierarchy is not None
    assert d.hierarchy.n_groups_per_level == (2,)
    assert d.hierarchy.sku_to_level_idx.shape == (2, 1)
    assert d.hierarchy.group_columns == ("category_1",)
    c = d.coords()
    assert "group_0" in c and len(c["group_0"]) == 2


def test_group_columns_inconsistent_per_sku_raises():
    df = pd.DataFrame(
        {
            "sku": ["a", "a"],
            "price": [10.0, 10.0],
            "quantity": [1.0, 1.0],
            "category_1": ["c0", "c1"],
        }
    )
    with pytest.raises(ValueError, match="constant within each SKU"):
        PricePanelData.from_frame(
            df, panel_columns=PanelColumns(group_columns=("category_1",))
        )


def _period_frame(period) -> pd.DataFrame:
    n = len(period)
    return pd.DataFrame(
        {
            "sku": ["a"] * n,
            "price": [10.0] * n,
            "quantity": [1.0] * n,
            "period": period,
        }
    )


def test_parse_period_index_datetime():
    values = pd.date_range("2024-01-01", periods=3, freq="W")
    idx = parse_period_index(values)
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.tz is None
    assert list(idx) == list(values)


def test_parse_period_index_strings():
    idx = parse_period_index(pd.Series(["2024-01-01", "2024-01-08"]))
    assert idx[0] == pd.Timestamp("2024-01-01")
    assert idx[1] == pd.Timestamp("2024-01-08")


def test_parse_period_index_tz_aware_to_naive_utc():
    values = pd.date_range("2024-01-01", periods=2, freq="D", tz="US/Eastern")
    idx = parse_period_index(values)
    assert idx.tz is None
    assert idx[0] == pd.Timestamp("2024-01-01 05:00:00")


def test_parse_period_index_rejects_numeric():
    with pytest.raises(ValueError, match="numeric"):
        parse_period_index(pd.Series([0, 1, 2]))
    with pytest.raises(ValueError, match="numeric"):
        parse_period_index(pd.Series([0.0, 1.0]))
    with pytest.raises(ValueError, match="numeric"):
        parse_period_index(pd.Series([0, 1, 2], dtype=object))


def test_parse_period_index_rejects_unparseable_strings():
    with pytest.raises(ValueError, match="could not be parsed"):
        parse_period_index(pd.Series(["week_1", "week_2"]))


def test_parse_period_index_rejects_nat():
    with pytest.raises(ValueError, match="NaN/NaT"):
        parse_period_index(pd.Series([pd.Timestamp("2024-01-01"), pd.NaT]))


def test_from_frame_integer_period_leaves_index_none():
    d = PricePanelData.from_frame(_period_frame([0, 1, 2]))
    assert d.period_index is None


def test_from_frame_datetime_period_attaches_index():
    periods = pd.date_range("2024-01-01", periods=3, freq="W")
    d = PricePanelData.from_frame(_period_frame(periods))
    assert d.period_index is not None
    assert len(d.period_index) == d.n_obs
    assert d.period_index.equals(pd.DatetimeIndex(periods))


def test_from_frame_parse_period_strings():
    d = PricePanelData.from_frame(
        _period_frame(["2024-01-01", "2024-01-08"]), parse_period=True
    )
    assert d.period_index is not None
    assert d.period_index[0] == pd.Timestamp("2024-01-01")


def test_from_frame_parse_period_rejects_integer():
    with pytest.raises(ValueError, match="numeric"):
        PricePanelData.from_frame(_period_frame([0, 1]), parse_period=True)


def test_from_frame_parse_period_requires_column():
    df = pd.DataFrame({"sku": ["a"], "price": [1.0], "quantity": [1.0]})
    with pytest.raises(ValueError, match="Missing period column"):
        PricePanelData.from_frame(df, parse_period=True)
