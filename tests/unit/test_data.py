import numpy as np
import pandas as pd
import pytest

from pypricing.data import PanelColumns, PricePanelData


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
