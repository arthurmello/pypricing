import numpy as np
import pandas as pd
import pytest

from pypricing import generate_mock_data


def test_shape_and_columns():
    df = generate_mock_data(
        n_periods=10,
        n_skus=5,
        n_categories=2,
        n_regions=3,
        n_controls=2,
        random_state=0,
    )
    assert len(df) == 10 * 5 * 3
    assert set(df.columns) >= {
        "sku",
        "period",
        "price",
        "quantity",
        "log_price",
        "log_quantity",
        "category",
        "region",
        "control_1",
        "control_2",
    }
    assert df["sku"].nunique() == 5
    assert df["category"].str.startswith("category_").all()
    assert df["region"].str.startswith("region_").all()


def test_no_optional_dims():
    df = generate_mock_data(n_periods=5, n_skus=4, random_state=1)
    assert len(df) == 5 * 4
    assert "category" not in df.columns
    assert "region" not in df.columns
    assert "control_1" not in df.columns
    assert "date" not in df.columns


def test_date_column_from_start():
    df = generate_mock_data(
        n_periods=10,
        n_skus=4,
        random_state=2,
        start_date="2020-01-06",
        freq="W",
    )
    assert len(df) == 10 * 4
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].nunique() == 10


def test_n_periods_positive():
    with pytest.raises(ValueError, match="n_periods"):
        generate_mock_data(0, n_skus=5, random_state=0)


@pytest.mark.parametrize("shape", ["log_log", "quadratic", "sigmoid"])
def test_demand_shape_smoke(shape):
    df = generate_mock_data(
        n_periods=6, n_skus=2, n_controls=1, random_state=0, shape=shape
    )
    assert len(df) == 12
    assert df["quantity"].gt(0).all()


def test_demand_shape_invalid():
    with pytest.raises(ValueError, match="not supported"):
        generate_mock_data(n_periods=3, n_skus=2, random_state=0, shape="linear")


def test_hierarchy_levels_columns_and_conflict():
    df = generate_mock_data(
        n_periods=4,
        n_skus=6,
        hierarchy_levels=(2, 3),
        random_state=1,
    )
    assert "category_1" in df.columns
    assert "category_2" in df.columns
    assert "category" not in df.columns
    assert len(df) == 4 * 6
    assert df["sku"].nunique() == 6

    with pytest.raises(ValueError, match="only one of"):
        generate_mock_data(
            n_periods=2,
            n_skus=4,
            n_categories=2,
            hierarchy_levels=(2,),
            random_state=0,
        )


@pytest.mark.parametrize("shape", ["log_log", "quadratic", "sigmoid"])
def test_hierarchy_demand_shapes(shape):
    df = generate_mock_data(
        n_periods=5,
        n_skus=4,
        hierarchy_levels=(2, 2),
        random_state=2,
        shape=shape,
    )
    assert len(df) == 20
    assert df["quantity"].gt(0).all()


def test_return_truth_log_log():
    out = generate_mock_data(
        n_periods=6,
        n_skus=3,
        n_controls=0,
        random_state=0,
        shape="log_log",
        return_truth=True,
    )
    assert isinstance(out, tuple)
    df, truth = out
    assert len(df) == 18
    assert truth.shape == "log_log"
    assert truth.sku_labels == ("sku_1", "sku_2", "sku_3")
    assert truth.elasticity_sku.shape == (3,)
    assert truth.alpha_sku.shape == (3,)
    assert truth.curvature_sku is None
    assert truth.noise_sigma > 0
    assert np.all(truth.elasticity_sku < 0)


def test_return_truth_quadratic_has_curvature():
    _df, truth = generate_mock_data(
        n_periods=4,
        n_skus=2,
        random_state=1,
        shape="quadratic",
        return_truth=True,
    )
    assert truth.curvature_sku is not None
    assert truth.curvature_sku.shape == (2,)


def test_round_quantity_false_keeps_continuous_qty():
    df, _truth = generate_mock_data(
        n_periods=8,
        n_skus=3,
        random_state=11,
        shape="log_log",
        include_seasonality=False,
        round_quantity=False,
        return_truth=True,
    )
    assert np.issubdtype(df["quantity"].dtype, np.floating)
    # Seed 11 previously floored sku_1 to qty==1 under integer rounding.
    assert df.loc[df["sku"] == "sku_1", "log_quantity"].std() > 0.05


def test_include_seasonality_flag():
    df_on = generate_mock_data(
        n_periods=20, n_skus=2, random_state=0, include_seasonality=True
    )
    df_off = generate_mock_data(
        n_periods=20, n_skus=2, random_state=0, include_seasonality=False
    )
    assert not np.allclose(
        df_on["log_quantity"].to_numpy(), df_off["log_quantity"].to_numpy()
    )
