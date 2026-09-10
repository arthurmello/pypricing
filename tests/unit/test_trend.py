from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data
from pypricing.model_components.posterior_mu import add_trend


def test_invalid_trend_value():
    with pytest.raises(ValueError, match="trend must be"):
        LogLogDemandModel(trend="linear")  # type: ignore[arg-type]


def test_trend_rejects_integer_period():
    df = generate_mock_data(n_periods=6, n_skus=2, random_state=0)
    with pytest.raises(ValueError, match="numeric"):
        LogLogDemandModel(trend="shared").build_model(df)


def test_trend_requires_period_column():
    df = generate_mock_data(
        n_periods=6,
        n_skus=2,
        random_state=0,
        start_date="2020-01-06",
    ).drop(columns=["period"])
    with pytest.raises(ValueError, match="Missing period column"):
        LogLogDemandModel(trend="sku").build_model(df)


def test_add_trend_shared():
    posterior = xr.Dataset(
        {"mu_trend": (("chain", "draw"), np.array([[0.1]], dtype=np.float64))},
        coords={"chain": [0], "draw": [0]},
    )
    mu = np.zeros((1, 1, 2), dtype=np.float64)
    out = add_trend(mu, posterior, np.array([0.0, 2.0]), np.array([0, 0]))
    assert out[0, 0, 0] == pytest.approx(0.0)
    assert out[0, 0, 1] == pytest.approx(0.2)


def test_add_trend_sku():
    posterior = xr.Dataset(
        {
            "trend_sku": (
                ("chain", "draw", "sku"),
                np.array([[[0.1, 0.5]]], dtype=np.float64),
            )
        },
        coords={"chain": [0], "draw": [0], "sku": [0, 1]},
    )
    mu = np.zeros((1, 1, 2), dtype=np.float64)
    out = add_trend(mu, posterior, np.array([2.0, 2.0]), np.array([0, 1]))
    assert out[0, 0, 0] == pytest.approx(0.2)
    assert out[0, 0, 1] == pytest.approx(1.0)


def test_compute_mu_includes_shared_trend():
    posterior = xr.Dataset(
        {
            "alpha_sku": (
                ("chain", "draw", "sku"),
                np.array([[[0.0]]], dtype=np.float64),
            ),
            "elasticity_sku": (
                ("chain", "draw", "sku"),
                np.array([[[-1.0]]], dtype=np.float64),
            ),
            "mu_trend": (("chain", "draw"), np.array([[0.1]], dtype=np.float64)),
        },
        coords={"chain": [0], "draw": [0], "sku": [0]},
    )
    model = LogLogDemandModel(trend="shared")
    mu = model.compute_mu_from_posterior(
        posterior=posterior,
        log_price=np.array([0.0], dtype=np.float64),
        obs_sku_idx=np.array([0], dtype=np.int64),
        X_control=None,
        t_years=np.array([2.0], dtype=np.float64),
    )
    assert mu[0, 0, 0] == pytest.approx(0.2)


def test_sku_trend_with_hierarchy_builds():
    df = generate_mock_data(
        n_periods=8,
        n_skus=4,
        hierarchy_levels=(2,),
        random_state=0,
        start_date="2020-01-06",
        include_seasonality=False,
    )
    model = LogLogDemandModel(
        trend="sku",
        panel_columns=PanelColumns(group_columns=("category_1",)),
    )
    model.build_model(df)
    assert "trend_sku" in model.model.named_vars
    assert "mu_trend" in model.model.named_vars
