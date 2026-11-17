"""Period effects and Mundlak: configuration, shapes, and prediction rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arviz as az

from pypricing import LogLogDemandModel, PanelColumns, PricePanelData, generate_mock_data
from pypricing.diagnostics.identification import within_period_price_share


def _panel(**kwargs):
    defaults = dict(
        n_periods=6,
        n_skus=3,
        n_controls=0,
        include_seasonality=False,
        random_state=0,
    )
    defaults.update(kwargs)
    return generate_mock_data(**defaults)


def test_time_effects_rejects_trend_seasonality_and_missing_region():
    with pytest.raises(ValueError, match="trend or seasonality"):
        LogLogDemandModel(time_effects="period", trend="shared")
    with pytest.raises(ValueError, match="trend or seasonality"):
        LogLogDemandModel(time_effects="period", seasonality="yearly")
    with pytest.raises(ValueError, match="region_col"):
        LogLogDemandModel(time_effects="region_period")
    with pytest.raises(ValueError, match="time_effects must be"):
        LogLogDemandModel(time_effects="week")  # type: ignore[arg-type]


def test_time_effects_requires_two_skus_per_cell():
    df = _panel(n_skus=1, n_periods=4)
    model = LogLogDemandModel(
        time_effects="period",
        panel_columns=PanelColumns(quantity_floor=1e-12),
    )
    with pytest.raises(ValueError, match="2 SKUs"):
        model.build_model(df)


def test_period_coords_and_prior_are_built():
    df = _panel()
    panel = PricePanelData.from_frame(df)
    assert panel.obs_period_idx is not None
    assert panel.obs_period_idx.shape == (len(df),)
    assert "period" in panel.coords()
    assert len(panel.coords()["period"]) == df["period"].nunique()

    model = LogLogDemandModel(
        time_effects="period",
        panel_columns=PanelColumns(quantity_floor=1e-12),
    )
    model.build_model(df)
    assert "period" in model.model.coords
    assert "lambda_period" in model.model.named_vars
    assert "sigma_lambda_period" in model.model.named_vars
    assert "region" not in model.model.coords

    plain = LogLogDemandModel(panel_columns=PanelColumns(quantity_floor=1e-12))
    plain.build_model(df)
    assert "lambda_period" not in plain.model.named_vars


def test_region_period_coords():
    df = _panel(n_regions=2)
    model = LogLogDemandModel(
        time_effects="region_period",
        panel_columns=PanelColumns(region_col="region", quantity_floor=1e-12),
    )
    model.build_model(df)
    assert "region" in model.model.coords
    assert "lambda_period" in model.model.named_vars
    assert "lambda_region_period" in model.model.named_vars


def test_mundlak_stores_sku_means_and_gamma():
    df = _panel()
    model = LogLogDemandModel(
        mundlak=True,
        panel_columns=PanelColumns(quantity_floor=1e-12),
    )
    model.build_model(df)
    assert model.sku_mean_log_price_ is not None
    assert model.sku_mean_log_price_.shape == (3,)
    assert "gamma_mundlak" in model.model.named_vars

    off = LogLogDemandModel(panel_columns=PanelColumns(quantity_floor=1e-12))
    off.build_model(df)
    assert off.sku_mean_log_price_ is None
    assert "gamma_mundlak" not in off.model.named_vars


def test_clone_keeps_time_effects_and_mundlak():
    model = LogLogDemandModel(time_effects="period", mundlak=True)
    copy = model.clone()
    assert copy.time_effects == "period"
    assert copy.mundlak is True
    assert copy.idata is None


def _shell_with_posterior():
    df = _panel(n_periods=4, n_skus=3)
    model = LogLogDemandModel(
        time_effects="period",
        panel_columns=PanelColumns(quantity_floor=1e-12),
    )
    model.build_model(df)
    n_period = len(model.period_levels_)
    n_draw = 800
    rng = np.random.default_rng(0)
    lam = rng.normal(size=(1, n_draw, n_period))
    sigma = np.full((1, n_draw), 0.4)
    posterior = xr.Dataset(
        {
            "lambda_period": (("chain", "draw", "period"), lam),
            "sigma_lambda_period": (("chain", "draw"), sigma),
            "sigma": (("chain", "draw"), np.ones((1, n_draw))),
            "alpha_sku": (("chain", "draw", "sku"), np.zeros((1, n_draw, 3))),
            "elasticity_sku": (("chain", "draw", "sku"), -np.ones((1, n_draw, 3))),
        },
        coords={
            "chain": [0],
            "draw": np.arange(n_draw),
            "period": np.asarray(model.period_levels_),
            "sku": np.arange(3),
        },
    )
    model.idata = az.InferenceData(posterior=posterior)
    return model, df, lam


def test_in_sample_period_uses_fitted_lambda_and_future_follows_period_effect():
    model, df, lam = _shell_with_posterior()
    periods = df["period"].to_numpy()
    insample = model._period_effect_draws(
        len(df), periods=periods, period_effect="draw", random_seed=1
    )
    codes = periods.astype(int)
    assert np.allclose(insample[:, :, 0], lam[:, :, codes[0]])

    future = np.array([10_000])
    neutral = model._period_effect_draws(
        1, periods=future, period_effect="neutral", random_seed=0
    )
    draw = model._period_effect_draws(
        1, periods=future, period_effect="draw", random_seed=0
    )
    assert np.allclose(neutral, 0.0)
    assert abs(float(draw.mean())) < 0.05
    assert float(draw.std()) > float(neutral.std())

    past = model.period_levels_[1]
    reused = model._period_effect_draws(1, periods=future, period_effect=past)
    assert np.allclose(reused[:, :, 0], lam[:, :, 1])

    with pytest.raises(ValueError, match="not a training period"):
        model._period_effect_draws(1, periods=future, period_effect=999)


def test_within_period_price_share_flags_a_common_price():
    df = _panel(n_periods=12, n_skus=4)
    # Base prices span much more than a typical price shock, so rebuild prices
    # with a small SKU level and a large SKU-period shock.
    rng = np.random.default_rng(0)
    sku_level = 0.05 * pd.Categorical(df["sku"]).codes
    df["price"] = np.exp(2.0 + sku_level + rng.normal(0.0, 0.4, size=len(df)))
    varied = LogLogDemandModel(
        time_effects="period",
        panel_columns=PanelColumns(quantity_floor=1e-12),
    )
    varied.build_model(df)
    assert within_period_price_share(varied) > 0.1

    flat = df.copy()
    flat["price"] = flat.groupby("period")["price"].transform("first")
    common = LogLogDemandModel(
        time_effects="period",
        panel_columns=PanelColumns(quantity_floor=1e-12),
    )
    common.build_model(flat)
    assert within_period_price_share(common) < 0.1
