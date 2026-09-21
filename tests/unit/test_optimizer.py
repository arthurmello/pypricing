"""Unit tests for revenue price optimization (no MCMC fit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import arviz as az
import xarray as xr

from pypricing import LogLogDemandModel, QuadraticLogDemandModel
from pypricing.optimizer import (
    mean_revenue_at_price,
    optimize_prices,
)


def _idata_log_log_single_sku(
    *, elasticity: float, alpha: float = 0.0
) -> az.InferenceData:
    """One SKU, one posterior draw; log_log only needs alpha and elasticity."""
    posterior = xr.Dataset(
        {
            "alpha_sku": (
                ("chain", "draw", "sku"),
                np.array([[[alpha]]], dtype=np.float64),
            ),
            "elasticity_sku": (
                ("chain", "draw", "sku"),
                np.array([[[elasticity]]], dtype=np.float64),
            ),
            "sigma": (("chain", "draw"), np.array([[0.01]], dtype=np.float64)),
        },
        coords={"chain": [0], "draw": [0], "sku": [0]},
    )
    return az.InferenceData(posterior=posterior)


def test_log_log_revenue_max_at_lower_bound_when_elasticity_lt_minus_one():
    """R ∝ p^(1+e); if e < -1, revenue increases as p decreases → optimum at low bound."""
    idata = _idata_log_log_single_sku(elasticity=-2.0)
    model = LogLogDemandModel()
    model.idata = idata
    model.sku_levels_ = pd.Index(["a"])
    model.control_names_ = ()
    model.log_price_midpoint_sku_ = np.array([1.0], dtype=np.float64)
    out = optimize_prices(model=model, price_bounds={"a": (1.0, 10.0)})
    assert out.loc["a", "optimal_price"] == pytest.approx(1.0, rel=1e-6)


def test_log_log_revenue_max_at_upper_bound_when_elasticity_gt_minus_one():
    """If -1 < e < 0, p^(1+e) increases in p → optimum at high bound."""
    idata = _idata_log_log_single_sku(elasticity=-0.5)
    model = LogLogDemandModel()
    model.idata = idata
    model.sku_levels_ = pd.Index(["a"])
    model.control_names_ = ()
    model.log_price_midpoint_sku_ = np.array([1.0], dtype=np.float64)
    out = optimize_prices(model=model, price_bounds={"a": (1.0, 10.0)})
    assert out.loc["a", "optimal_price"] == pytest.approx(10.0, rel=1e-6)


def test_mean_revenue_at_price_matches_closed_form_log_log():
    idata = _idata_log_log_single_sku(elasticity=-2.0, alpha=0.0)
    model = LogLogDemandModel()
    model.idata = idata
    model.sku_levels_ = pd.Index(["a"])
    model.control_names_ = ()
    model.log_price_midpoint_sku_ = np.array([1.0], dtype=np.float64)
    p = 3.0
    sigma = 0.01
    m = mean_revenue_at_price(
        model=model,
        price=p,
        sku_idx=0,
        X_row=None,
    )
    expected = p * np.exp(0.0 + (-2.0) * np.log(p) + 0.5 * sigma**2)
    assert m == pytest.approx(expected, rel=1e-10)


def test_log_log_straddling_minus_one_picks_better_endpoint():
    """A posterior on both sides of -1 is U-shaped; the max is the better bound."""
    posterior = xr.Dataset(
        {
            "alpha_sku": (
                ("chain", "draw", "sku"),
                np.zeros((1, 2, 1), dtype=np.float64),
            ),
            "elasticity_sku": (
                ("chain", "draw", "sku"),
                np.array([[[-2.0], [-0.2]]], dtype=np.float64),
            ),
            "sigma": (("chain", "draw"), np.zeros((1, 2), dtype=np.float64)),
        },
        coords={"chain": [0], "draw": [0, 1], "sku": [0]},
    )
    model = LogLogDemandModel()
    model.idata = az.InferenceData(posterior=posterior)
    model.sku_levels_ = pd.Index(["a"])
    model.control_names_ = ()
    out = optimize_prices(model=model, price_bounds={"a": (1.0, 10.0)})
    assert out.loc["a", "optimal_price"] == pytest.approx(10.0, rel=1e-6)
    low = mean_revenue_at_price(model=model, price=1.0, sku_idx=0, X_row=None)
    assert out.loc["a", "mean_revenue"] > low


def test_at_period_scales_expected_revenue_with_trend():
    idata = _idata_log_log_single_sku(elasticity=-2.0, alpha=0.0)
    idata.posterior["mu_trend"] = (("chain", "draw"), np.array([[1.0]]))
    model = LogLogDemandModel()
    model.idata = idata
    model.sku_levels_ = pd.Index(["a"])
    model.control_names_ = ()
    model.trend = "shared"
    model.t0_ = pd.Timestamp("2020-01-01")
    early = mean_revenue_at_price(
        model=model, price=2.0, sku_idx=0, X_row=None, at_period="2020-01-01"
    )
    late = mean_revenue_at_price(
        model=model, price=2.0, sku_idx=0, X_row=None, at_period="2021-01-01"
    )
    assert late > early * 2.0
    out = optimize_prices(
        model=model,
        price_bounds={"a": (1.0, 10.0)},
        at_period="2021-01-01",
    )
    assert out.loc["a", "optimal_price"] == pytest.approx(1.0, rel=1e-6)
    at_opt = mean_revenue_at_price(
        model=model,
        price=float(out.loc["a", "optimal_price"]),
        sku_idx=0,
        X_row=None,
        at_period="2021-01-01",
    )
    assert out.loc["a", "mean_revenue"] == pytest.approx(at_opt, rel=1e-10)


def test_search_finds_the_higher_of_two_revenue_peaks():
    """A unimodal polish over the whole interval can stop on the lower peak."""
    curvature = -12.0
    midpoint = float(np.log(10.0))
    peak_prices = (3.0, 30.0)
    peak_revenues = (100.0, 200.0)
    alphas = []
    elasticities = []
    for price, revenue in zip(peak_prices, peak_revenues):
        log_p = float(np.log(price))
        alphas.append(np.log(revenue) + curvature * log_p**2)
        elasticities.append(-1.0 + 2.0 * curvature * (midpoint - log_p))
    posterior = xr.Dataset(
        {
            "alpha_sku": (
                ("chain", "draw", "sku"),
                np.array(alphas, dtype=np.float64).reshape(1, 2, 1),
            ),
            "elasticity_sku": (
                ("chain", "draw", "sku"),
                np.array(elasticities, dtype=np.float64).reshape(1, 2, 1),
            ),
            "curvature_sku": (
                ("chain", "draw", "sku"),
                np.full((1, 2, 1), curvature, dtype=np.float64),
            ),
            "sigma": (("chain", "draw"), np.zeros((1, 2), dtype=np.float64)),
        },
        coords={"chain": [0], "draw": [0, 1], "sku": [0]},
    )
    model = QuadraticLogDemandModel()
    model.idata = az.InferenceData(posterior=posterior)
    model.sku_levels_ = pd.Index(["a"])
    model.control_names_ = ()
    model.log_price_midpoint_sku_ = np.array([midpoint], dtype=np.float64)
    out = optimize_prices(model=model, price_bounds={"a": (1.0, 50.0)})
    assert out.loc["a", "optimal_price"] == pytest.approx(30.0, rel=0.05)
    at_lower_peak = mean_revenue_at_price(
        model=model, price=3.0, sku_idx=0, X_row=None
    )
    assert out.loc["a", "mean_revenue"] > at_lower_peak


def test_price_bounds_missing_sku_raises():
    idata = _idata_log_log_single_sku(elasticity=-1.5)
    model = LogLogDemandModel()
    model.idata = idata
    model.sku_levels_ = pd.Index(["a", "b"])
    model.control_names_ = ()
    model.log_price_midpoint_sku_ = np.zeros(2)
    with pytest.raises(ValueError, match="Missing price_bounds"):
        optimize_prices(model=model, price_bounds={"a": (1.0, 2.0)})
