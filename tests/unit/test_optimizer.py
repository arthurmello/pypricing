"""Unit tests for revenue price optimization (no MCMC fit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import arviz as az
import xarray as xr

from pypricing import LogLogDemandModel
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
    m = mean_revenue_at_price(
        model=model,
        price=p,
        sku_idx=0,
        X_row=None,
    )
    expected = p * np.exp(0.0 + (-2.0) * np.log(p))
    assert m == pytest.approx(expected, rel=1e-10)


def test_price_bounds_missing_sku_raises():
    idata = _idata_log_log_single_sku(elasticity=-1.5)
    model = LogLogDemandModel()
    model.idata = idata
    model.sku_levels_ = pd.Index(["a", "b"])
    model.control_names_ = ()
    model.log_price_midpoint_sku_ = np.zeros(2)
    with pytest.raises(ValueError, match="Missing price_bounds"):
        optimize_prices(model=model, price_bounds={"a": (1.0, 2.0)})
