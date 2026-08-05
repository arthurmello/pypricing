from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pypricing import LogLogDemandModel, QuadraticLogDemandModel


def _idata_log_log_two_skus() -> az.InferenceData:
    posterior = xr.Dataset(
        {
            "elasticity_sku": (
                ("chain", "draw", "sku"),
                np.array([[[-1.0, -0.5], [-2.0, -1.5]]], dtype=np.float64),
            ),
            "sigma": (("chain", "draw"), np.array([[0.1, 0.1]], dtype=np.float64)),
        },
        coords={"chain": [0], "draw": [0, 1], "sku": [0, 1]},
    )
    return az.InferenceData(posterior=posterior)


def test_quantity_multiplier_summary_returns_all_skus_summary():
    model = LogLogDemandModel()
    model.idata = _idata_log_log_two_skus()
    model.sku_levels_ = pd.Index(["sku_a", "sku_b"])

    out = model.quantity_multiplier_summary(price_multiplier=2.0, hdi_prob=0.9)

    assert list(out.index) == ["sku_a", "sku_b"]
    assert {
        "quantity_multiplier_mean",
        "quantity_multiplier_hdi_lower",
        "quantity_multiplier_hdi_upper",
    }.issubset(out.columns)

    assert out.loc["sku_a", "quantity_multiplier_mean"] == pytest.approx(0.375)
    assert out.loc["sku_b", "quantity_multiplier_mean"] == pytest.approx(
        (2**-0.5 + 2**-1.5) / 2
    )


def test_quantity_multiplier_summary_non_log_log_raises():
    model = QuadraticLogDemandModel()
    model.idata = _idata_log_log_two_skus()
    model.sku_levels_ = pd.Index(["sku_a", "sku_b"])

    with pytest.raises(NotImplementedError, match="LogLogDemandModel"):
        model.quantity_multiplier_summary(price_multiplier=1.1)


def test_quantity_multiplier_summary_return_draws_returns_summary_and_draws():
    model = LogLogDemandModel()
    model.idata = _idata_log_log_two_skus()
    model.sku_levels_ = pd.Index(["sku_a", "sku_b"])

    summary, draws = model.quantity_multiplier_summary(
        price_multiplier=2.0,
        hdi_prob=0.9,
        return_draws=True,
    )

    assert isinstance(summary, pd.DataFrame)
    assert isinstance(draws, xr.DataArray)
    assert draws.dims == ("chain", "draw", "sku")
    assert np.allclose(
        draws.values,
        np.array([[[2**-1.0, 2**-0.5], [2**-2.0, 2**-1.5]]], dtype=np.float64),
    )
