from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pypricing import LogLogDemandModel, QuadraticLogDemandModel
from pypricing.posterior import (
    elasticities,
    summarize_quantity_multiplier_one_sku,
)


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


def test_model_sku_coord_is_the_label():
    df = pd.DataFrame(
        {
            "sku": ["9", "10", "9", "10"],
            "price": [10.0, 12.0, 11.0, 13.0],
            "quantity": [5.0, 4.0, 6.0, 3.0],
        }
    )
    model = LogLogDemandModel()
    model.build_model(df)
    assert [str(sku) for sku in model.model.coords["sku"]] == [
        str(sku) for sku in model.sku_levels_
    ]
    assert "10" in {str(sku) for sku in model.model.coords["sku"]}


def test_elasticities_follow_sku_labels_not_integer_positions():
    posterior = xr.Dataset(
        {
            "elasticity_sku": (
                ("chain", "draw", "sku"),
                np.array([[[-1.0, -3.0], [-1.0, -3.0]]], dtype=np.float64),
            ),
        },
        coords={"chain": [0], "draw": [0, 1], "sku": ["9", "10"]},
    )
    model = LogLogDemandModel()
    model.idata = az.InferenceData(posterior=posterior)
    model.sku_levels_ = pd.Index(["9", "10"])

    out = elasticities(model, hdi_prob=0.5)
    assert list(out.index) == ["9", "10"]
    assert out.loc["10", "mean"] == pytest.approx(-3.0)
    assert out.loc["9", "mean"] == pytest.approx(-1.0)

    one = summarize_quantity_multiplier_one_sku(
        model, sku="10", price_multiplier=2.0, hdi_prob=0.5
    )
    assert one["quantity_multiplier_mean"] == pytest.approx(2.0**-3.0)
