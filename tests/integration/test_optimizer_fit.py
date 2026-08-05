"""Integration smoke: optimize_prices after a short MCMC fit."""

from __future__ import annotations

import numpy as np
import pytest

from pypricing import (
    LogLogDemandModel,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
    generate_mock_data,
)


@pytest.mark.parametrize(
    "model_cls",
    [
        LogLogDemandModel,
        QuadraticLogDemandModel,
        SigmoidSaturationDemandModel,
    ],
)
def test_optimize_prices_smoke_after_fit(model_cls):
    df = generate_mock_data(
        n_periods=12,
        n_skus=3,
        n_controls=1,
        random_state=2,
    )
    model = model_cls()
    model.fit(
        df,
        draws=60,
        tune=60,
        chains=2,
        random_seed=3,
        progressbar=False,
        compute_convergence_checks=False,
    )
    bounds = {
        sku: (
            float(df.loc[df["sku"] == sku, "price"].min() * 0.5),
            float(df.loc[df["sku"] == sku, "price"].max() * 1.5),
        )
        for sku in model.sku_levels_
    }
    ctrl = df.groupby("sku", as_index=False).agg(control_1=("control_1", "median"))

    out = model.optimize_prices(price_bounds=bounds, controls_df=ctrl)
    assert len(out) == 3
    assert np.all(np.isfinite(out["optimal_price"].to_numpy()))
    assert np.all(np.isfinite(out["mean_revenue"].to_numpy()))
    for sku in model.sku_levels_:
        lo, hi = bounds[sku]
        assert lo <= out.loc[sku, "optimal_price"] <= hi


def test_controls_required_when_model_has_controls():
    df = generate_mock_data(n_periods=8, n_skus=2, n_controls=1, random_state=0)
    model = LogLogDemandModel()
    model.fit(
        df,
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    bounds = {sku: (0.5, 50.0) for sku in model.sku_levels_}
    with pytest.raises(ValueError, match="controls_df"):
        model.optimize_prices(price_bounds=bounds)
