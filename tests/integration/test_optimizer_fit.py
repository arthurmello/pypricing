"""Integration smoke: optimize_prices after a short MCMC fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def test_margin_objective_has_interior_optimum_for_log_log():
    df = generate_mock_data(n_periods=24, n_skus=2, n_controls=0, random_state=5)
    model = LogLogDemandModel()
    model.fit(
        df,
        draws=100,
        tune=100,
        chains=2,
        random_seed=1,
        progressbar=False,
        compute_convergence_checks=False,
    )
    cost = {
        sku: float(df.loc[df["sku"] == sku, "price"].median() * 0.4)
        for sku in model.sku_levels_
    }
    bounds = {sku: (c * 1.01, c * 20) for sku, c in cost.items()}
    out = model.optimize_prices(price_bounds=bounds, objective="margin", unit_cost=cost)
    e = model.elasticities()["mean"]
    for sku in model.sku_levels_:
        if e[sku] < -1.2:
            assert pd.isna(out.loc[sku, "at_bound"])
            lerner = cost[sku] * e[sku] / (1 + e[sku])
            assert abs(out.loc[sku, "optimal_price"] / lerner - 1) < 0.25
    assert (out["mean_margin"] < out["mean_revenue"]).all()
