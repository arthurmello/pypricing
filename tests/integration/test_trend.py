"""Integration smoke for linear time trend."""

from __future__ import annotations

import numpy as np
import pytest

from pypricing import LogLogDemandModel, generate_mock_data

_SMOKE = dict(
    draws=40,
    tune=40,
    chains=2,
    random_seed=0,
    progressbar=False,
    compute_convergence_checks=False,
)


def _dated_panel(**kwargs):
    defaults = dict(
        n_periods=12,
        n_skus=3,
        random_state=0,
        start_date="2020-01-06",
        include_seasonality=False,
        volume_trend=0.1,
    )
    defaults.update(kwargs)
    return generate_mock_data(**defaults)


def test_fit_shared_trend_smoke():
    df = _dated_panel()
    model = LogLogDemandModel(trend="shared")
    model.fit(df, **_SMOKE)
    assert "mu_trend" in model.idata.posterior
    assert "trend_sku" not in model.idata.posterior
    pred = model.predict(df.drop(columns=["quantity"]).copy(), hdi_prob=0.9)
    assert len(pred) == len(df)
    assert model.t0_ is not None


def test_fit_sku_trend_smoke():
    df = _dated_panel()
    model = LogLogDemandModel(trend="sku")
    model.fit(df, **_SMOKE)
    assert "trend_sku" in model.idata.posterior
    assert "mu_trend" in model.idata.posterior


def test_predict_without_period_raises():
    df = _dated_panel()
    model = LogLogDemandModel(trend="shared")
    model.fit(df, **_SMOKE)
    with pytest.raises(ValueError, match="datetime"):
        model.predict(df.drop(columns=["quantity", "period"]))


def test_save_and_load_shared_trend(tmp_path):
    df = _dated_panel(n_skus=2, n_periods=10)
    model = LogLogDemandModel(trend="shared")
    model.fit(df, **_SMOKE)
    path = tmp_path / "trend_model.nc"
    model.save(path)
    loaded = LogLogDemandModel.load(path)
    assert loaded.trend == "shared"
    df_pred = df.drop(columns=["quantity"]).copy()
    out1 = model.predict(df_pred, hdi_prob=0.9, random_seed=1)
    out2 = loaded.predict(df_pred, hdi_prob=0.9, random_seed=1)
    assert np.allclose(
        out1["quantity_mean"].to_numpy(),
        out2["quantity_mean"].to_numpy(),
    )
