"""Integration smoke for Fourier seasonality."""

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


def _weekly_panel(**kwargs):
    defaults = dict(
        n_periods=20,
        n_skus=3,
        random_state=0,
        start_date="2020-01-06",
        freq="W",
        include_seasonality=True,
    )
    defaults.update(kwargs)
    return generate_mock_data(**defaults)


def test_fit_yearly_seasonality_smoke():
    df = _weekly_panel()
    model = LogLogDemandModel(seasonality="yearly")
    model.fit(df, **_SMOKE)
    assert "beta_season" in model.idata.posterior
    assert model.seasonality_ == ("yearly",)
    assert model.idata.posterior["beta_season"].values.shape[-1] == 4
    pred = model.predict(df.drop(columns=["quantity"]).copy(), hdi_prob=0.9)
    assert len(pred) == len(df)


def test_fit_weekly_and_yearly_smoke():
    df = generate_mock_data(
        n_periods=28,
        n_skus=2,
        random_state=1,
        start_date="2020-01-01",
        freq="D",
        include_seasonality=True,
    )
    model = LogLogDemandModel(seasonality=("yearly", "weekly"))
    model.fit(df, **_SMOKE)
    assert model.seasonality_ == ("yearly", "weekly")
    assert model.idata.posterior["beta_season"].values.shape[-1] == 6


def test_predict_without_period_raises():
    df = _weekly_panel(n_skus=2, n_periods=12)
    model = LogLogDemandModel(seasonality="yearly")
    model.fit(df, **_SMOKE)
    with pytest.raises(ValueError, match="datetime"):
        model.predict(df.drop(columns=["quantity", "period"]))


def test_save_and_load_yearly_seasonality(tmp_path):
    df = _weekly_panel(n_skus=2, n_periods=12)
    model = LogLogDemandModel(seasonality="auto")
    model.fit(df, **_SMOKE)
    path = tmp_path / "season_model.nc"
    model.save(path)
    loaded = LogLogDemandModel.load(path)
    assert loaded.seasonality == "auto"
    assert loaded.seasonality_ == ("yearly",)
    df_pred = df.drop(columns=["quantity"]).copy()
    out1 = model.predict(df_pred, hdi_prob=0.9, random_seed=1)
    out2 = loaded.predict(df_pred, hdi_prob=0.9, random_seed=1)
    assert np.allclose(
        out1["quantity_mean"].to_numpy(),
        out2["quantity_mean"].to_numpy(),
    )
