"""Integration smoke: control-function IV on log-log."""

from __future__ import annotations

import numpy as np
import pytest

from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
    generate_mock_data,
)


def _endogenous_panel(*, seed: int = 0):
    return generate_mock_data(
        n_periods=12,
        n_skus=3,
        n_controls=1,
        n_instruments=1,
        endogeneity=1.0,
        include_seasonality=False,
        round_quantity=False,
        random_state=seed,
    )


_IV_COLS = PanelColumns(quantity_floor=1e-12)


def test_log_log_iv_fit_predict_optimize():
    df = _endogenous_panel()
    model = LogLogDemandModel(panel_columns=_IV_COLS)
    model.fit(
        df,
        draws=50,
        tune=50,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert model.iv_names_ == ("iv_1",)
    assert "rho" in model.idata.posterior
    assert "pi" in model.idata.posterior
    assert "obs_price" not in model.idata.posterior

    summary = model.fit_summary()
    assert "rho" in summary.index or any("rho" in str(i) for i in summary.index)

    diag = model.run_diagnostics()
    assert "rho_mean" in diag
    assert "weak_iv" in diag

    df_pred = df.drop(columns=["quantity"]).copy()
    out = model.predict(df_pred, hdi_prob=0.9, random_seed=123)
    assert len(out) == len(df_pred)
    assert np.all(np.isfinite(out["quantity_mean"].to_numpy()))

    bounds = {
        sku: (
            float(df.loc[df["sku"] == sku, "price"].min() * 0.5),
            float(df.loc[df["sku"] == sku, "price"].max() * 1.5),
        )
        for sku in model.sku_levels_
    }
    ctrl = df.groupby("sku", as_index=False).agg(control_1=("control_1", "median"))
    opt = model.optimize_prices(price_bounds=bounds, controls_df=ctrl)
    assert len(opt) == 3
    assert np.all(np.isfinite(opt["optimal_price"].to_numpy()))


def test_log_log_iv_save_load_roundtrip(tmp_path):
    df = _endogenous_panel(seed=1)
    model = LogLogDemandModel(panel_columns=_IV_COLS)
    model.fit(
        df,
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    path = tmp_path / "iv_model.nc"
    model.save(path)
    loaded = LogLogDemandModel.load(path)
    assert loaded.iv_names_ == ("iv_1",)

    df_pred = df.drop(columns=["quantity"]).copy()
    out1 = model.predict(df_pred, hdi_prob=0.9, random_seed=123)
    out2 = loaded.predict(df_pred, hdi_prob=0.9, random_seed=123)
    assert np.allclose(
        out1["quantity_mean"].to_numpy(), out2["quantity_mean"].to_numpy()
    )


def test_iv_disabled_with_empty_tuple():
    df = _endogenous_panel(seed=2)
    model = LogLogDemandModel(
        panel_columns=PanelColumns(quantity_floor=1e-12, iv_columns=())
    )
    model.fit(
        df,
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert model.iv_names_ == ()
    assert "rho" not in model.idata.posterior


@pytest.mark.parametrize(
    "model_cls",
    [QuadraticLogDemandModel, SigmoidSaturationDemandModel],
)
def test_iv_not_supported_on_nonlinear_curves(model_cls):
    df = _endogenous_panel(seed=3)
    model = model_cls(panel_columns=_IV_COLS)
    with pytest.raises(ValueError, match="not supported"):
        model.fit(
            df,
            draws=10,
            tune=10,
            chains=1,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )
