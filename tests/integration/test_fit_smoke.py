"""Integration smoke: fit, predict, plots, train/test, save/load."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")

import pymc as pm

from pypricing import (
    LogLogDemandModel,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
    generate_mock_data,
)


def test_fit_runs_default():
    df = generate_mock_data(
        n_periods=15,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )
    idata = LogLogDemandModel().fit(
        df,
        draws=80,
        tune=80,
        chains=2,
        random_seed=42,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "elasticity_sku" in idata.posterior
    assert "alpha_sku" in idata.posterior


def test_fit_stores_artifacts_and_diagnostics():
    df = generate_mock_data(
        n_periods=10,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )
    model = LogLogDemandModel()
    model.fit(
        df,
        draws=50,
        tune=50,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )

    assert model.idata is not None
    assert hasattr(model, "model")
    assert model.sku_levels_ is not None
    assert model.control_names_ == ("control_1",)

    diag = model.run_diagnostics()
    assert "n_divergent" in diag

    summary = model.fit_summary()
    assert isinstance(summary, pd.DataFrame)


def test_predict_without_quantity():
    df_train = generate_mock_data(
        n_periods=8,
        n_skus=4,
        n_controls=2,
        random_state=1,
    )
    model = LogLogDemandModel()
    model.fit(
        df_train,
        draws=60,
        tune=60,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )

    df_pred = df_train.drop(columns=["quantity"])
    ds = model.sample_posterior_predictive(df_pred, random_seed=123)
    assert "quantity" in ds
    assert ds["quantity"].shape[-1] == len(df_pred)

    out = model.predict(df_pred, hdi_prob=0.9, random_seed=123)
    assert "quantity_mean" in out.columns
    assert "quantity_hdi_lower" in out.columns
    assert "quantity_hdi_upper" in out.columns
    assert len(out) == len(df_pred)


def test_predict_unknown_sku_raises():
    df_train = generate_mock_data(
        n_periods=6,
        n_skus=3,
        n_controls=0,
        random_state=2,
    )
    model = LogLogDemandModel()
    model.fit(
        df_train,
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )

    df_pred = df_train.drop(columns=["quantity"]).copy()
    df_pred.loc[:, "sku"] = "sku_UNKNOWN"
    with pytest.raises(ValueError, match="Unknown sku"):
        model.sample_posterior_predictive(df_pred, random_seed=0)


def test_predict_missing_control_raises():
    df_train = generate_mock_data(
        n_periods=6,
        n_skus=3,
        n_controls=1,
        random_state=3,
    )
    model = LogLogDemandModel()
    model.fit(
        df_train,
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )

    df_pred = df_train.drop(columns=["quantity", "control_1"]).copy()
    with pytest.raises(ValueError, match="Missing control columns"):
        model.sample_posterior_predictive(df_pred, random_seed=0)


def test_custom_priors_model_config_smoke():
    df = generate_mock_data(
        n_periods=8,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )

    model_config = {
        "alpha_sku": {"dist": pm.Normal, "kwargs": {"mu": 0.0, "sigma": 1.0}},
        "sigma": {"dist": pm.HalfNormal, "kwargs": {"sigma": 0.2}},
    }

    model = LogLogDemandModel(model_config=model_config)
    idata = model.fit(
        df,
        draws=80,
        tune=80,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )

    assert "alpha_sku" in idata.posterior
    assert "elasticity_sku" in idata.posterior


def test_plotting_smoke():
    df = generate_mock_data(
        n_periods=8,
        n_skus=3,
        n_controls=2,
        random_state=1,
    )
    model = LogLogDemandModel()
    model.fit(
        df,
        draws=80,
        tune=80,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )

    ax1 = model.plot_elasticity_posterior(hdi_prob=0.8)
    assert ax1 is not None

    sku_value = model.sku_levels_[0]
    price_grid = [df["price"].min(), float(df["price"].mean()), df["price"].max()]
    controls = {
        c: float(df.loc[df[model.sku_col] == sku_value, c].iloc[0])
        for c in model.control_names_
    }

    ax2 = model.plot_response_curve(
        sku=sku_value,
        price_grid=price_grid,
        controls=controls,
        hdi_prob=0.8,
    )
    assert ax2 is not None

    ax_loc = model.plot_local_elasticity_vs_price(
        sku=sku_value,
        price_grid=price_grid,
        controls=controls,
        hdi_prob=0.8,
    )
    assert ax_loc is not None

    ax_rev = model.plot_revenue_vs_price(
        sku=sku_value,
        price_grid=price_grid,
        controls=controls,
        hdi_prob=0.8,
        random_seed=0,
    )
    assert ax_rev is not None

    ax3 = model.plot_posterior_predictive_calibration(
        hdi_prob=0.8,
        random_seed=0,
    )
    assert ax3 is not None

    bounds = {
        sku: (
            float(df.loc[df["sku"] == sku, "price"].min() * 0.5),
            float(df.loc[df["sku"] == sku, "price"].max() * 1.5),
        )
        for sku in model.sku_levels_
    }
    ctrl = df.groupby("sku", as_index=False).agg(
        **{c: (c, "median") for c in model.control_names_}
    )
    opt_df = model.optimize_prices(price_bounds=bounds, controls_df=ctrl)
    ax5, ax6 = model.plot_optimization_summary(opt_df, controls_df=ctrl)
    assert ax5 is not None
    assert ax6 is not None


def test_fit_train_test_metrics():
    df = generate_mock_data(
        n_periods=10,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )
    model = LogLogDemandModel()
    out = model.fit_train_test(
        df,
        test_size=0.3,
        period_col="period",
        hdi_prob=0.9,
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert set(["rmse", "hdi_coverage", "test_predictions"]).issubset(out.keys())
    assert out["rmse"] >= 0
    assert 0.0 <= out["hdi_coverage"] <= 1.0
    assert len(out["test_predictions"]) == out["n_test"]


def test_check_falsification_returns_lead_summary():
    df = generate_mock_data(
        n_periods=10,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )
    model = LogLogDemandModel()
    sample_kw = dict(
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    model.fit(df, **sample_kw)
    out = model.check_falsification()
    explicit = model.check_falsification(**sample_kw)
    assert out["lead_coef_mean"] == pytest.approx(explicit["lead_coef_mean"])

    lo, hi = out["lead_coef_hdi"]
    assert lo <= out["lead_coef_mean"] <= hi
    assert out["n"] < len(df)
    assert list(out["elasticity"].columns) == ["baseline", "with_lead", "shift"]
    assert len(out["elasticity"]) == 3
    assert model.control_names_ == ("control_1",)


def test_check_instruments_reports_all_sections():
    df = generate_mock_data(
        n_periods=12,
        n_skus=3,
        n_controls=1,
        n_instruments=2,
        random_state=0,
    )
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
    out = model.check_instruments()

    assert out["first_stage"]["f"] > 0
    assert out["overid"]["applicable"]
    assert out["overid"]["n_instruments"] == 2
    assert 0.0 <= out["overid"]["p_value"] <= 1.0
    lo, hi = out["rho"]["hdi"]
    assert lo <= out["rho"]["mean"] <= hi
    assert list(out["elasticity"].columns) == ["iv", "ols", "shift"]
    assert len(out["elasticity"]) == 3

    no_ols = model.check_instruments(compare_ols=False)
    assert no_ols["elasticity"] is None


def test_check_instruments_requires_instruments():
    df = generate_mock_data(n_periods=6, n_skus=2, random_state=0)
    model = LogLogDemandModel()
    model.fit(
        df,
        draws=20,
        tune=20,
        chains=1,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    with pytest.raises(ValueError, match="no instruments"):
        model.check_instruments()


def test_check_sensitivity_per_sku_and_pooled():
    df = generate_mock_data(
        n_periods=20,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )
    model = LogLogDemandModel()
    model.fit(
        df,
        draws=100,
        tune=100,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    out = model.check_sensitivity()

    assert list(out.index) == [*model.sku_levels_, "pooled"]
    assert list(out.columns) == ["estimate", "se", "t", "partial_r2", "rv", "rv_qa"]
    assert ((out["rv"] >= out["rv_qa"]) & (out["rv_qa"] >= 0)).all()
    posterior = model.idata.posterior["elasticity_sku"].mean(("chain", "draw"))
    per_sku = out.loc[list(model.sku_levels_)]
    gap = np.abs(per_sku["estimate"].to_numpy() - posterior.to_numpy())
    assert (gap < 3 * per_sku["se"].to_numpy()).all()


def test_check_fit_in_and_out_of_sample():
    df = generate_mock_data(
        n_periods=12,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )
    model = LogLogDemandModel()
    train_df, test_df = model.train_test_split(df, test_size=0.25)
    model.fit(
        train_df,
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )

    in_sample = model.check_fit(random_seed=0, min_rows=1)
    assert list(in_sample.index) == ["overall", "low", "mid", "high"]
    assert in_sample.loc["overall", "n"] == len(train_df)

    by_control = model.check_fit(by="control_1", random_seed=0, min_rows=1)
    assert by_control.loc[["low", "mid", "high"], "n"].sum() == len(train_df)

    oos = model.check_fit(test_df, random_seed=0, min_rows=1)
    assert oos.loc["overall", "n"] == len(test_df)
    assert oos.loc[["low", "mid", "high"], "n"].sum() == len(test_df)

    with pytest.raises(ValueError, match="not found"):
        model.check_fit(by="missing_col")


def test_fit_sigmoid_smoke():
    df = generate_mock_data(
        n_periods=8,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )
    model = SigmoidSaturationDemandModel()
    model.fit(
        df,
        draws=60,
        tune=60,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "elasticity_sku" in model.idata.posterior

    df_pred = df.drop(columns=["quantity"]).copy()
    out = model.predict(df_pred, hdi_prob=0.9, random_seed=123)
    assert "quantity_mean" in out.columns
    assert len(out) == len(df_pred)


def test_fit_quadratic_smoke():
    df = generate_mock_data(
        n_periods=8,
        n_skus=3,
        n_controls=1,
        random_state=1,
    )
    model = QuadraticLogDemandModel()
    model.fit(
        df,
        draws=60,
        tune=60,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "elasticity_sku" in model.idata.posterior
    assert "curvature_sku" in model.idata.posterior

    df_pred = df.drop(columns=["quantity"]).copy()
    out = model.predict(df_pred, hdi_prob=0.9, random_seed=123)
    assert "quantity_mean" in out.columns
    assert len(out) == len(df_pred)


def test_save_and_load_predict_roundtrip(tmp_path):
    df = generate_mock_data(
        n_periods=8,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )
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

    path = tmp_path / "elasticity_model.nc"
    model.save(path)
    loaded = LogLogDemandModel.load(path)

    df_pred = df.drop(columns=["quantity"]).copy()
    out1 = model.predict(df_pred, hdi_prob=0.9, random_seed=123)
    out2 = loaded.predict(df_pred, hdi_prob=0.9, random_seed=123)

    assert np.allclose(
        out1["quantity_mean"].to_numpy(),
        out2["quantity_mean"].to_numpy(),
    )
    assert np.allclose(
        out1["quantity_hdi_lower"].to_numpy(),
        out2["quantity_hdi_lower"].to_numpy(),
    )
    assert np.allclose(
        out1["quantity_hdi_upper"].to_numpy(),
        out2["quantity_hdi_upper"].to_numpy(),
    )


def test_save_and_load_predict_roundtrip_sigmoid(tmp_path):
    df = generate_mock_data(
        n_periods=8,
        n_skus=3,
        n_controls=1,
        random_state=0,
    )
    model = SigmoidSaturationDemandModel()
    model.fit(
        df,
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )

    path = tmp_path / "elasticity_model_sigmoid.nc"
    model.save(path)
    loaded = SigmoidSaturationDemandModel.load(path)

    df_pred = df.drop(columns=["quantity"]).copy()
    out1 = model.predict(df_pred, hdi_prob=0.9, random_seed=123)
    out2 = loaded.predict(df_pred, hdi_prob=0.9, random_seed=123)

    assert np.allclose(
        out1["quantity_mean"].to_numpy(),
        out2["quantity_mean"].to_numpy(),
    )
