"""Lewbel instruments: heteroskedastic prices are identified, homoskedastic ones are not."""

from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    QuadraticLogDemandModel,
    generate_mock_data,
)
from pypricing.diagnostics import DiagnosticWarning
from pypricing.endogeneity import EndogeneityMethod, LewbelMethod

_PANELS = PanelColumns(quantity_floor=1e-12, control_columns=("hetero",))


def _prices(*, heteroskedasticity: bool, seed: int = 0):
    return generate_mock_data(
        n_periods=300,
        n_skus=1,
        heteroskedasticity=heteroskedasticity,
        endogeneity=1.5,
        noise_sigma=0.3,
        price_shock_sigma=0.02,
        true_elasticity=-1.5,
        include_seasonality=False,
        random_state=seed,
        return_truth=True,
    )


def test_lewbel_method_satisfies_the_interface():
    assert isinstance(LewbelMethod(), EndogeneityMethod)


def test_heteroskedastic_prices_recover_elasticity():
    df, truth = _prices(heteroskedasticity=True)
    model = LogLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiagnosticWarning)
        checked = model.endogeneity_check(df, method="lewbel")
    assert checked["verdict"] == "ok"
    assert checked["estimate"].method == "lewbel"
    estimate = checked["estimate"].elasticity
    truth_elasticity = float(truth.elasticity_sku[0])
    assert estimate == pytest.approx(truth_elasticity, abs=0.15)
    prepared = model.endogeneity_method_.prepare(model)
    design = np.column_stack([prepared.exogenous, prepared.log_price])
    ols_coef, _, _, _ = np.linalg.lstsq(design, prepared.log_quantity, rcond=None)
    ols = float(ols_coef[-1])
    assert abs(estimate - truth_elasticity) < abs(ols - truth_elasticity)
    assert model.iv_names_ == ()
    free = {variable.name for variable in model.model.free_RVs}
    observed = {variable.name for variable in model.model.observed_RVs}
    assert "rho" in free
    assert "pi" in free
    assert "rho_copula" not in free
    assert "obs_price" in observed


def test_homoskedastic_prices_are_not_identified():
    df, _truth = _prices(heteroskedasticity=False)
    model = LogLogDemandModel(panel_columns=PanelColumns(quantity_floor=1e-12))
    with pytest.warns(DiagnosticWarning, match="variation"):
        checked = model.endogeneity_check(df, method="lewbel")
    assert checked["verdict"] == "not identified"
    assert "estimate" not in checked
    with pytest.warns(DiagnosticWarning, match="variation"):
        with pytest.raises(ValueError, match="not identified"):
            model.endogeneity_method_.quick_estimate(model)


def test_homoskedastic_residual_is_not_identified():
    residual = np.array([1.0, -1.0, 1.0, -1.0])
    df = pd.DataFrame(
        {
            "sku": ["a"] * 4,
            "price": np.exp(residual),
            "quantity": np.ones(4),
            "scale": [-1.0, -1.0, 1.0, 1.0],
        }
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(quantity_floor=1e-12, control_columns=("scale",))
    )
    with pytest.warns(DiagnosticWarning, match="homoskedastic"):
        checked = model.endogeneity_check(df, method="lewbel")
    assert checked["verdict"] == "not identified"
    assert "estimate" not in checked


def test_scale_driver_must_be_an_exogenous_regressor():
    df, _truth = _prices(heteroskedasticity=True)
    model = LogLogDemandModel(panel_columns=PanelColumns(quantity_floor=1e-12))
    with pytest.warns(DiagnosticWarning, match="variation"):
        checked = model.endogeneity_check(df, method="lewbel")
    assert checked["verdict"] == "not identified"
    assert "hetero" in df.columns


def test_nonlinear_models_warn_when_lewbel_is_set():
    with pytest.warns(DiagnosticWarning, match="whole curve"):
        QuadraticLogDemandModel(endogeneity="lewbel")


def test_summarize_posterior_reads_rho():
    df, _truth = _prices(heteroskedasticity=True)
    model = LogLogDemandModel(panel_columns=_PANELS, endogeneity="lewbel")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        model.build_model(df)
    rho = np.linspace(-0.2, 0.4, 20).reshape(2, 10)
    model.idata = az.InferenceData(
        posterior=xr.Dataset({"rho": (("chain", "draw"), rho)})
    )
    summary = model.endogeneity_method_.summarize_posterior(model)
    assert summary["rho"]["mean"] == pytest.approx(float(rho.mean()))
    assert summary["rho"]["hdi_includes_zero"]
