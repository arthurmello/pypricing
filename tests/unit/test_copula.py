"""Gaussian copula control: identification, recovery, and the two warnings."""

from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pytest
import xarray as xr

from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
    generate_mock_data,
)
from pypricing.diagnostics import DiagnosticWarning
from pypricing.endogeneity import CopulaMethod, EndogeneityMethod

_PANELS = PanelColumns(quantity_floor=1e-12)


def _prices(*, distribution: str, seed: int, shape: str = "log_log"):
    return generate_mock_data(
        n_periods=300,
        n_skus=1,
        price_distribution=distribution,
        endogeneity=1.5,
        noise_sigma=0.35,
        price_shock_sigma=0.02,
        true_elasticity=-1.5,
        include_seasonality=False,
        shape=shape,
        random_state=seed,
        return_truth=True,
    )


def _messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(item.message)
        for item in caught
        if issubclass(item.category, DiagnosticWarning)
    ]


def test_copula_method_satisfies_the_interface():
    assert isinstance(CopulaMethod(), EndogeneityMethod)


def test_gamma_prices_recover_elasticity():
    df, truth = _prices(distribution="gamma", seed=0)
    model = LogLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiagnosticWarning)
        checked = model.endogeneity_check(df, method="copula")
    assert checked["verdict"] == "ok"
    estimate = checked["estimate"].elasticity
    assert estimate == pytest.approx(float(truth.elasticity_sku[0]), abs=0.2)
    prepared = model.endogeneity_method_.prepare(model)
    design = np.column_stack([prepared.exogenous, prepared.log_price])
    ols_coef, _, _, _ = np.linalg.lstsq(design, prepared.log_quantity, rcond=None)
    ols = float(ols_coef[-1])
    truth_elasticity = float(truth.elasticity_sku[0])
    assert abs(estimate - truth_elasticity) < abs(ols - truth_elasticity)
    free = {variable.name for variable in model.model.free_RVs}
    assert "rho_copula" in free
    assert "rho" not in free
    assert "obs_price" not in {variable.name for variable in model.model.observed_RVs}


def test_normal_prices_are_not_identified():
    df, _truth = _prices(distribution="normal", seed=4)
    model = LogLogDemandModel(panel_columns=_PANELS)
    with pytest.warns(DiagnosticWarning, match="not identified"):
        checked = model.endogeneity_check(df, method="copula")
    assert checked["verdict"] == "not identified"
    assert "estimate" not in checked
    with pytest.warns(DiagnosticWarning, match="not identified"):
        with pytest.raises(ValueError, match="not identified"):
            model.endogeneity_method_.quick_estimate(model)


def test_nonlinear_models_warn_when_endogeneity_is_set():
    with pytest.warns(DiagnosticWarning, match="whole curve"):
        QuadraticLogDemandModel(endogeneity="copula")
    with pytest.warns(DiagnosticWarning, match="whole curve"):
        SigmoidSaturationDemandModel(endogeneity="iv")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiagnosticWarning)
        LogLogDemandModel(endogeneity="copula")


def test_quadratic_warns_when_the_control_tracks_curvature():
    df, _truth = _prices(distribution="gamma", seed=0, shape="quadratic")
    model = QuadraticLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DiagnosticWarning)
        checked = model.endogeneity_check(df, method="copula")
    messages = _messages(caught)
    assert any("whole curve" in message for message in messages)
    assert any("0.8" in message for message in messages)
    assert checked["verdict"] == "weak"
    assert np.isfinite(checked["estimate"].elasticity)


def test_sigmoid_control_can_stay_clear_of_the_nonlinear_term():
    df, _truth = _prices(distribution="gamma", seed=0, shape="sigmoid")
    model = SigmoidSaturationDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DiagnosticWarning)
        checked = model.endogeneity_check(df, method="copula")
    messages = _messages(caught)
    assert any("whole curve" in message for message in messages)
    assert not any("0.8" in message for message in messages)
    assert checked["verdict"] == "ok"


def test_summarize_posterior_reports_curvature_correlation():
    df = generate_mock_data(
        n_periods=8,
        n_skus=2,
        price_distribution="gamma",
        include_seasonality=False,
        shape="quadratic",
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        model = QuadraticLogDemandModel(panel_columns=_PANELS, endogeneity="copula")
        model.build_model(df)
    rho = np.linspace(-0.4, 0.6, 20).reshape(2, 10)
    curvature = np.stack([rho, -2.0 * rho], axis=-1)
    model.idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "rho_copula": (("chain", "draw"), rho),
                "curvature_sku": (("chain", "draw", "sku"), curvature),
            },
            coords={"sku": list(model.sku_levels_)},
        )
    )
    summary = model.endogeneity_method_.summarize_posterior(model)
    correlation = summary["posterior_correlation"]["curvature_sku"]
    labels = list(model.sku_levels_)
    assert correlation[str(labels[0])] == pytest.approx(1.0)
    assert correlation[str(labels[1])] == pytest.approx(-1.0)
    assert summary["rho_copula"]["mean"] == pytest.approx(float(rho.mean()))


def test_clone_keeps_the_endogeneity_setting():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        model = QuadraticLogDemandModel(endogeneity="copula")
        assert model.clone().endogeneity == "copula"


def test_endogeneity_name_is_rejected_when_unknown():
    with pytest.raises(ValueError, match="endogeneity"):
        LogLogDemandModel(endogeneity="latent")
