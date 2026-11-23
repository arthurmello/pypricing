"""Phase-0 harness: the endogeneity interface, with IV as the only method."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import xarray as xr

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data
from pypricing.diagnostics.identification import first_stage_exogenous
from pypricing.endogeneity import (
    EndogeneityMethod,
    IVMethod,
    exogenous_design,
)


def _panel(**kwargs):
    defaults = dict(
        n_periods=80,
        n_skus=2,
        n_instruments=1,
        instrument_coef=0.8,
        include_seasonality=False,
        random_state=0,
    )
    defaults.update(kwargs)
    return generate_mock_data(**defaults)


def _built(df, **panel_kwargs):
    model = LogLogDemandModel(
        panel_columns=PanelColumns(quantity_floor=1e-12, **panel_kwargs)
    )
    model.build_model(df)
    return model


def test_iv_method_satisfies_the_interface_and_builds_the_price_equation():
    assert isinstance(IVMethod(), EndogeneityMethod)
    df = _panel()
    model = _built(df)
    assert isinstance(model.endogeneity_method_, IVMethod)
    assert model.endogeneity_method_.name == "iv"
    free = {v.name for v in model.model.free_RVs}
    observed = {v.name for v in model.model.observed_RVs}
    assert {"rho", "pi", "sigma_price"} <= free
    assert "obs_price" in observed
    np.testing.assert_allclose(exogenous_design(model), first_stage_exogenous(model))


def test_no_instruments_leaves_the_method_unset():
    df = generate_mock_data(n_periods=8, n_skus=2, random_state=0)
    model = _built(df, iv_columns=())
    assert model.endogeneity_method_ is None
    assert "rho" not in {v.name for v in model.model.free_RVs}
    report = IVMethod().check_assumptions(model)
    assert report.verdict() == "not identified"
    with pytest.raises(ValueError, match="not identified"):
        IVMethod().quick_estimate(model)


def test_strong_instruments_are_ok_and_a_constant_instrument_is_not_identified():
    strong = _built(_panel())
    assert strong.endogeneity_method_.check_assumptions(strong).verdict() == "ok"

    weak = _built(_panel(instrument_coef=0.0, n_periods=100))
    assert weak.endogeneity_method_.check_assumptions(weak).verdict() == "weak"

    frame = _panel()
    frame["iv_1"] = 1.0
    flat = _built(frame)
    report = flat.endogeneity_method_.check_assumptions(flat)
    assert report.verdict() == "not identified"
    with pytest.raises(ValueError, match="not identified"):
        flat.endogeneity_method_.quick_estimate(flat)


def test_quick_estimate_recovers_elasticity_when_price_is_endogenous():
    df, truth = generate_mock_data(
        n_periods=160,
        n_skus=1,
        n_instruments=1,
        instrument_coef=0.8,
        endogeneity=1.2,
        noise_sigma=0.15,
        price_shock_sigma=0.02,
        true_elasticity=-1.5,
        include_seasonality=False,
        random_state=0,
        return_truth=True,
    )
    model = _built(df)
    prepared = model.endogeneity_method_.prepare(model)
    ols_design = np.column_stack([prepared.exogenous, prepared.log_price])
    ols_coef, _, _, _ = np.linalg.lstsq(
        ols_design, prepared.log_quantity, rcond=None
    )
    ols = float(ols_coef[-1])
    estimate = model.endogeneity_method_.quick_estimate(model)
    truth_elasticity = float(truth.elasticity_sku[0])
    assert abs(estimate.elasticity - truth_elasticity) < abs(ols - truth_elasticity)
    assert estimate.elasticity == pytest.approx(truth_elasticity, abs=0.25)
    assert estimate.method == "iv"
    assert np.isfinite(estimate.parameters["rho"])


def test_summarize_posterior_requires_a_fit():
    model = _built(_panel(n_periods=8))
    with pytest.raises(RuntimeError, match="not been fit"):
        model.endogeneity_method_.summarize_posterior(model)


def test_summarize_posterior_reads_rho():
    model = _built(_panel(n_periods=8))
    rho = np.array([[0.2, 0.4, -0.1], [0.0, 0.3, 0.5]])
    model.idata = az.InferenceData(
        posterior=xr.Dataset({"rho": (("chain", "draw"), rho)})
    )
    summary = model.endogeneity_method_.summarize_posterior(model, hdi_prob=0.9)
    assert summary["rho"]["mean"] == pytest.approx(float(rho.mean()))
    assert summary["rho"]["hdi"][0] <= 0.0 <= summary["rho"]["hdi"][1]
    assert summary["rho"]["hdi_includes_zero"] is True
