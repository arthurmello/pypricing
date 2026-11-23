"""compare_endogeneity runs the fast check for each method and leaves the model alone."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pypricing import LogLogDemandModel, PanelColumns, QuadraticLogDemandModel, generate_mock_data
from pypricing.diagnostics import DiagnosticWarning

_PANELS = PanelColumns(quantity_floor=1e-12)


def _gamma_frame():
    return generate_mock_data(
        n_periods=80,
        n_skus=1,
        price_distribution="gamma",
        endogeneity=1.5,
        noise_sigma=0.35,
        price_shock_sigma=0.02,
        include_seasonality=False,
        random_state=0,
    )


def test_compare_endogeneity_reports_each_default_method_and_ols():
    df = _gamma_frame()
    model = LogLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DiagnosticWarning)
        table = model.compare_endogeneity(df)
    assert model.data is None
    assert model.endogeneity_method_ is None
    assert list(table.index) == ["ols", "copula", "latent", "lewbel"]
    assert table.loc["ols", "verdict"] == "uncorrected"
    assert np.isfinite(table.loc["ols", "elasticity"])
    assert table.loc["lewbel", "verdict"] == "not identified"
    assert np.isnan(table.loc["lewbel", "elasticity"])
    fresh = LogLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        checked = fresh.endogeneity_check(df, method="copula")
    assert table.loc["copula", "verdict"] == checked["verdict"]
    if checked["verdict"] != "not identified":
        assert table.loc["copula", "elasticity"] == pytest.approx(
            checked["estimate"].elasticity
        )
    messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, DiagnosticWarning)
    ]
    assert any("experimental" in message for message in messages)


def test_compare_includes_iv_when_the_frame_has_instruments():
    df = generate_mock_data(
        n_periods=80,
        n_skus=2,
        n_instruments=1,
        instrument_coef=0.8,
        include_seasonality=False,
        random_state=0,
    )
    model = LogLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        table = model.compare_endogeneity(df)
    assert list(table.index) == ["ols", "copula", "iv", "latent", "lewbel"]
    assert table.loc["iv", "verdict"] == "ok"
    assert np.isfinite(table.loc["iv", "elasticity"])


def test_iv_without_instruments_is_a_row():
    df = _gamma_frame()
    model = LogLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        table = model.compare_endogeneity(df, methods=("copula", "iv"))
    assert table.loc["iv", "verdict"] == "not identified"
    assert "instrument" in table.loc["iv", "note"]
    assert np.isnan(table.loc["iv", "elasticity"])


def test_compare_needs_at_least_two_methods():
    df = _gamma_frame()
    model = LogLogDemandModel(panel_columns=_PANELS)
    with pytest.raises(ValueError, match="at least two"):
        model.compare_endogeneity(df, methods=("copula",))
    with pytest.raises(TypeError, match="sequence"):
        model.compare_endogeneity(df, methods="copula")
    with pytest.raises(ValueError, match="duplicates"):
        model.compare_endogeneity(df, methods=("copula", "copula"))


def test_nonlinear_compare_warns_about_the_whole_curve():
    df = _gamma_frame()
    model = QuadraticLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DiagnosticWarning)
        table = model.compare_endogeneity(df, methods=("copula", "lewbel"))
    assert list(table.index) == ["ols", "copula", "lewbel"]
    assert any("whole curve" in str(item.message) for item in caught)
