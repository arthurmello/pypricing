"""Latent IV: a real price mixture is identified, a normal price is not."""

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
from pypricing.endogeneity import EndogeneityMethod, LatentMethod

_PANELS = PanelColumns(quantity_floor=1e-12)
_TRUTH = -1.5


def latent_iv_frame(*, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Price is a class shift plus a shock shared with demand.

    The class is drawn independently of the demand shock. That is the case
    latent IV is built for. The generator's discrete prices are not: there the
    price point itself moves with the demand shock.
    """
    rng = np.random.default_rng(seed)
    demand_shock = rng.normal(0, 0.4, n)
    latent_class = rng.choice(3, n, p=[0.3, 0.4, 0.3])
    gap = 2.5
    log_price = (
        np.array([0.0, gap, 2.0 * gap])[latent_class]
        + 1.8 * demand_shock
        + rng.normal(0, 0.35, n)
    )
    log_quantity = 1.0 + _TRUTH * log_price + demand_shock
    return pd.DataFrame(
        {
            "sku": np.repeat("sku_1", n),
            "price": np.exp(log_price),
            "quantity": np.exp(log_quantity),
        }
    )


def _messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(item.message)
        for item in caught
        if issubclass(item.category, DiagnosticWarning)
    ]


def test_latent_method_satisfies_the_interface():
    method = LatentMethod()
    assert isinstance(method, EndogeneityMethod)
    assert method.experimental is True


def test_a_price_mixture_recovers_elasticity():
    df = latent_iv_frame()
    model = LogLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DiagnosticWarning)
        checked = model.endogeneity_check(df, method="latent")
    messages = _messages(caught)
    assert any("experimental" in message for message in messages)
    assert not any("not identified" in message for message in messages)
    assert checked["verdict"] == "ok"
    assert checked["method"] == "latent"
    estimate = checked["estimate"].elasticity
    assert estimate == pytest.approx(_TRUTH, abs=0.15)
    prepared = model.endogeneity_method_.prepare(model)
    design = np.column_stack([prepared.exogenous, prepared.log_price])
    ols = float(np.linalg.lstsq(design, prepared.log_quantity, rcond=None)[0][-1])
    assert abs(estimate - _TRUTH) < abs(ols - _TRUTH)
    assert prepared.n_classes == 3
    free = {variable.name for variable in model.model.free_RVs}
    observed = {variable.name for variable in model.model.observed_RVs}
    assert "rho_latent" in free
    assert "latent_pi" in free
    assert "rho" not in free
    assert "rho_copula" not in free
    assert "obs_latent_price" in observed


def test_a_normal_price_is_not_identified():
    df = generate_mock_data(
        n_periods=300,
        n_skus=1,
        price_distribution="normal",
        endogeneity=1.5,
        noise_sigma=0.35,
        price_shock_sigma=0.02,
        true_elasticity=_TRUTH,
        include_seasonality=False,
        random_state=4,
    )
    model = LogLogDemandModel(panel_columns=_PANELS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DiagnosticWarning)
        checked = model.endogeneity_check(df, method="latent")
    messages = _messages(caught)
    assert any("experimental" in message for message in messages)
    assert any("not identified" in message for message in messages)
    assert checked["verdict"] == "not identified"
    assert "estimate" not in checked
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DiagnosticWarning)
        with pytest.raises(ValueError, match="not identified"):
            model.endogeneity_method_.quick_estimate(model)
    assert any("not identified" in message for message in _messages(caught))


def test_latent_iv_warns_that_it_is_experimental():
    with pytest.warns(DiagnosticWarning, match="experimental"):
        LogLogDemandModel(endogeneity="latent")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DiagnosticWarning)
        QuadraticLogDemandModel(endogeneity="latent")
    messages = _messages(caught)
    assert any("experimental" in message for message in messages)
    assert any("whole curve" in message for message in messages)


def test_summarize_posterior_reads_rho_latent():
    df = latent_iv_frame(n=40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        model = LogLogDemandModel(panel_columns=_PANELS, endogeneity="latent")
        model.build_model(df)
    rho = np.linspace(-0.2, 0.4, 20).reshape(2, 10)
    model.idata = az.InferenceData(
        posterior=xr.Dataset({"rho_latent": (("chain", "draw"), rho)})
    )
    summary = model.endogeneity_method_.summarize_posterior(model)
    assert summary["rho_latent"]["mean"] == pytest.approx(float(rho.mean()))
    assert summary["rho_latent"]["hdi_includes_zero"]
    assert summary["n_classes"] >= 2
