"""Unit tests for control-function IV graph construction (no MCMC)."""

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
from pypricing.model_components.iv_terms import (
    WEAK_IV_F_THRESHOLD,
    first_stage_partial_f,
)

_IV_MODELS = (
    (LogLogDemandModel, "log_log"),
    (QuadraticLogDemandModel, "quadratic"),
    (SigmoidSaturationDemandModel, "sigmoid"),
)


@pytest.mark.parametrize("model_cls, shape", _IV_MODELS)
def test_iv_graph_has_price_equation(model_cls, shape):
    df = generate_mock_data(
        n_periods=8,
        n_skus=2,
        n_instruments=1,
        include_seasonality=False,
        random_state=0,
        shape=shape,
    )
    model = model_cls(panel_columns=PanelColumns(quantity_floor=1e-12))
    model.build_model(df)
    free = {v.name for v in model.model.free_RVs}
    observed = {v.name for v in model.model.observed_RVs}
    assert "rho" in free
    assert "pi" in free
    assert "sigma_price" in free
    assert "obs" in observed
    assert "obs_price" in observed
    assert model.iv_names_ == ("iv_1",)


def test_first_stage_partial_f_matches_demeaned_regression():
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n)
    y = 0.8 * z + rng.normal(scale=0.1, size=n)
    got = first_stage_partial_f(y, z, np.ones((n, 1)))
    y_d = y - y.mean()
    z_d = z - z.mean()
    beta = np.dot(z_d, y_d) / np.dot(z_d, z_d)
    resid = y_d - beta * z_d
    ssr_u = float(resid @ resid)
    ssr_r = float(y_d @ y_d)
    expected = ((ssr_r - ssr_u) / 1.0) / (ssr_u / (n - 2))
    assert got == pytest.approx(expected, rel=1e-8)
    assert got > WEAK_IV_F_THRESHOLD


def test_first_stage_partial_f_is_small_for_an_irrelevant_instrument():
    rng = np.random.default_rng(1)
    n = 400
    z = rng.normal(size=n)
    y = rng.normal(size=n)
    got = first_stage_partial_f(y, z, np.ones((n, 1)))
    assert got < WEAK_IV_F_THRESHOLD


def test_first_stage_partial_f_is_zero_when_instrument_duplicates_a_control():
    rng = np.random.default_rng(2)
    n = 80
    x = rng.normal(size=(n, 1))
    y = x[:, 0] + rng.normal(scale=0.2, size=n)
    got = first_stage_partial_f(y, x, x)
    assert got == pytest.approx(0.0, abs=1e-8)
