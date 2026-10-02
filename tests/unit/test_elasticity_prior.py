"""Default elasticity prior is centered on published evidence."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
    generate_mock_data,
)
from pypricing.model_components.priors import (
    DEFAULT_ELASTICITY_MU,
    DEFAULT_ELASTICITY_SIGMA,
)

_MODELS = [
    LogLogDemandModel,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
]


def _frame(*, hierarchical: bool):
    if hierarchical:
        return generate_mock_data(
            n_periods=4,
            n_skus=4,
            hierarchy_levels=(2,),
            n_controls=0,
            include_seasonality=False,
            random_state=0,
        )
    return generate_mock_data(
        n_periods=4,
        n_skus=2,
        n_controls=0,
        include_seasonality=False,
        random_state=0,
    )


def _built(cls, *, hierarchical: bool = False, model_config=None):
    kwargs = {
        "panel_columns": PanelColumns(
            quantity_floor=1e-12,
            group_columns=("category_1",) if hierarchical else None,
        )
    }
    if model_config is not None:
        kwargs["model_config"] = model_config
    model = cls(**kwargs)
    model.build_model(_frame(hierarchical=hierarchical))
    return model


def _params(rv):
    n_params = len(rv.owner.op.ndims_params)
    return [np.asarray(inp.eval(), dtype=float) for inp in rv.owner.inputs[-n_params:]]


def test_named_defaults():
    assert DEFAULT_ELASTICITY_MU == -2.6
    assert DEFAULT_ELASTICITY_SIGMA == 1.5


@pytest.mark.parametrize("cls", _MODELS, ids=[c.__name__ for c in _MODELS])
def test_flat_elasticity_prior(cls):
    rv = _built(cls).model["elasticity_sku"]
    assert type(rv.owner.op).__name__ == "NormalRV"
    mu, sigma = _params(rv)
    assert np.allclose(mu, -2.6)
    assert np.allclose(sigma, 1.5)


@pytest.mark.parametrize("cls", _MODELS, ids=[c.__name__ for c in _MODELS])
def test_hierarchical_elasticity_prior(cls):
    graph = _built(cls, hierarchical=True).model
    mu, sigma = _params(graph["mu_elasticity"])
    assert type(graph["mu_elasticity"].owner.op).__name__ == "NormalRV"
    assert np.allclose(mu, -2.6)
    assert np.allclose(sigma, 1.5)
    for name in ("sigma_elasticity_sku", "sigma_elasticity_group_0"):
        scale = graph[name]
        assert type(scale.owner.op).__name__ == "HalfNormalRV"
        assert np.allclose(_params(scale)[-1], 1.0)


def test_flat_prior_predictive_mass_in_published_range():
    model = _built(LogLogDemandModel)
    with model.model:
        prior = pm.sample_prior_predictive(
            draws=2000,
            var_names=["elasticity_sku"],
            random_seed=0,
        )
    draws = prior.prior["elasticity_sku"].values
    frac = float(np.mean((draws >= -4.0) & (draws <= 0.0)))
    assert frac >= 0.70


@pytest.mark.parametrize("cls", _MODELS, ids=[c.__name__ for c in _MODELS])
def test_model_config_overrides_flat_elasticity_prior(cls):
    model = _built(
        cls,
        model_config={
            "elasticity_sku": {
                "dist": pm.Normal,
                "kwargs": {"mu": -1.0, "sigma": 2.0},
            }
        },
    )
    mu, sigma = _params(model.model["elasticity_sku"])
    assert np.allclose(mu, -1.0)
    assert np.allclose(sigma, 2.0)


def test_model_config_overrides_hierarchical_location():
    model = _built(
        LogLogDemandModel,
        hierarchical=True,
        model_config={
            "mu_elasticity": {
                "dist": pm.Normal,
                "kwargs": {"mu": -1.0, "sigma": 2.0},
            }
        },
    )
    mu, sigma = _params(model.model["mu_elasticity"])
    assert np.allclose(mu, -1.0)
    assert np.allclose(sigma, 2.0)
