from pypricing.model_components.priors import resolve_prior
import pymc as pm
from typing import Any
import pytensor.tensor as pt
from pypricing.data import PricePanelData


def get_sigma(model_config: dict[str, Any] | None) -> Any:
    return resolve_prior(
        model_config=model_config,
        param_name="sigma",
        default_dist=pm.HalfNormal,
        default_kwargs={"sigma": 0.5},
    )


def get_controls_term(model_config: dict[str, Any] | None, data: PricePanelData) -> Any:
    controls_term = 0.0
    k = data.control_matrix.shape[1]
    if k > 0:
        X = pt.constant(data.control_matrix)
        beta = resolve_prior(
            model_config=model_config,
            param_name="beta_control",
            default_dist=pm.Normal,
            default_kwargs={"mu": 0.0, "sigma": 0.5},
            shape=k,
        )
        controls_term = pt.dot(X, beta)
    return controls_term
