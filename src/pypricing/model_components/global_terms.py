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


def get_linear_term(
    model_config: dict[str, Any] | None,
    matrix: Any,
    *,
    param_name: str,
    default_sigma: float = 0.5,
) -> Any:
    k = matrix.shape[1]
    if k == 0:
        return 0.0
    beta = resolve_prior(
        model_config=model_config,
        param_name=param_name,
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": default_sigma},
        shape=k,
    )
    return pt.dot(pt.constant(matrix), beta)


def get_controls_term(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    *,
    param_name: str = "beta_control",
) -> Any:
    return get_linear_term(
        model_config,
        data.control_matrix,
        param_name=param_name,
    )
