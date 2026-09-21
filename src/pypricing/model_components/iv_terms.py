"""Control-function IV terms for log-demand models."""

from __future__ import annotations

from typing import Any

import pymc as pm
import pytensor.tensor as pt

from pypricing.data import PricePanelData
from pypricing.model_components.global_terms import get_controls_term, get_linear_term
from pypricing.model_components.priors import resolve_prior
from pypricing.model_components.sku_effects import get_sku_effect
from pypricing.model_components.time_terms import (
    TrendKind,
    get_season_term,
    get_trend_term,
)


def get_control_function_term(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    *,
    trend: TrendKind | None,
    t0: Any,
    seasonality_components: tuple[str, ...] | None,
) -> Any:
    """Price equation residual times shared ``rho``; 0.0 when there are no instruments."""
    if data.n_iv == 0:
        return 0.0

    obs_sku = pt.constant(data.obs_sku_idx)
    log_p = pt.constant(data.log_price)

    alpha_price_sku = get_sku_effect(
        "alpha_price",
        data,
        model_config,
        mu_default_mu=2.0,
        mu_default_sigma=2.0,
    )
    iv_term = get_linear_term(
        model_config,
        data.iv_matrix,
        param_name="pi",
        default_sigma=1.0,
    )
    controls_price = get_controls_term(
        model_config, data, param_name="beta_control_price"
    )
    trend_price = get_trend_term(
        model_config,
        data,
        trend=trend,
        t0=t0,
        param_suffix="_price",
    )
    season_price = get_season_term(
        model_config,
        data,
        components=seasonality_components,
        param_name="beta_season_price",
    )

    mu_price = (
        alpha_price_sku[obs_sku]
        + iv_term
        + controls_price
        + trend_price
        + season_price
    )
    sigma_price = resolve_prior(
        model_config=model_config,
        param_name="sigma_price",
        default_dist=pm.HalfNormal,
        default_kwargs={"sigma": 0.5},
    )
    pm.Normal("obs_price", mu=mu_price, sigma=sigma_price, observed=data.log_price)

    residual = log_p - mu_price
    rho = resolve_prior(
        model_config=model_config,
        param_name="rho",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": 1.0},
    )
    return rho * residual
