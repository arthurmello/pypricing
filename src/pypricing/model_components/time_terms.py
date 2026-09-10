"""Time-varying terms for mean log-quantity (trend, later seasonality)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pypricing.data import PricePanelData, period_to_t_years
from pypricing.model_components.hierarchical_effects import (
    get_noncentered_hierarchical_effect_per_sku,
)
from pypricing.model_components.priors import resolve_prior

TrendKind = Literal["sku", "shared"]

_TREND_PRIOR_SIGMA = 0.05


def get_trend_term(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    *,
    trend: TrendKind | None,
    t0: pd.Timestamp | None,
) -> Any:
    """Additive ``trend * t`` in log-quantity; ``t`` is years since ``t0``."""
    if trend is None:
        return 0.0
    if data.period_index is None:
        raise ValueError("trend requires a datetime period_index on the panel")
    if t0 is None:
        raise ValueError("trend requires t0 (minimum training period)")

    t = pt.constant(period_to_t_years(data.period_index, t0))
    if trend == "shared":
        mu_trend = resolve_prior(
            model_config=model_config,
            param_name="mu_trend",
            default_dist=pm.Normal,
            default_kwargs={"mu": 0.0, "sigma": _TREND_PRIOR_SIGMA},
        )
        return mu_trend * t

    trend_sku = _trend_sku_effect(model_config, data)
    obs_sku = pt.constant(data.obs_sku_idx)
    return trend_sku[obs_sku] * t


def _trend_sku_effect(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
) -> Any:
    if data.hierarchy is not None:
        lin = get_noncentered_hierarchical_effect_per_sku(
            "trend",
            n_skus=data.n_skus,
            hierarchy=data.hierarchy,
            model_config=model_config,
            mu_default_mu=0.0,
            mu_default_sigma=_TREND_PRIOR_SIGMA,
            level_scale_default_sigma=_TREND_PRIOR_SIGMA,
            sku_scale_default_sigma=_TREND_PRIOR_SIGMA,
        )
        return pm.Deterministic("trend_sku", lin, dims="sku")

    mu_trend = resolve_prior(
        model_config=model_config,
        param_name="mu_trend",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": _TREND_PRIOR_SIGMA},
    )
    sigma_sku = resolve_prior(
        model_config=model_config,
        param_name="sigma_trend_sku",
        default_dist=pm.HalfNormal,
        default_kwargs={"sigma": _TREND_PRIOR_SIGMA},
    )
    eta_sku = resolve_prior(
        model_config=model_config,
        param_name="eta_trend_sku",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": 1.0},
        dims="sku",
    )
    return pm.Deterministic("trend_sku", mu_trend + sigma_sku * eta_sku, dims="sku")
