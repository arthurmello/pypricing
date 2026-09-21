"""Control-function IV terms for log-demand models."""

from __future__ import annotations

from typing import Any

import numpy as np
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

# Rule of thumb for one endogenous regressor (Staiger and Stock).
WEAK_IV_F_THRESHOLD = 10.0


def _ssr_and_rank(y: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(resid @ resid), int(rank)


def first_stage_partial_f(
    log_price: np.ndarray,
    instruments: np.ndarray,
    exogenous: np.ndarray,
) -> float:
    """Partial F for excluded instruments in a linear price equation.

    ``exogenous`` is partialled out first (SKU intercepts, controls, trend,
    seasonality). The statistic is the homoskedastic F for the joint hypothesis
    that every instrument coefficient is zero. Instruments with no remaining
    variation return ``0``.
    """
    y = np.asarray(log_price, dtype=np.float64).reshape(-1)
    Z = np.asarray(instruments, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    X = np.asarray(exogenous, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = y.shape[0]
    if X.size == 0 or X.shape[1] == 0:
        X = np.ones((n, 1), dtype=np.float64)
    ssr_r, rank_r = _ssr_and_rank(y, X)
    ssr_u, rank_u = _ssr_and_rank(y, np.column_stack([X, Z]))
    df_num = rank_u - rank_r
    df_den = n - rank_u
    if df_num <= 0 or df_den <= 0:
        return 0.0
    gap = max(ssr_r - ssr_u, 0.0)
    return (gap / df_num) / (ssr_u / df_den)


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
