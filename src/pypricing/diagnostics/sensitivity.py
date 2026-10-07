"""Sensitivity to unobserved confounding (Cinelli & Hazlett, 2020)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from pypricing.diagnostics.identification import first_stage_exogenous

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel


def partial_r2(t: float, df: int) -> float:
    """Share of the outcome's residual variance explained by the regressor."""
    return float(t**2 / (t**2 + df))


def robustness_value(
    t: float, df: int, *, q: float = 1.0, alpha: float | None = None
) -> float:
    """Minimum confounder strength that would remove ``q`` of the estimate.

    Strength is the partial R^2 of the confounder with both treatment and
    outcome. With ``alpha``, it's the strength that would make the adjusted
    estimate no longer significant at that level instead.
    """
    fq = q * abs(t) / np.sqrt(df)
    if alpha is not None:
        f_crit = abs(stats.t.ppf(alpha / 2, df - 1)) / np.sqrt(df - 1)
        if fq <= f_crit:
            return 0.0
        if fq > 1 / f_crit:
            return float((fq**2 - f_crit**2) / (1 + fq**2))
        fq = fq - f_crit
    return float(0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2))


def _ols(
    y: np.ndarray, X: np.ndarray, slope_cols: slice
) -> tuple[np.ndarray, np.ndarray, int]:
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    df = len(y) - int(rank)
    sigma2 = float(resid @ resid) / df
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return coef[slope_cols], np.sqrt(np.diag(cov)[slope_cols]), df


def _row(estimate: float, se: float, df: int, q: float, alpha: float) -> dict:
    t = estimate / se
    return {
        "estimate": estimate,
        "se": se,
        "t": t,
        "partial_r2": partial_r2(t, df),
        "rv": robustness_value(t, df, q=q),
        "rv_qa": robustness_value(t, df, q=q, alpha=alpha),
    }


def check_sensitivity(
    model: DemandModel, *, q: float = 1.0, alpha: float = 0.05
) -> pd.DataFrame:
    """Robustness values of the price elasticity, per SKU and pooled.

    Fits ``log Q`` on ``log P`` by OLS with the model's exogenous terms (SKU
    intercepts, controls, trend, seasonality) and reports, for each slope,
    how strong a hidden confounder would need to be -- as partial R^2 with
    both price and demand -- to remove ``q`` of it (``rv``) or make it
    insignificant at ``alpha`` (``rv_qa``).

    The estimates are linear OLS elasticities, not the model's posterior. OLS
    fits each SKU on its own, while the model pools SKUs toward a shared mean,
    so SKUs with little price variation can differ a lot -- and those are the
    ones with low robustness values. The specification matches the log-log
    model without instruments and approximates the other curves. With
    instruments, this describes the uninstrumented estimate; see
    ``check_instruments`` for the IV side.
    """
    model._require_fitted()
    data = model.data
    quantity = data[model.quantity_col].to_numpy(dtype=np.float64)
    log_q = np.log(np.maximum(quantity, model.quantity_floor))
    log_p = np.log(data[model.price_col].to_numpy(dtype=np.float64))
    exog = first_stage_exogenous(model)
    n_exog = exog.shape[1]

    sku = pd.Categorical(data[model.sku_col], categories=model.sku_levels_)
    sku_dummies = pd.get_dummies(sku).to_numpy(dtype=np.float64)
    n_sku = sku_dummies.shape[1]

    rows = {}
    est, se, df = _ols(
        log_q,
        np.column_stack([exog, sku_dummies * log_p[:, None]]),
        slice(n_exog, n_exog + n_sku),
    )
    for level, e, s in zip(model.sku_levels_, est, se):
        rows[level] = _row(float(e), float(s), df, q, alpha)

    est, se, df = _ols(
        log_q, np.column_stack([exog, log_p]), slice(n_exog, n_exog + 1)
    )
    rows["pooled"] = _row(float(est[0]), float(se[0]), df, q, alpha)

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = model.sku_col
    return out
