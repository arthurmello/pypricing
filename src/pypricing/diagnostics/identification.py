"""Instrument diagnostics: first-stage strength, overidentification, IV vs OLS."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from pypricing.posterior import get_elasticity_means

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel

# Rule of thumb for one endogenous regressor (Staiger and Stock).
WEAK_IV_F_THRESHOLD = 10.0
OVERID_ALPHA = 0.05


def _lstsq_fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, int]:
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    return X @ coef, int(rank)


def _ssr_and_rank(y: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    fitted, rank = _lstsq_fit(y, X)
    resid = y - fitted
    return float(resid @ resid), rank


def _as_2d(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.size == 0 or a.shape[1] == 0:
        return np.ones((n, 1), dtype=np.float64)
    return a


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
    n = y.shape[0]
    Z = np.asarray(instruments, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    X = _as_2d(exogenous, n)
    ssr_r, rank_r = _ssr_and_rank(y, X)
    ssr_u, rank_u = _ssr_and_rank(y, np.column_stack([X, Z]))
    df_num = rank_u - rank_r
    df_den = n - rank_u
    if df_num <= 0 or df_den <= 0:
        return 0.0
    gap = max(ssr_r - ssr_u, 0.0)
    return (gap / df_num) / (ssr_u / df_den)


def sargan_test(
    log_quantity: np.ndarray,
    log_price: np.ndarray,
    instruments: np.ndarray,
    exogenous: np.ndarray,
) -> dict[str, float]:
    """Sargan overidentification test from a linear 2SLS demand equation.

    Fits ``log Q = X b + e * log P`` by 2SLS, then regresses the residuals on
    all exogenous variables and instruments: ``n * R^2 ~ chi2(k - 1)`` with
    ``k`` instruments, under the null that all instruments are valid.
    """
    q = np.asarray(log_quantity, dtype=np.float64).reshape(-1)
    p = np.asarray(log_price, dtype=np.float64).reshape(-1)
    n = q.shape[0]
    X = _as_2d(exogenous, n)
    W = np.column_stack([X, np.asarray(instruments, dtype=np.float64)])

    _, rank_x = _lstsq_fit(q, X)
    _, rank_w = _lstsq_fit(q, W)
    df = rank_w - rank_x - 1
    if df <= 0:
        raise ValueError(
            "Sargan test needs more instruments than endogenous regressors "
            "(at least 2 instruments with independent variation)."
        )

    p_hat, _ = _lstsq_fit(p, W)
    coef, _, _, _ = np.linalg.lstsq(np.column_stack([X, p_hat]), q, rcond=None)
    resid = q - np.column_stack([X, p]) @ coef

    ssr, _ = _ssr_and_rank(resid, W)
    sst = float(((resid - resid.mean()) ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    statistic = n * r2
    return {
        "statistic": float(statistic),
        "p_value": float(stats.chi2.sf(statistic, df)),
        "df": int(df),
    }


def first_stage_exogenous(model: DemandModel) -> np.ndarray:
    """Exogenous regressors of the training frame: SKU, controls, trend, season."""
    assert model.data is not None and model.sku_levels_ is not None
    data = model.data
    sku = pd.Categorical(data[model.sku_col], categories=model.sku_levels_)
    dummies = pd.get_dummies(sku).to_numpy(dtype=np.float64)
    parts: list[np.ndarray] = [dummies]
    if model.control_names_:
        parts.append(
            data.loc[:, list(model.control_names_)].to_numpy(dtype=np.float64)
        )
    if model.trend is not None:
        t = model._t_years_for_frame(data).reshape(-1, 1)
        parts.append(dummies * t if model.trend == "sku" else t)
    season = model._season_features_for_frame(data)
    if season is not None and season.size:
        parts.append(season)
    return np.column_stack(parts)


def _instruments_and_log_price(model: DemandModel) -> tuple[np.ndarray, np.ndarray]:
    instruments = model.data.loc[:, list(model.iv_names_)].to_numpy(dtype=np.float64)
    log_price = np.log(model.data[model.price_col].to_numpy(dtype=np.float64))
    return instruments, log_price


def iv_diagnostics(model: DemandModel, *, hdi_prob: float = 0.9) -> dict[str, Any]:
    """Endogeneity interval for ``rho`` and the first-stage partial F."""
    assert model.idata is not None
    post = model.idata.posterior
    if "rho" not in post and "pi" not in post:
        return {}

    out: dict[str, Any] = {}
    alpha = (1.0 - hdi_prob) / 2.0
    if "rho" in post:
        rho = np.asarray(post["rho"].values, dtype=np.float64).ravel()
        out["rho_mean"] = float(np.mean(rho))
        lo, hi = np.quantile(rho, [alpha, 1.0 - alpha])
        out["rho_hdi"] = (float(lo), float(hi))
        out["rho_hdi_includes_zero"] = bool(lo <= 0.0 <= hi)
    if model.iv_names_ and model.data is not None:
        instruments, log_price = _instruments_and_log_price(model)
        f_stat = first_stage_partial_f(
            log_price, instruments, first_stage_exogenous(model)
        )
        out["first_stage_f"] = f_stat
        out["weak_iv"] = bool(f_stat < WEAK_IV_F_THRESHOLD)
    return out


def _overid(model: DemandModel) -> dict[str, Any]:
    n_iv = len(model.iv_names_)
    if n_iv < 2:
        return {"applicable": False, "n_instruments": n_iv}
    instruments, log_price = _instruments_and_log_price(model)
    quantity = model.data[model.quantity_col].to_numpy(dtype=np.float64)
    log_quantity = np.log(np.maximum(quantity, model.quantity_floor))
    result = sargan_test(
        log_quantity, log_price, instruments, first_stage_exogenous(model)
    )
    return {"applicable": True, "n_instruments": n_iv, **result}


def check_instruments(
    model: DemandModel,
    *,
    hdi_prob: float = 0.9,
    compare_ols: bool = True,
    **sample_kwargs: Any,
) -> dict[str, Any]:
    """First-stage strength, overidentification test, and IV vs OLS elasticities.

    Warns when the first stage is weak (F below 10) or the Sargan test rejects
    (p below 0.05). The Sargan test uses a pooled linear 2SLS demand equation,
    not the fitted curve, so it checks whether the instruments agree with each
    other rather than the model's exact specification.

    With ``compare_ols``, refits the model without instruments on the same
    data; ``sample_kwargs`` default to those of the last ``fit()``.
    """
    model._require_fitted()
    if not model.iv_names_:
        raise ValueError(
            "Model has no instruments; set PanelColumns.iv_columns or add iv_* columns."
        )

    diag = iv_diagnostics(model, hdi_prob=hdi_prob)
    first_stage = {"f": diag["first_stage_f"], "weak": diag["weak_iv"]}
    if first_stage["weak"]:
        warnings.warn(
            f"Weak instruments: first-stage F = {first_stage['f']:.1f} "
            f"(< {WEAK_IV_F_THRESHOLD:.0f}). IV estimates will be noisy and "
            "pulled toward OLS.",
            stacklevel=3,
        )

    overid = _overid(model)
    if overid["applicable"] and overid["p_value"] < OVERID_ALPHA:
        warnings.warn(
            f"Sargan test rejects (p = {overid['p_value']:.3f}): the instruments "
            "imply different elasticities, so at least one may affect demand "
            "other than through price.",
            stacklevel=3,
        )

    elasticity = None
    if compare_ols:
        ols = model.clone(iv_columns=())
        ols.fit(model.data, **{**model.fit_kwargs_, **sample_kwargs})
        elasticity = pd.DataFrame(
            {"iv": get_elasticity_means(model), "ols": get_elasticity_means(ols)}
        )
        elasticity["shift"] = elasticity["iv"] - elasticity["ols"]

    return {
        "first_stage": first_stage,
        "overid": overid,
        "rho": {
            "mean": diag["rho_mean"],
            "hdi": diag["rho_hdi"],
            "hdi_includes_zero": diag["rho_hdi_includes_zero"],
        },
        "elasticity": elasticity,
    }
