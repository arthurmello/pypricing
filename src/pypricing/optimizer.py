"""Revenue-optimal prices from a fitted model posterior (per-SKU 1D search)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel


def mean_revenue_at_price(
    *,
    model: "DemandModel",
    price: float,
    sku_idx: int,
    X_row: np.ndarray | None,
) -> float:
    """
    Posterior mean of revenue ``price * exp(mu)`` for one SKU at a single price.

    ``mu`` is the mean log-quantity from the demand curve (sigma not folded into
    ``exp(mu)``; same convention as :meth:`DemandModel.sample_posterior_predictive`).
    """
    if getattr(model, "cross_elasticity", None) is not None:
        raise NotImplementedError(
            "mean_revenue_at_price does not include competitor prices when "
            "cross_elasticity is enabled; use sample_posterior_predictive with a "
            "full market cell or joint optimization (not implemented)."
        )

    if price <= 0:
        return float("-inf")

    log_p = float(np.log(price))
    assert model.idata is not None
    t_years = model._t_years_for_counterfactual(1)
    X_season = model._season_features_for_counterfactual(1)
    mu = model.compute_mu_from_posterior(
        posterior=model.idata.posterior,
        log_price=np.array([log_p], dtype=np.float64),
        obs_sku_idx=np.array([sku_idx], dtype=np.int64),
        X_control=X_row,
        t_years=t_years,
        X_season=X_season,
    )
    q = np.exp(mu)
    revenue = price * q
    return float(np.mean(revenue))


def _negative_mean_revenue(
    log_price: float,
    model: "DemandModel",
    sku_idx: int,
    X_row: np.ndarray | None,
) -> float:
    p = float(np.exp(log_price))
    return -mean_revenue_at_price(
        model=model,
        price=p,
        sku_idx=sku_idx,
        X_row=X_row,
    )


def optimize_price_one_sku(
    *,
    model: "DemandModel",
    sku_idx: int,
    price_low: float,
    price_high: float,
    X_row: np.ndarray | None,
    options: dict[str, Any] | None = None,
) -> float:
    """Maximize posterior mean revenue for one SKU over ``[price_low, price_high]``."""
    from scipy.optimize import minimize_scalar

    if price_low <= 0 or price_high <= 0 or price_low >= price_high:
        raise ValueError("Need 0 < price_low < price_high")

    opts = {"xatol": 1e-10, "maxiter": 500}
    if options:
        opts.update(options)

    res = minimize_scalar(
        _negative_mean_revenue,
        bounds=(np.log(price_low), np.log(price_high)),
        method="bounded",
        args=(
            model,
            sku_idx,
            X_row,
        ),
        options=opts,
    )
    return float(np.exp(res.x))


def control_matrix_for_skus(
    model: "DemandModel",
    controls_df: pd.DataFrame | None,
) -> np.ndarray | None:
    assert model.sku_levels_ is not None
    sku_levels = model.sku_levels_
    control_names = model.control_names_
    sku_col = model.sku_col

    if not control_names:
        return None
    if controls_df is None:
        raise ValueError(
            "Model was fit with control columns; pass `controls_df` with one row per "
            f"SKU in {sku_col!r} and columns {list(control_names)}."
        )
    missing = [c for c in control_names if c not in controls_df.columns]
    if missing:
        raise ValueError(f"controls_df missing columns: {missing}")
    if sku_col not in controls_df.columns:
        raise ValueError(f"controls_df missing {sku_col!r}")

    rows = []
    for sku in sku_levels:
        sub = controls_df.loc[controls_df[sku_col] == sku]
        if len(sub) == 0:
            raise ValueError(f"No row in controls_df for sku={sku!r}")
        if len(sub) > 1:
            raise ValueError(f"Multiple rows in controls_df for sku={sku!r}")
        rows.append(sub.iloc[0][list(control_names)].to_numpy(dtype=np.float64))
    return np.stack(rows, axis=0)


def _normalize_price_bounds(
    sku_levels: pd.Index,
    price_bounds: Mapping[Any, tuple[float, float]] | pd.Series,
) -> np.ndarray:
    """Shape (n_skus, 2) with columns [low, high]."""
    out = np.empty((len(sku_levels), 2), dtype=np.float64)
    for i, sku in enumerate(sku_levels):
        if sku in price_bounds:
            key: Any = sku
        elif str(sku) in price_bounds:
            key = str(sku)
        else:
            raise ValueError(f"Missing price_bounds for sku={sku!r}")
        low, high = price_bounds[key]
        out[i, 0] = low
        out[i, 1] = high
    return out


def optimize_prices(
    *,
    model: "DemandModel",
    price_bounds: Mapping[Any, tuple[float, float]] | pd.Series,
    controls_df: pd.DataFrame | None = None,
    minimize_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Maximize posterior **mean** total revenue by choosing one price per SKU.

    With independent per-SKU demand (no cross-price in the model), this runs one
    bounded 1D optimization per SKU on log-price. Not available when
    ``cross_elasticity`` is set on the model.

    Parameters
    ----------
    model
        Fitted pricing model instance.
    price_bounds
        Mapping each SKU to ``(low, high)`` strictly positive with ``low < high``.
    controls_df
        One row per SKU with ``sku_col`` and all ``control_names`` if controls were used.
    minimize_options
        Forwarded to :func:`scipy.optimize.minimize_scalar` ``options``.

    Notes
    -----
    For ``log_log`` demand, expected revenue in ``p`` is proportional to ``p^(1+e)``
    for each posterior draw; if elasticity is roughly constant, the optimum often lies
    on the **boundary** of ``price_bounds``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``optimal_price``, ``mean_revenue`` (posterior mean revenue at that
        price for that SKU), ``at_bound`` (``"low"`` / ``"high"`` / ``None``, whether
        the optimum landed on the corresponding ``price_bounds`` edge rather than an
        interior price); index = SKU.
    """
    model._require_fitted()
    if getattr(model, "cross_elasticity", None) is not None:
        raise NotImplementedError(
            "optimize_prices is not supported when cross_elasticity is enabled "
            "(revenue depends on all SKU prices in the market cell)."
        )
    assert model.sku_levels_ is not None
    bounds_mat = _normalize_price_bounds(model.sku_levels_, price_bounds)
    X = control_matrix_for_skus(model, controls_df)

    rows: list[dict[str, Any]] = []
    at_bound_skus: dict[Any, str] = {}
    for i, sku in enumerate(model.sku_levels_):
        low, high = bounds_mat[i]
        X_row = None if X is None else X[i : i + 1, :]
        opt_p = optimize_price_one_sku(
            model=model,
            sku_idx=i,
            price_low=low,
            price_high=high,
            X_row=X_row,
            options=minimize_options,
        )
        mrev = mean_revenue_at_price(
            model=model,
            price=opt_p,
            sku_idx=i,
            X_row=X_row,
        )

        tol = 1e-3 * (high - low)
        at_bound: str | None = None
        if abs(opt_p - low) < tol:
            at_bound = "low"
        elif abs(opt_p - high) < tol:
            at_bound = "high"
        if at_bound is not None:
            at_bound_skus[sku] = at_bound

        rows.append(
            {
                "sku": sku,
                "optimal_price": opt_p,
                "mean_revenue": mrev,
                "at_bound": at_bound,
            }
        )

    if at_bound_skus:
        msg = (
            f"{len(at_bound_skus)}/{len(model.sku_levels_)} SKU(s) optimized to a "
            f"price_bounds edge, not an interior price: {at_bound_skus}."
        )
        if getattr(model, "model_name", None) == "log_log":
            msg += (
                " This is expected for log_log demand: revenue is proportional to "
                "price^(1+elasticity), which is monotonic in price for any constant "
                "elasticity != -1, so the constrained optimum is always a bound, not "
                "a genuine trade-off point. If you want a real interior optimum, use "
                "QuadraticLogDemandModel or SigmoidSaturationDemandModel instead."
            )
        else:
            msg += (
                " Verify this reflects a genuine trade-off and not just where you "
                "set the bound."
            )
        warnings.warn(msg, stacklevel=2)

    return pd.DataFrame(rows).set_index("sku")
