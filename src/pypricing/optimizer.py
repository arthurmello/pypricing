"""Revenue-optimal prices from a fitted model posterior (per-SKU 1D search)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel

_N_PRICE_GRID = 64


def mean_revenue_at_price(
    *,
    model: "DemandModel",
    price: float,
    sku_idx: int,
    X_row: np.ndarray | None,
    at_period: pd.Timestamp | str | None = None,
) -> float:
    """
    Posterior mean of expected revenue for one SKU at a single price.

    Expected quantity on a draw is ``exp(mu + sigma**2 / 2)``: ``mu`` is the
    demand-curve mean log-quantity, and the lognormal factor is the mean of
    ``exp(mu + sigma * e)``. That is the same target as ``quantity_mean`` from
    :meth:`DemandModel.predict`.
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
    posterior = model.idata.posterior
    t_years = model._t_years_for_counterfactual(1, at_period)
    X_season = model._season_features_for_counterfactual(1, at_period)
    mu = model.compute_mu_from_posterior(
        posterior=posterior,
        log_price=np.array([log_p], dtype=np.float64),
        obs_sku_idx=np.array([sku_idx], dtype=np.int64),
        X_control=X_row,
        t_years=t_years,
        X_season=X_season,
    )
    sigma = np.asarray(posterior["sigma"].values, dtype=np.float64).reshape(
        mu.shape[0], mu.shape[1]
    )
    expected_q = np.exp(mu + 0.5 * np.square(sigma)[:, :, None])
    return float(np.mean(price * expected_q))


def _negative_mean_revenue(
    log_price: float,
    model: "DemandModel",
    sku_idx: int,
    X_row: np.ndarray | None,
    at_period: pd.Timestamp | str | None,
) -> float:
    p = float(np.exp(log_price))
    return -mean_revenue_at_price(
        model=model,
        price=p,
        sku_idx=sku_idx,
        X_row=X_row,
        at_period=at_period,
    )


def optimize_price_one_sku(
    *,
    model: "DemandModel",
    sku_idx: int,
    price_low: float,
    price_high: float,
    X_row: np.ndarray | None,
    at_period: pd.Timestamp | str | None = None,
    options: dict[str, Any] | None = None,
) -> float:
    """Maximize posterior mean expected revenue over ``[price_low, price_high]``.

    Scores a log-price grid, including both bounds, then polishes the best grid
    point. The grid is what keeps a second peak from being missed by a single
    unimodal search.
    """
    from scipy.optimize import minimize_scalar

    if price_low <= 0 or price_high <= 0 or price_low >= price_high:
        raise ValueError("Need 0 < price_low < price_high")

    log_low = float(np.log(price_low))
    log_high = float(np.log(price_high))
    grid = np.linspace(log_low, log_high, _N_PRICE_GRID)
    revenues = np.array(
        [
            mean_revenue_at_price(
                model=model,
                price=float(np.exp(log_p)),
                sku_idx=sku_idx,
                X_row=X_row,
                at_period=at_period,
            )
            for log_p in grid
        ]
    )
    best_i = int(np.argmax(revenues))
    best_log = float(grid[best_i])
    best_rev = float(revenues[best_i])

    step = float(grid[1] - grid[0])
    win_lo = max(log_low, best_log - step)
    win_hi = min(log_high, best_log + step)
    if win_hi <= win_lo:
        return float(np.exp(best_log))

    opts = {"xatol": 1e-10, "maxiter": 500}
    if options:
        opts.update(options)
    res = minimize_scalar(
        _negative_mean_revenue,
        bounds=(win_lo, win_hi),
        method="bounded",
        args=(model, sku_idx, X_row, at_period),
        options=opts,
    )
    polished = float(np.exp(res.x))
    polished_rev = mean_revenue_at_price(
        model=model,
        price=polished,
        sku_idx=sku_idx,
        X_row=X_row,
        at_period=at_period,
    )
    if polished_rev >= best_rev:
        return polished
    return float(np.exp(best_log))


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
    at_period: pd.Timestamp | str | None = None,
    minimize_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Maximize posterior mean expected revenue by choosing one price per SKU.

    Expected revenue on a draw is ``price * exp(mu + sigma**2 / 2)``. With
    independent per-SKU demand (no cross-price in the model), each SKU is a 1D
    search on log-price inside its bounds. Not available when
    ``cross_elasticity`` is set on the model.

    Parameters
    ----------
    model
        Fitted pricing model instance.
    price_bounds
        Mapping each SKU to ``(low, high)`` strictly positive with ``low < high``.
    controls_df
        One row per SKU with ``sku_col`` and all ``control_names`` if controls were used.
    at_period
        Calendar date for trend and seasonality. Defaults to the last training date.
        Ignored when the model has neither.
    minimize_options
        Forwarded to the local :func:`scipy.optimize.minimize_scalar` polish.

    Notes
    -----
    For ``log_log`` demand, expected revenue on a draw is proportional to
    ``p^(1+e)``. That is monotone in price unless ``e == -1``, and a posterior
    that straddles ``-1`` is U-shaped, so the maximum on an interval is always
    an endpoint of ``price_bounds``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``optimal_price``, ``mean_revenue`` (posterior mean expected
        revenue at that price for that SKU), ``at_bound`` (``"low"`` / ``"high"``
        / ``None``, whether the optimum landed on the corresponding
        ``price_bounds`` edge rather than an interior price); index = SKU.
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
            at_period=at_period,
            options=minimize_options,
        )
        mrev = mean_revenue_at_price(
            model=model,
            price=opt_p,
            sku_idx=i,
            X_row=X_row,
            at_period=at_period,
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
                "price^(1+elasticity). Any draw with elasticity other than -1 is "
                "monotone in price, and a posterior that straddles -1 is U-shaped, "
                "so the constrained optimum is always a price_bounds endpoint. An "
                "interior revenue maximum needs a price-dependent elasticity "
                "(QuadraticLogDemandModel or SigmoidSaturationDemandModel)."
            )
        else:
            msg += (
                " Verify this reflects a genuine trade-off and not just where you "
                "set the bound."
            )
        warnings.warn(msg, stacklevel=2)

    return pd.DataFrame(rows).set_index("sku")
