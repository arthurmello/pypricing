from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:  # pragma: no cover
    from pypricing.models.basic import DemandModel


def _extract_hdi(draws: np.ndarray, *, hdi_prob: float) -> tuple[float, float]:
    hdi = az.hdi(draws, hdi_prob=hdi_prob)
    # `az.hdi` typically returns an xarray.DataArray with `hdi` coordinate.
    if isinstance(hdi, xr.DataArray):
        lower = float(hdi.sel(hdi="lower").values)
        upper = float(hdi.sel(hdi="higher").values)
        return lower, upper
    if isinstance(hdi, (np.ndarray, list, tuple)):
        arr = np.asarray(hdi).reshape(-1)
        return float(arr[0]), float(arr[1])
    raise TypeError(f"Unexpected HDI return type: {type(hdi)}")


def _sku_price_grid_design(
    model: "DemandModel",
    sku: Any,
    price_grid: Iterable[float],
    controls: dict[str, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, Any]:
    """
    Arrays for counterfactuals along ``price_grid`` for one SKU (fixed controls).

    Returns ``price_grid_arr``, ``log_price``, ``obs_sku_idx`` (length n), ``X_control``
    (n × n_controls or None), ``sku_value``.
    """
    model._require_fitted()
    sku_levels = model.sku_levels_
    if sku_levels is None or len(sku_levels) == 0:
        raise RuntimeError(
            "Model does not have stored sku_levels_. Fit the model first."
        )

    if sku in sku_levels:
        sku_value = sku
    elif isinstance(sku, (int, np.integer)) and 0 <= int(sku) < len(sku_levels):
        sku_value = sku_levels[int(sku)]
    else:
        raise ValueError(
            f"Invalid `sku`: expected one of {list(sku_levels)[:5]}... "
            "or a valid integer index"
        )

    price_grid_arr = np.asarray(list(price_grid), dtype=float)
    if (price_grid_arr <= 0).any():
        raise ValueError("price_grid values must be strictly positive.")
    log_price = np.log(price_grid_arr)
    n = int(log_price.shape[0])
    sku_idx = int(sku_levels.get_loc(sku_value))
    obs_sku_idx = np.full(n, sku_idx, dtype=np.int64)

    if model.control_names_:
        if controls is None:
            raise ValueError("Model has control variables; provide `controls` dict.")
        missing = [c for c in model.control_names_ if c not in controls]
        if missing:
            raise ValueError(f"Missing control values for: {missing}")
        row = np.array(
            [[float(controls[c]) for c in model.control_names_]],
            dtype=np.float64,
        )
        X_control = np.repeat(row, n, axis=0)
    else:
        X_control = None

    return price_grid_arr, log_price, obs_sku_idx, X_control, sku_value


def _sku_price_grid_predictions(
    model: "DemandModel",
    sku: Any,
    price_grid: Iterable[float],
    controls: dict[str, float] | None,
    *,
    hdi_prob: float,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, Any]:
    """Predict quantity along ``price_grid`` for one SKU; return ``(pred_df, sku_value)``."""
    price_grid_arr, _, _, _, sku_value = _sku_price_grid_design(
        model, sku, price_grid, controls
    )
    n = len(price_grid_arr)
    df_pred = pd.DataFrame(
        {
            model.sku_col: [sku_value] * n,
            model.price_col: price_grid_arr,
        }
    )
    if model.control_names_:
        assert controls is not None
        for c in model.control_names_:
            df_pred[c] = float(controls[c])

    pred = model.predict(df_pred, hdi_prob=hdi_prob, random_seed=random_seed)
    return pred, sku_value


def _posterior_local_elasticity(
    model: "DemandModel",
    *,
    log_price: np.ndarray,
    obs_sku_idx: np.ndarray,
    X_control: np.ndarray | None,
    fd_step: float | None = None,
) -> np.ndarray:
    """
    Posterior draws of local own-price elasticity ``d mu / d log p`` on the mean
    log-quantity path (holding controls fixed), shape ``(chain, draw, n_points)``.

    Uses a central difference on ``mu`` from :meth:`DemandModel.compute_mu_from_posterior`.
    """
    if getattr(model, "cross_elasticity", None) is not None:
        raise NotImplementedError(
            "Local elasticity along price for one SKU is not implemented when "
            "cross_elasticity is enabled (counterfactuals need a full market cell)."
        )
    model._require_fitted()
    assert model.idata is not None
    post = model.idata.posterior
    log_price = np.asarray(log_price, dtype=np.float64)
    if fd_step is None:
        h = np.cbrt(np.finfo(np.float64).eps) * np.maximum(1.0, np.abs(log_price))
    else:
        h = np.full_like(log_price, float(fd_step))

    mu_p = model.compute_mu_from_posterior(
        posterior=post,
        log_price=log_price + h,
        obs_sku_idx=obs_sku_idx,
        X_control=X_control,
        X_cross=None,
    )
    mu_m = model.compute_mu_from_posterior(
        posterior=post,
        log_price=log_price - h,
        obs_sku_idx=obs_sku_idx,
        X_control=X_control,
        X_cross=None,
    )
    denom = (2.0 * h)[np.newaxis, np.newaxis, :]
    return (mu_p - mu_m) / denom


def plot_elasticity_posterior(
    model: "DemandModel",
    *,
    hdi_prob: float = 0.9,
    ax=None,
):
    """
    Plot posterior mean + horizontal HDI interval per SKU elasticity.

    The plot is intentionally simple (no coloring, no true-value markers).
    """
    model._require_fitted()
    assert model.idata is not None

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    elasticity = model.idata.posterior["elasticity_sku"]
    sku_codes = elasticity.coords["sku"].values

    if model.sku_levels_ is not None and len(model.sku_levels_) == len(sku_codes):
        ylabels = [str(x) for x in model.sku_levels_]
    else:
        ylabels = [str(x) for x in sku_codes]

    for i, sku_code in enumerate(sku_codes):
        draws = elasticity.sel(sku=sku_code).values.flatten()
        mean = float(np.mean(draws))
        lower, upper = _extract_hdi(draws, hdi_prob=hdi_prob)

        ax.plot(mean, i, "o", markersize=7, color="C0")
        ax.plot([lower, upper], [i, i], linewidth=2, alpha=0.8, color="C0")

    ax.set_yticks(range(len(sku_codes)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Own-price elasticity")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
    ax.set_title(
        f"Estimated own-price elasticity per SKU\n(mean + {int(hdi_prob * 100)}% HDI)"
    )
    plt.tight_layout()
    return ax


def plot_cross_price_effects_heatmap(
    model: "DemandModel",
    *,
    agg: Literal["mean", "median"] = "mean",
    cmap: str = "coolwarm",
    ax=None,
):
    """
    Heatmap of cross-price coefficients ``gamma_pair`` on the mean log-quantity path.

    Cell ``(i, j)`` summarizes the posterior of the coefficient on competitor ``j``'s
    **log price** in the linear predictor for focal SKU ``i``. The diagonal (own
    price enters via ``elasticity_sku``, not ``gamma_pair``) and any directed pair
    omitted by the cross-price structure (e.g. ``within_group``) are masked.

    Parameters
    ----------
    model
        Fitted model with ``cross_elasticity`` and ``gamma_pair`` in the posterior.
    agg
        How to summarize posterior draws per pair: ``mean`` or ``median``.
    cmap
        Matplotlib colormap (diverging around zero is recommended).
    ax
        Axes to draw on, or None for a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    model._require_fitted()
    assert model.idata is not None
    if getattr(model, "cross_elasticity", None) is None:
        raise ValueError(
            "plot_cross_price_effects_heatmap requires a model fit with "
            "cross_elasticity enabled."
        )
    if "gamma_pair" not in model.idata.posterior:
        raise ValueError("Posterior is missing gamma_pair.")
    pf = getattr(model, "cross_pairs_", None)
    if pf is None:
        raise RuntimeError(
            "Model is missing cross_pairs_; refit with cross_elasticity."
        )
    sku_levels = model.sku_levels_
    if sku_levels is None:
        raise RuntimeError("Model has no sku_levels_.")
    n = len(sku_levels)
    gam = model.idata.posterior["gamma_pair"].values
    if gam.ndim != 3:
        raise ValueError(f"Unexpected gamma_pair ndim/shape: {gam.shape}")
    if agg == "mean":
        g_summary = gam.mean(axis=(0, 1))
    elif agg == "median":
        g_summary = np.median(gam.reshape(-1, gam.shape[-1]), axis=0)
    else:
        raise ValueError("agg must be 'mean' or 'median'.")
    if int(g_summary.shape[0]) != pf.n_pairs:
        raise RuntimeError("gamma_pair length does not match cross_pairs_.")

    mat = np.full((n, n), np.nan, dtype=float)
    for p in range(pf.n_pairs):
        mat[int(pf.pair_from[p]), int(pf.pair_to[p])] = float(g_summary[p])

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        raise RuntimeError("No finite cross-price cells to plot.")
    vmax = float(np.nanmax(np.abs(finite)))
    if vmax == 0.0:
        vmax = 1.0

    labels = [str(s) for s in sku_levels]
    if ax is None:
        fig_w = max(5.5, 0.65 * n)
        fig_h = max(4.8, 0.55 * n)
        _, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Competitor {model.sku_col} (log price)")
    ax.set_ylabel(f"Focal {model.sku_col}")
    stat = "Posterior mean" if agg == "mean" else "Posterior median"
    ax.set_title(f"Cross-price effects on mean log quantity\n({stat} of γ)")
    plt.colorbar(
        im,
        ax=ax,
        label="γ (on competitor log price)",
        fraction=0.046,
        pad=0.04,
    )
    plt.tight_layout()
    return ax


def plot_response_curve(
    model: "DemandModel",
    *,
    sku: Any,
    price_grid: Iterable[float],
    controls: dict[str, float] | None = None,
    hdi_prob: float = 0.9,
    ax=None,
):
    """
    Plot predicted quantity vs price for a single SKU.

    Requires `sku` + `price_grid` and (if the fitted model included them) the
    same set of `control_*` features via `controls`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    pred, sku_value = _sku_price_grid_predictions(
        model,
        sku,
        price_grid,
        controls,
        hdi_prob=hdi_prob,
    )

    ax.plot(
        pred[model.price_col],
        pred["quantity_mean"],
        marker="o",
        label="Posterior mean",
        color="C0",
    )
    ax.fill_between(
        pred[model.price_col],
        pred["quantity_hdi_lower"],
        pred["quantity_hdi_upper"],
        alpha=0.25,
        color="C0",
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.set_xlabel("price")
    ax.set_ylabel("quantity")
    ax.set_title(f"Predicted response for {model.sku_col}={sku_value}")
    ax.legend(loc="best")
    plt.tight_layout()
    return ax


def plot_local_elasticity_vs_price(
    model: "DemandModel",
    *,
    sku: Any,
    price_grid: Iterable[float],
    controls: dict[str, float] | None = None,
    hdi_prob: float = 0.9,
    fd_step: float | None = None,
    ax=None,
):
    """
    Plot local own-price elasticity along ``price_grid`` for one SKU.

    Elasticity is ``d mu / d (log p)`` on the mean log-quantity path from the
    fitted surface (holding controls fixed). For ``Q_mean approx exp(mu)`` this
    matches ``d log Q_mean / d log p``. On a plain log-log model it is **flat**
    in price and aligns with ``elasticity_sku``.

    Posterior uncertainty is shown by taking the central difference per posterior
    draw, then summarizing with mean and HDI at each price.

    Parameters match :func:`plot_response_curve` plus optional ``fd_step`` for
    the log-price step in the finite difference (default: scale with ``|log p|``).

    Raises
    ------
    NotImplementedError
        When ``cross_elasticity`` is enabled (counterfactuals need a full market
        cell).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    price_grid_arr, log_price, obs_sku_idx, X_control, sku_value = (
        _sku_price_grid_design(model, sku, price_grid, controls)
    )
    elast = _posterior_local_elasticity(
        model,
        log_price=log_price,
        obs_sku_idx=obs_sku_idx,
        X_control=X_control,
        fd_step=fd_step,
    )
    mean = elast.mean(axis=(0, 1))
    n = mean.shape[0]
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    for i in range(n):
        lo, hi = _extract_hdi(elast[:, :, i].ravel(), hdi_prob=hdi_prob)
        lower[i] = lo
        upper[i] = hi

    p = price_grid_arr
    ax.plot(
        p,
        mean,
        marker="o",
        label="Posterior mean",
        color="C2",
    )
    ax.fill_between(
        p,
        lower,
        upper,
        alpha=0.25,
        color="C2",
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.35)
    ax.set_xlabel("price")
    ax.set_ylabel(r"local elasticity $d\mu / d\log p$")
    ax.set_title(f"Local own-price elasticity vs price ({model.sku_col}={sku_value})")
    ax.legend(loc="best")
    plt.tight_layout()
    return ax


def plot_revenue_vs_price(
    model: "DemandModel",
    *,
    sku: Any,
    price_grid: Iterable[float],
    controls: dict[str, float] | None = None,
    hdi_prob: float = 0.9,
    random_seed: int | None = None,
    ax=None,
):
    """
    Plot posterior predictive revenue (price × quantity) vs price for one SKU.

    Uses the same prediction path as :func:`plot_response_curve`. The HDI band
    scales quantity intervals by price point-wise (exact for the predictive
    distribution of ``price * quantity`` when price is fixed and quantity is
    positive).

    Parameters match :func:`plot_response_curve` plus optional ``random_seed``
    for :meth:`~pypricing.models.basic.DemandModel.predict`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    pred, sku_value = _sku_price_grid_predictions(
        model,
        sku,
        price_grid,
        controls,
        hdi_prob=hdi_prob,
        random_seed=random_seed,
    )

    p = pred[model.price_col].to_numpy(dtype=float)
    rev_mean = p * pred["quantity_mean"].to_numpy(dtype=float)
    rev_lo = p * pred["quantity_hdi_lower"].to_numpy(dtype=float)
    rev_hi = p * pred["quantity_hdi_upper"].to_numpy(dtype=float)

    ax.plot(
        p,
        rev_mean,
        marker="o",
        label="Posterior mean revenue",
        color="C1",
    )
    ax.fill_between(
        p,
        rev_lo,
        rev_hi,
        alpha=0.25,
        color="C1",
        label=f"{int(hdi_prob * 100)}% HDI",
    )
    ax.set_xlabel("price")
    ax.set_ylabel("revenue (price × quantity)")
    ax.set_title(f"Predicted revenue for {model.sku_col}={sku_value}")
    ax.legend(loc="best")
    plt.tight_layout()
    return ax


@dataclass
class _OptSummaryAlign:
    """Optimizer output and reference prices aligned to ``sku_levels_`` order."""

    skus: list[Any]
    labels: list[str]
    ref_prices: np.ndarray
    opt_prices: np.ndarray
    opt_mean_revenue: np.ndarray


def _align_optimization_summary(
    model: "DemandModel",
    opt_df: pd.DataFrame,
    reference_prices: Mapping[Any, float] | pd.Series | None,
) -> _OptSummaryAlign:
    """Align optimizer output and reference prices to ``model.sku_levels_`` order."""
    required = {"optimal_price", "mean_revenue"}
    miss = required - set(opt_df.columns)
    if miss:
        raise ValueError(f"opt_df missing columns: {sorted(miss)}")

    skus = list(model.sku_levels_)
    if len(skus) == 0:
        raise RuntimeError("Model has no sku_levels_.")

    aligned = opt_df.reindex(skus)
    if aligned["optimal_price"].isna().any():
        raise ValueError("opt_df must include every SKU in model.sku_levels_.")

    if reference_prices is None:
        if model.data is None:
            raise ValueError("reference_prices is required when model.data is missing.")
        ref_series = (
            model.data.groupby(model.sku_col, sort=False)[model.price_col]
            .mean()
            .reindex(skus)
        )
    elif isinstance(reference_prices, pd.Series):
        ref_series = reference_prices.reindex(skus)
    else:
        ref_series = pd.Series(dict(reference_prices)).reindex(skus)

    if ref_series.isna().any():
        raise ValueError("reference_prices missing values for some SKUs.")

    return _OptSummaryAlign(
        skus=skus,
        labels=[str(s) for s in skus],
        ref_prices=ref_series.to_numpy(dtype=float),
        opt_prices=aligned["optimal_price"].to_numpy(dtype=float),
        opt_mean_revenue=aligned["mean_revenue"].to_numpy(dtype=float),
    )


def _draw_optimization_summary_bars(
    *,
    ax_price,
    ax_rev,
    align: _OptSummaryAlign,
    ref_revenues: np.ndarray | None,
) -> None:
    """Draw price bars; if ``ax_rev`` and ``ref_revenues`` are set, draw revenue bars too."""
    n = len(align.skus)
    x = np.arange(n)
    width = 0.35

    ax_price.bar(
        x - width / 2,
        align.ref_prices,
        width,
        label="Reference price",
        color="C0",
    )
    ax_price.bar(
        x + width / 2,
        align.opt_prices,
        width,
        label="Optimal price",
        color="C1",
    )
    ax_price.set_ylabel("price")
    ax_price.set_title("Price optimization summary")
    ax_price.legend(loc="best")

    if ax_rev is not None and ref_revenues is not None:
        ax_price.set_xticks(x)
        ax_price.set_xticklabels([])

        ax_rev.bar(
            x - width / 2,
            ref_revenues,
            width,
            label="At reference price",
            color="C0",
        )
        ax_rev.bar(
            x + width / 2,
            align.opt_mean_revenue,
            width,
            label="At optimal price",
            color="C1",
        )
        ax_rev.set_ylabel("posterior mean revenue")
        ax_rev.set_xticks(x)
        ax_rev.set_xticklabels(align.labels, rotation=45, ha="right")
        ax_rev.legend(loc="best")
    else:
        ax_price.set_xticks(x)
        ax_price.set_xticklabels(align.labels, rotation=45, ha="right")


def plot_optimization_summary(
    model: "DemandModel",
    opt_df: pd.DataFrame,
    *,
    reference_prices: Mapping[Any, float] | pd.Series | None = None,
    controls_df: pd.DataFrame | None = None,
    include_revenue_comparison: bool = True,
    ax=None,
) -> tuple[Any, Any | None]:
    """
    Visual summary of :func:`~pypricing.optimizer.optimize_prices` output.

    Top panel: grouped bars of reference price vs ``optimal_price`` per SKU.
    Bottom panel (unless ``ax`` is passed): grouped bars of posterior **mean**
    revenue at the reference price vs at the optimal price (same definition as
    ``mean_revenue`` in the optimizer output).

    Parameters
    ----------
    model
        Fitted model (same instance used for ``optimize_prices``).
    opt_df
        DataFrame returned by ``optimize_prices`` (columns ``optimal_price``,
        ``mean_revenue``; index = SKU).
    reference_prices
        Baseline price per SKU for comparison. If omitted, uses the mean of
        ``model.data`` ``price_col`` per ``sku_col``.
    controls_df
        Same ``controls_df`` passed to ``optimize_prices`` when the model has
        control features; required for the revenue panel in that case.
    include_revenue_comparison
        If False, only the price panel is drawn (or the only panel when ``ax``
        is set).
    ax
        If provided, draw only the price comparison on this axes (no revenue
        panel).

    Returns
    -------
    tuple
        ``(ax_prices, ax_revenue)`` where ``ax_revenue`` is None when only one
        panel is shown.

    Raises
    ------
    NotImplementedError
        If the model uses ``cross_elasticity`` (revenue comparison uses
        ``mean_revenue_at_price``, which is not defined in that case).
    """
    from pypricing.optimizer import control_matrix_for_skus, mean_revenue_at_price

    model._require_fitted()
    if getattr(model, "cross_elasticity", None) is not None:
        raise NotImplementedError(
            "plot_optimization_summary does not support cross_elasticity models."
        )

    align = _align_optimization_summary(model, opt_df, reference_prices)
    n = len(align.skus)

    show_revenue = include_revenue_comparison and ax is None
    ax_rev = None
    if ax is None:
        if show_revenue:
            _, (ax_price, ax_rev) = plt.subplots(
                2,
                1,
                figsize=(max(8.0, 0.45 * n), 7.0),
                sharex=True,
                layout="constrained",
            )
        else:
            _, ax_price = plt.subplots(figsize=(max(8.0, 0.45 * n), 4.5))
    else:
        ax_price = ax

    ref_revenues: np.ndarray | None = None
    if show_revenue:
        X = control_matrix_for_skus(model, controls_df)
        ref_revenues = np.empty(n, dtype=float)
        for i in range(n):
            X_row = None if X is None else X[i : i + 1, :]
            ref_revenues[i] = mean_revenue_at_price(
                model=model,
                price=float(align.ref_prices[i]),
                sku_idx=i,
                X_row=X_row,
            )

    _draw_optimization_summary_bars(
        ax_price=ax_price,
        ax_rev=ax_rev,
        align=align,
        ref_revenues=ref_revenues,
    )
    if ax is None and not show_revenue:
        plt.tight_layout()

    return ax_price, ax_rev


def plot_posterior_predictive_calibration(
    model: "DemandModel",
    *,
    df: pd.DataFrame | None = None,
    hdi_prob: float = 0.94,
    style: Literal["series", "calibration"] = "series",
    random_seed: int | None = None,
    ax=None,
):
    """
    Compare observed quantities on a panel to posterior predictive summaries.

    Uses :meth:`~pypricing.models.basic.DemandModel.predict` (mean log-quantity
    path, then Gaussian noise in log space) so intervals reflect epistemic +
    residual uncertainty.

    Parameters
    ----------
    model
        Fitted demand model.
    df
        Panel used for prediction. Defaults to the model's training frame
        ``model.data``.
    hdi_prob
        Credible level for predictive intervals.
    style
        - ``"series"``: observations ordered by ``period_col`` then ``sku_col``
          (when present); posterior mean line, HDI band, observed markers.
        - ``"calibration"``: scatter of observed vs posterior mean quantity
          with a y = x reference line.
    random_seed
        Passed to :meth:`~pypricing.models.basic.DemandModel.predict`.
    ax
        Matplotlib axes, or None to create a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    model._require_fitted()
    assert model.idata is not None

    if df is None:
        if model.data is None:
            raise RuntimeError(
                "No training data on the model; pass `df` explicitly "
                "(e.g. the same frame you passed to fit())."
            )
        work = model.data.copy()
    else:
        work = df.copy()

    if model.quantity_col not in work.columns:
        raise ValueError(
            f"DataFrame must include quantity column {model.quantity_col!r}."
        )

    pred = model.predict(work, hdi_prob=hdi_prob, random_seed=random_seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5) if style == "series" else (6, 6))

    y_obs = pred[model.quantity_col].to_numpy(dtype=float)
    y_mean = pred["quantity_mean"].to_numpy(dtype=float)
    y_lo = pred["quantity_hdi_lower"].to_numpy(dtype=float)
    y_hi = pred["quantity_hdi_upper"].to_numpy(dtype=float)

    if style == "calibration":
        lo = float(np.nanmin([y_obs.min(), y_mean.min()]))
        hi = float(np.nanmax([y_obs.max(), y_mean.max()]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            pad = 1.0 if hi == 0 else abs(hi) * 0.05
            lo, hi = lo - pad, hi + pad
        ax.scatter(
            y_mean,
            y_obs,
            alpha=0.55,
            s=22,
            edgecolors="none",
            color="C0",
            label="Observations",
        )
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Predicted quantity (posterior mean)")
        ax.set_ylabel(f"Observed {model.quantity_col}")
        ax.set_title(
            f"Calibration: observed vs predicted mean\n"
            f"({int(hdi_prob * 100)}% predictive HDI available in series view)"
        )
        ax.legend(loc="best")
        plt.tight_layout()
        return ax

    # series view
    sort_cols: list[str] = []
    if model.period_col in pred.columns:
        sort_cols.append(model.period_col)
    if model.sku_col in pred.columns:
        sort_cols.append(model.sku_col)
    if sort_cols:
        pred = pred.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        y_obs = pred[model.quantity_col].to_numpy(dtype=float)
        y_mean = pred["quantity_mean"].to_numpy(dtype=float)
        y_lo = pred["quantity_hdi_lower"].to_numpy(dtype=float)
        y_hi = pred["quantity_hdi_upper"].to_numpy(dtype=float)

    x = np.arange(len(pred))
    ax.fill_between(
        x,
        y_lo,
        y_hi,
        alpha=0.25,
        color="C0",
        label=f"{int(hdi_prob * 100)}% predictive HDI",
    )
    ax.plot(x, y_mean, color="C0", linewidth=1.5, label="Posterior mean")
    ax.scatter(
        x,
        y_obs,
        color="black",
        s=14,
        alpha=0.75,
        label="Observed",
        zorder=3,
    )
    xlab = "Observation index"
    if sort_cols:
        xlab += f" (sorted by {', '.join(sort_cols)})"
    ax.set_xlabel(xlab)
    ax.set_ylabel(model.quantity_col)
    ax.set_title("Posterior predictive check on panel (training-style data)")
    ax.legend(loc="best")
    plt.tight_layout()
    return ax
