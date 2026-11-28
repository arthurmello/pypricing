"""Time-varying terms for mean log-quantity (trend and seasonality)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pypricing.data import PricePanelData, period_to_t_years
from pypricing.model_components.hierarchical_effects import (
    get_noncentered_hierarchical_effect_per_sku,
)
from pypricing.model_components.priors import resolve_prior

TrendKind = Literal["sku", "shared"]
TimeEffectsKind = Literal["period", "region_period", "group_period"]
SeasonComponent = Literal["yearly", "weekly"]
PanelFrequency = Literal["D", "W", "M", "Q", "irregular"]
_PERIOD_EFFECT_SIGMA = 0.5
_REGION_PERIOD_EFFECT_SIGMA = 0.25

_TREND_PRIOR_SIGMA = 0.05
_SEASON_PRIOR_SIGMA = 0.5
_YEARLY_HARMONICS = 2
_WEEKLY_HARMONICS = 1
_SEASON_COMPONENTS = frozenset({"yearly", "weekly"})


def normalize_seasonality(
    value: str | Sequence[str] | None,
) -> Literal["auto"] | tuple[str, ...] | None:
    """None | ``\"auto\"`` | unique ``(\"yearly\", \"weekly\")`` components."""
    if value is None:
        return None
    if isinstance(value, str):
        if value == "auto":
            return "auto"
        value = (value,)
    if isinstance(value, Sequence):
        if len(value) == 0:
            raise ValueError("seasonality must not be an empty list")
        if "auto" in value:
            raise ValueError("'auto' cannot be mixed with seasonality components")
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            if item not in _SEASON_COMPONENTS:
                raise ValueError(
                    "seasonality components must be 'yearly' or 'weekly'; "
                    f"got {item!r}"
                )
            if item not in seen:
                out.append(item)
                seen.add(item)
        return tuple(out)
    raise ValueError(
        "seasonality must be None, 'auto', 'yearly', 'weekly', or a sequence of those"
    )


def infer_panel_frequency(dates: pd.DatetimeIndex) -> PanelFrequency:
    """Classify unique timestamps by median spacing (calendar days)."""
    unique = pd.DatetimeIndex(np.sort(pd.unique(pd.DatetimeIndex(dates))))
    if len(unique) < 2:
        return "irregular"
    delta_days = (unique[1:] - unique[:-1]).days.to_numpy(dtype=float)
    med = float(np.median(delta_days))
    if 0.5 <= med < 2.5:
        return "D"
    if 5.0 <= med <= 10.0:
        return "W"
    if 25.0 <= med <= 40.0:
        return "M"
    if 80.0 <= med <= 110.0:
        return "Q"
    return "irregular"


def seasonality_components_for_frequency(
    freq: PanelFrequency,
    dates: pd.DatetimeIndex,
) -> tuple[str, ...]:
    if freq == "D":
        return ("yearly", "weekly")
    if freq in ("W", "M", "Q"):
        return ("yearly",)
    n_weekdays = int(pd.DatetimeIndex(np.unique(dates)).dayofweek.nunique())
    if n_weekdays >= 4:
        return ("yearly", "weekly")
    return ("yearly",)


def fourier_features(
    period_index: pd.DatetimeIndex,
    components: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Sin/cos features at ``period_index``. Yearly uses day-of-year; weekly day-of-week."""
    idx = pd.DatetimeIndex(period_index)
    n = len(idx)
    cols: list[np.ndarray] = []
    names: list[str] = []
    if "yearly" in components:
        doy = idx.dayofyear.to_numpy(dtype=np.float64)
        for k in range(1, _YEARLY_HARMONICS + 1):
            phase = 2.0 * np.pi * k * (doy - 1.0) / 365.25
            cols.append(np.sin(phase))
            cols.append(np.cos(phase))
            names.append(f"yearly_sin_{k}")
            names.append(f"yearly_cos_{k}")
    if "weekly" in components:
        dow = idx.dayofweek.to_numpy(dtype=np.float64)
        for k in range(1, _WEEKLY_HARMONICS + 1):
            phase = 2.0 * np.pi * k * dow / 7.0
            cols.append(np.sin(phase))
            cols.append(np.cos(phase))
            names.append(f"weekly_sin_{k}")
            names.append(f"weekly_cos_{k}")
    if not cols:
        return np.empty((n, 0), dtype=np.float64), ()
    return np.column_stack(cols).astype(np.float64), tuple(names)


def get_period_effect_term(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    kind: TimeEffectsKind | None,
    *,
    param_suffix: str = "",
) -> Any:
    """Additive period effect in log-quantity (or in the price equation).

    ``kind="period"`` is one zero-sum λ per period, shared by every SKU.
    ``kind="region_period"`` adds a zero-sum region-by-period deviation on top
    of that national effect. ``kind="group_period"`` adds a zero-sum
    deviation per period for each group of the first ``group_columns`` level
    (e.g. category × week), absorbing demand shocks a retailer may time its
    promotions around. ``param_suffix`` (for example ``"_price"``) gives
    the first stage its own period effects, so the control-function residual is
    price variation left after conditioning on the period.
    """
    if kind is None:
        return 0.0
    if data.obs_period_idx is None or data.period_levels is None:
        raise ValueError("time_effects requires a period column")

    sigma_name = f"sigma_lambda_period{param_suffix}"
    lambda_name = f"lambda_period{param_suffix}"
    sigma = resolve_prior(
        model_config=model_config,
        param_name=sigma_name,
        default_dist=pm.HalfNormal,
        default_kwargs={"sigma": _PERIOD_EFFECT_SIGMA},
    )
    lam = pm.ZeroSumNormal(lambda_name, sigma=sigma, dims="period")
    term = lam[pt.constant(data.obs_period_idx)]
    if kind == "period":
        return term
    if kind == "group_period":
        if data.hierarchy is None:
            raise ValueError("time_effects='group_period' requires group_columns")
        group_idx = np.asarray(data.hierarchy.sku_to_level_idx, dtype=np.int64)[
            np.asarray(data.obs_sku_idx), 0
        ]
        sigma_group = resolve_prior(
            model_config=model_config,
            param_name=f"sigma_lambda_group_period{param_suffix}",
            default_dist=pm.HalfNormal,
            default_kwargs={"sigma": _REGION_PERIOD_EFFECT_SIGMA},
        )
        lam_group = pm.ZeroSumNormal(
            f"lambda_group_period{param_suffix}",
            sigma=sigma_group,
            n_zerosum_axes=2,
            dims=("group_0", "period"),
        )
        return term + lam_group[pt.constant(group_idx), pt.constant(data.obs_period_idx)]
    if kind != "region_period":
        raise ValueError(
            "time_effects must be 'period', 'region_period' or 'group_period'; "
            f"got {kind!r}"
        )
    if data.obs_region_idx is None or data.region_levels is None:
        raise ValueError("time_effects='region_period' requires a region column")

    sigma_region = resolve_prior(
        model_config=model_config,
        param_name=f"sigma_lambda_region_period{param_suffix}",
        default_dist=pm.HalfNormal,
        default_kwargs={"sigma": _REGION_PERIOD_EFFECT_SIGMA},
    )
    lam_region = pm.ZeroSumNormal(
        f"lambda_region_period{param_suffix}",
        sigma=sigma_region,
        n_zerosum_axes=2,
        dims=("region", "period"),
    )
    return term + lam_region[
        pt.constant(data.obs_region_idx), pt.constant(data.obs_period_idx)
    ]


def get_trend_term(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    *,
    trend: TrendKind | None,
    t0: pd.Timestamp | None,
    param_suffix: str = "",
) -> Any:
    """Additive ``trend * t`` in log-quantity; ``t`` is years since ``t0``."""
    if trend is None:
        return 0.0
    if data.period_index is None:
        raise ValueError("trend requires a datetime period_index on the panel")
    if t0 is None:
        raise ValueError("trend requires t0 (minimum training period)")

    t = pt.constant(period_to_t_years(data.period_index, t0))
    basename = f"trend{param_suffix}"
    if trend == "shared":
        mu_trend = resolve_prior(
            model_config=model_config,
            param_name=f"mu_{basename}",
            default_dist=pm.Normal,
            default_kwargs={"mu": 0.0, "sigma": _TREND_PRIOR_SIGMA},
        )
        return mu_trend * t

    trend_sku = _trend_sku_effect(model_config, data, basename=basename)
    obs_sku = pt.constant(data.obs_sku_idx)
    return trend_sku[obs_sku] * t


def get_season_term(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    *,
    components: tuple[str, ...] | None,
    param_name: str = "beta_season",
) -> Any:
    """Shared Fourier seasonality: ``X_season @ beta_season``."""
    if not components:
        return 0.0
    if data.period_index is None:
        raise ValueError("seasonality requires a datetime period_index on the panel")
    X, _names = fourier_features(data.period_index, components)
    k = X.shape[1]
    if k == 0:
        return 0.0
    beta = resolve_prior(
        model_config=model_config,
        param_name=param_name,
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": _SEASON_PRIOR_SIGMA},
        shape=k,
    )
    return pt.dot(pt.constant(X), beta)


def _trend_sku_effect(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    *,
    basename: str = "trend",
) -> Any:
    sku_name = f"{basename}_sku"
    if data.hierarchy is not None:
        lin = get_noncentered_hierarchical_effect_per_sku(
            basename,
            n_skus=data.n_skus,
            hierarchy=data.hierarchy,
            model_config=model_config,
            mu_default_mu=0.0,
            mu_default_sigma=_TREND_PRIOR_SIGMA,
            level_scale_default_sigma=_TREND_PRIOR_SIGMA,
            sku_scale_default_sigma=_TREND_PRIOR_SIGMA,
        )
        return pm.Deterministic(sku_name, lin, dims="sku")

    mu_trend = resolve_prior(
        model_config=model_config,
        param_name=f"mu_{basename}",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": _TREND_PRIOR_SIGMA},
    )
    sigma_sku = resolve_prior(
        model_config=model_config,
        param_name=f"sigma_{basename}_sku",
        default_dist=pm.HalfNormal,
        default_kwargs={"sigma": _TREND_PRIOR_SIGMA},
    )
    eta_sku = resolve_prior(
        model_config=model_config,
        param_name=f"eta_{basename}_sku",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": 1.0},
        dims="sku",
    )
    return pm.Deterministic(sku_name, mu_trend + sigma_sku * eta_sku, dims="sku")
