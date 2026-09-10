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
SeasonComponent = Literal["yearly", "weekly"]
PanelFrequency = Literal["D", "W", "M", "Q", "irregular"]

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


def get_season_term(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    *,
    components: tuple[str, ...] | None,
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
        param_name="beta_season",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": _SEASON_PRIOR_SIGMA},
        shape=k,
    )
    return pt.dot(pt.constant(X), beta)


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
