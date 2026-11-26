"""Promo response beyond the log-linear discount term.

``PromoSpec`` adds terms to mean log-quantity, all in the depth index
``z = -log(1 - d)`` (``0`` at regular price, growing with the discount):

- ``depth="quadratic"``: ``promo_curvature_sku * z**2``, so the lift can
  saturate (negative curvature) or accelerate for deep discounts.
- ``type_col``: one additive ``promo_type_effect`` per promo mechanic (for
  example ``"featured"`` or ``"bundle"``), shared across SKUs. Rows at regular
  price, or with a missing / ``"none"`` type, are the reference.
- ``lags=L``: ``promo_lag[k] * z[t-k]`` for ``k = 1..L``, the same SKU's depth
  ``k`` rows earlier (by ``period_col``, within region). A negative value is a
  post-promo dip (pull-forward / stockpiling). Rows without history use 0, so
  predict on frames that include the preceding periods when lags matter.

Separately, ``varying_controls`` on the model gives the named control columns
a per-SKU slope: the shared ``beta_control`` plus a zero-mean deviation pooled
through ``group_columns`` (``beta_control_dev_sku``). Use it for competitor
prices whose effect differs by category or brand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from pypricing.data import PanelColumns, PricePanelData
from pypricing.model_components.priors import resolve_prior
from pypricing.model_components.sku_effects import get_sku_effect

DepthKind = Literal["log_linear", "quadratic"]
NO_PROMO_TYPE = "none"


@dataclass(frozen=True)
class PromoSpec:
    """Extra promo terms. Requires ``PanelColumns.discount_col``."""

    depth: DepthKind = "log_linear"
    type_col: str | None = None
    lags: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.lags, int) or self.lags < 0:
            raise ValueError(
                f"PromoSpec.lags must be a non-negative int; got {self.lags!r}"
            )
        if self.depth not in ("log_linear", "quadratic"):
            raise ValueError(
                f"PromoSpec.depth must be 'log_linear' or 'quadratic'; got {self.depth!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {"depth": self.depth, "type_col": self.type_col, "lags": self.lags}

    @classmethod
    def from_json(cls, raw: dict[str, Any] | None) -> PromoSpec | None:
        if raw is None:
            return None
        return cls(**raw)


@dataclass
class ExtraFeatures:
    """Row-aligned design pieces for the extra terms; ``None`` means absent."""

    depth_sq: np.ndarray | None = None
    type_onehot: np.ndarray | None = None
    type_levels: tuple[str, ...] = ()
    lag_matrix: np.ndarray | None = None
    varying_X: np.ndarray | None = None
    varying_names: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return (
            self.depth_sq is None
            and self.type_onehot is None
            and self.lag_matrix is None
            and self.varying_X is None
        )


def depth_index(log_discount_index: np.ndarray | None, n: int) -> np.ndarray:
    """``z = -log(1 - d)``; zeros without a discount column."""
    if log_discount_index is None:
        return np.zeros(n, dtype=np.float64)
    return -np.asarray(log_discount_index, dtype=np.float64)


def promo_type_values(df: pd.DataFrame, type_col: str, z: np.ndarray) -> np.ndarray:
    """Promo type per row as strings; regular-price rows are ``"none"``."""
    if type_col not in df.columns:
        raise ValueError(f"Missing promo type column: {type_col!r}")
    raw = df[type_col]
    values = (
        raw.astype("string").fillna(NO_PROMO_TYPE).str.strip().to_numpy(dtype=object)
    )
    values[(values == "") | (z <= 1e-12)] = NO_PROMO_TYPE
    return values


def lagged_depth(
    df: pd.DataFrame, panel_columns: PanelColumns, z: np.ndarray, lags: int
) -> np.ndarray:
    """``(n_obs, lags)``: depth of the same SKU (and region) 1..lags rows earlier."""
    period_col = panel_columns.period_col
    if period_col not in df.columns:
        raise ValueError(f"PromoSpec.lags requires column {period_col!r}")
    keys = [panel_columns.sku_col]
    if panel_columns.region_col is not None and panel_columns.region_col in df.columns:
        keys.append(panel_columns.region_col)
    tmp = pd.DataFrame({f"k{i}": df[k].to_numpy() for i, k in enumerate(keys)})
    tmp["_p"] = df[period_col].to_numpy()
    tmp["_z"] = z
    tmp["_row"] = np.arange(len(df))
    key_cols = [f"k{i}" for i in range(len(keys))]
    tmp = tmp.sort_values([*key_cols, "_p"], kind="stable")
    out = np.zeros((len(df), lags), dtype=np.float64)
    grouped = tmp.groupby(key_cols, sort=False)["_z"]
    rows = tmp["_row"].to_numpy()
    for k in range(1, lags + 1):
        out[rows, k - 1] = grouped.shift(k).fillna(0.0).to_numpy()
    return out


def build_extra_features(
    *,
    spec: PromoSpec | None,
    log_discount_index: np.ndarray | None,
    n_obs: int,
    df: pd.DataFrame | None = None,
    type_levels: tuple[str, ...] | None = None,
    panel_columns: PanelColumns | None = None,
    X_control: np.ndarray | None = None,
    varying_idx: tuple[int, ...] = (),
    varying_names: tuple[str, ...] = (),
) -> ExtraFeatures:
    """Features for ``spec`` and varying controls.

    ``type_levels=None`` learns promo types from ``df``. ``varying_idx`` are
    column positions of the varying controls in ``X_control``.
    """
    feats = ExtraFeatures()
    if varying_idx:
        if X_control is None:
            raise ValueError("varying controls need the control matrix")
        feats.varying_X = np.asarray(X_control, dtype=np.float64)[:, list(varying_idx)]
        feats.varying_names = tuple(varying_names)
    if spec is None:
        return feats
    z = depth_index(log_discount_index, n_obs)
    if spec.depth == "quadratic":
        feats.depth_sq = z**2
    if spec.type_col is not None:
        if df is None:
            raise ValueError("promo type features need the data frame")
        values = promo_type_values(df, spec.type_col, z)
        if type_levels is None:
            type_levels = tuple(sorted({str(v) for v in values if v != NO_PROMO_TYPE}))
        unknown = sorted({str(v) for v in values} - set(type_levels) - {NO_PROMO_TYPE})
        if unknown:
            raise ValueError(
                f"Unknown promo types {unknown}; fitted on {list(type_levels)}"
            )
        onehot = np.zeros((n_obs, len(type_levels)), dtype=np.float64)
        for k, level in enumerate(type_levels):
            onehot[:, k] = values == level
        feats.type_onehot = onehot
        feats.type_levels = tuple(type_levels)
    if spec.lags:
        if df is None or panel_columns is None:
            raise ValueError("lag features need the data frame and panel columns")
        feats.lag_matrix = lagged_depth(df, panel_columns, z, spec.lags)
    return feats


def extra_mu_term(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    feats: ExtraFeatures | None,
) -> Any:
    """PyMC term for the extras; call inside an active ``pm.Model``."""
    if feats is None or feats.is_empty():
        return 0.0
    obs_sku = pt.constant(data.obs_sku_idx)
    term: Any = 0.0
    if feats.depth_sq is not None:
        curvature = get_sku_effect(
            "promo_curvature",
            data,
            model_config,
            mu_default_mu=0.0,
            mu_default_sigma=1.0,
            level_scale_default_sigma=0.3,
            sku_scale_default_sigma=0.3,
        )
        term = term + curvature[obs_sku] * pt.constant(feats.depth_sq)
    if feats.type_onehot is not None and feats.type_levels:
        pm.modelcontext(None).add_coord("promo_type", list(feats.type_levels))
        effect = resolve_prior(
            model_config=model_config,
            param_name="promo_type_effect",
            default_dist=pm.Normal,
            default_kwargs={"mu": 0.0, "sigma": 0.5},
            dims="promo_type",
        )
        term = term + pt.dot(pt.constant(feats.type_onehot), effect)
    if feats.varying_X is not None:
        dev = _varying_slopes(model_config, data, feats.varying_names)
        term = term + pt.sum(dev[obs_sku] * pt.constant(feats.varying_X), axis=1)
    if feats.lag_matrix is not None:
        pm.modelcontext(None).add_coord(
            "lag", list(range(1, feats.lag_matrix.shape[1] + 1))
        )
        lag = resolve_prior(
            model_config=model_config,
            param_name="promo_lag",
            default_dist=pm.Normal,
            default_kwargs={"mu": 0.0, "sigma": 0.5},
            dims="lag",
        )
        term = term + pt.dot(pt.constant(feats.lag_matrix), lag)
    return term


def _varying_slopes(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
    names: tuple[str, ...],
) -> Any:
    """``(n_skus, n_varying)`` zero-mean slope deviations, non-centered."""
    model = pm.modelcontext(None)
    model.add_coord("varying_control", list(names))
    n_v = len(names)
    lin: Any = pt.zeros((data.n_skus, n_v))
    if data.hierarchy is not None:
        idx_mat = np.asarray(data.hierarchy.sku_to_level_idx, dtype=np.int64)
        for level in range(len(data.hierarchy.n_groups_per_level)):
            sigma = resolve_prior(
                model_config=model_config,
                param_name=f"sigma_control_dev_group_{level}",
                default_dist=pm.HalfNormal,
                default_kwargs={"sigma": 0.3},
                dims="varying_control",
            )
            eta = resolve_prior(
                model_config=model_config,
                param_name=f"eta_control_dev_group_{level}",
                default_dist=pm.Normal,
                default_kwargs={"mu": 0.0, "sigma": 1.0},
                dims=(f"group_{level}", "varying_control"),
            )
            lin = lin + sigma[None, :] * eta[pt.constant(idx_mat[:, level])]
    sigma_sku = resolve_prior(
        model_config=model_config,
        param_name="sigma_control_dev_sku",
        default_dist=pm.HalfNormal,
        default_kwargs={"sigma": 0.1},
        dims="varying_control",
    )
    eta_sku = resolve_prior(
        model_config=model_config,
        param_name="eta_control_dev_sku",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": 1.0},
        dims=("sku", "varying_control"),
    )
    lin = lin + sigma_sku[None, :] * eta_sku
    return pm.Deterministic(
        "beta_control_dev_sku", lin, dims=("sku", "varying_control")
    )


def add_extra(
    mu: np.ndarray,
    posterior: xr.Dataset,
    feats: ExtraFeatures | None,
    obs_sku_idx: np.ndarray,
) -> np.ndarray:
    """Numpy counterpart of :func:`extra_mu_term` on posterior draws."""
    if feats is None:
        return mu
    if feats.depth_sq is not None and "promo_curvature_sku" in posterior:
        curvature = posterior["promo_curvature_sku"].values
        mu = mu + curvature[:, :, obs_sku_idx] * feats.depth_sq[None, None, :]
    if feats.type_onehot is not None and "promo_type_effect" in posterior:
        effect = posterior["promo_type_effect"].values
        mu = mu + np.einsum("cdk,ok->cdo", effect, feats.type_onehot, optimize=True)
    if feats.varying_X is not None and "beta_control_dev_sku" in posterior:
        dev = posterior["beta_control_dev_sku"].values[:, :, obs_sku_idx, :]
        mu = mu + np.einsum("cdov,ov->cdo", dev, feats.varying_X, optimize=True)
    if feats.lag_matrix is not None and "promo_lag" in posterior:
        lag = posterior["promo_lag"].values
        mu = mu + np.einsum("cdk,ok->cdo", lag, feats.lag_matrix, optimize=True)
    return mu
