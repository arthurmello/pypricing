"""Promo response beyond the log-linear discount term.

``PromoSpec`` adds terms to mean log-quantity, all in the depth index
``z = -log(1 - d)`` (``0`` at regular price, growing with the discount):

- ``depth="quadratic"``: ``promo_curvature_sku * z**2``, so the lift can
  saturate (negative curvature) or accelerate for deep discounts.
- ``type_col``: one additive ``promo_type_effect`` per promo mechanic (for
  example ``"featured"`` or ``"bundle"``), shared across SKUs. Rows at regular
  price, or with a missing / ``"none"`` type, are the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from pypricing.data import PricePanelData
from pypricing.model_components.priors import resolve_prior
from pypricing.model_components.sku_effects import get_sku_effect

DepthKind = Literal["log_linear", "quadratic"]
NO_PROMO_TYPE = "none"


@dataclass(frozen=True)
class PromoSpec:
    """Extra promo terms. Requires ``PanelColumns.discount_col``."""

    depth: DepthKind = "log_linear"
    type_col: str | None = None

    def __post_init__(self) -> None:
        if self.depth not in ("log_linear", "quadratic"):
            raise ValueError(
                f"PromoSpec.depth must be 'log_linear' or 'quadratic'; got {self.depth!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {"depth": self.depth, "type_col": self.type_col}

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

    def is_empty(self) -> bool:
        return self.depth_sq is None and self.type_onehot is None


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


def build_extra_features(
    *,
    spec: PromoSpec | None,
    log_discount_index: np.ndarray | None,
    n_obs: int,
    df: pd.DataFrame | None = None,
    type_levels: tuple[str, ...] | None = None,
) -> ExtraFeatures:
    """Features for ``spec``. ``type_levels=None`` learns them from ``df``."""
    feats = ExtraFeatures()
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
    return term


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
    return mu
