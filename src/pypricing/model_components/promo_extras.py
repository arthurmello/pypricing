"""Promo response beyond the log-linear discount term.

``PromoSpec`` adds terms to mean log-quantity, all in the depth index
``z = -log(1 - d)`` (``0`` at regular price, growing with the discount):

- ``depth="quadratic"``: ``promo_curvature_sku * z**2``, so the lift can
  saturate (negative curvature) or accelerate for deep discounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytensor.tensor as pt
import xarray as xr

from pypricing.data import PricePanelData
from pypricing.model_components.sku_effects import get_sku_effect

DepthKind = Literal["log_linear", "quadratic"]


@dataclass(frozen=True)
class PromoSpec:
    """Extra promo terms. Requires ``PanelColumns.discount_col``."""

    depth: DepthKind = "log_linear"

    def __post_init__(self) -> None:
        if self.depth not in ("log_linear", "quadratic"):
            raise ValueError(
                f"PromoSpec.depth must be 'log_linear' or 'quadratic'; got {self.depth!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {"depth": self.depth}

    @classmethod
    def from_json(cls, raw: dict[str, Any] | None) -> PromoSpec | None:
        if raw is None:
            return None
        return cls(**raw)


@dataclass
class ExtraFeatures:
    """Row-aligned design pieces for the extra terms; ``None`` means absent."""

    depth_sq: np.ndarray | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in vars(self).values())


def depth_index(log_discount_index: np.ndarray | None, n: int) -> np.ndarray:
    """``z = -log(1 - d)``; zeros without a discount column."""
    if log_discount_index is None:
        return np.zeros(n, dtype=np.float64)
    return -np.asarray(log_discount_index, dtype=np.float64)


def build_extra_features(
    *,
    spec: PromoSpec | None,
    log_discount_index: np.ndarray | None,
    n_obs: int,
) -> ExtraFeatures:
    feats = ExtraFeatures()
    if spec is None:
        return feats
    z = depth_index(log_discount_index, n_obs)
    if spec.depth == "quadratic":
        feats.depth_sq = z**2
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
    return mu
