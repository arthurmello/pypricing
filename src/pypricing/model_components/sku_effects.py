"""Per-SKU effects: hierarchical or flat, selected from panel structure."""

from __future__ import annotations

from typing import Any

import pymc as pm

from pypricing.data import PricePanelData
from pypricing.model_components.hierarchical_effects import (
    get_noncentered_hierarchical_effect_per_sku,
)
from pypricing.model_components.priors import resolve_prior


def get_sku_effect(
    basename: str,
    data: PricePanelData,
    model_config: dict[str, Any] | None,
    *,
    mu_default_mu: float,
    mu_default_sigma: float,
    level_scale_default_sigma: float = 1.0,
    sku_scale_default_sigma: float = 1.0,
) -> Any:
    """
    Build ``{basename}_sku`` inside an active ``pm.Model``.

    Uses non-centered hierarchy when the panel has group indices; otherwise a
    flat Normal prior per SKU.
    """
    param_name = f"{basename}_sku"
    if data.hierarchy is not None:
        lin = get_noncentered_hierarchical_effect_per_sku(
            basename,
            n_skus=data.n_skus,
            hierarchy=data.hierarchy,
            model_config=model_config,
            mu_default_mu=mu_default_mu,
            mu_default_sigma=mu_default_sigma,
            level_scale_default_sigma=level_scale_default_sigma,
            sku_scale_default_sigma=sku_scale_default_sigma,
        )
        return pm.Deterministic(param_name, lin, dims="sku")

    return resolve_prior(
        model_config=model_config,
        param_name=param_name,
        default_dist=pm.Normal,
        default_kwargs={"mu": mu_default_mu, "sigma": mu_default_sigma},
        dims="sku",
    )
