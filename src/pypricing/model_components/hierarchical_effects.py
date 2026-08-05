"""Non-centered hierarchical effects: global mean + level random + SKU random."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pypricing.data import HierarchyIndex


def get_noncentered_hierarchical_effect_per_sku(
    basename: str,
    *,
    n_skus: int,
    hierarchy: HierarchyIndex,
    model_config: dict[str, Any] | None,
    mu_default_mu: float,
    mu_default_sigma: float,
    level_scale_default_sigma: float = 1.0,
    sku_scale_default_sigma: float = 1.0,
) -> Any:
    """
    Vector of length ``n_skus``: ``mu + sum_L sigma_L * eta_L[idx_L] + sigma_sku * eta_sku``.

    Priors are non-centered: ``eta`` are standard normal; ``sigma`` are HalfNormal.
    """
    from pypricing.model_components.priors import resolve_prior

    idx_mat = np.asarray(hierarchy.sku_to_level_idx, dtype=np.int64)
    n_groups_per_level = hierarchy.n_groups_per_level
    if idx_mat.shape != (n_skus, len(n_groups_per_level)):
        raise ValueError("sku_to_level_idx shape must be (n_skus, n_levels)")

    mu = resolve_prior(
        model_config=model_config,
        param_name=f"mu_{basename}",
        default_dist=pm.Normal,
        default_kwargs={"mu": mu_default_mu, "sigma": mu_default_sigma},
    )

    lin = pt.ones((n_skus,)) * mu

    for level, n_g in enumerate(n_groups_per_level):
        sigma = resolve_prior(
            model_config=model_config,
            param_name=f"sigma_{basename}_group_{level}",
            default_dist=pm.HalfNormal,
            default_kwargs={"sigma": level_scale_default_sigma},
        )
        eta = resolve_prior(
            model_config=model_config,
            param_name=f"eta_{basename}_group_{level}",
            default_dist=pm.Normal,
            default_kwargs={"mu": 0.0, "sigma": 1.0},
            dims=f"group_{level}",
        )
        idx = pt.constant(idx_mat[:, level])
        lin = lin + sigma * eta[idx]

    sigma_sku = resolve_prior(
        model_config=model_config,
        param_name=f"sigma_{basename}_sku",
        default_dist=pm.HalfNormal,
        default_kwargs={"sigma": sku_scale_default_sigma},
    )
    eta_sku = resolve_prior(
        model_config=model_config,
        param_name=f"eta_{basename}_sku",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": 1.0},
        dims="sku",
    )
    return lin + sigma_sku * eta_sku
