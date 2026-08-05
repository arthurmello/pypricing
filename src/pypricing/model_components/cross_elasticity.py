"""Cross-price (cross-elasticity) pair sets and design matrices for panel demand models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pypricing.data import CrossPairIndex, HierarchyIndex

CROSS_ELASTICITY_ALL = "all"
CROSS_ELASTICITY_WITHIN_GROUP = "within_group"

_VALID_CROSS_MODES = frozenset({CROSS_ELASTICITY_ALL, CROSS_ELASTICITY_WITHIN_GROUP})


def parse_cross_elasticity_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    key = mode.strip().lower()
    if key not in _VALID_CROSS_MODES:
        allowed = ", ".join(sorted(_VALID_CROSS_MODES))
        raise ValueError(
            f"cross_elasticity {mode!r} is invalid; use None or one of: {allowed}"
        )
    return key


@dataclass
class CrossElasticitySpec:
    """Cross-price term configuration. Pass ``None`` on the model to disable."""

    mode: Literal["all", "within_group"] | str
    group_level: int | None = None

    def __post_init__(self) -> None:
        parsed = parse_cross_elasticity_mode(self.mode)
        if parsed is None:
            raise ValueError("CrossElasticitySpec.mode must be 'all' or 'within_group'")
        self.mode = parsed


def _cross_pair_index(
    pair_from: list[int],
    pair_to: list[int],
    pair_pool: list[int],
    n_pool: int,
) -> CrossPairIndex:
    return CrossPairIndex(
        pair_from=np.asarray(pair_from, dtype=np.int64),
        pair_to=np.asarray(pair_to, dtype=np.int64),
        pair_pool=np.asarray(pair_pool, dtype=np.int64),
        n_pool=int(n_pool),
    )


def enumerate_cross_pairs(
    *,
    mode: str,
    n_skus: int,
    hierarchy: HierarchyIndex | None,
    cross_elasticity_group_level: int | None,
) -> CrossPairIndex:
    """
    Directed pairs (i, j), i != j, with pool index for hierarchical priors.

    Pooling: ``all`` without hierarchy uses a single pool. ``all`` with hierarchy
    pools by focal SKU's coarsest group (``group_0``). ``within_group`` pools by
    the chosen hierarchy level (same group for i and j).
    """
    if n_skus < 2:
        raise ValueError("cross_elasticity requires at least two SKUs")

    sku_to_level_idx = None if hierarchy is None else hierarchy.sku_to_level_idx
    n_groups_per_level = None if hierarchy is None else hierarchy.n_groups_per_level

    if mode == CROSS_ELASTICITY_ALL:
        pool_level = 0
        if sku_to_level_idx is None or n_groups_per_level is None:
            pair_from: list[int] = []
            pair_to: list[int] = []
            pair_pool: list[int] = []
            for i in range(n_skus):
                for j in range(n_skus):
                    if i == j:
                        continue
                    pair_from.append(i)
                    pair_to.append(j)
                    pair_pool.append(0)
            return _cross_pair_index(pair_from, pair_to, pair_pool, n_pool=1)
        n_cross_pool = int(n_groups_per_level[pool_level])
        pair_from = []
        pair_to = []
        pair_pool = []
        for i in range(n_skus):
            for j in range(n_skus):
                if i == j:
                    continue
                pair_from.append(i)
                pair_to.append(j)
                pair_pool.append(int(sku_to_level_idx[i, pool_level]))
        return _cross_pair_index(pair_from, pair_to, pair_pool, n_pool=n_cross_pool)

    # within_group
    if sku_to_level_idx is None or n_groups_per_level is None:
        raise ValueError(
            "cross_elasticity='within_group' requires group_columns (hierarchy) in the data"
        )
    if cross_elasticity_group_level is None:
        raise ValueError("cross_elasticity='within_group' requires group_level")
    level = int(cross_elasticity_group_level)
    if level < 0 or level >= len(n_groups_per_level):
        raise ValueError("group_level must satisfy 0 <= level < len(group_columns)")
    n_cross_pool = int(n_groups_per_level[level])
    pair_from = []
    pair_to = []
    pair_pool = []
    for i in range(n_skus):
        for j in range(n_skus):
            if i == j:
                continue
            if int(sku_to_level_idx[i, level]) != int(sku_to_level_idx[j, level]):
                continue
            pair_from.append(i)
            pair_to.append(j)
            pair_pool.append(int(sku_to_level_idx[i, level]))
    if not pair_from:
        raise ValueError(
            "no cross-price pairs for within_group mode; need at least two SKUs "
            "in the same group at the chosen level"
        )
    return _cross_pair_index(pair_from, pair_to, pair_pool, n_pool=n_cross_pool)


def build_gamma_pair(
    model_config: dict[str, Any] | None,
    *,
    pair_pool_idx: np.ndarray,
) -> Any:
    """Non-centered hierarchical ``gamma_pair`` (requires active ``pm.Model`` context)."""
    from pypricing.model_components.priors import resolve_prior

    import pymc as pm
    import pytensor.tensor as pt

    pair_pool_idx_t = pt.constant(np.asarray(pair_pool_idx, dtype=np.int64))

    mu_cross = resolve_prior(
        model_config=model_config,
        param_name="mu_cross_pool",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": 0.1},
        dims="cross_pool",
    )
    sigma_cross = resolve_prior(
        model_config=model_config,
        param_name="sigma_cross_pool",
        default_dist=pm.HalfNormal,
        default_kwargs={"sigma": 0.1},
        dims="cross_pool",
    )
    eta_cross = resolve_prior(
        model_config=model_config,
        param_name="eta_cross_pair",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": 1.0},
        dims="cross_pair",
    )
    gamma = pm.Deterministic(
        "gamma_pair",
        mu_cross[pair_pool_idx_t] + sigma_cross[pair_pool_idx_t] * eta_cross,
        dims="cross_pair",
    )
    return gamma


def validate_cross_elasticity_config(
    *,
    cross_elasticity: CrossElasticitySpec | None,
    group_columns: tuple[str, ...] | None,
) -> None:
    if cross_elasticity is None:
        return
    if cross_elasticity.mode == CROSS_ELASTICITY_ALL:
        if cross_elasticity.group_level is not None:
            raise ValueError(
                "group_level must be None when cross_elasticity mode is 'all'"
            )
    elif cross_elasticity.mode == CROSS_ELASTICITY_WITHIN_GROUP:
        if group_columns is None or len(group_columns) == 0:
            raise ValueError("cross_elasticity='within_group' requires group_columns")
        if cross_elasticity.group_level is None:
            raise ValueError("cross_elasticity='within_group' requires group_level")
        if not (0 <= cross_elasticity.group_level < len(group_columns)):
            raise ValueError("group_level must satisfy 0 <= level < len(group_columns)")
