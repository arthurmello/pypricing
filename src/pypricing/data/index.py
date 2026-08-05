"""Panel index types: hierarchy and directed cross-price pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HierarchyIndex:
    """Derived SKU→group indices when ``group_columns`` are set on the panel."""

    group_columns: tuple[str, ...]
    sku_to_level_idx: np.ndarray
    """Shape ``(n_skus, n_levels)``: group index for each SKU at each hierarchy level."""
    n_groups_per_level: tuple[int, ...]

    @property
    def n_levels(self) -> int:
        return len(self.group_columns)


@dataclass
class CrossPairIndex:
    """Directed cross-price pairs and pooling indices (post-build / runtime)."""

    pair_from: np.ndarray
    pair_to: np.ndarray
    pair_pool: np.ndarray
    n_pool: int

    @property
    def n_pairs(self) -> int:
        return int(len(self.pair_from))
