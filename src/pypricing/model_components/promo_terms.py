"""Log-linear temporary-discount term, shared by every demand curve."""

from __future__ import annotations

from typing import Any

import pytensor.tensor as pt

from pypricing.data import PricePanelData
from pypricing.model_components.priors import (
    DEFAULT_PROMO_ELASTICITY_MU,
    DEFAULT_PROMO_ELASTICITY_SIGMA,
)
from pypricing.model_components.sku_effects import get_sku_effect


def get_promo_term(
    model_config: dict[str, Any] | None,
    data: PricePanelData,
) -> Any:
    """``promo_elasticity_sku * log(1 - d)``; ``0.0`` when there is no discount column.

    Discounts sit in a narrow band, so the promo response is log-linear on every
    curve. ``promo_elasticity_sku`` pools through ``group_columns`` the same way
    ``elasticity_sku`` does.
    """
    if data.log_discount_index is None:
        return 0.0

    obs_sku = pt.constant(data.obs_sku_idx)
    promo_elasticity_sku = get_sku_effect(
        "promo_elasticity",
        data,
        model_config,
        mu_default_mu=DEFAULT_PROMO_ELASTICITY_MU,
        mu_default_sigma=DEFAULT_PROMO_ELASTICITY_SIGMA,
    )
    return promo_elasticity_sku[obs_sku] * pt.constant(data.log_discount_index)
