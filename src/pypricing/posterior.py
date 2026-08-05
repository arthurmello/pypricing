from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:  # pragma: no cover
    from pypricing.models.basic import DemandModel


def _require_log_log(model: "DemandModel") -> None:
    if getattr(model, "model_name", "log_log") != "log_log":
        raise NotImplementedError(
            "Quantity-change posterior from elasticity currently supports only "
            "LogLogDemandModel. For other shapes, compute from posterior predictive."
        )


def quantity_multiplier_posterior(
    model: "DemandModel",
    *,
    price_multiplier: float,
) -> xr.DataArray:
    """Posterior of ``price_multiplier ** elasticity`` (log-log only)."""
    model._require_fitted()
    assert model.idata is not None
    _require_log_log(model)

    elasticity = model.idata.posterior["elasticity_sku"]
    return float(price_multiplier) ** elasticity


def summarize_quantity_multiplier_by_sku(
    model: "DemandModel",
    *,
    price_multiplier: float,
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    """Per-SKU mean and HDI of the quantity multiplier posterior."""
    mult = quantity_multiplier_posterior(
        model=model,
        price_multiplier=price_multiplier,
    )
    mean = mult.mean(dim=("chain", "draw"))
    hdi = az.hdi(mult, hdi_prob=hdi_prob)
    hdi_mult = hdi[mult.name] if isinstance(hdi, xr.Dataset) else hdi

    out = pd.DataFrame(
        {
            "sku": np.asarray(model.sku_levels_),
            "quantity_multiplier_mean": mean.to_numpy(),
            "quantity_multiplier_hdi_lower": hdi_mult.sel(hdi="lower").to_numpy(),
            "quantity_multiplier_hdi_upper": hdi_mult.sel(hdi="higher").to_numpy(),
        }
    )
    return out.set_index("sku")


def summarize_quantity_multiplier_one_sku(
    model: "DemandModel",
    *,
    sku: Any,
    price_multiplier: float,
    hdi_prob: float = 0.9,
) -> dict[str, Any]:
    """Draws, mean, and HDI of the quantity multiplier for one SKU."""
    mult = quantity_multiplier_posterior(
        model=model,
        price_multiplier=price_multiplier,
    )

    assert model.sku_levels_ is not None
    sku_levels = model.sku_levels_
    if sku in sku_levels:
        sku_idx = int(np.where(np.asarray(sku_levels) == sku)[0][0])
    elif isinstance(sku, (int, np.integer)) and 0 <= int(sku) < len(sku_levels):
        sku_idx = int(sku)
    else:
        raise ValueError("Invalid `sku` for summarize_quantity_multiplier_one_sku.")

    draws = mult.sel(sku=sku_idx).values.flatten()
    hdi = az.hdi(draws, hdi_prob=hdi_prob)
    if isinstance(hdi, xr.DataArray):
        lower = float(hdi.sel(hdi="lower").values)
        upper = float(hdi.sel(hdi="higher").values)
    else:
        arr = np.asarray(hdi).reshape(-1)
        lower = float(arr[0])
        upper = float(arr[1])

    return {
        "quantity_multiplier_draws": draws,
        "quantity_multiplier_mean": float(np.mean(draws)),
        "quantity_multiplier_hdi": (lower, upper),
    }
