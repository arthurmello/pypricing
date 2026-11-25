from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypricing.arviz_compat import hdi_bounds

if TYPE_CHECKING:  # pragma: no cover
    from pypricing.models.basic import DemandModel


def _require_log_log(model: "DemandModel") -> None:
    if getattr(model, "model_name", "log_log") != "log_log":
        raise NotImplementedError(
            "Quantity-change posterior from elasticity currently supports only "
            "LogLogDemandModel. For other shapes, compute from posterior predictive."
        )


def elasticities(model: "DemandModel", *, hdi_prob: float = 0.9) -> pd.DataFrame:
    """Posterior mean and HDI of own-price elasticity, indexed by SKU label."""
    model._require_fitted()
    assert model.idata is not None
    draws = model.idata.posterior["elasticity_sku"]
    mean = draws.mean(dim=("chain", "draw"))
    lower, upper = hdi_bounds(draws, hdi_prob)
    return pd.DataFrame(
        {
            "mean": np.asarray(mean),
            "hdi_lower": np.asarray(lower),
            "hdi_upper": np.asarray(upper),
        },
        index=_sku_index(model, mean.coords["sku"].values),
    )


def _sku_index(model: "DemandModel", coord_values: Any) -> pd.Index:
    values = np.asarray(coord_values)
    levels = model.sku_levels_
    if (
        levels is not None
        and len(levels) == len(values)
        and np.array_equal(values, np.arange(len(values)))
    ):
        return pd.Index(levels, name=model.sku_col)
    return pd.Index(values, name=model.sku_col)


def get_elasticity_means(model: "DemandModel") -> pd.Series:
    """Posterior mean own-price elasticity per SKU."""
    model._require_fitted()
    draws = model.idata.posterior["elasticity_sku"]
    return pd.Series(
        draws.mean(dim=("chain", "draw")).to_numpy(),
        index=pd.Index(model.sku_levels_, name=model.sku_col),
    )


def quantity_multiplier_posterior(
    model: "DemandModel",
    *,
    price_multiplier: float,
) -> xr.DataArray:
    """Posterior of ``price_multiplier ** elasticity`` (log-log only).

    ``elasticity_sku`` is the regular-price elasticity, so the multiplier
    scales the regular price and holds the discount fraction fixed.
    """
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
    lower, upper = hdi_bounds(mult, hdi_prob)

    out = pd.DataFrame(
        {
            "sku": np.asarray(model.sku_levels_),
            "quantity_multiplier_mean": mean.to_numpy(),
            "quantity_multiplier_hdi_lower": np.asarray(lower),
            "quantity_multiplier_hdi_upper": np.asarray(upper),
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

    label = sku_levels[sku_idx]
    if label in mult.coords["sku"]:
        draws = mult.sel(sku=label).values.flatten()
    else:
        draws = mult.isel(sku=sku_idx).values.flatten()
    lower, upper = hdi_bounds(draws, hdi_prob)
    lower = float(np.asarray(lower))
    upper = float(np.asarray(upper))

    return {
        "quantity_multiplier_draws": draws,
        "quantity_multiplier_mean": float(np.mean(draws)),
        "quantity_multiplier_hdi": (lower, upper),
    }
