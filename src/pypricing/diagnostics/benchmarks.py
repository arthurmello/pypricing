from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel

EXTREME_THRESHOLD = 0.99

# Bijmolt, van Heerde & Pieters (2005), Figure 2; n = 1,851
EDGES = np.array([-18.9, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 4])
COUNTS = np.array([15, 5, 9, 9, 14, 20, 32, 64, 145, 281, 468, 464, 284, 24, 15, 2])
CDF_AT_EDGES = np.concatenate([[0], np.cumsum(COUNTS)]) / COUNTS.sum()


def literature_cdf(elasticity):
    """Share of published estimates at or below this value (i.e. at least as elastic)."""
    return np.interp(elasticity, EDGES, CDF_AT_EDGES)


def share_less_elastic(elasticity):
    """Share of published estimates closer to zero than this value.

    >>> round(float(share_less_elastic(-3.7)), 3)
    0.785
    """
    return 1.0 - literature_cdf(elasticity)


def _flat_draws(model: DemandModel, var_name: str, dim: str) -> np.ndarray:
    draws = model.idata.posterior[var_name]
    return draws.stack(sample=("chain", "draw")).transpose(dim, "sample").values


def elasticity_benchmarks(
    model: DemandModel, *, extreme_threshold: float = EXTREME_THRESHOLD
) -> pd.DataFrame:
    """Own-price elasticity per SKU against published estimates.

    Warns when a SKU's posterior mean is more elastic than ``extreme_threshold``
    of the literature, or positive. For the quadratic and sigmoid models,
    ``elasticity_sku`` is the elasticity at each SKU's reference price.
    """
    model._require_fitted()
    flat = _flat_draws(model, "elasticity_sku", "sku")
    mean = flat.mean(axis=1)
    out = pd.DataFrame(
        {
            "elasticity_mean": mean,
            "prob_positive": (flat > 0).mean(axis=1),
            "share_less_elastic": share_less_elastic(mean),
        },
        index=pd.Index(model.sku_levels_, name=model.sku_col),
    )
    out["extreme"] = out["share_less_elastic"] >= extreme_threshold
    out["positive"] = out["elasticity_mean"] > 0
    _warn(out, extreme_threshold)
    return out


def cross_elasticity_summary(model: DemandModel) -> pd.DataFrame | None:
    """Posterior summary of ``gamma_pair`` per directed SKU pair.

    No sign warning: substitutes should be positive, complements negative,
    and the model can't tell which pairs are which. ``None`` when the model
    has no cross-price terms.
    """
    model._require_fitted()
    if model.cross_pairs_ is None or "gamma_pair" not in model.idata.posterior:
        return None
    flat = _flat_draws(model, "gamma_pair", "cross_pair")
    levels = np.asarray(model.sku_levels_)
    return pd.DataFrame(
        {
            "focal": levels[model.cross_pairs_.pair_from],
            "competitor": levels[model.cross_pairs_.pair_to],
            "gamma_mean": flat.mean(axis=1),
            "prob_positive": (flat > 0).mean(axis=1),
        }
    )


def check_benchmarks(
    model: DemandModel, *, extreme_threshold: float = EXTREME_THRESHOLD
) -> dict[str, pd.DataFrame | None]:
    return {
        "own": elasticity_benchmarks(model, extreme_threshold=extreme_threshold),
        "cross": cross_elasticity_summary(model),
    }


def _warn(out: pd.DataFrame, extreme_threshold: float) -> None:
    extreme = out.index[out["extreme"]].tolist()
    if extreme:
        warnings.warn(
            f"{len(extreme)} SKU(s) more elastic than {extreme_threshold:.0%} of "
            f"published estimates: {extreme}",
            stacklevel=3,
        )
    positive = out.index[out["positive"]].tolist()
    if positive:
        warnings.warn(
            f"{len(positive)} SKU(s) with positive own-price elasticity: {positive}",
            stacklevel=3,
        )
