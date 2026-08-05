"""Helpers for statistical recovery checks."""

from __future__ import annotations

import numpy as np
import xarray as xr


def posterior_hdi_by_sku(
    posterior: xr.Dataset,
    var_name: str,
    *,
    hdi_prob: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (mean, lower, upper) per SKU for ``var_name`` with dims
    ``(chain, draw, sku)``.
    """
    da = posterior[var_name]
    # Flatten chain/draw
    flat = da.stack(sample=("chain", "draw"))
    mean = flat.mean("sample").values.astype(np.float64)
    # Quantile HDI approximation via equal-tail for speed/stability in tests
    alpha = (1.0 - hdi_prob) / 2.0
    lower = flat.quantile(alpha, dim="sample").values.astype(np.float64)
    upper = flat.quantile(1.0 - alpha, dim="sample").values.astype(np.float64)
    return mean, lower, upper


def fraction_truth_in_interval(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    covered = (truth >= lower) & (truth <= upper)
    return float(np.mean(covered))


def max_rhat(summary) -> float:
    """Max ``r_hat`` from an ArviZ / model ``fit_summary`` DataFrame."""
    if "r_hat" not in summary.columns:
        raise KeyError("summary missing r_hat column")
    vals = summary["r_hat"].to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.max(vals))
