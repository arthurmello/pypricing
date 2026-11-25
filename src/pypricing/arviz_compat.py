"""Operations that differ between ArviZ 0.x and ArviZ 1.x.

ArviZ 1 replaces ``InferenceData`` with ``xarray.DataTree``. Sampling can
finish and then fail while attaching results, so fit, save, load, and HDI
all go through this module.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import xarray as xr


def add_dataset_group(idata: Any, name: str, dataset: xr.Dataset) -> None:
    """Attach ``dataset`` as a group, replacing an existing group of that name."""
    with np.errstate(all="ignore"):
        if hasattr(idata, "add_groups"):
            if name in idata:
                delattr(idata, name)
            idata.add_groups(**{name: dataset})
            return
        if name in idata:
            del idata[name]
        idata[name] = dataset


def read_netcdf(path: str | Path) -> Any:
    """Load a fit file written by either ArviZ generation."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"fit_data group is not defined in the InferenceData scheme",
        )
        return az.from_netcdf(path)


def hdi_bounds(data: Any, prob: float) -> tuple[Any, Any]:
    """Lower and upper HDI, on ArviZ 0.x (``hdi``) and 1.x (``ci_bound``)."""
    result = _call_hdi(data, prob)
    if isinstance(result, xr.Dataset):
        name = getattr(data, "name", None)
        if name in result:
            result = result[name]
        else:
            result = next(iter(result.data_vars.values()))
    if isinstance(result, xr.DataArray):
        dim = _interval_dim(result)
        lower_name, upper_name = _interval_labels(result.coords[dim].values)
        return (
            result.sel({dim: lower_name}),
            result.sel({dim: upper_name}),
        )
    values = np.asarray(result)
    if values.shape[-1] == 2:
        return values[..., 0], values[..., 1]
    if values.shape[0] == 2:
        return values[0], values[1]
    raise TypeError(f"Unexpected HDI return type: {type(result)}")


def summary(data: Any, var_names: list[str] | None = None, **kwargs: Any) -> Any:
    """``az.summary``, accepting the 0.x ``hdi_prob`` keyword on 1.x."""
    try:
        return az.summary(data, var_names=var_names, **kwargs)
    except TypeError as exc:
        if "hdi_prob" not in str(exc) or "hdi_prob" not in kwargs:
            raise
        translated = dict(kwargs)
        translated["ci_prob"] = translated.pop("hdi_prob")
        return az.summary(data, var_names=var_names, **translated)


def _call_hdi(data: Any, prob: float) -> Any:
    try:
        return az.hdi(data, hdi_prob=prob)
    except TypeError as exc:
        if "hdi_prob" not in str(exc):
            raise
        return az.hdi(data, prob=prob)


def _interval_dim(result: xr.DataArray) -> str:
    for name in ("hdi", "ci_bound"):
        if name in result.dims:
            return name
    raise TypeError(f"HDI result has no interval dimension: {result.dims}")


def _interval_labels(labels: Any) -> tuple[str, str]:
    names = [str(label) for label in labels]
    lower = "lower" if "lower" in names else names[0]
    if "higher" in names:
        upper = "higher"
    elif "upper" in names:
        upper = "upper"
    else:
        upper = names[-1]
    return lower, upper
