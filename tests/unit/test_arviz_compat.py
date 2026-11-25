"""Helpers shared by ArviZ 0.x and 1.x.

The module is loaded by path. Importing ``pypricing`` also imports PyMC, and
current PyMC requires ArviZ < 1, so this file is what the ArviZ 1 CI job runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _compat():
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "pypricing"
        / "arviz_compat.py"
    )
    spec = importlib.util.spec_from_file_location("pypricing_arviz_compat", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _posterior():
    values = np.random.default_rng(0).normal(size=(2, 40))
    try:
        return az.from_dict(posterior={"sigma": values})
    except TypeError:
        return az.from_dict({"posterior": {"sigma": values}})


def test_datatree_group_replaces_and_reloads(tmp_path):
    compat = _compat()
    tree = xr.DataTree()
    first = pd.DataFrame({"price": [1.0]}).to_xarray()
    second = pd.DataFrame({"sku": ["10", "9"], "price": [4.0, 5.0]}).to_xarray()
    compat.add_dataset_group(tree, "fit_data", first)
    compat.add_dataset_group(tree, "fit_data", second)
    tree.attrs["id"] = "m1"
    path = tmp_path / "fit.nc"
    tree.to_netcdf(path)

    loaded = compat.read_netcdf(path)
    assert loaded.attrs["id"] == "m1"
    assert np.allclose(np.asarray(loaded.fit_data["price"]), [4.0, 5.0])


def test_inferencedata_group_roundtrip(tmp_path):
    try:
        idata = az.from_dict(posterior={"sigma": np.zeros((1, 4))})
    except TypeError:
        pytest.skip("this ArviZ builds a DataTree, not InferenceData")
    if not hasattr(idata, "add_groups"):
        pytest.skip("this ArviZ builds a DataTree, not InferenceData")

    compat = _compat()
    dataset = pd.DataFrame({"sku": ["10", "9"], "price": [1.0, 2.0]}).to_xarray()
    compat.add_dataset_group(idata, "fit_data", dataset)
    idata.attrs["id"] = "m2"
    path = tmp_path / "fit.nc"
    idata.to_netcdf(path)

    loaded = compat.read_netcdf(path)
    assert loaded.attrs["id"] == "m2"
    assert "fit_data" in loaded
    assert np.allclose(np.asarray(loaded.fit_data["price"]), [1.0, 2.0])


def test_hdi_bounds_for_draws_and_dataarray():
    compat = _compat()
    draws = np.linspace(-2.0, 2.0, 51)
    lower, upper = compat.hdi_bounds(draws, 0.8)
    assert float(np.asarray(lower)) < float(np.asarray(upper))

    quantity = xr.DataArray(
        np.random.default_rng(1).normal(size=(2, 30, 3)),
        dims=("chain", "draw", "obs"),
        name="quantity",
    )
    lower, upper = compat.hdi_bounds(quantity, 0.9)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    assert lower.shape == (3,)
    assert np.all(lower <= upper)


def test_summary_accepts_hdi_prob():
    compat = _compat()
    table = compat.summary(_posterior(), var_names=["sigma"], hdi_prob=0.9)
    assert "r_hat" in table.columns
