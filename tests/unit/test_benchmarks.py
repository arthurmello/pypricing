from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pypricing import LogLogDemandModel
from pypricing.data.index import CrossPairIndex
from pypricing.diagnostics.benchmarks import (
    cross_elasticity_summary,
    elasticity_benchmarks,
    share_less_elastic,
)


def _model(elasticity_draws: np.ndarray, gamma_draws: np.ndarray | None = None):
    """``elasticity_draws`` has shape (draws, skus); one chain."""
    n_draws, n_skus = elasticity_draws.shape
    data_vars = {
        "elasticity_sku": (("chain", "draw", "sku"), elasticity_draws[None]),
    }
    coords = {"chain": [0], "draw": np.arange(n_draws), "sku": np.arange(n_skus)}
    if gamma_draws is not None:
        data_vars["gamma_pair"] = (("chain", "draw", "cross_pair"), gamma_draws[None])
        coords["cross_pair"] = np.arange(gamma_draws.shape[1])

    model = LogLogDemandModel()
    model.idata = az.InferenceData(posterior=xr.Dataset(data_vars, coords=coords))
    model.sku_levels_ = pd.Index([f"sku_{i}" for i in range(n_skus)])
    return model


def test_share_less_elastic_matches_literature():
    assert share_less_elastic(-3.7) == pytest.approx(0.785, abs=1e-3)
    assert share_less_elastic(-100.0) == pytest.approx(1.0)
    assert share_less_elastic(100.0) == pytest.approx(0.0)


def test_elasticity_benchmarks_typical_values_do_not_warn():
    model = _model(np.array([[-1.0, -2.5], [-1.4, -2.7]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = elasticity_benchmarks(model)
    assert list(out.index) == ["sku_0", "sku_1"]
    assert out.loc["sku_0", "elasticity_mean"] == pytest.approx(-1.2)
    assert out.loc["sku_1", "prob_positive"] == 0.0
    assert not out["extreme"].any()
    assert not out["positive"].any()


def test_elasticity_benchmarks_warns_on_extreme_value():
    model = _model(np.array([[-1.0, -15.0], [-1.0, -15.0]]))
    with pytest.warns(UserWarning, match="more elastic than 99%") as record:
        out = elasticity_benchmarks(model)
    assert "sku_1" in str(record[0].message)
    assert out["extreme"].tolist() == [False, True]


def test_elasticity_benchmarks_warns_on_positive_value():
    model = _model(np.array([[-1.0, 0.5], [-1.0, -0.1]]))
    with pytest.warns(UserWarning, match="positive own-price"):
        out = elasticity_benchmarks(model)
    assert out.loc["sku_1", "positive"]
    assert out.loc["sku_1", "prob_positive"] == pytest.approx(0.5)


def test_extreme_threshold_is_configurable():
    model = _model(np.array([[-4.0], [-4.0]]))
    with pytest.warns(UserWarning, match="more elastic than 50%"):
        out = elasticity_benchmarks(model, extreme_threshold=0.5)
    assert out["extreme"].all()


def test_cross_elasticity_summary_labels_pairs_without_warning():
    model = _model(
        np.array([[-1.0, -1.0], [-1.0, -1.0]]),
        gamma_draws=np.array([[0.2, -0.3], [0.4, -0.1]]),
    )
    model.cross_pairs_ = CrossPairIndex(
        pair_from=np.array([0, 1]),
        pair_to=np.array([1, 0]),
        pair_pool=np.array([0, 0]),
        n_pool=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = cross_elasticity_summary(model)
    assert out["focal"].tolist() == ["sku_0", "sku_1"]
    assert out["competitor"].tolist() == ["sku_1", "sku_0"]
    np.testing.assert_allclose(out["gamma_mean"], [0.3, -0.2])
    np.testing.assert_allclose(out["prob_positive"], [1.0, 0.0])


def test_cross_elasticity_summary_is_none_without_cross_terms():
    model = _model(np.array([[-1.0], [-1.0]]))
    assert cross_elasticity_summary(model) is None
    assert set(model.check_benchmarks()) == {"own", "cross"}
