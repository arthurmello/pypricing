"""Lewbel sampler matches the fast check and recovers elasticity under heteroskedasticity."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data
from pypricing.diagnostics import DiagnosticWarning

_DRAWS = 300
_TUNE = 300
_CHAINS = 2


def test_lewbel_posterior_matches_the_check_and_the_truth():
    df, truth = generate_mock_data(
        n_periods=300,
        n_skus=1,
        heteroskedasticity=True,
        endogeneity=1.5,
        noise_sigma=0.3,
        price_shock_sigma=0.02,
        true_elasticity=-1.5,
        include_seasonality=False,
        random_state=0,
        return_truth=True,
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(quantity_floor=1e-12, control_columns=("hetero",)),
        endogeneity="lewbel",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        checked = model.endogeneity_check(df, method="lewbel")
        assert checked["verdict"] == "ok"
        quick = float(checked["estimate"].elasticity)
        model.fit(
            df,
            draws=_DRAWS,
            tune=_TUNE,
            chains=_CHAINS,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
            target_accept=0.9,
        )
    posterior_mean = float(
        model.idata.posterior["elasticity_sku"].mean(dim=("chain", "draw")).item()
    )
    truth_elasticity = float(truth.elasticity_sku[0])
    assert posterior_mean == pytest.approx(quick, abs=0.15)
    assert posterior_mean == pytest.approx(truth_elasticity, abs=0.25)
    summary = model.endogeneity_method_.summarize_posterior(model)
    assert np.isfinite(summary["rho"]["mean"])
