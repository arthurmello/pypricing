"""Latent IV sampler matches the fast check and recovers elasticity on a price mixture."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pypricing import LogLogDemandModel, PanelColumns
from pypricing.diagnostics import DiagnosticWarning
from tests.unit.test_latent import latent_iv_frame

_DRAWS = 300
_TUNE = 300
_CHAINS = 2
_TRUTH = -1.5


def test_latent_posterior_matches_the_check_and_the_truth():
    df = latent_iv_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        model = LogLogDemandModel(
            panel_columns=PanelColumns(quantity_floor=1e-12),
            endogeneity="latent",
        )
        checked = model.endogeneity_check(df, method="latent")
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
    assert posterior_mean == pytest.approx(quick, abs=0.15)
    assert posterior_mean == pytest.approx(_TRUTH, abs=0.25)
    summary = model.endogeneity_method_.summarize_posterior(model)
    assert np.isfinite(summary["rho_latent"]["mean"])
    assert summary["n_classes"] == 3
