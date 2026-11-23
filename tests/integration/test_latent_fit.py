"""Short MCMC smoke for experimental latent IV."""

from __future__ import annotations

import warnings

import numpy as np

from pypricing import LogLogDemandModel, PanelColumns
from pypricing.diagnostics import DiagnosticWarning
from tests.unit.test_latent import latent_iv_frame


def test_latent_fit_samples_rho_and_predicts():
    df = latent_iv_frame(n=80, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        model = LogLogDemandModel(
            panel_columns=PanelColumns(quantity_floor=1e-12),
            endogeneity="latent",
        )
        model.fit(
            df,
            draws=40,
            tune=40,
            chains=2,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )
    assert "rho_latent" in model.idata.posterior
    summary = model.endogeneity_method_.summarize_posterior(model)
    assert np.isfinite(summary["rho_latent"]["mean"])
    assert summary["n_classes"] >= 2
    predicted = model.predict(df.drop(columns=["quantity"]), random_seed=0)
    assert len(predicted) == len(df)
    assert np.all(np.isfinite(predicted["quantity_mean"].to_numpy()))
