"""Short MCMC smoke for Lewbel instruments."""

from __future__ import annotations

import warnings

import numpy as np

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data
from pypricing.diagnostics import DiagnosticWarning


def test_lewbel_fit_samples_rho_and_predicts():
    df = generate_mock_data(
        n_periods=60,
        n_skus=1,
        heteroskedasticity=True,
        endogeneity=1.2,
        noise_sigma=0.3,
        price_shock_sigma=0.02,
        include_seasonality=False,
        random_state=0,
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(quantity_floor=1e-12, control_columns=("hetero",)),
        endogeneity="lewbel",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiagnosticWarning)
        model.fit(
            df,
            draws=40,
            tune=40,
            chains=2,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )
    assert "rho" in model.idata.posterior
    summary = model.endogeneity_method_.summarize_posterior(model)
    assert np.isfinite(summary["rho"]["mean"])
    predicted = model.predict(df.drop(columns=["quantity"]), random_seed=0)
    assert len(predicted) == len(df)
    assert np.all(np.isfinite(predicted["quantity_mean"].to_numpy()))
