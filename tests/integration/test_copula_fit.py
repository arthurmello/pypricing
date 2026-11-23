"""Short MCMC smoke for the copula control."""

from __future__ import annotations

import warnings

import numpy as np

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data
from pypricing.diagnostics import DiagnosticWarning


def test_copula_fit_samples_rho_and_predicts():
    df = generate_mock_data(
        n_periods=40,
        n_skus=2,
        price_distribution="gamma",
        endogeneity=1.2,
        noise_sigma=0.3,
        include_seasonality=False,
        random_state=0,
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(quantity_floor=1e-12),
        endogeneity="copula",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiagnosticWarning)
        model.fit(
            df,
            draws=40,
            tune=40,
            chains=2,
            random_seed=0,
            progressbar=False,
            compute_convergence_checks=False,
        )
    assert "rho_copula" in model.idata.posterior
    summary = model.endogeneity_method_.summarize_posterior(model)
    assert np.isfinite(summary["rho_copula"]["mean"])
    predicted = model.predict(df.drop(columns=["quantity"]), random_seed=0)
    assert len(predicted) == len(df)
    assert np.all(np.isfinite(predicted["quantity_mean"].to_numpy()))
