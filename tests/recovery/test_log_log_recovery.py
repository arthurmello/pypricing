"""Parameter recovery for LogLogDemandModel on synthetic log-log DGP."""

from __future__ import annotations

import numpy as np

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data

from tests.recovery.helpers import (
    fraction_truth_in_interval,
    max_rhat,
    posterior_hdi_by_sku,
)

# Longer MCMC than integration smoke; still kept modest for local runs.
_RECOVERY_DRAWS = 400
_RECOVERY_TUNE = 400
_RECOVERY_CHAINS = 2
_HDI_PROB = 0.9
_MIN_ELASTICITY_COVERAGE = 0.75
# Continuous qty can be < 1; default floor=1 would clamp and break recovery.
_RECOVERY_PANEL_COLUMNS = PanelColumns(quantity_floor=1e-12)


def _recovery_panel(*, n_periods: int, n_skus: int, seed: int):
    """Continuous qty, no seasonality, stronger price walks — ID-friendly DGP."""
    return generate_mock_data(
        n_periods=n_periods,
        n_skus=n_skus,
        n_controls=0,
        random_state=seed,
        shape="log_log",
        include_seasonality=False,
        price_shock_sigma=0.08,
        return_truth=True,
    )


def test_log_log_elasticity_recovery_hdi_coverage():
    df, truth = _recovery_panel(n_periods=60, n_skus=4, seed=11)
    assert truth.shape == "log_log"
    assert truth.curvature_sku is None
    assert df["log_quantity"].groupby(df["sku"]).std().min() > 0.05

    model = LogLogDemandModel(panel_columns=_RECOVERY_PANEL_COLUMNS)
    idata = model.fit(
        df,
        draws=_RECOVERY_DRAWS,
        tune=_RECOVERY_TUNE,
        chains=_RECOVERY_CHAINS,
        random_seed=11,
        progressbar=False,
        compute_convergence_checks=False,
        target_accept=0.9,
    )

    label_to_i = {lab: i for i, lab in enumerate(truth.sku_labels)}
    order = [label_to_i[str(s)] for s in model.sku_levels_]
    true_e = truth.elasticity_sku[order]

    mean_e, lo_e, hi_e = posterior_hdi_by_sku(
        idata.posterior, "elasticity_sku", hdi_prob=_HDI_PROB
    )
    coverage = fraction_truth_in_interval(true_e, lo_e, hi_e)
    assert coverage >= _MIN_ELASTICITY_COVERAGE, (
        f"elasticity HDI coverage {coverage:.2f} < {_MIN_ELASTICITY_COVERAGE}; "
        f"truth={true_e}, mean={mean_e}, lo={lo_e}, hi={hi_e}"
    )

    assert np.all(mean_e < 0.0)

    mae = float(np.mean(np.abs(mean_e - true_e)))
    assert mae < 0.35, f"elasticity MAE {mae:.3f} too large"


def test_log_log_recovery_diagnostics_reasonable():
    df, _truth = _recovery_panel(n_periods=40, n_skus=3, seed=3)
    model = LogLogDemandModel(panel_columns=_RECOVERY_PANEL_COLUMNS)
    model.fit(
        df,
        draws=_RECOVERY_DRAWS,
        tune=_RECOVERY_TUNE,
        chains=_RECOVERY_CHAINS,
        random_seed=3,
        progressbar=False,
        compute_convergence_checks=False,
        target_accept=0.9,
    )

    diag = model.run_diagnostics()
    assert diag["n_divergent"] is not None
    assert diag["n_divergent"] <= 10
    if diag.get("max_rhat") is not None:
        assert diag["max_rhat"] < 1.15
    else:
        summary = model.fit_summary()
        assert max_rhat(summary) < 1.15

    sigma_mean = float(model.idata.posterior["sigma"].mean())
    assert 0.02 < sigma_mean < 1.0
