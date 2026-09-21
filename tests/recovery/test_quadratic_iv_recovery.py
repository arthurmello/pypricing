"""IV recovery for quadratic: associational elasticity is biased; CF-IV recovers it."""

from __future__ import annotations

import numpy as np

from pypricing import PanelColumns, QuadraticLogDemandModel, generate_mock_data

from tests.recovery.helpers import (
    fraction_truth_in_interval,
    posterior_hdi_by_sku,
)

_RECOVERY_DRAWS = 400
_RECOVERY_TUNE = 400
_RECOVERY_CHAINS = 2
_HDI_PROB = 0.9
_MIN_ELASTICITY_COVERAGE = 0.75


def _endogenous_panel(*, seed: int):
    return generate_mock_data(
        n_periods=80,
        n_skus=3,
        n_controls=0,
        n_instruments=1,
        instrument_coef=0.35,
        endogeneity=1.4,
        noise_sigma=0.25,
        price_shock_sigma=0.01,
        include_seasonality=False,
        round_quantity=False,
        shape="quadratic",
        random_state=seed,
        return_truth=True,
    )


def _fit(df, panel_columns: PanelColumns, seed: int) -> QuadraticLogDemandModel:
    model = QuadraticLogDemandModel(panel_columns=panel_columns)
    model.fit(
        df,
        draws=_RECOVERY_DRAWS,
        tune=_RECOVERY_TUNE,
        chains=_RECOVERY_CHAINS,
        random_seed=seed,
        progressbar=False,
        compute_convergence_checks=False,
        target_accept=0.9,
    )
    return model


def _true_elasticity(model: QuadraticLogDemandModel, truth) -> np.ndarray:
    label_to_i = {lab: i for i, lab in enumerate(truth.sku_labels)}
    order = [label_to_i[str(s)] for s in model.sku_levels_]
    return truth.elasticity_sku[order]


def test_quadratic_endogenous_prices_bias_associational_elasticity():
    df, truth = _endogenous_panel(seed=21)
    model = _fit(df, PanelColumns(quantity_floor=1e-12, iv_columns=()), seed=21)
    true_e = _true_elasticity(model, truth)
    mean_e, lo_e, hi_e = posterior_hdi_by_sku(
        model.idata.posterior, "elasticity_sku", hdi_prob=_HDI_PROB
    )
    coverage = fraction_truth_in_interval(true_e, lo_e, hi_e)
    bias = float(np.mean(mean_e - true_e))
    assert bias > 0.2, f"expected OLS-style elasticity toward 0; bias={bias:.3f}"
    assert coverage < 0.5, (
        f"associational HDI still covered truth; coverage={coverage:.2f}"
    )


def test_quadratic_iv_recovers_elasticity_under_endogenous_prices():
    df, truth = _endogenous_panel(seed=21)
    model = _fit(df, PanelColumns(quantity_floor=1e-12, iv_columns=("iv_1",)), seed=21)
    true_e = _true_elasticity(model, truth)
    mean_e, lo_e, hi_e = posterior_hdi_by_sku(
        model.idata.posterior, "elasticity_sku", hdi_prob=_HDI_PROB
    )
    coverage = fraction_truth_in_interval(true_e, lo_e, hi_e)
    assert coverage >= _MIN_ELASTICITY_COVERAGE, (
        f"IV elasticity HDI coverage {coverage:.2f} < {_MIN_ELASTICITY_COVERAGE}; "
        f"truth={true_e}, mean={mean_e}, lo={lo_e}, hi={hi_e}"
    )
    mae = float(np.mean(np.abs(mean_e - true_e)))
    assert mae < 0.45, f"IV elasticity MAE {mae:.3f} too large"
    assert np.all(mean_e < 0.0)
