"""Recovery: common shocks bias elasticity; period effects and Mundlak undo it."""

from __future__ import annotations

import numpy as np
import pymc as pm

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data

from tests.recovery.helpers import fraction_truth_in_interval, posterior_hdi_by_sku

_DRAWS = 300
_TUNE = 300
_CHAINS = 2
_HDI_PROB = 0.9
_MIN_COVERAGE = 0.75
_PANEL = PanelColumns(quantity_floor=1e-12)


def _fit(df, **model_kwargs):
    model_kwargs.setdefault("panel_columns", _PANEL)
    model = LogLogDemandModel(**model_kwargs)
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
    return model


def _elasticity(model, truth):
    label_to_i = {lab: i for i, lab in enumerate(truth.sku_labels)}
    order = [label_to_i[str(s)] for s in model.sku_levels_]
    true_e = truth.elasticity_sku[order]
    mean_e, lo_e, hi_e = posterior_hdi_by_sku(
        model.idata.posterior, "elasticity_sku", hdi_prob=_HDI_PROB
    )
    return true_e, mean_e, lo_e, hi_e


def test_common_shock_biases_elasticity_toward_zero():
    df, truth = generate_mock_data(
        n_periods=36,
        n_skus=4,
        n_controls=0,
        include_seasonality=False,
        price_shock_sigma=0.02,
        noise_sigma=0.12,
        common_shock_sigma=0.45,
        common_endogeneity=1.5,
        true_elasticity=-1.0,
        shape="log_log",
        random_state=4,
        return_truth=True,
    )
    model = _fit(df)
    true_e, mean_e, _lo, _hi = _elasticity(model, truth)
    bias = float(np.mean(mean_e - true_e))
    assert bias > 0.15, f"expected elasticity biased toward 0; bias={bias:.3f}"


def test_period_effects_recover_elasticity_and_lambda():
    # Within-period price variation has to be large enough that the elasticity
    # is not shrunk into lambda under the period-effect prior.
    df, truth = generate_mock_data(
        n_periods=30,
        n_skus=4,
        n_controls=0,
        include_seasonality=False,
        price_shock_sigma=0.25,
        noise_sigma=0.08,
        common_shock_sigma=0.4,
        common_endogeneity=1.2,
        true_elasticity=-1.0,
        shape="log_log",
        random_state=5,
        return_truth=True,
    )
    model = _fit(df, time_effects="period")
    true_e, _mean, lo_e, hi_e = _elasticity(model, truth)
    coverage = fraction_truth_in_interval(true_e, lo_e, hi_e)
    assert coverage >= _MIN_COVERAGE, f"elasticity coverage {coverage:.2f}"

    lam = model.idata.posterior["lambda_period"].mean(("chain", "draw"))
    fitted = np.asarray(lam.values, dtype=np.float64)
    corr = float(np.corrcoef(fitted, truth.lambda_period)[0, 1])
    assert corr > 0.7, f"lambda correlation {corr:.2f}"


def test_region_period_effects_recover_elasticity():
    df, truth = generate_mock_data(
        n_periods=16,
        n_skus=8,
        n_regions=2,
        n_controls=0,
        include_seasonality=False,
        price_shock_sigma=0.15,
        noise_sigma=0.12,
        common_shock_sigma=0.4,
        common_endogeneity=1.2,
        common_shock_by_region=True,
        true_elasticity=-1.0,
        shape="log_log",
        random_state=6,
        return_truth=True,
    )
    model = _fit(
        df,
        time_effects="region_period",
        panel_columns=PanelColumns(region_col="region", quantity_floor=1e-12),
    )
    true_e, _mean, lo_e, hi_e = _elasticity(model, truth)
    coverage = fraction_truth_in_interval(true_e, lo_e, hi_e)
    assert coverage >= _MIN_COVERAGE, f"elasticity coverage {coverage:.2f}"

    national = model.idata.posterior["lambda_period"].mean(("chain", "draw"))
    regional = model.idata.posterior["lambda_region_period"].mean(("chain", "draw"))
    corrs = []
    for r, region in enumerate(model.region_levels_):
        fitted = np.asarray(national.values, dtype=np.float64) + np.asarray(
            regional.sel(region=region).values, dtype=np.float64
        )
        corrs.append(float(np.corrcoef(fitted, truth.lambda_period[r])[0, 1]))
    assert min(corrs) > 0.6, f"region-period correlations {corrs}"


def test_mundlak_removes_shrinkage_bias():
    df, truth = generate_mock_data(
        n_periods=8,
        n_skus=12,
        hierarchy_levels=(3,),
        n_controls=0,
        include_seasonality=False,
        price_shock_sigma=0.1,
        noise_sigma=0.08,
        mundlak_coef=1.5,
        true_elasticity=-1.0,
        shape="log_log",
        random_state=7,
        return_truth=True,
    )
    panel = PanelColumns(
        group_columns=("category_1",),
        quantity_floor=1e-12,
    )
    # Center the elasticity prior on the truth so the comparison is the
    # Mundlak bias, not the default -2.6 location.
    config = {
        "mu_elasticity": {"dist": pm.Normal, "kwargs": {"mu": -1.0, "sigma": 1.0}}
    }
    plain = _fit(df, panel_columns=panel, model_config=config)
    corrected = _fit(df, panel_columns=panel, mundlak=True, model_config=config)
    true_plain, mean_plain, _lo, _hi = _elasticity(plain, truth)
    true_m, mean_m, lo_m, hi_m = _elasticity(corrected, truth)
    bias_plain = float(np.mean(mean_plain - true_plain))
    bias_m = float(np.mean(mean_m - true_m))
    assert bias_plain > bias_m + 0.05, (
        f"mundlak should pull elasticity back from zero; "
        f"plain={bias_plain:.3f}, mundlak={bias_m:.3f}"
    )
    coverage = fraction_truth_in_interval(true_m, lo_m, hi_m)
    assert coverage >= 0.6, f"mundlak coverage {coverage:.2f}"
