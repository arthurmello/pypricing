"""Regular-price and promo elasticities recover on a split-price DGP."""

from __future__ import annotations

import arviz as az
import numpy as np

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data

_DRAWS = 400
_TUNE = 400
_CHAINS = 2
_HDI_PROB = 0.9
_TRUE_REGULAR = -2.0
_TRUE_PROMO = -3.5


def _hdi_by_sku(posterior, var_name: str):
    da = posterior[var_name]
    mean = np.asarray(da.mean(("chain", "draw")).values, dtype=np.float64)
    interval = az.hdi(da, hdi_prob=_HDI_PROB)
    if hasattr(interval, "data_vars"):
        interval = interval[next(iter(interval.data_vars))]
    lower = np.asarray(interval.sel(hdi="lower").values, dtype=np.float64)
    upper = np.asarray(interval.sel(hdi="higher").values, dtype=np.float64)
    return mean, lower, upper


def _fit(df, columns, seed: int) -> LogLogDemandModel:
    model = LogLogDemandModel(panel_columns=columns)
    model.fit(
        df,
        draws=_DRAWS,
        tune=_TUNE,
        chains=_CHAINS,
        random_seed=seed,
        progressbar=False,
        compute_convergence_checks=False,
        target_accept=0.9,
    )
    return model


def test_regular_and_promo_elasticities_cover_truth_and_single_price_lies_between():
    df, truth = generate_mock_data(
        n_periods=80,
        n_skus=3,
        promotions=True,
        promo_frequency=0.4,
        promo_depth=0.2,
        true_elasticity=_TRUE_REGULAR,
        true_promo_elasticity=_TRUE_PROMO,
        price_shock_sigma=0.08,
        noise_sigma=0.1,
        include_seasonality=False,
        random_state=7,
        return_truth=True,
    )
    assert np.all(truth.elasticity_sku == _TRUE_REGULAR)
    assert truth.promo_elasticity == _TRUE_PROMO
    columns = PanelColumns(discount_col="discount_rate", quantity_floor=1e-12)
    split = _fit(df, columns, seed=7)

    regular_mean, regular_lo, regular_hi = _hdi_by_sku(
        split.idata.posterior, "elasticity_sku"
    )
    promo_mean, promo_lo, promo_hi = _hdi_by_sku(
        split.idata.posterior, "promo_elasticity_sku"
    )
    assert np.all((regular_lo <= _TRUE_REGULAR) & (_TRUE_REGULAR <= regular_hi)), (
        f"regular HDI missed {_TRUE_REGULAR}: mean={regular_mean}, "
        f"lo={regular_lo}, hi={regular_hi}"
    )
    assert np.all((promo_lo <= _TRUE_PROMO) & (_TRUE_PROMO <= promo_hi)), (
        f"promo HDI missed {_TRUE_PROMO}: mean={promo_mean}, "
        f"lo={promo_lo}, hi={promo_hi}"
    )

    pooled = _fit(df, PanelColumns(quantity_floor=1e-12), seed=7)
    pooled_mean, _, _ = _hdi_by_sku(pooled.idata.posterior, "elasticity_sku")
    assert np.all((pooled_mean > _TRUE_PROMO) & (pooled_mean < _TRUE_REGULAR)), (
        f"single-price elasticity {pooled_mean} is not between "
        f"{_TRUE_PROMO} and {_TRUE_REGULAR}"
    )
