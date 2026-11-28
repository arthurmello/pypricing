"""Category x week effects and seasonal forecasting of period effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pypricing import LogLogDemandModel, PanelColumns, PromoSpec, generate_promo_panel
from pypricing.promo import plan_frame


def _confounded_panel(seed: int = 0) -> tuple[pd.DataFrame, float]:
    """Promos timed in category-weeks with a demand shock: naive lift is biased up."""
    df, truth = generate_promo_panel(
        n_brands=3, n_categories=3, n_weeks=110, promo_prob=0.0, random_state=seed
    )
    rng = np.random.default_rng(seed)
    cells = df[["category", "period"]].drop_duplicates()
    shock = pd.Series(
        rng.normal(0, 0.35, len(cells)), index=pd.MultiIndex.from_frame(cells)
    )
    s = shock.reindex(pd.MultiIndex.from_frame(df[["category", "period"]])).to_numpy()
    # Promote more often when the category shock is high.
    p = 1 / (1 + np.exp(-(s * 6 - 1.8)))
    on = rng.random(len(df)) < p
    d = np.where(on, 0.2, 0.0)
    regular = df["price"] / (1 - df["discount"])
    slope = 2.0
    base = np.log(df["quantity"].clip(lower=1))
    q = np.exp(base + slope * -np.log1p(-d) + s)
    out = df.assign(discount=d, price=regular * (1 - d), quantity=np.round(q))
    return out, slope


@pytest.mark.parametrize("seed", [3])
def test_group_period_effects_remove_timing_bias(seed):
    df, slope = _confounded_panel(seed)
    kw = dict(
        draws=150,
        tune=200,
        chains=2,
        random_seed=1,
        progressbar=False,
        compute_convergence_checks=False,
    )
    cols = PanelColumns(discount_col="discount", group_columns=("category",))
    naive = LogLogDemandModel(panel_columns=cols, center_price=True)
    naive.fit(df, **kw)
    fe = LogLogDemandModel(
        panel_columns=cols, center_price=True, time_effects="group_period"
    )
    fe.fit(df, **kw)
    est_naive = -float(naive.idata.posterior["mu_promo_elasticity"].mean())
    est_fe = -float(fe.idata.posterior["mu_promo_elasticity"].mean())
    assert abs(est_fe - slope) < abs(est_naive - slope)
    assert abs(est_fe - slope) < 0.4


def test_seasonal_period_effect_in_plans(tmp_path):
    df, _ = generate_promo_panel(
        n_brands=2, n_categories=2, n_weeks=110, random_state=2
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(
            discount_col="discount", group_columns=("category",)
        ),
        promo=PromoSpec(lags=1),
        time_effects="group_period",
        center_price=True,
    )
    model.fit(
        df,
        draws=80,
        tune=120,
        chains=2,
        random_seed=1,
        progressbar=False,
        compute_convergence_checks=False,
    )
    frame = plan_frame(model, 3, unit_cost_col=None)
    seasonal = model.expected_quantity_draws(frame, period_effect="seasonal")
    neutral = model.expected_quantity_draws(frame, period_effect="neutral")
    assert not np.allclose(seasonal, neutral)
    path = tmp_path / "gp.nc"
    model.save(str(path))
    loaded = LogLogDemandModel.load(str(path))
    np.testing.assert_allclose(
        loaded.expected_quantity_draws(frame, period_effect="seasonal"), seasonal
    )
