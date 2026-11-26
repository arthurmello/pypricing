"""Integration checks for PromoSpec terms on the synthetic promo panel."""

from __future__ import annotations

import numpy as np
import pytest

from pypricing import LogLogDemandModel, PanelColumns, PromoSpec, generate_promo_panel

SAMPLE = dict(
    draws=150,
    tune=150,
    chains=2,
    random_seed=3,
    progressbar=False,
    compute_convergence_checks=False,
)


@pytest.fixture(scope="module")
def panel():
    df, truth = generate_promo_panel(
        n_brands=3, n_categories=2, n_weeks=80, promo_prob=0.25, random_state=4
    )
    return df, truth


def _model(**kwargs):
    cols = PanelColumns(discount_col="discount", group_columns=("category",))
    return LogLogDemandModel(panel_columns=cols, **kwargs)


def test_quadratic_depth_finds_saturation(panel, tmp_path):
    df, truth = panel
    model = _model(promo=PromoSpec(depth="quadratic"))
    model.fit(df, **SAMPLE)
    curvature = float(model.idata.posterior["mu_promo_curvature"].mean())
    assert curvature < -0.3
    pred = model.predict(df.head(4))
    assert np.isfinite(pred["quantity_mean"]).all()

    path = tmp_path / "m.nc"
    model.save(str(path))
    loaded = LogLogDemandModel.load(str(path))
    assert loaded.promo == PromoSpec(depth="quadratic")
    np.testing.assert_allclose(
        loaded.predict(df.head(4), random_seed=0)["quantity_mean"],
        model.predict(df.head(4), random_seed=0)["quantity_mean"],
    )


def test_promo_requires_discount_col():
    with pytest.raises(ValueError, match="discount_col"):
        LogLogDemandModel(promo=PromoSpec(depth="quadratic"))


def test_promo_type_effects(panel):
    df, truth = panel
    model = _model(promo=PromoSpec(type_col="promo_type"))
    model.fit(df, **SAMPLE)
    assert model.promo_type_levels_ == ("bundle", "featured", "pct_off")
    effect = model.idata.posterior["promo_type_effect"].mean(("chain", "draw"))
    featured = float(
        effect.sel(promo_type="featured") - effect.sel(promo_type="pct_off")
    )
    assert featured > 0
    bad = df.head(3).copy()
    bad["discount"] = 0.2
    bad["promo_type"] = "flash_sale"
    with pytest.raises(ValueError, match="Unknown promo types"):
        model.predict(bad)


def test_post_promo_dip(panel):
    df, truth = panel
    model = _model(promo=PromoSpec(lags=1))
    model.fit(df, **SAMPLE)
    lag = float(model.idata.posterior["promo_lag"].mean())
    assert lag < 0


def test_varying_competitor_effects(panel):
    df, truth = panel
    model = _model(varying_controls=("control_log_competitor_price",))
    model.fit(df, **SAMPLE)
    post = model.idata.posterior
    slope = (
        post["beta_control"].values[..., 0][..., None]
        + post["beta_control_dev_sku"].values[..., 0]
    )
    by_sku = dict(zip(model.sku_levels_, slope.mean(axis=(0, 1))))
    cat_a = np.mean([v for k, v in by_sku.items() if k.endswith("cat_A")])
    cat_b = np.mean([v for k, v in by_sku.items() if k.endswith("cat_B")])
    tc = truth["competitor_effect_category"]
    assert (cat_a - cat_b) * (tc["cat_A"] - tc["cat_B"]) > 0
    ctrl = df.groupby("sku", as_index=False)["control_log_competitor_price"].median()
    bounds = {s: (10.0, 200.0) for s in model.sku_levels_}
    out = model.optimize_prices(price_bounds=bounds, controls_df=ctrl)
    assert np.isfinite(out["mean_revenue"]).all()


def test_varying_controls_must_be_controls(panel):
    df, _ = panel
    model = _model(varying_controls=("not_a_control",))
    with pytest.raises(ValueError, match="varying_controls"):
        model.build_model(df)
