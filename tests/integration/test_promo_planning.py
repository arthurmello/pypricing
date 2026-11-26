"""Integration checks for plan frames, scenarios and the promo optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    PromoSpec,
    add_event_columns,
    generate_promo_panel,
)
from pypricing.promo import WINDOW_COL, apply_promos, compare_scenarios, plan_frame


@pytest.fixture(scope="module")
def fitted():
    df, _ = generate_promo_panel(
        n_brands=2, n_categories=2, n_weeks=70, promo_prob=0.3, random_state=7
    )
    df = add_event_columns(df)
    model = LogLogDemandModel(
        panel_columns=PanelColumns(
            discount_col="discount", group_columns=("category",)
        ),
        promo=PromoSpec(type_col="promo_type", lags=1, pressure_level=0),
    )
    model.fit(
        df,
        draws=120,
        tune=150,
        chains=2,
        random_seed=2,
        progressbar=False,
        compute_convergence_checks=False,
    )
    return model, df


def test_plan_frame_layout(fitted):
    model, df = fitted
    frame = plan_frame(model, 3)
    n_skus = df["sku"].nunique()
    assert len(frame) == n_skus * 4  # 3 window periods + 1 lag tail
    assert frame[WINDOW_COL].sum() == n_skus * 3
    assert (frame["discount"] == 0).all()
    assert (frame["promo_type"] == "none").all()
    assert frame["period"].min() > df["period"].max()
    last = df[df["period"] == df["period"].max()].set_index("sku")
    regular = last["price"] / (1 - last["discount"])
    first = frame[frame["period"] == frame["period"].min()].set_index("sku")
    np.testing.assert_allclose(first["price"], regular.loc[first.index])


def test_apply_promos_only_touches_window(fitted):
    model, _ = fitted
    frame = plan_frame(model, 2)
    sku = frame["sku"].iloc[0]
    out = apply_promos(
        model,
        frame,
        pd.DataFrame({"sku": [sku], "discount": [0.25], "promo_type": ["featured"]}),
    )
    hit = (out["sku"] == sku) & out[WINDOW_COL]
    assert np.allclose(out.loc[hit, "discount"], 0.25)
    assert (out.loc[~hit, "discount"] == 0).all()
    assert (out.loc[hit, "promo_type"] == "featured").all()
    np.testing.assert_allclose(out.loc[hit, "price"], frame.loc[hit, "price"] * 0.75)


def test_compare_scenarios(fitted):
    model, _ = fitted
    frame = plan_frame(model, 2)
    skus = frame[["sku"]].drop_duplicates()
    result = compare_scenarios(
        model,
        frame,
        {"shallow": skus.assign(discount=0.1), "deep": skus.assign(discount=0.4)},
        objective="units",
    )
    s = result.summary
    assert list(s.index) == ["baseline", "shallow", "deep"]
    assert (
        s.loc["deep", "units"] > s.loc["shallow", "units"] > s.loc["baseline", "units"]
    )
    assert s.loc["deep", "promo_cost"] > s.loc["shallow", "promo_cost"] > 0
    assert np.isclose(s["prob_best"].sum(), 1.0)
    assert result.draws.shape[1] == 3


def test_promo_options_table(fitted):
    from pypricing.promo import promo_options

    model, df = fitted
    frame = plan_frame(model, 2)
    opts = promo_options(
        model, frame, discounts=(0.1, 0.6), promo_types=("pct_off", "featured")
    )
    n_units = df["sku"].nunique()
    assert len(opts.table) == n_units * 5  # no promo + 2 depths x 2 types
    none = opts.table[opts.table["discount"] == 0]
    assert (none["incremental_margin"] == 0).all()
    deep = opts.table[opts.table["discount"] == 0.6]
    assert deep["extrapolated"].all()
    assert opts.draws["margin"].shape[1:] == (n_units, 5)


def test_optimize_respects_constraints(fitted):
    from pypricing.promo import PromoConstraints, optimize_promotions

    model, _ = fitted
    frame = plan_frame(model, 2)
    c = PromoConstraints(max_promos=1, max_discount=0.2)
    result = optimize_promotions(
        model, frame, discounts=(0.1, 0.2, 0.3), objective="revenue", constraints=c
    )
    assert (result.plan["discount"] > 0).sum() <= 1
    assert (result.plan["discount"] <= 0.2).all()
    assert result.comparison.index[0] == "baseline"
    assert "best" in result.comparison.index
    assert result.usage["n_promos"] <= 1


def test_optimize_by_category_units_and_budget(fitted):
    from pypricing.promo import PromoConstraints, optimize_promotions

    model, _ = fitted
    frame = plan_frame(model, 2)
    result = optimize_promotions(
        model,
        frame,
        discounts=(0.1, 0.3),
        unit_cols=("category",),
        objective="units",
        constraints=PromoConstraints(budget=1e9),
        n_alternatives=2,
    )
    assert list(result.plan["category"]) == sorted(frame["category"].unique())
    # Units always rise with a deeper discount, so with no real budget limit the
    # deepest option wins everywhere.
    assert (result.plan["discount"] == 0.3).all()
    assert len(result.alternatives) >= 1


def test_infeasible_plan_raises(fitted):
    from pypricing.promo import PromoConstraints, optimize_promotions

    model, _ = fitted
    frame = plan_frame(model, 2)
    with pytest.raises(ValueError, match="(?i)margin|feasible"):
        optimize_promotions(
            model,
            frame,
            discounts=(0.2,),
            constraints=PromoConstraints(min_margin_rate=0.99),
        )
