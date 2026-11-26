"""Synthetic retail promo panel: brands x categories, weekly, with known truth.

Built for testing the promo features end to end: discount depth with saturation,
promo mechanics, post-promo dip, within-category cannibalization, competitor
prices with category-specific effects, retail events, and unit costs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PROMO_DEPTHS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
PROMO_TYPES = ("pct_off", "featured", "bundle")


def black_friday(year: int) -> pd.Timestamp:
    """US Black Friday: the day after the fourth Thursday of November."""
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + pd.Timedelta(weeks=3, days=1)


def generate_promo_panel(
    *,
    n_brands: int = 4,
    n_categories: int = 3,
    n_weeks: int = 104,
    start: str = "2024-01-01",
    promo_prob: float = 0.15,
    sigma: float = 0.15,
    random_state: int | None = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Weekly panel with one SKU per brand x category, and the true parameters.

    Columns: ``sku``, ``brand``, ``category``, ``period`` (week start), ``price``
    (paid), ``discount`` (fraction of regular price), ``promo_type`` (``"none"``
    when not on promo), ``quantity``, ``unit_cost``, ``competitor_price`` and
    ``control_log_competitor_price`` (log of the competitor price relative to
    the SKU's regular price).

    Mean log quantity::

        alpha + e_cat * log(R) + a_sku * z + c * z**2 + type_effect
        + lag * z[t-1] + pressure * mean_z_other_skus_in_category
        + comp_cat * log(competitor / R) + events + seasonality

    with ``z = -log(1 - discount)``.
    """
    rng = np.random.default_rng(random_state)
    brands = [f"brand_{i + 1}" for i in range(n_brands)]
    categories = [f"cat_{chr(65 + j)}" for j in range(n_categories)]
    periods = pd.date_range(start, periods=n_weeks, freq="W-MON")

    cat_elasticity = rng.uniform(-2.2, -1.2, n_categories)
    cat_comp = np.linspace(0.9, 0.1, n_categories)
    truth: dict[str, Any] = {
        "elasticity_category": dict(zip(categories, cat_elasticity)),
        "competitor_effect_category": dict(zip(categories, cat_comp)),
        "promo_curvature": -1.2,
        "type_effect": {"pct_off": 0.0, "featured": 0.2, "bundle": 0.08},
        "lag": -0.35,
        "pressure": -0.8,
        "black_friday": 0.7,
        "christmas": 0.3,
        "sigma": sigma,
        "promo_slope_sku": {},
    }

    rows: list[dict[str, Any]] = []
    skus: list[tuple[str, str, str, float, float, float]] = []
    for j, cat in enumerate(categories):
        for brand in brands:
            sku = f"{brand}|{cat}"
            base_price = float(np.exp(rng.uniform(np.log(15), np.log(120))))
            alpha = float(rng.normal(4.5, 0.4)) - cat_elasticity[j] * np.log(base_price)
            slope = float(rng.uniform(1.8, 3.2))
            truth["promo_slope_sku"][sku] = slope
            skus.append((sku, brand, cat, base_price, alpha, slope))

    n_skus = len(skus)
    discount = np.zeros((n_skus, n_weeks))
    ptype = np.full((n_skus, n_weeks), "none", dtype=object)
    on = rng.random((n_skus, n_weeks)) < promo_prob
    discount[on] = rng.choice(PROMO_DEPTHS, size=int(on.sum()))
    ptype[on] = rng.choice(PROMO_TYPES, size=int(on.sum()), p=[0.6, 0.25, 0.15])
    z = -np.log1p(-discount)

    regular = np.empty((n_skus, n_weeks))
    for s, (_sku, _b, _c, base_price, _a, _sl) in enumerate(skus):
        steps = np.where(rng.random(n_weeks) < 0.04, rng.normal(0, 0.06, n_weeks), 0.0)
        regular[s] = base_price * np.exp(np.cumsum(steps))

    comp_noise = np.cumsum(rng.normal(0, 0.03, (n_categories, n_weeks)), axis=1)
    comp_promo = (rng.random((n_categories, n_weeks)) < 0.12) * rng.uniform(
        0.1, 0.3, (n_categories, n_weeks)
    )

    bf = {black_friday(y) for y in range(periods[0].year, periods[-1].year + 1)}
    t = np.arange(n_weeks)
    season = 0.12 * np.sin(2 * np.pi * (periods.dayofyear.to_numpy() / 365.25))

    cat_of = np.array([categories.index(c) for (_s, _b, c, *_r) in skus])
    for s, (sku, brand, cat, _bp, alpha, slope) in enumerate(skus):
        j = cat_of[s]
        same = np.flatnonzero((cat_of == j) & (np.arange(n_skus) != s))
        for w in range(n_weeks):
            week = periods[w]
            in_bf = any(week <= d < week + pd.Timedelta(days=7) for d in bf)
            in_xmas = week.month == 12 and 14 <= week.day <= 28
            comp_price = (
                regular[s, w] * np.exp(0.05 + comp_noise[j, w]) * (1 - comp_promo[j, w])
            )
            log_comp_rel = float(np.log(comp_price / regular[s, w]))
            mu = (
                alpha
                + cat_elasticity[j] * np.log(regular[s, w])
                + slope * z[s, w]
                + truth["promo_curvature"] * z[s, w] ** 2
                + truth["type_effect"].get(ptype[s, w], 0.0)
                + (truth["lag"] * z[s, w - 1] if w > 0 else 0.0)
                + truth["pressure"] * (z[same, w].mean() if len(same) else 0.0)
                + cat_comp[j] * log_comp_rel
                + truth["black_friday"] * in_bf
                + truth["christmas"] * in_xmas
                + season[w]
                + 0.02 * t[w] / 52
            )
            q = float(np.exp(mu + rng.normal(0, sigma)))
            rows.append(
                {
                    "sku": sku,
                    "brand": brand,
                    "category": cat,
                    "period": week,
                    "price": regular[s, w] * (1 - discount[s, w]),
                    "discount": discount[s, w],
                    "promo_type": ptype[s, w],
                    "quantity": round(q),
                    "unit_cost": regular[s, w] * 0.55,
                    "competitor_price": comp_price,
                    "control_log_competitor_price": log_comp_rel,
                }
            )
    return pd.DataFrame(rows), truth
