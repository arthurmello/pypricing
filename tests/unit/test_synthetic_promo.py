import numpy as np
import pandas as pd

from pypricing import generate_promo_panel
from pypricing.synthetic_promo import black_friday


def test_black_friday_dates():
    assert black_friday(2024) == pd.Timestamp("2024-11-29")
    assert black_friday(2025) == pd.Timestamp("2025-11-28")


def test_promo_panel_shape_and_consistency():
    df, truth = generate_promo_panel(n_brands=2, n_categories=2, n_weeks=20)
    assert len(df) == 2 * 2 * 20
    on = df["discount"] > 0
    assert (df.loc[~on, "promo_type"] == "none").all()
    assert (df.loc[on, "promo_type"] != "none").all()
    assert np.all((df["discount"] >= 0) & (df["discount"] < 1))
    assert set(truth["competitor_effect_category"]) == set(df["category"])
