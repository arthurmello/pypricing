import numpy as np
import pandas as pd

from pypricing import PanelColumns, aggregate_panel


def test_aggregate_two_skus_fixed_weights():
    df = pd.DataFrame(
        {
            "sku": ["a", "b", "a", "b"],
            "brand": ["x", "x", "x", "x"],
            "category": ["c", "c", "c", "c"],
            "period": [1, 1, 2, 2],
            "price": [8.0, 20.0, 10.0, 20.0],
            "discount": [0.2, 0.0, 0.0, 0.0],
            "quantity": [30.0, 10.0, 10.0, 10.0],
            "unit_cost": [5.0, 12.0, 5.0, 12.0],
            "promo_type": ["featured", "none", "none", "none"],
            "control_comp": [1.0, 3.0, 1.0, 3.0],
        }
    )
    out = aggregate_panel(
        df,
        ["brand", "category"],
        panel_columns=PanelColumns(discount_col="discount"),
        type_col="promo_type",
    )
    assert len(out) == 2
    # Fixed weights: mean quantity a = 20, b = 10.
    w = np.array([20.0, 10.0])
    p1 = (8 * w[0] + 20 * w[1]) / w.sum()
    r1 = (10 * w[0] + 20 * w[1]) / w.sum()
    first = out.iloc[0]
    assert first["sku"] == "x|c"
    assert np.isclose(first["price"], p1)
    assert np.isclose(first["discount"], 1 - p1 / r1)
    assert first["quantity"] == 40
    assert first["promo_type"] == "featured"
    assert out.iloc[1]["discount"] == 0
    assert out.iloc[1]["promo_type"] == "none"
    assert np.isclose(first["control_comp"], (1 * 20 + 3 * 10) / 30)
