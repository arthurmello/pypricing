import numpy as np
import pandas as pd

from pypricing.model_components.cross_elasticity import (
    CROSS_ELASTICITY_ALL,
    enumerate_cross_pairs,
)
from pypricing.data import PanelColumns, PricePanelData


def test_enumerate_all_pairs_count():
    pairs = enumerate_cross_pairs(
        mode=CROSS_ELASTICITY_ALL,
        n_skus=3,
        hierarchy=None,
        cross_elasticity_group_level=None,
    )
    assert pairs.n_pairs == 6
    assert pairs.n_pool == 1
    assert np.all(pairs.pair_pool == 0)


def test_build_cross_log_price_matrix_matches_directed_pairs():
    df = pd.DataFrame(
        {
            "sku": ["a", "a", "b", "b"],
            "period": [0, 1, 0, 1],
            "price": [10.0, 10.0, 20.0, 20.0],
        }
    )
    sku_levels = pd.Index(["a", "b"])
    pair_from = np.array([0, 1], dtype=np.int64)
    pair_to = np.array([1, 0], dtype=np.int64)
    cat = pd.Categorical(df["sku"], categories=sku_levels)
    obs_sku_idx = cat.codes.astype(np.int64)
    X = PricePanelData.build_cross_log_price_matrix(
        df,
        panel_columns=PanelColumns(),
        sku_levels=sku_levels,
        pair_from_idx=pair_from,
        pair_to_idx=pair_to,
        obs_sku_idx=obs_sku_idx,
    )
    assert X.shape == (4, 2)
    # Focal a (sku 0): cross to b uses b's log price in the same (period,) cell.
    assert np.allclose(X[0, 0], np.log(20.0))
    assert np.allclose(X[1, 0], np.log(20.0))
    assert np.allclose(X[2, 1], np.log(10.0))
    assert np.allclose(X[3, 1], np.log(10.0))
