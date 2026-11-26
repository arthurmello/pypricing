import numpy as np
import pandas as pd
import pytest

from pypricing.model_components.promo_extras import PromoSpec, build_extra_features


def test_type_onehot_reference_rows():
    df = pd.DataFrame({"t": ["featured", None, "bundle", "featured", ""]})
    log_idx = np.log(1 - np.array([0.2, 0.3, 0.1, 0.0, 0.2]))
    feats = build_extra_features(
        spec=PromoSpec(type_col="t"), log_discount_index=log_idx, n_obs=5, df=df
    )
    assert feats.type_levels == ("bundle", "featured")
    np.testing.assert_array_equal(
        feats.type_onehot, [[0, 1], [0, 0], [1, 0], [0, 0], [0, 0]]
    )


def test_quadratic_depth_feature():
    log_idx = np.log(1 - np.array([0.0, 0.5]))
    feats = build_extra_features(
        spec=PromoSpec(depth="quadratic"), log_discount_index=log_idx, n_obs=2
    )
    np.testing.assert_allclose(feats.depth_sq, [0.0, np.log(2) ** 2])


def test_bad_depth():
    with pytest.raises(ValueError):
        PromoSpec(depth="cubic")


def test_lagged_depth_by_sku_and_period():
    from pypricing import PanelColumns

    df = pd.DataFrame(
        {
            "sku": ["a", "b", "a", "b", "a"],
            "period": [2, 1, 1, 2, 3],
        }
    )
    d = np.array([0.0, 0.5, 0.2, 0.0, 0.1])
    feats = build_extra_features(
        spec=PromoSpec(lags=2),
        log_discount_index=np.log(1 - d),
        n_obs=5,
        df=df,
        panel_columns=PanelColumns(),
    )
    z = -np.log(1 - d)
    expected = np.array(
        [
            [z[2], 0.0],  # a@2: a@1
            [0.0, 0.0],  # b@1
            [0.0, 0.0],  # a@1
            [z[1], 0.0],  # b@2: b@1
            [z[0], z[2]],  # a@3: a@2, a@1
        ]
    )
    np.testing.assert_allclose(feats.lag_matrix, expected)


def test_group_pressure_excludes_own_row():
    from pypricing import PanelColumns

    df = pd.DataFrame({"sku": ["a", "b", "c", "a"], "period": [1, 1, 1, 2]})
    d = np.array([0.2, 0.4, 0.0, 0.3])
    z = -np.log(1 - d)
    feats = build_extra_features(
        spec=PromoSpec(pressure_level=0),
        log_discount_index=np.log(1 - d),
        n_obs=4,
        df=df,
        panel_columns=PanelColumns(),
        row_group=np.array([0, 0, 0, 0]),
    )
    np.testing.assert_allclose(
        feats.pressure, [(z[1] + z[2]) / 2, (z[0] + z[2]) / 2, (z[0] + z[1]) / 2, 0.0]
    )
