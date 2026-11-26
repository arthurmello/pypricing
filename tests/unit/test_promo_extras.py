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
