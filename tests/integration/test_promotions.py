"""Fit, predict, and save/load with a discount column on all three curves."""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
    generate_mock_data,
)

_MODELS = [
    (LogLogDemandModel, "log_log"),
    (QuadraticLogDemandModel, "quadratic"),
    (SigmoidSaturationDemandModel, "sigmoid"),
]


@pytest.mark.parametrize("cls,shape", _MODELS, ids=[s for _, s in _MODELS])
def test_fit_predict_save_load_with_discounts(cls, shape, tmp_path):
    df = generate_mock_data(
        n_periods=16,
        n_skus=3,
        promotions=True,
        promo_frequency=0.5,
        promo_depth=0.2,
        include_seasonality=False,
        random_state=0,
        shape=shape,
    )
    model = cls(
        panel_columns=PanelColumns(discount_col="discount_rate", quantity_floor=1e-12)
    )
    model.fit(
        df,
        draws=40,
        tune=40,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "promo_elasticity_sku" in model.idata.posterior
    summary = model.fit_summary()
    assert any("promo_elasticity_sku" in str(name) for name in summary.index)

    pred = df.drop(columns=["quantity"])
    out = model.predict(pred, hdi_prob=0.9, random_seed=1)
    assert np.isfinite(out["quantity_mean"]).all()
    assert len(out) == len(pred)

    with pytest.raises(ValueError, match="discount column"):
        model.predict(pred.drop(columns=["discount_rate"]))

    axes = model.plot_elasticity_posterior()
    assert len(np.atleast_1d(axes)) == 2

    path = tmp_path / f"{shape}.nc"
    model.save(str(path))
    loaded = cls.load(str(path))
    assert loaded.panel_columns.discount_col == "discount_rate"
    out_loaded = loaded.predict(pred, hdi_prob=0.9, random_seed=1)
    assert np.allclose(out["quantity_mean"], out_loaded["quantity_mean"])
