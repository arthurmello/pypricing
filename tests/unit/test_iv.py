"""Unit tests for control-function IV graph construction (no MCMC)."""

from __future__ import annotations

import pytest

from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
    generate_mock_data,
)


@pytest.mark.parametrize("model_cls", [LogLogDemandModel, QuadraticLogDemandModel])
def test_iv_graph_has_price_equation(model_cls):
    df = generate_mock_data(
        n_periods=8,
        n_skus=2,
        n_instruments=1,
        include_seasonality=False,
        round_quantity=False,
        random_state=0,
        shape="log_log" if model_cls is LogLogDemandModel else "quadratic",
    )
    model = model_cls(panel_columns=PanelColumns(quantity_floor=1e-12))
    model.build_model(df)
    free = {v.name for v in model.model.free_RVs}
    observed = {v.name for v in model.model.observed_RVs}
    assert "rho" in free
    assert "pi" in free
    assert "sigma_price" in free
    assert "obs" in observed
    assert "obs_price" in observed
    assert model.iv_names_ == ("iv_1",)


def test_sigmoid_iv_raises_before_sampling():
    df = generate_mock_data(
        n_periods=6,
        n_skus=2,
        n_instruments=1,
        include_seasonality=False,
        round_quantity=False,
        random_state=0,
    )
    with pytest.raises(ValueError, match="not supported"):
        SigmoidSaturationDemandModel(
            panel_columns=PanelColumns(quantity_floor=1e-12)
        ).build_model(df)
