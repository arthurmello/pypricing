from __future__ import annotations

import pandas as pd

from pypricing import CrossElasticitySpec, PanelColumns, QuadraticLogDemandModel


def _model() -> QuadraticLogDemandModel:
    return QuadraticLogDemandModel(
        panel_columns=PanelColumns(quantity_floor=0.5),
        cross_elasticity=CrossElasticitySpec(mode="all"),
        trend="shared",
        seasonality="yearly",
        sampler_config={"draws": 123},
    )


def test_clone_keeps_configuration_and_is_unfitted():
    model = _model()
    model.idata = object()
    copy = model.clone()

    assert type(copy) is QuadraticLogDemandModel
    assert copy.panel_columns == model.panel_columns
    assert copy.cross_elasticity == model.cross_elasticity
    assert copy.trend == "shared"
    assert copy.seasonality == model.seasonality
    assert copy.sampler_config["draws"] == 123
    assert copy.idata is None


def test_clone_pins_controls_once_fitted_and_applies_overrides():
    model = _model()
    model.sku_levels_ = pd.Index(["a", "b"])
    model.control_names_ = ("control_1",)

    assert model.clone().panel_columns.control_columns == ("control_1",)
    copy = model.clone(control_columns=("x",), iv_columns=())
    assert copy.panel_columns.control_columns == ("x",)
    assert copy.panel_columns.iv_columns == ()
    assert model.panel_columns.control_columns is None


def test_clone_of_unfitted_model_keeps_auto_detection():
    assert _model().clone().panel_columns.control_columns is None
