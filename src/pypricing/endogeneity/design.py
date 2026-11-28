"""Exogenous columns of the price equation, shared by every endogeneity method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel


def log_regular_price(model: DemandModel) -> np.ndarray:
    """Log regular price in training-row order.

    With no discount column this is log paid price. The discount index is not
    part of the endogenous regressor.
    """
    from pypricing.data.price_panel import decompose_log_price

    assert model.data is not None
    price = model.data[model.price_col].to_numpy(dtype=np.float64)
    discount = None
    discount_col = model.panel_columns.discount_col
    if discount_col is not None:
        discount = model.data[discount_col].to_numpy(dtype=np.float64)
    _paid, regular, _index = decompose_log_price(price, discount)
    return regular


def exogenous_design(model: DemandModel) -> np.ndarray:
    """SKU intercepts, controls, trend, seasonality, and period effects.

    Excluded instruments are not in this matrix. Instrument diagnostics
    return the same columns from ``first_stage_exogenous``.
    """
    assert model.data is not None and model.sku_levels_ is not None
    data = model.data
    sku = pd.Categorical(data[model.sku_col], categories=model.sku_levels_)
    dummies = pd.get_dummies(sku).to_numpy(dtype=np.float64)
    parts: list[np.ndarray] = [dummies]
    if model.control_names_:
        parts.append(
            data.loc[:, list(model.control_names_)].to_numpy(dtype=np.float64)
        )
    if model.trend is not None:
        t = model._t_years_for_frame(data).reshape(-1, 1)
        parts.append(dummies * t if model.trend == "sku" else t)
    season = model._season_features_for_frame(data)
    if season is not None and season.size:
        parts.append(season)
    if model.time_effects is not None and model.period_col in data.columns:
        if (
            model.time_effects == "region_period"
            and model.region_col is not None
            and model.region_col in data.columns
        ):
            cell = (
                data[model.region_col].astype(str)
                + "|"
                + data[model.period_col].astype(str)
            )
            parts.append(pd.get_dummies(cell).to_numpy(dtype=np.float64))
        elif model.time_effects == "group_period" and model.time_group_codes_ is not None:
            sku_idx = pd.Categorical(
                data[model.sku_col], categories=model.sku_levels_
            ).codes
            cell = (
                pd.Series(model.time_group_codes_[sku_idx]).astype(str).to_numpy()
                + "|"
                + data[model.period_col].astype(str).to_numpy()
            )
            parts.append(pd.get_dummies(cell).to_numpy(dtype=np.float64))
        else:
            period = pd.Categorical(data[model.period_col])
            parts.append(pd.get_dummies(period).to_numpy(dtype=np.float64))
    return np.column_stack(parts)
