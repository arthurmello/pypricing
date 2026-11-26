"""Aggregate a SKU panel to coarser units (e.g. brand x category) before fitting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pypricing.data import PanelColumns
from pypricing.model_components.promo_extras import NO_PROMO_TYPE


def aggregate_panel(
    df: pd.DataFrame,
    by: Sequence[str],
    *,
    panel_columns: PanelColumns | None = None,
    unit_cost_col: str | None = "unit_cost",
    type_col: str | None = None,
    sep: str = "|",
) -> pd.DataFrame:
    """One row per ``by`` combination, period (and region).

    Prices are averaged with fixed weights: each SKU's mean quantity over the
    whole frame, so a change in the mix of SKUs sold does not show up as a
    price change. ``price`` is the weighted paid price, ``discount`` the
    weighted discount (``1 - paid / regular``), ``quantity`` the sum, unit
    cost and control columns weighted means. ``type_col`` takes the most
    common promo type among discounted SKUs. Columns that are constant within
    each ``by`` group are kept. The new SKU label joins the ``by`` values with
    ``sep``.
    """
    cols = panel_columns or PanelColumns()
    by = list(by)
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise ValueError(f"by columns not in frame: {missing}")
    keys = [*by, cols.period_col]
    if cols.region_col is not None and cols.region_col in df.columns:
        keys.append(cols.region_col)

    work = df.copy()
    weight = (
        work.groupby(cols.sku_col)[cols.quantity_col].transform("mean").clip(lower=1e-9)
    )
    paid = work[cols.price_col].to_numpy(dtype=np.float64)
    if cols.discount_col is not None and cols.discount_col in work.columns:
        d = work[cols.discount_col].to_numpy(dtype=np.float64)
    else:
        d = np.zeros(len(work))
    work["_w"] = weight
    work["_paid_w"] = paid * weight
    work["_reg_w"] = paid / (1.0 - d) * weight

    control_cols = list(
        cols.control_columns or [c for c in df.columns if c.startswith("control_")]
    )
    mean_cols = [c for c in [unit_cost_col, *control_cols] if c and c in work.columns]
    for c in mean_cols:
        work[f"_{c}_w"] = work[c].to_numpy(dtype=np.float64) * weight

    grouped = work.groupby(keys, sort=True, dropna=False)
    out = grouped.agg(
        **{
            cols.quantity_col: (cols.quantity_col, "sum"),
            "_w": ("_w", "sum"),
            "_paid_w": ("_paid_w", "sum"),
            "_reg_w": ("_reg_w", "sum"),
            **{f"_{c}_w": (f"_{c}_w", "sum") for c in mean_cols},
        }
    ).reset_index()
    out[cols.price_col] = out["_paid_w"] / out["_w"]
    regular = out["_reg_w"] / out["_w"]
    if cols.discount_col is not None:
        out[cols.discount_col] = (1.0 - out[cols.price_col] / regular).clip(lower=0.0)
    for c in mean_cols:
        out[c] = out[f"_{c}_w"] / out["_w"]

    if type_col is not None and type_col in work.columns:
        on = work[d > 0]
        mode = (
            on.groupby(keys, dropna=False)[type_col]
            .agg(lambda s: s.value_counts().index[0])
            .rename(type_col)
        )
        out = out.merge(mode.reset_index(), on=keys, how="left")
        out[type_col] = out[type_col].fillna(NO_PROMO_TYPE)
        if cols.discount_col is not None:
            out.loc[out[cols.discount_col] <= 0, type_col] = NO_PROMO_TYPE

    used = set(keys) | {
        cols.sku_col,
        cols.price_col,
        cols.quantity_col,
        type_col,
        *mean_cols,
    }
    if cols.discount_col:
        used.add(cols.discount_col)
    for c in df.columns:
        if c in used:
            continue
        if (df.groupby(by, dropna=False)[c].nunique(dropna=False) <= 1).all():
            first = df.groupby(by, dropna=False)[c].first().rename(c).reset_index()
            out = out.merge(first, on=by, how="left")

    out[cols.sku_col] = out[by].astype(str).agg(sep.join, axis=1)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])
