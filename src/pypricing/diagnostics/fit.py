from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel

BUCKET_LABELS = ("low", "mid", "high")
MIN_BUCKET_ROWS = 10

_BUCKET_COL = "_fit_bucket"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def hdi_coverage(y_true, lower, upper) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def rank_within_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    reference: pd.DataFrame | None = None,
) -> pd.Series:
    """Share of the group's ``reference`` values at or below each row's value.

    ``reference`` defaults to ``df`` itself (in-sample). For out-of-sample
    rows, pass the training frame so ranks are relative to the history the
    model was fitted on; values beyond that range rank 0 or 1.
    """
    reference = df if reference is None else reference
    ref_sorted = {
        key: np.sort(values.to_numpy(dtype=float))
        for key, values in reference.groupby(group_col, observed=True)[value_col]
    }
    missing = set(df[group_col].unique()) - ref_sorted.keys()
    if missing:
        raise ValueError(
            f"No reference {value_col!r} values for {group_col} "
            f"{sorted(map(str, missing))}."
        )

    ranks = pd.Series(np.nan, index=df.index, dtype=float)
    for key, idx in df.groupby(group_col, observed=True).groups.items():
        ref = ref_sorted[key]
        values = df.loc[idx, value_col].to_numpy(dtype=float)
        ranks.loc[idx] = np.searchsorted(ref, values, side="right") / len(ref)
    return ranks


def assign_buckets(ranks: pd.Series) -> pd.Series:
    edges = np.linspace(0.0, 1.0, len(BUCKET_LABELS) + 1)
    return pd.cut(ranks, bins=edges, labels=BUCKET_LABELS, include_lowest=True)


def _metrics(frame, quantity_col, quantity_floor) -> dict:
    y = frame[quantity_col].to_numpy(dtype=float)
    log_y = np.log(np.maximum(y, quantity_floor))
    log_pred = frame["log_quantity_mean"].to_numpy(dtype=float)
    return {
        "n": len(frame),
        "rmse": rmse(y, frame["quantity_mean"].to_numpy(dtype=float)),
        "bias_log": float(np.mean(log_y - log_pred)),
        "rmse_log": rmse(log_y, log_pred),
        "hdi_coverage": hdi_coverage(
            y,
            frame["quantity_hdi_lower"].to_numpy(dtype=float),
            frame["quantity_hdi_upper"].to_numpy(dtype=float),
        ),
    }


def fit_metrics(
    pred: pd.DataFrame,
    *,
    quantity_col: str,
    bucket_col: str | None = None,
    quantity_floor: float = 1.0,
    min_rows: int = MIN_BUCKET_ROWS,
) -> pd.DataFrame:
    """Metrics for all rows (``overall``) plus one row per bucket.

    Buckets with fewer than ``min_rows`` rows keep their ``n`` but get NaN
    metrics, since a handful of rows can't support a comparison.
    """
    if pred.empty:
        raise ValueError("Cannot compute fit metrics on an empty frame.")
    rows = {"overall": _metrics(pred, quantity_col, quantity_floor)}
    if bucket_col is not None:
        for name, group in pred.groupby(bucket_col, observed=False):
            if len(group) < min_rows:
                rows[name] = {"n": len(group)}
            else:
                rows[name] = _metrics(group, quantity_col, quantity_floor)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["n"] = out["n"].astype(int)
    return out


def check_fit(
    model: DemandModel,
    df: pd.DataFrame | None = None,
    *,
    by: str | None = None,
    hdi_prob: float = 0.94,
    min_rows: int = MIN_BUCKET_ROWS,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Fit metrics bucketed by ``by`` (default: price), ranked within SKU.

    Ranks are always relative to the model's training frame, so on a test
    frame "high" means high compared with the prices the model learned from.
    """
    model._require_fitted()
    if model.data is None:
        raise RuntimeError("Model has no training data to rank against.")
    by = model.price_col if by is None else by
    work = model.data if df is None else df
    for frame, name in ((work, "df"), (model.data, "training data")):
        if by not in frame.columns:
            raise ValueError(f"Column {by!r} not found in {name}.")

    pred = model.predict(work, hdi_prob=hdi_prob, random_seed=random_seed)
    ranks = rank_within_group(pred, by, model.sku_col, reference=model.data)
    pred[_BUCKET_COL] = assign_buckets(ranks)
    return fit_metrics(
        pred,
        quantity_col=model.quantity_col,
        bucket_col=_BUCKET_COL,
        quantity_floor=model.quantity_floor,
        min_rows=min_rows,
    )
