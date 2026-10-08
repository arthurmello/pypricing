from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pypricing.diagnostics.fit import (
    BUCKET_LABELS,
    assign_buckets,
    fit_metrics,
    rank_within_group,
)


def _pred_frame(log_offset: float, n_per_sku: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2 * n_per_sku
    quantity = rng.uniform(5.0, 50.0, n)
    return pd.DataFrame(
        {
            "sku": np.repeat(["a", "b"], n_per_sku),
            "price": np.tile(np.arange(1.0, n_per_sku + 1), 2),
            "quantity": quantity,
            "quantity_mean": quantity,
            "log_quantity_mean": np.log(quantity) - log_offset,
            "quantity_hdi_lower": quantity - 1.0,
            "quantity_hdi_upper": quantity + 1.0,
        }
    )


def test_rank_within_group_in_sample_is_per_sku():
    df = pd.DataFrame(
        {"sku": ["a", "a", "a", "b", "b", "b"], "price": [1, 2, 3, 100, 200, 300]}
    )
    ranks = rank_within_group(df, "price", "sku")
    np.testing.assert_allclose(ranks, [1 / 3, 2 / 3, 1, 1 / 3, 2 / 3, 1])


def test_rank_within_group_uses_reference_out_of_sample():
    train = pd.DataFrame({"sku": ["a"] * 4, "price": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"sku": ["a"] * 3, "price": [0.5, 2.5, 9.0]})
    ranks = rank_within_group(test, "price", "sku", reference=train)
    np.testing.assert_allclose(ranks, [0.0, 0.5, 1.0])


def test_rank_within_group_raises_for_sku_missing_from_reference():
    train = pd.DataFrame({"sku": ["a"], "price": [1.0]})
    test = pd.DataFrame({"sku": ["a", "z"], "price": [1.0, 1.0]})
    with pytest.raises(ValueError, match="No reference"):
        rank_within_group(test, "price", "sku", reference=train)


def test_assign_buckets_splits_into_thirds():
    ranks = pd.Series([0.0, 0.2, 1 / 3, 0.5, 2 / 3, 0.9, 1.0])
    buckets = assign_buckets(ranks)
    assert list(buckets) == ["low", "low", "low", "mid", "mid", "high", "high"]


def test_fit_metrics_overall_row_and_known_values():
    pred = _pred_frame(log_offset=0.1)
    out = fit_metrics(pred, quantity_col="quantity")
    assert list(out.index) == ["overall"]
    row = out.loc["overall"]
    assert row["n"] == len(pred)
    assert row["rmse"] == pytest.approx(0.0)
    assert row["bias_log"] == pytest.approx(0.1)
    assert row["rmse_log"] == pytest.approx(0.1)
    assert row["hdi_coverage"] == pytest.approx(1.0)


def test_fit_metrics_by_bucket_has_all_labels_in_order():
    pred = _pred_frame(log_offset=0.0)
    pred["bucket"] = assign_buckets(rank_within_group(pred, "price", "sku"))
    out = fit_metrics(pred, quantity_col="quantity", bucket_col="bucket")
    assert list(out.index) == ["overall", *BUCKET_LABELS]
    assert out.loc[list(BUCKET_LABELS), "n"].sum() == len(pred)


def test_fit_metrics_small_and_empty_buckets_get_nan_metrics():
    pred = _pred_frame(log_offset=0.0)
    pred["bucket"] = pd.Categorical(
        ["low"] * (len(pred) - 3) + ["mid"] * 3, categories=BUCKET_LABELS
    )
    out = fit_metrics(pred, quantity_col="quantity", bucket_col="bucket", min_rows=5)
    assert out.loc["mid", "n"] == 3
    assert np.isnan(out.loc["mid", "rmse_log"])
    assert out.loc["high", "n"] == 0
    assert np.isnan(out.loc["high", "hdi_coverage"])
    assert not np.isnan(out.loc["low", "rmse_log"])


def test_fit_metrics_raises_on_empty_frame():
    pred = _pred_frame(log_offset=0.0).iloc[0:0]
    with pytest.raises(ValueError, match="empty"):
        fit_metrics(pred, quantity_col="quantity")
