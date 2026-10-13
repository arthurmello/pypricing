"""Discount column: validation, regular price, promo term, optimizer."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import arviz as az
import xarray as xr

from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    PricePanelData,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
    apply_discount,
    generate_mock_data,
)
from pypricing.model_components.priors import (
    DEFAULT_PROMO_ELASTICITY_MU,
    DEFAULT_PROMO_ELASTICITY_SIGMA,
)
from pypricing.optimizer import optimize_prices

_MODELS = [
    LogLogDemandModel,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
]


def _frame(**extra) -> pd.DataFrame:
    data = {
        "sku": ["a", "a", "b", "b"],
        "price": [80.0, 100.0, 50.0, 40.0],
        "quantity": [10.0, 8.0, 12.0, 9.0],
    }
    data.update(extra)
    return pd.DataFrame(data)


def _params(rv):
    n_params = len(rv.owner.op.ndims_params)
    return [np.asarray(inp.eval(), dtype=float) for inp in rv.owner.inputs[-n_params:]]


def test_named_promo_prior():
    assert DEFAULT_PROMO_ELASTICITY_MU == -3.5
    assert DEFAULT_PROMO_ELASTICITY_SIGMA == 1.5


def test_discount_below_zero_shows_example_rows():
    df = _frame(discount_rate=[0.1, -0.2, 0.0, 0.1])
    with pytest.raises(ValueError, match=">= 0") as exc:
        PricePanelData.from_frame(df, panel_columns=PanelColumns(discount_col="discount_rate"))
    assert "-0.2" in str(exc.value)


def test_discount_at_least_one_shows_example_rows():
    df = _frame(discount_rate=[0.1, 1.0, 0.0, 0.1])
    with pytest.raises(ValueError, match="must be < 1") as exc:
        PricePanelData.from_frame(df, panel_columns=PanelColumns(discount_col="discount_rate"))
    assert "1.0" in str(exc.value) or "1" in str(exc.value)


def test_discount_nan_shows_example_rows():
    df = _frame(discount_rate=[0.1, np.nan, 0.0, 0.1])
    with pytest.raises(ValueError, match="NaN") as exc:
        PricePanelData.from_frame(df, panel_columns=PanelColumns(discount_col="discount_rate"))
    assert "NaN" in str(exc.value)


def test_discount_must_be_numeric():
    df = _frame(discount_rate=["20%", "0", "0", "0"])
    with pytest.raises(ValueError, match="numeric"):
        PricePanelData.from_frame(df, panel_columns=PanelColumns(discount_col="discount_rate"))


@pytest.mark.parametrize(
    "columns",
    [
        PanelColumns(discount_col="control_1", control_columns=("control_1",)),
        PanelColumns(discount_col="iv_cost", iv_columns=("iv_cost",)),
        PanelColumns(discount_col="category", group_columns=("category",)),
    ],
)
def test_discount_must_not_overlap_other_columns(columns):
    df = _frame(control_1=[0.1, 0.0, 0.2, 0.0], iv_cost=[0.0, 0.1, 0.0, 0.2], category=["x", "x", "y", "y"])
    with pytest.raises(ValueError, match="must not overlap"):
        PricePanelData.from_frame(df, panel_columns=columns)


def test_price_80_at_20_percent_off_is_regular_100():
    df = pd.DataFrame(
        {
            "sku": ["a"],
            "price": [80.0],
            "quantity": [5.0],
            "discount_rate": [0.2],
        }
    )
    panel = PricePanelData.from_frame(
        df, panel_columns=PanelColumns(discount_col="discount_rate")
    )
    assert np.allclose(panel.log_price, np.log(80.0))
    assert np.allclose(np.exp(panel.log_regular_price), 100.0)
    assert np.allclose(panel.log_discount_index, np.log(0.8))


def test_no_discount_column_keeps_paid_price_as_regular_price():
    df = _frame()
    panel = PricePanelData.from_frame(df)
    assert panel.log_discount_index is None
    assert panel.log_regular_price is panel.log_price


@pytest.mark.parametrize("cls", _MODELS, ids=[c.__name__ for c in _MODELS])
def test_no_discount_column_omits_promo_parameter(cls):
    df = generate_mock_data(
        n_periods=6,
        n_skus=2,
        n_controls=0,
        include_seasonality=False,
        random_state=0,
        shape={"LogLogDemandModel": "log_log", "QuadraticLogDemandModel": "quadratic", "SigmoidSaturationDemandModel": "sigmoid"}[cls.__name__],
    )
    model = cls(panel_columns=PanelColumns(quantity_floor=1e-12))
    model.build_model(df)
    names = set(model.model.named_vars)
    assert "elasticity_sku" in names
    assert "promo_elasticity_sku" not in names
    assert "mu_promo_elasticity" not in names


def test_promo_prior_and_hierarchy():
    df = generate_mock_data(
        n_periods=24,
        n_skus=4,
        hierarchy_levels=(2,),
        promotions=True,
        promo_frequency=0.5,
        promo_depth=0.2,
        include_seasonality=False,
        random_state=0,
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(
            discount_col="discount_rate",
            quantity_floor=1e-12,
            group_columns=("category_1",),
        )
    )
    model.build_model(df)
    assert "promo_elasticity_sku" in model.model.named_vars
    mu, sigma = _params(model.model["mu_promo_elasticity"])
    assert np.allclose(mu, -3.5)
    assert np.allclose(sigma, 1.5)


def test_flat_promo_prior():
    df = generate_mock_data(
        n_periods=16,
        n_skus=2,
        promotions=True,
        promo_frequency=0.5,
        include_seasonality=False,
        random_state=1,
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(discount_col="discount_rate", quantity_floor=1e-12)
    )
    model.build_model(df)
    rv = model.model["promo_elasticity_sku"]
    assert type(rv.owner.op).__name__ == "NormalRV"
    mu, sigma = _params(rv)
    assert np.allclose(mu, -3.5)
    assert np.allclose(sigma, 1.5)


def test_identification_warnings_and_diagnostics():
    df = pd.DataFrame(
        {
            "sku": [
                "no_promo",
                "no_promo",
                "flat_regular",
                "flat_regular",
                "ok",
                "ok",
            ],
            "price": [10.0, 12.0, 100.0, 50.0, 100.0, 40.0],
            "quantity": [5.0, 4.0, 5.0, 8.0, 5.0, 7.0],
            "discount_rate": [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
        }
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(discount_col="discount_rate", quantity_floor=1e-12)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.build_model(df)
    texts = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    promo_msg = " ".join(t for t in texts if "promo_elasticity_sku" in t)
    regular_msg = " ".join(t for t in texts if "group_columns" in t)
    assert "no_promo" in promo_msg
    assert "discount = 0 on every row" in promo_msg
    assert "flat_regular" in regular_msg
    assert "elasticity_sku" in regular_msg

    model.fit(
        df,
        draws=20,
        tune=20,
        chains=1,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    diag = model.run_diagnostics(
        fit=False, benchmarks=False, sensitivity=False, instruments=False
    )
    assert list(diag["no_discount_variation_skus"]) == ["no_promo"]
    assert list(diag["no_regular_price_variation_skus"]) == ["flat_regular"]


def test_apply_discount_keeps_price_and_discount_consistent():
    df = pd.DataFrame(
        {
            "sku": ["a", "a"],
            "price": [100.0, 80.0],
            "discount_rate": [0.0, 0.2],
            "quantity": [1.0, 1.0],
        }
    )
    original_price = df["price"].to_numpy().copy()
    out = apply_discount(df, 0.5)
    assert np.allclose(df["price"], original_price)
    regular = out["price"] / (1.0 - out["discount_rate"])
    assert np.allclose(regular, [100.0, 100.0])
    assert np.allclose(out["discount_rate"], 0.5)
    assert np.allclose(out["price"], [50.0, 50.0])

    cleared = apply_discount(out, 0.0)
    assert np.allclose(cleared["price"], 100.0)
    assert np.allclose(cleared["discount_rate"], 0.0)


def test_optimize_prices_uses_discount_zero_and_returns_regular_price():
    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "alpha_sku": (("chain", "draw", "sku"), np.zeros((1, 1, 1))),
                "elasticity_sku": (("chain", "draw", "sku"), np.array([[[-0.5]]])),
                "promo_elasticity_sku": (
                    ("chain", "draw", "sku"),
                    np.array([[[-8.0]]]),
                ),
                "sigma": (("chain", "draw"), np.array([[0.0]])),
            },
            coords={"chain": [0], "draw": [0], "sku": [0]},
        )
    )
    model = LogLogDemandModel(panel_columns=PanelColumns(discount_col="discount_rate"))
    model.idata = idata
    model.sku_levels_ = pd.Index(["a"])
    model.control_names_ = ()
    seen: dict[str, np.ndarray] = {}
    real = model.compute_mu_from_posterior

    def _spy(**kwargs):
        seen["log_discount_index"] = kwargs["log_discount_index"]
        seen["log_price"] = kwargs["log_price"]
        return real(**kwargs)

    model.compute_mu_from_posterior = _spy  # type: ignore[method-assign]
    out = optimize_prices(model=model, price_bounds={"a": (1.0, 10.0)})
    assert np.all(seen["log_discount_index"] == 0.0)
    assert out.loc["a", "optimal_price"] == pytest.approx(10.0, rel=1e-6)


def test_mock_promotions_match_regular_and_promo_elasticities():
    df, truth = generate_mock_data(
        n_periods=12,
        n_skus=2,
        promotions=True,
        promo_frequency=0.5,
        promo_depth=0.25,
        true_elasticity=-2.0,
        true_promo_elasticity=-3.5,
        noise_sigma=1e-8,
        include_seasonality=False,
        random_state=0,
        return_truth=True,
    )
    assert "discount_rate" in df.columns
    assert set(df["discount_rate"].unique()).issubset({0.0, 0.25})
    assert np.allclose(df["log_price"], np.log(df["price"]))
    regular = df["price"] / (1.0 - df["discount_rate"])
    assert np.all(truth.elasticity_sku == -2.0)
    assert truth.promo_elasticity == -3.5
    label_to_i = {lab: i for i, lab in enumerate(truth.sku_labels)}
    expected = []
    for row in df.itertuples(index=False):
        s = label_to_i[row.sku]
        expected.append(
            truth.alpha_sku[s]
            + truth.elasticity_sku[s] * np.log(row.price / (1.0 - row.discount_rate))
            + truth.promo_elasticity * np.log(1.0 - row.discount_rate)
        )
    assert np.allclose(df["log_quantity"], expected, atol=1e-6)
    assert np.all(regular > df["price"] - 1e-9)
