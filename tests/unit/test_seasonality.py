from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data
from pypricing.model_components.posterior_mu import add_season
from pypricing.model_components.time_terms import (
    fourier_features,
    infer_panel_frequency,
    normalize_seasonality,
    seasonality_components_for_frequency,
)

_AUTO_COLS = PanelColumns(quantity_floor=1e-12)


def _auto_recipe(dates) -> tuple[str, tuple[str, ...]]:
    idx = pd.DatetimeIndex(dates)
    freq = infer_panel_frequency(idx)
    return freq, seasonality_components_for_frequency(freq, idx)


def test_normalize_seasonality():
    assert normalize_seasonality(None) is None
    assert normalize_seasonality("auto") == "auto"
    assert normalize_seasonality("yearly") == ("yearly",)
    assert normalize_seasonality(["weekly", "yearly", "weekly"]) == (
        "weekly",
        "yearly",
    )
    with pytest.raises(ValueError, match="empty"):
        normalize_seasonality([])
    with pytest.raises(ValueError, match="auto"):
        normalize_seasonality(["auto", "yearly"])
    with pytest.raises(ValueError, match="components"):
        normalize_seasonality("monthly")
    with pytest.raises(ValueError, match="components"):
        LogLogDemandModel(seasonality="monthly")


@pytest.mark.parametrize(
    "dates, expected_freq, expected_components",
    [
        pytest.param(
            pd.date_range("2020-01-01", periods=20, freq="D"),
            "D",
            ("yearly", "weekly"),
            id="daily",
        ),
        pytest.param(
            pd.date_range("2020-01-01", periods=15, freq="B"),
            "D",
            ("yearly", "weekly"),
            id="business-day",
        ),
        pytest.param(
            pd.date_range("2020-01-01", periods=12, freq="2D"),
            "D",
            ("yearly", "weekly"),
            id="every-2-days-still-daily",
        ),
        pytest.param(
            pd.date_range("2020-01-06", periods=20, freq="W"),
            "W",
            ("yearly",),
            id="weekly-sunday",
        ),
        pytest.param(
            pd.date_range("2020-01-06", periods=16, freq="W-MON"),
            "W",
            ("yearly",),
            id="weekly-monday",
        ),
        pytest.param(
            pd.date_range("2020-01-03", periods=12, freq="5D"),
            "W",
            ("yearly",),
            id="every-5-days-weekly-band",
        ),
        pytest.param(
            pd.date_range("2020-01-01", periods=12, freq="MS"),
            "M",
            ("yearly",),
            id="month-start",
        ),
        pytest.param(
            pd.date_range("2020-01-31", periods=10, freq="ME"),
            "M",
            ("yearly",),
            id="month-end",
        ),
        pytest.param(
            pd.date_range("2020-01-01", periods=8, freq="QS"),
            "Q",
            ("yearly",),
            id="quarter-start",
        ),
        pytest.param(
            pd.DatetimeIndex(["2020-01-01"]),
            "irregular",
            ("yearly",),
            id="single-timestamp",
        ),
        pytest.param(
            pd.date_range("2020-01-06", periods=10, freq="14D"),
            "irregular",
            ("yearly",),
            id="biweekly-one-weekday",
        ),
        pytest.param(
            pd.date_range("2020-01-04", periods=8, freq="7D"),
            "W",
            ("yearly",),
            id="saturdays-are-weekly",
        ),
        pytest.param(
            pd.DatetimeIndex(
                [
                    "2020-01-06",
                    "2020-01-22",
                    "2020-03-05",
                    "2020-04-14",
                    "2020-07-02",
                    "2020-09-21",
                    "2020-11-11",
                ]
            ),
            "irregular",
            ("yearly", "weekly"),
            id="irregular-four-plus-weekdays",
        ),
        pytest.param(
            pd.DatetimeIndex(
                [
                    "2020-01-06",
                    "2020-02-12",
                    "2020-04-03",
                    "2020-06-15",
                    "2020-09-02",
                    "2020-11-20",
                ]
            ),
            "irregular",
            ("yearly",),
            id="irregular-three-weekdays",
        ),
        pytest.param(
            pd.DatetimeIndex(["2020-01-01", "2020-01-22", "2020-03-04", "2020-08-05"]),
            "irregular",
            ("yearly",),
            id="sparse-one-weekday",
        ),
        pytest.param(
            pd.date_range("2020-01-01", periods=48, freq="h"),
            "irregular",
            ("yearly",),
            id="hourly-two-days",
        ),
        pytest.param(
            pd.date_range("2020-01-06", periods=24 * 8, freq="h"),
            "irregular",
            ("yearly", "weekly"),
            id="hourly-spanning-weekdays",
        ),
        pytest.param(
            pd.date_range("2020-01-01", periods=10, freq="3D"),
            "irregular",
            ("yearly", "weekly"),
            id="every-3-days-irregular-many-weekdays",
        ),
    ],
)
def test_auto_frequency_and_components(dates, expected_freq, expected_components):
    freq, components = _auto_recipe(dates)
    assert freq == expected_freq
    assert components == expected_components


def test_auto_ignores_duplicates_and_order():
    base = pd.date_range("2020-01-01", periods=15, freq="D")
    mixed = np.concatenate([base.to_numpy(), base.to_numpy(), base[[0, 3]].to_numpy()])
    rng = np.random.default_rng(0)
    rng.shuffle(mixed)
    freq, components = _auto_recipe(pd.DatetimeIndex(mixed))
    assert freq == "D"
    assert components == ("yearly", "weekly")


def test_auto_weekday_threshold_is_four():
    three = pd.DatetimeIndex(
        ["2020-01-06", "2020-02-04", "2020-04-03", "2020-06-01", "2020-09-07"]
    )
    assert three.dayofweek.nunique() == 3
    assert _auto_recipe(three) == ("irregular", ("yearly",))

    four = pd.DatetimeIndex(
        ["2020-01-06", "2020-02-04", "2020-04-03", "2020-06-03", "2020-09-09"]
    )
    assert four.dayofweek.nunique() == 4
    assert _auto_recipe(four) == ("irregular", ("yearly", "weekly"))


def test_fourier_yearly_matches_mock_phase():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-07-02"])
    X, names = fourier_features(idx, ("yearly",))
    assert names[:2] == ("yearly_sin_1", "yearly_cos_1")
    phase = 2 * np.pi * (idx.dayofyear.to_numpy(dtype=float) - 1) / 365.25
    assert X[0, 0] == pytest.approx(np.sin(phase[0]))
    assert X[1, 0] == pytest.approx(np.sin(phase[1]))


def test_fourier_weekly_monday_is_zero_phase():
    idx = pd.DatetimeIndex(["2020-01-06"])  # Monday
    X, names = fourier_features(idx, ("weekly",))
    assert names == ("weekly_sin_1", "weekly_cos_1")
    assert X[0, 0] == pytest.approx(0.0)
    assert X[0, 1] == pytest.approx(1.0)


def test_seasonality_requires_period_column():
    df = generate_mock_data(
        n_periods=6,
        n_skus=2,
        random_state=0,
        start_date="2020-01-06",
        include_seasonality=False,
    ).drop(columns=["period"])
    with pytest.raises(ValueError, match="Missing period column"):
        LogLogDemandModel(seasonality="yearly").build_model(df)


def test_seasonality_only_counterfactual_period():
    df = generate_mock_data(
        n_periods=8,
        n_skus=2,
        random_state=0,
        start_date="2020-01-06",
        include_seasonality=False,
    )
    model = LogLogDemandModel(seasonality="yearly")
    model.build_model(df)
    assert model.t0_ is None
    assert model.default_counterfactual_period() == pd.Timestamp(df["period"].max())


def test_seasonality_rejects_integer_period():
    df = generate_mock_data(n_periods=6, n_skus=2, random_state=0)
    with pytest.raises(ValueError, match="numeric"):
        LogLogDemandModel(seasonality="yearly").build_model(df)


def _mock_panel(*, freq: str, n_periods: int, start_date: str) -> pd.DataFrame:
    return generate_mock_data(
        n_periods=n_periods,
        n_skus=2,
        random_state=0,
        start_date=start_date,
        freq=freq,
        include_seasonality=False,
    )


@pytest.mark.parametrize(
    "freq, n_periods, start_date, expected",
    [
        pytest.param("D", 21, "2020-01-01", ("yearly", "weekly"), id="daily"),
        pytest.param("B", 15, "2020-01-01", ("yearly", "weekly"), id="business-day"),
        pytest.param("W", 12, "2020-01-06", ("yearly",), id="weekly"),
        pytest.param("W-MON", 12, "2020-01-06", ("yearly",), id="weekly-monday"),
        pytest.param("MS", 12, "2020-01-01", ("yearly",), id="monthly"),
        pytest.param("QS", 8, "2020-01-01", ("yearly",), id="quarterly"),
    ],
)
def test_auto_resolves_on_mock_panel(freq, n_periods, start_date, expected):
    df = _mock_panel(freq=freq, n_periods=n_periods, start_date=start_date)
    model = LogLogDemandModel(seasonality="auto", panel_columns=_AUTO_COLS)
    model.build_model(df)
    assert model.seasonality_ == expected
    assert "beta_season" in model.model.named_vars


def test_auto_resolves_irregular_weekday_rich_panel():
    dates = pd.DatetimeIndex(
        [
            "2020-01-06",
            "2020-01-22",
            "2020-03-05",
            "2020-04-14",
            "2020-07-02",
            "2020-09-21",
            "2020-11-11",
        ]
    )
    df = _mock_panel(freq="D", n_periods=len(dates), start_date="2020-01-01")
    mapping = dict(zip(sorted(df["period"].unique()), dates, strict=True))
    df = df.copy()
    df["period"] = df["period"].map(mapping)
    model = LogLogDemandModel(seasonality="auto", panel_columns=_AUTO_COLS)
    model.build_model(df)
    assert model.seasonality_ == ("yearly", "weekly")


def test_add_season_dot():
    posterior = xr.Dataset(
        {
            "beta_season": (
                ("chain", "draw", "beta_season_dim_0"),
                np.array([[[0.2, -0.1]]], dtype=np.float64),
            )
        },
        coords={"chain": [0], "draw": [0], "beta_season_dim_0": [0, 1]},
    )
    mu = np.zeros((1, 1, 2), dtype=np.float64)
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    out = add_season(mu, posterior, X)
    assert out[0, 0, 0] == pytest.approx(0.2)
    assert out[0, 0, 1] == pytest.approx(-0.1)


def test_compute_mu_includes_season():
    X = np.array([[1.0, 0.0]], dtype=np.float64)
    posterior = xr.Dataset(
        {
            "alpha_sku": (
                ("chain", "draw", "sku"),
                np.array([[[0.0]]], dtype=np.float64),
            ),
            "elasticity_sku": (
                ("chain", "draw", "sku"),
                np.array([[[-1.0]]], dtype=np.float64),
            ),
            "beta_season": (
                ("chain", "draw", "beta_season_dim_0"),
                np.array([[[0.15, 0.0]]], dtype=np.float64),
            ),
        },
        coords={
            "chain": [0],
            "draw": [0],
            "sku": [0],
            "beta_season_dim_0": [0, 1],
        },
    )
    model = LogLogDemandModel(seasonality="yearly")
    mu = model.compute_mu_from_posterior(
        posterior=posterior,
        log_price=np.array([0.0], dtype=np.float64),
        obs_sku_idx=np.array([0], dtype=np.int64),
        X_control=None,
        X_season=X,
    )
    assert mu[0, 0, 0] == pytest.approx(0.15)
