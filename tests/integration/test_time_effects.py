"""Short MCMC: period effects sample cleanly, round-trip, and predict."""

from __future__ import annotations

import numpy as np
import pytest

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data

from tests.conftest import SMOKE_CHAINS, SMOKE_DRAWS, SMOKE_TUNE


def test_period_effects_smoke_round_trip_and_prediction(tmp_path):
    df = generate_mock_data(
        n_periods=16,
        n_skus=4,
        n_controls=0,
        include_seasonality=False,
        price_shock_sigma=0.08,
        common_shock_sigma=0.25,
        common_endogeneity=1.0,
        random_state=0,
    )
    model = LogLogDemandModel(
        time_effects="period",
        mundlak=True,
        panel_columns=PanelColumns(quantity_floor=1e-12),
    )
    model.fit(
        df,
        draws=SMOKE_DRAWS,
        tune=SMOKE_TUNE,
        chains=SMOKE_CHAINS,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
        target_accept=0.95,
    )
    n_divergent = int(model.idata.sample_stats["diverging"].sum().item())
    assert n_divergent == 0, f"n_divergent={n_divergent}"
    assert "lambda_period" in model.idata.posterior
    assert "gamma_mundlak" in model.idata.posterior

    path = tmp_path / "period.nc"
    model.save(str(path))
    loaded = LogLogDemandModel.load(str(path))
    assert loaded.time_effects == "period"
    assert loaded.mundlak is True
    assert loaded.sku_mean_log_price_ is not None
    assert np.allclose(loaded.sku_mean_log_price_, model.sku_mean_log_price_)

    future = df.copy()
    future["period"] = 10_000
    neutral = loaded.predict(future, period_effect="neutral", random_seed=0)
    draw = loaded.predict(future, period_effect="draw", random_seed=0)
    assert np.allclose(
        neutral["log_quantity_mean"], draw["log_quantity_mean"], atol=0.08
    )
    neutral_width = neutral["quantity_hdi_upper"] - neutral["quantity_hdi_lower"]
    draw_width = draw["quantity_hdi_upper"] - draw["quantity_hdi_lower"]
    assert float(draw_width.mean()) > float(neutral_width.mean())

    with pytest.raises(ValueError, match="not a training period"):
        loaded.predict(future, period_effect=999)
