"""Integration smoke: cross-elasticity fits and config validation."""

from __future__ import annotations

import pytest

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data
from pypricing.model_components.cross_elasticity import CrossElasticitySpec


def test_cross_elasticity_all_log_log_fit():
    df = generate_mock_data(
        n_periods=8,
        n_skus=3,
        cross_elasticity="all",
        random_state=2,
    )
    m = LogLogDemandModel(cross_elasticity=CrossElasticitySpec(mode="all"))
    idata = m.fit(
        df,
        draws=60,
        tune=60,
        chains=2,
        random_seed=1,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "gamma_pair" in idata.posterior
    ax_all = m.plot_cross_price_effects_heatmap(agg="median")
    assert ax_all is not None


def test_cross_elasticity_within_group_hierarchy_fit():
    df = generate_mock_data(
        n_periods=6,
        n_skus=4,
        hierarchy_levels=(2, 2),
        cross_elasticity="within_group",
        cross_elasticity_group_level=0,
        random_state=0,
    )
    m = LogLogDemandModel(
        panel_columns=PanelColumns(group_columns=("category_1", "category_2")),
        cross_elasticity=CrossElasticitySpec(mode="within_group", group_level=0),
    )
    idata = m.fit(
        df,
        draws=50,
        tune=50,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "gamma_pair" in idata.posterior
    ax_h = m.plot_cross_price_effects_heatmap(agg="mean")
    assert ax_h is not None


def test_cross_elasticity_invalid_config_raises():
    with pytest.raises(ValueError, match="group_level"):
        LogLogDemandModel(
            cross_elasticity=CrossElasticitySpec(mode="all", group_level=0),
        )
    with pytest.raises(ValueError, match="within_group"):
        LogLogDemandModel(
            cross_elasticity=CrossElasticitySpec(mode="within_group", group_level=0),
        )


def test_plot_cross_price_effects_heatmap_requires_cross():
    df = generate_mock_data(
        n_periods=6,
        n_skus=2,
        n_controls=0,
        random_state=1,
    )
    m = LogLogDemandModel()
    m.fit(
        df,
        draws=30,
        tune=30,
        chains=2,
        random_seed=1,
        progressbar=False,
        compute_convergence_checks=False,
    )
    with pytest.raises(ValueError, match="cross_elasticity"):
        m.plot_cross_price_effects_heatmap()
