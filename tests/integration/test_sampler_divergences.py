"""Hierarchical and cross-price fits stay nearly divergence-free at target_accept=0.95."""

from __future__ import annotations

from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data
from pypricing.model_components.cross_elasticity import CrossElasticitySpec

_DRAWS = 200
_TUNE = 400
_CHAINS = 2
# Same absolute cap as the log-log recovery diagnostics.
_MAX_DIVERGENCES = 10


def _assert_few_divergences(model: LogLogDemandModel) -> None:
    diag = model.run_diagnostics()
    n_divergent = diag["n_divergent"]
    assert n_divergent is not None
    assert n_divergent <= _MAX_DIVERGENCES, (
        f"n_divergent={n_divergent} exceeds {_MAX_DIVERGENCES}"
    )


def test_hierarchical_fit_few_divergences():
    df = generate_mock_data(
        n_periods=40,
        n_skus=8,
        hierarchy_levels=(2, 2),
        n_controls=0,
        random_state=0,
        shape="log_log",
        include_seasonality=False,
        price_shock_sigma=0.08,
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(group_columns=("category_1", "category_2")),
    )
    model.fit(
        df,
        draws=_DRAWS,
        tune=_TUNE,
        chains=_CHAINS,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
        target_accept=0.95,
    )
    assert model.idata is not None
    assert "mu_elasticity" in model.idata.posterior
    _assert_few_divergences(model)


def test_cross_price_fit_few_divergences():
    df = generate_mock_data(
        n_periods=24,
        n_skus=4,
        hierarchy_levels=(2, 2),
        cross_elasticity="within_group",
        cross_elasticity_group_level=0,
        n_controls=0,
        random_state=1,
        shape="log_log",
        include_seasonality=False,
        price_shock_sigma=0.08,
    )
    model = LogLogDemandModel(
        panel_columns=PanelColumns(group_columns=("category_1", "category_2")),
        cross_elasticity=CrossElasticitySpec(mode="within_group", group_level=0),
    )
    model.fit(
        df,
        draws=_DRAWS,
        tune=_TUNE,
        chains=_CHAINS,
        random_seed=1,
        progressbar=False,
        compute_convergence_checks=False,
        target_accept=0.95,
    )
    assert model.idata is not None
    assert "gamma_pair" in model.idata.posterior
    _assert_few_divergences(model)
