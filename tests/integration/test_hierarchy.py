"""Integration smoke: hierarchical partial pooling fits."""

from __future__ import annotations

from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
    generate_mock_data,
)


def test_group_columns_hierarchical_fit():
    df = generate_mock_data(
        n_periods=12,
        n_skus=6,
        hierarchy_levels=(2, 3),
        n_controls=1,
        random_state=0,
    )
    assert "category_1" in df.columns and "category_2" in df.columns
    idata = LogLogDemandModel(
        panel_columns=PanelColumns(group_columns=("category_1", "category_2")),
    ).fit(
        df,
        draws=60,
        tune=60,
        chains=2,
        random_seed=42,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "elasticity_sku" in idata.posterior
    assert "mu_elasticity" in idata.posterior


def test_group_columns_quadratic_and_sigmoid_smoke():
    df = generate_mock_data(
        n_periods=10,
        n_skus=5,
        hierarchy_levels=(2, 2),
        n_controls=1,
        random_state=3,
    )
    q = QuadraticLogDemandModel(
        panel_columns=PanelColumns(group_columns=("category_1", "category_2"))
    )
    q.fit(
        df,
        draws=50,
        tune=50,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "curvature_sku" in q.idata.posterior
    assert "mu_curvature" in q.idata.posterior

    df2 = generate_mock_data(
        n_periods=10,
        n_skus=5,
        hierarchy_levels=(2, 2),
        shape="sigmoid",
        random_state=1,
    )
    s = SigmoidSaturationDemandModel(
        panel_columns=PanelColumns(group_columns=("category_1", "category_2"))
    )
    s.fit(
        df2,
        draws=50,
        tune=50,
        chains=2,
        random_seed=0,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "elasticity_sku" in s.idata.posterior
