from __future__ import annotations

import pytest

from pypricing.diagnostics.sensitivity import partial_r2, robustness_value

# Cinelli & Hazlett (2020), Darfur example: "directly harmed", t = 4.18, df = 783.
DARFUR_T = 4.18
DARFUR_DF = 783


def test_robustness_value_matches_paper():
    assert robustness_value(DARFUR_T, DARFUR_DF) == pytest.approx(0.139, abs=1e-3)


def test_robustness_value_for_significance_matches_paper():
    rv_qa = robustness_value(DARFUR_T, DARFUR_DF, alpha=0.05)
    assert rv_qa == pytest.approx(0.076, abs=1e-3)


def test_partial_r2_matches_paper():
    assert partial_r2(DARFUR_T, DARFUR_DF) == pytest.approx(0.022, abs=1e-3)


def test_robustness_value_ignores_sign_and_scales_with_q():
    assert robustness_value(-DARFUR_T, DARFUR_DF) == robustness_value(
        DARFUR_T, DARFUR_DF
    )
    assert robustness_value(DARFUR_T, DARFUR_DF, q=0.5) < robustness_value(
        DARFUR_T, DARFUR_DF
    )


def test_insignificant_estimate_has_zero_rv_for_significance():
    assert robustness_value(1.0, 500, alpha=0.05) == 0.0
    assert robustness_value(1.0, 500) > 0.0


def test_very_strong_estimate_uses_extreme_case():
    rv_qa = robustness_value(60.0, 100, alpha=0.05)
    assert 0.0 < rv_qa < 1.0
