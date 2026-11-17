from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pypricing.diagnostics import DiagnosticWarning
from pypricing.diagnostics.identification import check_instruments, sargan_test


def _simulate(n: int, *, direct_effect: float, seed: int):
    """Demand with endogenous price; ``direct_effect`` makes z2 an invalid instrument."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    demand_shock = rng.normal(size=n)
    log_p = 0.6 * z[:, 0] + 0.6 * z[:, 1] + 0.5 * demand_shock + rng.normal(
        scale=0.3, size=n
    )
    log_q = 2.0 - 1.5 * log_p + direct_effect * z[:, 1] + demand_shock
    return log_q, log_p, z, np.ones((n, 1))


def test_sargan_does_not_reject_valid_instruments():
    out = sargan_test(*_simulate(2000, direct_effect=0.0, seed=0))
    assert out["df"] == 1
    assert out["p_value"] > 0.05


def test_sargan_rejects_when_an_instrument_affects_demand_directly():
    out = sargan_test(*_simulate(2000, direct_effect=0.8, seed=0))
    assert out["p_value"] < 0.01


def _instrument_model() -> SimpleNamespace:
    return SimpleNamespace(iv_names_=("iv_1",), _require_fitted=lambda: None)


def _stub_instrument_checks(monkeypatch: pytest.MonkeyPatch, *, includes_zero: bool) -> None:
    lo, hi = (-0.4, 0.2) if includes_zero else (0.2, 0.8)
    monkeypatch.setattr(
        "pypricing.diagnostics.identification.iv_diagnostics",
        lambda model, hdi_prob=0.9: {
            "first_stage_f": 20.0,
            "weak_iv": False,
            "rho_mean": 0.1,
            "rho_hdi": (lo, hi),
            "rho_hdi_includes_zero": includes_zero,
        },
    )
    monkeypatch.setattr(
        "pypricing.diagnostics.identification._overid",
        lambda model: {"applicable": False, "n_instruments": 1},
    )


def test_check_instruments_warns_when_rho_includes_zero(monkeypatch):
    _stub_instrument_checks(monkeypatch, includes_zero=True)
    with pytest.warns(DiagnosticWarning, match="No evidence of endogeneity"):
        check_instruments(_instrument_model(), compare_ols=False)


def test_check_instruments_silent_when_rho_excludes_zero(monkeypatch):
    _stub_instrument_checks(monkeypatch, includes_zero=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiagnosticWarning)
        check_instruments(_instrument_model(), compare_ols=False)


def test_sargan_requires_more_than_one_instrument():
    log_q, log_p, z, x = _simulate(200, direct_effect=0.0, seed=1)
    with pytest.raises(ValueError, match="more instruments"):
        sargan_test(log_q, log_p, z[:, :1], x)
