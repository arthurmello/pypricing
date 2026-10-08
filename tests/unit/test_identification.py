from __future__ import annotations

import numpy as np
import pytest

from pypricing.diagnostics.identification import sargan_test


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


def test_sargan_requires_more_than_one_instrument():
    log_q, log_p, z, x = _simulate(200, direct_effect=0.0, seed=1)
    with pytest.raises(ValueError, match="more instruments"):
        sargan_test(log_q, log_p, z[:, :1], x)
