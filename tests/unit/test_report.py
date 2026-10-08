from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pypricing.diagnostics import DiagnosticWarning
from pypricing.diagnostics.report import (
    DiagnosticsReport,
    _collect,
    _fit_flags,
    _sampler_flags,
)


def _sampler(**overrides):
    return {"n_divergent": 0, "max_rhat": 1.0, "floor_censored_skus": {}, **overrides}


def _skip_all_but_sampler(report: DiagnosticsReport) -> DiagnosticsReport:
    for name in ("fit", "benchmarks", "sensitivity", "instruments", "falsification"):
        report.skipped[name] = "disabled"
    return report


def test_report_indexes_sampler_results():
    report = DiagnosticsReport(sampler=_sampler(n_divergent=3))
    assert report["n_divergent"] == 3
    assert "max_rhat" in report
    assert report.get("missing", "x") == "x"
    with pytest.raises(KeyError):
        report["missing"]


def test_report_prints_status_and_flags():
    report = _skip_all_but_sampler(DiagnosticsReport(sampler=_sampler(n_divergent=2)))
    report.flags["sampler"] = _sampler_flags(report.sampler)

    text = str(report)

    assert "sampler        WARN" in text
    assert "falsification  SKIPPED  disabled" in text
    assert "2 divergent transitions." in text
    assert not report.ok


def test_report_ok_without_flags():
    report = _skip_all_but_sampler(DiagnosticsReport(sampler=_sampler()))
    report.flags["sampler"] = _sampler_flags(report.sampler)
    assert report.ok
    assert "Flags:" not in str(report)


def test_sampler_flags():
    flags = _sampler_flags(
        _sampler(n_divergent=1, max_rhat=1.05, floor_censored_skus={"a": 0.6})
    )
    assert len(flags) == 3


def test_fit_flags_only_off_target_rows():
    table = pd.DataFrame(
        {"hdi_coverage": [0.93, 0.70, float("nan")]}, index=["overall", "high", "low"]
    )
    flags = _fit_flags(table, hdi_prob=0.94)
    assert len(flags) == 1
    assert "'high'" in flags[0]


def test_collect_keeps_diagnostic_warnings_and_reemits_others():
    def check():
        warnings.warn("model issue", DiagnosticWarning)
        warnings.warn("unrelated", RuntimeWarning)
        return 42

    with pytest.warns(RuntimeWarning, match="unrelated"):
        result, messages = _collect(check)

    assert result == 42
    assert messages == ["model issue"]
