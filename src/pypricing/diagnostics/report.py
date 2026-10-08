"""Run all model checks at once and summarize them in a printable report."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypricing.diagnostics import DiagnosticWarning
from pypricing.diagnostics.benchmarks import check_benchmarks
from pypricing.diagnostics.falsification import check_falsification
from pypricing.diagnostics.fit import check_fit
from pypricing.diagnostics.identification import check_instruments
from pypricing.diagnostics.sensitivity import check_sensitivity

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel

RHAT_THRESHOLD = 1.01
COVERAGE_TOLERANCE = 0.1
SECTIONS = (
    "sampler",
    "fit",
    "benchmarks",
    "sensitivity",
    "instruments",
    "falsification",
)


@dataclass(repr=False)
class DiagnosticsReport:
    """Results of ``run_diagnostics``; print it for a summary.

    Each check's full output is an attribute (``None`` when skipped, with the
    reason in ``skipped``). Indexing reads the sampler results, so
    ``report["n_divergent"]`` works as before.
    """

    sampler: dict[str, Any]
    fit: pd.DataFrame | None = None
    benchmarks: dict[str, pd.DataFrame | None] | None = None
    sensitivity: pd.DataFrame | None = None
    instruments: dict[str, Any] | None = None
    falsification: dict[str, Any] | None = None
    hdi_prob: float = 0.94
    flags: dict[str, list[str]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.sampler[key]

    def __contains__(self, key: object) -> bool:
        return key in self.sampler

    def get(self, key: str, default: Any = None) -> Any:
        return self.sampler.get(key, default)

    @property
    def ok(self) -> bool:
        return not any(self.flags.values())

    def __str__(self) -> str:
        lines = []
        for name in SECTIONS:
            if name in self.skipped:
                lines.append(f"{name:<14} SKIPPED  {self.skipped[name]}")
                continue
            status = "WARN" if self.flags.get(name) else "OK"
            lines.append(f"{name:<14} {status:<8} {_SUMMARIES[name](self)}")
        issues = [msg for name in SECTIONS for msg in self.flags.get(name, [])]
        if issues:
            lines.append("")
            lines.append("Flags:")
            lines.extend(f"  - {msg}" for msg in issues)
        return "\n".join(lines)

    __repr__ = __str__


def _summarize_sampler(report: DiagnosticsReport) -> str:
    s = report.sampler
    rhat = "n/a" if s.get("max_rhat") is None else f"{s['max_rhat']:.3f}"
    return f"{s.get('n_divergent')} divergences, max r-hat {rhat}"


def _summarize_fit(report: DiagnosticsReport) -> str:
    overall = report.fit.loc["overall"]
    return (
        f"{report.hdi_prob:.0%} HDI coverage {overall['hdi_coverage']:.2f}, "
        f"rmse_log {overall['rmse_log']:.2f}, bias_log {overall['bias_log']:+.2f}"
    )


def _summarize_benchmarks(report: DiagnosticsReport) -> str:
    own = report.benchmarks["own"]["elasticity_mean"]
    text = f"own elasticities in [{own.min():.2f}, {own.max():.2f}]"
    cross = report.benchmarks["cross"]
    if cross is not None:
        text += f"; {len(cross)} cross-price pairs"
    return text


def _summarize_sensitivity(report: DiagnosticsReport) -> str:
    table = report.sensitivity
    per_sku = table.drop(index="pooled")
    weakest = per_sku["rv_qa"].idxmin()
    return (
        f"pooled RV {table.loc['pooled', 'rv']:.2f} "
        f"(at alpha: {table.loc['pooled', 'rv_qa']:.2f}); "
        f"weakest {weakest} ({per_sku.loc[weakest, 'rv_qa']:.2f})"
    )


def _summarize_instruments(report: DiagnosticsReport) -> str:
    out = report.instruments
    text = f"first-stage F {out['first_stage']['f']:.1f}"
    overid = out["overid"]
    if overid["applicable"]:
        text += f", Sargan p {overid['p_value']:.2f}"
    lo, hi = out["rho"]["hdi"]
    text += f", rho HDI ({lo:.2f}, {hi:.2f})"
    if out["elasticity"] is not None:
        text += f", max |IV - OLS| {out['elasticity']['shift'].abs().max():.2f}"
    return text


def _summarize_falsification(report: DiagnosticsReport) -> str:
    out = report.falsification
    lo, hi = out["lead_coef_hdi"]
    return (
        f"lead coef {out['lead_coef_mean']:.2f}, HDI ({lo:.2f}, {hi:.2f}), "
        f"max elasticity shift {out['elasticity']['shift'].abs().max():.2f}"
    )


_SUMMARIES: dict[str, Callable[[DiagnosticsReport], str]] = {
    "sampler": _summarize_sampler,
    "fit": _summarize_fit,
    "benchmarks": _summarize_benchmarks,
    "sensitivity": _summarize_sensitivity,
    "instruments": _summarize_instruments,
    "falsification": _summarize_falsification,
}


def _collect(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, list[str]]:
    """Run a check, returning its result and its ``DiagnosticWarning`` messages.

    Other warnings (e.g. from refits) are re-emitted unchanged.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    messages = []
    for w in caught:
        if issubclass(w.category, DiagnosticWarning):
            messages.append(str(w.message))
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return result, messages


def _sampler_flags(sampler: dict[str, Any]) -> list[str]:
    flags = []
    if sampler.get("n_divergent"):
        flags.append(f"{sampler['n_divergent']} divergent transitions.")
    if sampler.get("max_rhat") is not None and sampler["max_rhat"] > RHAT_THRESHOLD:
        flags.append(f"max r-hat {sampler['max_rhat']:.3f} > {RHAT_THRESHOLD}.")
    if sampler.get("floor_censored_skus"):
        flags.append(
            f"SKU(s) mostly at the quantity floor: {list(sampler['floor_censored_skus'])}."
        )
    return flags


def _fit_flags(table: pd.DataFrame, hdi_prob: float) -> list[str]:
    coverage = table["hdi_coverage"].dropna()
    off = coverage[(coverage - hdi_prob).abs() > COVERAGE_TOLERANCE]
    return [
        f"HDI coverage {value:.2f} in '{name}' rows (target {hdi_prob:.2f})."
        for name, value in off.items()
    ]


def run_diagnostics(
    model: DemandModel,
    sampler: dict[str, Any],
    *,
    fit: bool = True,
    benchmarks: bool = True,
    sensitivity: bool = True,
    instruments: bool = True,
    falsification: bool = False,
    compare_ols: bool = False,
    hdi_prob: float = 0.94,
) -> DiagnosticsReport:
    """Run the selected checks with their default settings.

    ``falsification`` and ``compare_ols`` refit the model (with the sampler
    settings of the last ``fit()``), so they are off by default. For custom
    settings, call the individual ``check_*`` methods.
    """
    report = DiagnosticsReport(sampler=sampler, hdi_prob=hdi_prob)
    report.flags["sampler"] = _sampler_flags(sampler)

    if fit:
        report.fit = check_fit(model, hdi_prob=hdi_prob)
        report.flags["fit"] = _fit_flags(report.fit, hdi_prob)
    else:
        report.skipped["fit"] = "disabled"

    if benchmarks:
        report.benchmarks, report.flags["benchmarks"] = _collect(check_benchmarks, model)
    else:
        report.skipped["benchmarks"] = "disabled"

    if sensitivity:
        report.sensitivity = check_sensitivity(model)
    else:
        report.skipped["sensitivity"] = "disabled"

    if not instruments:
        report.skipped["instruments"] = "disabled"
    elif not model.iv_names_:
        report.skipped["instruments"] = "model has no instruments"
    else:
        report.instruments, report.flags["instruments"] = _collect(
            check_instruments, model, compare_ols=compare_ols
        )

    if falsification:
        report.falsification, report.flags["falsification"] = _collect(
            check_falsification, model, hdi_prob=hdi_prob
        )
    else:
        report.skipped["falsification"] = "refits the model; pass falsification=True"

    return report
