"""Lewbel instruments from a heteroskedastic price residual."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from pypricing.diagnostics import DiagnosticWarning
from pypricing.endogeneity.base import Assumption, AssumptionReport, QuickEstimate
from pypricing.endogeneity.design import exogenous_design, log_regular_price
from pypricing.endogeneity.iv import IVMethod

if TYPE_CHECKING:
    from pypricing.data import PricePanelData
    from pypricing.model_components.time_terms import TimeEffectsKind, TrendKind
    from pypricing.models.basic import DemandModel

# Breusch-Pagan. Above this p-value the residual is treated as homoskedastic.
HETEROSKEDASTICITY_ALPHA = 0.05
_COLUMN_STD_MIN = 1e-8


@dataclass(frozen=True)
class LewbelPrepared:
    """Generated instruments and the price residual they were built from."""

    log_price: np.ndarray
    log_quantity: np.ndarray
    exogenous: np.ndarray
    instruments: np.ndarray
    price_residual: np.ndarray


class LewbelMethod:
    """Instruments ``(X - mean(X)) * v`` from the price residual (Lewbel 2012).

    Log regular price is residualized on the exogenous design. Each exogenous
    column with variation is demeaned and multiplied by that residual. Those
    generated instruments are estimated with the IV control function. The
    correction is not identified when the residual is homoskedastic
    (Breusch-Pagan p above 0.05), or when every exogenous regressor is constant.
    """

    name = "lewbel"

    def prepare(self, model: DemandModel) -> LewbelPrepared:
        """Generated instruments from the exogenous design and the price residual."""
        self._require_built(model)
        log_price = log_regular_price(model)
        exogenous = exogenous_design(model)
        residual = _residual(log_price, exogenous)
        quantity = model.data[model.quantity_col].to_numpy(dtype=np.float64)
        log_quantity = np.log(np.maximum(quantity, model.quantity_floor))
        return LewbelPrepared(
            log_price=log_price,
            log_quantity=log_quantity,
            exogenous=exogenous,
            instruments=_generated_instruments(exogenous, residual),
            price_residual=residual,
        )

    def check_assumptions(self, model: DemandModel) -> AssumptionReport:
        """Heteroskedasticity of the price residual, then the IV relevance check.

        No varying exogenous regressor, or a Breusch-Pagan p-value above 0.05,
        is not identified. When the residual is heteroskedastic, instrument
        strength is the same first-stage F the IV method uses.
        """
        self._require_built(model)
        prepared = self.prepare(model)
        scale = _varying_columns(prepared.exogenous)
        if scale is None or prepared.instruments.shape[1] == 0:
            detail = (
                "Lewbel needs a regressor with variation to interact with "
                "the price residual."
            )
            warnings.warn(detail, DiagnosticWarning, stacklevel=2)
            return AssumptionReport(
                method=self.name,
                assumptions=(Assumption("heteroskedasticity", "fail", detail),),
            )

        p_value = _breusch_pagan(prepared.price_residual, scale)
        if p_value > HETEROSKEDASTICITY_ALPHA:
            detail = (
                "Price residual is homoskedastic "
                f"(Breusch-Pagan p = {p_value:.3f}); "
                "Lewbel instruments are not identified."
            )
            warnings.warn(detail, DiagnosticWarning, stacklevel=2)
            return AssumptionReport(
                method=self.name,
                assumptions=(Assumption("heteroskedasticity", "fail", detail),),
            )

        assumptions = [
            Assumption(
                "heteroskedasticity",
                "pass",
                "Price residual variance moves with the exogenous regressors "
                f"(Breusch-Pagan p = {p_value:.3f}).",
            )
        ]
        iv_report = IVMethod(instruments=prepared.instruments).check_assumptions(
            model
        )
        for item in iv_report.assumptions:
            assumptions.append(item)
            if item.status in {"fail", "warn"}:
                warnings.warn(item.detail, DiagnosticWarning, stacklevel=2)
        return AssumptionReport(method=self.name, assumptions=tuple(assumptions))

    def quick_estimate(
        self, model: DemandModel, report: AssumptionReport | None = None
    ) -> QuickEstimate:
        """Pooled control-function elasticity from the generated instruments.

        Raises when the assumption verdict is ``not identified``. A ``weak``
        verdict still returns the estimate. Pass ``report`` to reuse a check
        that already ran.
        """
        if report is None:
            report = self.check_assumptions(model)
        if report.verdict() == "not identified":
            failed = [
                item.detail for item in report.assumptions if item.status == "fail"
            ]
            raise ValueError("Lewbel is not identified: " + " ".join(failed))
        prepared = self.prepare(model)
        estimate = IVMethod(instruments=prepared.instruments).quick_estimate(
            model, report
        )
        return QuickEstimate(
            method=self.name,
            elasticity=estimate.elasticity,
            parameters=estimate.parameters,
        )

    def pymc_term(
        self,
        model_config: dict[str, Any] | None,
        data: PricePanelData,
        *,
        trend: TrendKind | None,
        t0: Any,
        seasonality_components: tuple[str, ...] | None,
        time_effects: TimeEffectsKind | None = None,
        model: DemandModel | None = None,
    ) -> Any:
        """IV control function on the generated instruments, or 0 when none exist."""
        if model is None:
            raise ValueError("LewbelMethod.pymc_term requires the demand model.")
        prepared = self.prepare(model)
        if prepared.instruments.shape[1] == 0:
            return 0.0
        return IVMethod(instruments=prepared.instruments).pymc_term(
            model_config,
            data,
            trend=trend,
            t0=t0,
            seasonality_components=seasonality_components,
            time_effects=time_effects,
        )

    def summarize_posterior(
        self, model: DemandModel, *, hdi_prob: float = 0.9
    ) -> dict[str, Any]:
        """Posterior mean and interval for ``rho``, the IV control coefficient."""
        return IVMethod().summarize_posterior(model, hdi_prob=hdi_prob)

    @staticmethod
    def _require_built(model: DemandModel) -> None:
        if model.data is None or model.sku_levels_ is None:
            raise RuntimeError("Build the model before preparing Lewbel.")


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ coef


def _varying_columns(exogenous: np.ndarray) -> np.ndarray | None:
    keep = [
        exogenous[:, j]
        for j in range(exogenous.shape[1])
        if float(np.std(exogenous[:, j])) > _COLUMN_STD_MIN
    ]
    if not keep:
        return None
    return np.column_stack(keep)


def _generated_instruments(exogenous: np.ndarray, residual: np.ndarray) -> np.ndarray:
    scale = _varying_columns(exogenous)
    if scale is None:
        return np.zeros((residual.shape[0], 0), dtype=np.float64)
    columns = []
    for j in range(scale.shape[1]):
        demeaned = scale[:, j] - float(scale[:, j].mean())
        column = demeaned * residual
        if float(np.std(column)) > _COLUMN_STD_MIN:
            columns.append(column)
    if not columns:
        return np.zeros((residual.shape[0], 0), dtype=np.float64)
    return np.column_stack(columns)


def _breusch_pagan(residual: np.ndarray, scale: np.ndarray) -> float:
    """p-value of n R² from residual squared on the scale regressors."""
    squared = residual**2
    design = np.column_stack([np.ones(residual.shape[0]), scale])
    coef, _, rank, _ = np.linalg.lstsq(design, squared, rcond=None)
    if rank <= 1:
        return 1.0
    fitted = design @ coef
    total = float(np.sum((squared - squared.mean()) ** 2))
    if total <= 0.0:
        return 1.0
    r2 = 1.0 - float(np.sum((squared - fitted) ** 2)) / total
    lm = residual.shape[0] * max(r2, 0.0)
    return float(stats.chi2.sf(lm, rank - 1))
