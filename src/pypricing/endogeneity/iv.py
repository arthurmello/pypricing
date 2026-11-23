"""Control-function instrumental variables, as one endogeneity method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pypricing.diagnostics.identification import (
    WEAK_IV_F_THRESHOLD,
    _instruments_and_log_price,
    first_stage_partial_f,
    iv_diagnostics,
)
from pypricing.endogeneity.base import Assumption, AssumptionReport, QuickEstimate
from pypricing.endogeneity.design import exogenous_design
from pypricing.model_components.iv_terms import get_control_function_term

if TYPE_CHECKING:
    from pypricing.data import PricePanelData
    from pypricing.model_components.time_terms import TimeEffectsKind, TrendKind
    from pypricing.models.basic import DemandModel


@dataclass(frozen=True)
class IVPrepared:
    """First-stage residual and the columns that produced it."""

    log_price: np.ndarray
    log_quantity: np.ndarray
    exogenous: np.ndarray
    instruments: np.ndarray
    residual: np.ndarray


class IVMethod:
    """Control function for a price instrumented by excluded ``iv_*`` columns.

    The price equation is the existing PyMC first stage. Demand gets
    ``rho`` times that equation's residual. ``predict`` and ``optimize_prices``
    do not include the residual.
    """

    name = "iv"

    def prepare(self, model: DemandModel) -> IVPrepared:
        """Residual of log regular price on the exogenous design plus instruments."""
        self._require_instruments(model)
        instruments, log_price = _instruments_and_log_price(model)
        exogenous = exogenous_design(model)
        fitted = _fitted(log_price, np.column_stack([exogenous, instruments]))
        quantity = model.data[model.quantity_col].to_numpy(dtype=np.float64)
        log_quantity = np.log(np.maximum(quantity, model.quantity_floor))
        return IVPrepared(
            log_price=log_price,
            log_quantity=log_quantity,
            exogenous=exogenous,
            instruments=instruments,
            residual=log_price - fitted,
        )

    def check_assumptions(self, model: DemandModel) -> AssumptionReport:
        """Instruments present, and a non-zero partial F after exogenous columns.

        F below 10 warns (``weak``). F of 0, or no instruments, is
        ``not identified``. The exclusion restriction is not checked.
        """
        if model.data is None or model.sku_levels_ is None:
            raise RuntimeError("Build the model before checking IV assumptions.")
        if not model.iv_names_:
            return AssumptionReport(
                method=self.name,
                assumptions=(
                    Assumption(
                        "instruments",
                        "fail",
                        "No instruments. Set PanelColumns.iv_columns or add iv_* columns.",
                    ),
                ),
            )
        instruments, log_price = _instruments_and_log_price(model)
        exogenous = exogenous_design(model)
        f_stat = first_stage_partial_f(log_price, instruments, exogenous)
        assumptions = [
            Assumption(
                "instruments",
                "pass",
                f"{len(model.iv_names_)} instrument(s).",
            )
        ]
        if f_stat == 0.0:
            assumptions.append(
                Assumption(
                    "relevance",
                    "fail",
                    "Instruments have no independent variation after the "
                    "exogenous regressors.",
                )
            )
        elif f_stat < WEAK_IV_F_THRESHOLD:
            assumptions.append(
                Assumption(
                    "relevance",
                    "warn",
                    f"Weak instruments: first-stage F = {f_stat:.1f} "
                    f"(< {WEAK_IV_F_THRESHOLD:.0f}).",
                )
            )
        else:
            assumptions.append(
                Assumption(
                    "relevance",
                    "pass",
                    f"First-stage F = {f_stat:.1f}.",
                )
            )
        return AssumptionReport(method=self.name, assumptions=tuple(assumptions))

    def quick_estimate(self, model: DemandModel) -> QuickEstimate:
        """Pooled control-function elasticity: log quantity on price and the residual.

        Raises when :meth:`check_assumptions` returns ``not identified``.
        A ``weak`` verdict still returns the estimate.
        """
        report = self.check_assumptions(model)
        if report.verdict() == "not identified":
            failed = [item.detail for item in report.assumptions if item.status == "fail"]
            raise ValueError("IV is not identified: " + " ".join(failed))
        prepared = self.prepare(model)
        design = np.column_stack(
            [prepared.exogenous, prepared.log_price, prepared.residual]
        )
        coef = _fitted_coef(prepared.log_quantity, design)
        return QuickEstimate(
            method=self.name,
            elasticity=float(coef[-2]),
            parameters={"rho": float(coef[-1])},
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
    ) -> Any:
        """``rho * residual`` inside the PyMC demand mean, or 0 with no instruments."""
        return get_control_function_term(
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
        """Posterior mean and interval for ``rho``."""
        model._require_fitted()
        diag = iv_diagnostics(model, hdi_prob=hdi_prob)
        if "rho_mean" not in diag:
            raise ValueError(
                "Posterior has no rho. Fit a model with instruments first."
            )
        return {
            "rho": {
                "mean": diag["rho_mean"],
                "hdi": diag["rho_hdi"],
                "hdi_includes_zero": diag["rho_hdi_includes_zero"],
            }
        }

    @staticmethod
    def _require_instruments(model: DemandModel) -> None:
        if model.data is None or model.sku_levels_ is None:
            raise RuntimeError("Build the model before preparing IV.")
        if not model.iv_names_:
            raise ValueError(
                "IVMethod requires instruments. Set PanelColumns.iv_columns "
                "or add iv_* columns."
            )


def _fitted_coef(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(x, np.asarray(y, dtype=np.float64), rcond=None)
    return np.asarray(coef, dtype=np.float64)


def _fitted(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    coef = _fitted_coef(y, x)
    return x @ coef
