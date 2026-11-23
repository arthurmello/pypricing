"""Gaussian-copula control function for a non-normal price."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm, normaltest, rankdata, shapiro

from pypricing.diagnostics import DiagnosticWarning
from pypricing.endogeneity.base import Assumption, AssumptionReport, QuickEstimate
from pypricing.endogeneity.design import exogenous_design, log_regular_price
from pypricing.model_components.priors import resolve_prior

if TYPE_CHECKING:
    from pypricing.data import PricePanelData
    from pypricing.model_components.time_terms import TimeEffectsKind, TrendKind
    from pypricing.models.basic import DemandModel

# A normal price makes the copula control an affine function of price.
NORMALITY_ALPHA = 0.05
# Above this, the control is hard to tell apart from the curve's bend.
NONLINEAR_CORR_THRESHOLD = 0.8


@dataclass(frozen=True)
class CopulaPrepared:
    """Copula control and the columns the pooled estimator uses."""

    log_price: np.ndarray
    log_quantity: np.ndarray
    exogenous: np.ndarray
    control: np.ndarray
    price_residual: np.ndarray


class CopulaMethod:
    """Control function from the Gaussian copula of the price residual.

    Log regular price is residualized on the exogenous design (SKU intercepts,
    controls, trend, season, period effects). The control is the normal score
    of that residual's rank. It is identified when the residual is not normal:
    a normal residual makes the score collinear with price. Estimation adds
    ``rho_copula`` times the control. ``predict`` and ``optimize_prices`` use
    the structural curve, with the control at 0.
    """

    name = "copula"

    def prepare(self, model: DemandModel) -> CopulaPrepared:
        """Normal scores of the log-price residual, standardized to unit variance."""
        self._require_built(model)
        log_price = log_regular_price(model)
        exogenous = exogenous_design(model)
        residual = _residual(log_price, exogenous)
        quantity = model.data[model.quantity_col].to_numpy(dtype=np.float64)
        log_quantity = np.log(np.maximum(quantity, model.quantity_floor))
        return CopulaPrepared(
            log_price=log_price,
            log_quantity=log_quantity,
            exogenous=exogenous,
            control=_normal_scores(residual),
            price_residual=residual,
        )

    def check_assumptions(self, model: DemandModel) -> AssumptionReport:
        """Normality of the price residual, and overlap with nonlinear curve terms.

        A residual consistent with a normal distribution (Shapiro-Wilk, or
        D'Agostino's test above 5000 rows, p above 0.05) is not identified.
        On a quadratic or sigmoid curve, a correlation above 0.8 between the
        control and the curve's nonlinear term warns.
        """
        self._require_built(model)
        prepared = self.prepare(model)
        assumptions: list[Assumption] = []
        p_value = _normality_pvalue(prepared.price_residual)
        if p_value > NORMALITY_ALPHA:
            detail = (
                "Price residual is consistent with a normal distribution "
                f"(p = {p_value:.3f}); the copula control is not identified."
            )
            assumptions.append(Assumption("non_normal_price", "fail", detail))
            warnings.warn(detail, DiagnosticWarning, stacklevel=2)
        else:
            assumptions.append(
                Assumption(
                    "non_normal_price",
                    "pass",
                    f"Price residual is not normal (p = {p_value:.3f}).",
                )
            )

        design = np.column_stack([prepared.exogenous, prepared.log_price])
        if _matrix_rank(design) == _matrix_rank(
            np.column_stack([design, prepared.control])
        ):
            detail = "Copula control is collinear with price after the exogenous regressors."
            assumptions.append(Assumption("separate_from_price", "fail", detail))
            warnings.warn(detail, DiagnosticWarning, stacklevel=2)
        else:
            assumptions.append(
                Assumption(
                    "separate_from_price",
                    "pass",
                    "Copula control is not a linear function of price.",
                )
            )

        nonlinear = _nonlinear_term(model, prepared.log_price)
        if nonlinear is not None:
            correlation = _partial_correlation(
                prepared.control, nonlinear, design
            )
            if correlation is None:
                assumptions.append(
                    Assumption(
                        "nonlinear_terms",
                        "pass",
                        "Nonlinear term has no variation beyond price.",
                    )
                )
            elif abs(correlation) > NONLINEAR_CORR_THRESHOLD:
                detail = (
                    "Correlation between the copula control and the curve's "
                    f"nonlinear term is {correlation:.2f}, above "
                    f"{NONLINEAR_CORR_THRESHOLD:.1f}."
                )
                assumptions.append(Assumption("nonlinear_terms", "warn", detail))
                warnings.warn(detail, DiagnosticWarning, stacklevel=2)
            else:
                assumptions.append(
                    Assumption(
                        "nonlinear_terms",
                        "pass",
                        "Correlation between the copula control and the curve's "
                        f"nonlinear term is {correlation:.2f}.",
                    )
                )
        return AssumptionReport(method=self.name, assumptions=tuple(assumptions))

    def quick_estimate(
        self, model: DemandModel, report: AssumptionReport | None = None
    ) -> QuickEstimate:
        """Pooled elasticity from log quantity on price and the copula control.

        Raises when the assumption verdict is ``not identified``. A ``weak``
        verdict still returns the estimate. Pass ``report`` to reuse a check
        that already ran.
        """
        if report is None:
            report = self.check_assumptions(model)
        if report.verdict() == "not identified":
            failed = [item.detail for item in report.assumptions if item.status == "fail"]
            raise ValueError("Copula is not identified: " + " ".join(failed))
        prepared = self.prepare(model)
        design = np.column_stack(
            [prepared.exogenous, prepared.log_price, prepared.control]
        )
        coef = _lstsq_coef(prepared.log_quantity, design)
        return QuickEstimate(
            method=self.name,
            elasticity=float(coef[-2]),
            parameters={"rho_copula": float(coef[-1])},
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
        """``rho_copula`` times the copula control inside the demand mean."""
        del data, trend, t0, seasonality_components, time_effects
        if model is None:
            raise RuntimeError("CopulaMethod.pymc_term requires the demand model.")
        prepared = self.prepare(model)
        rho = resolve_prior(
            model_config=model_config,
            param_name="rho_copula",
            default_dist=pm.Normal,
            default_kwargs={"mu": 0.0, "sigma": 1.0},
        )
        return rho * pt.constant(prepared.control)

    def summarize_posterior(
        self, model: DemandModel, *, hdi_prob: float = 0.9
    ) -> dict[str, Any]:
        """Posterior interval for ``rho_copula`` and its correlation with curvature.

        Quadratic models report the correlation with ``curvature_sku``. Sigmoid
        models report the correlation with ``log_price_center_sku``. Both are
        per SKU, across posterior draws.
        """
        model._require_fitted()
        assert model.idata is not None
        post = model.idata.posterior
        if "rho_copula" not in post:
            raise ValueError(
                "Posterior has no rho_copula. Fit with endogeneity='copula' first."
            )
        rho = _stacked_draws(post["rho_copula"])
        alpha = (1.0 - hdi_prob) / 2.0
        lo, hi = np.quantile(rho, [alpha, 1.0 - alpha])
        out: dict[str, Any] = {
            "rho_copula": {
                "mean": float(np.mean(rho)),
                "hdi": (float(lo), float(hi)),
                "hdi_includes_zero": bool(lo <= 0.0 <= hi),
            }
        }
        correlation: dict[str, dict[str, float]] = {}
        for name in ("curvature_sku", "log_price_center_sku"):
            if name not in post:
                continue
            correlation[name] = _draw_correlations(rho, post[name])
        if correlation:
            out["posterior_correlation"] = correlation
        return out

    @staticmethod
    def _require_built(model: DemandModel) -> None:
        if model.data is None or model.sku_levels_ is None:
            raise RuntimeError("Build the model before using the copula method.")


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) - x @ _lstsq_coef(y, x)


def _lstsq_coef(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(x, np.asarray(y, dtype=np.float64), rcond=None)
    return np.asarray(coef, dtype=np.float64)


def _normal_scores(x: np.ndarray) -> np.ndarray:
    """Van der Waerden scores, then centered and scaled to unit variance."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    ranks = rankdata(values, method="average")
    scores = norm.ppf(ranks / (values.shape[0] + 1.0))
    scale = float(np.std(scores))
    if scale == 0.0:
        return np.zeros_like(scores)
    return (scores - float(np.mean(scores))) / scale


def _normality_pvalue(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.shape[0] < 8 or float(np.std(values)) == 0.0:
        return 1.0
    if values.shape[0] <= 5000:
        return float(shapiro(values).pvalue)
    return float(normaltest(values).pvalue)


def _matrix_rank(x: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(x, tol=1e-8))


def _partial_correlation(
    left: np.ndarray, right: np.ndarray, controls: np.ndarray
) -> float | None:
    left_r = _residual(left, controls)
    right_r = _residual(right, controls)
    if float(np.std(left_r)) < 1e-8 or float(np.std(right_r)) < 1e-8:
        return None
    return float(np.corrcoef(left_r, right_r)[0, 1])


def _nonlinear_term(model: DemandModel, log_price: np.ndarray) -> np.ndarray | None:
    """Column that carries the curve's bend, in training-row order."""
    if model.model_name == "quadratic":
        return np.asarray(log_price, dtype=np.float64) ** 2
    if model.model_name == "sigmoid":
        if model.log_price_midpoint_sku_ is None or model.data is None:
            return None
        sku = pd.Categorical(
            model.data[model.sku_col], categories=model.sku_levels_
        )
        center = np.exp(np.asarray(model.log_price_midpoint_sku_, dtype=np.float64))
        distance = np.exp(np.asarray(log_price, dtype=np.float64)) - center[sku.codes]
        return distance**2
    return None


def _stacked_draws(da: Any) -> np.ndarray:
    stacked = da.stack(sample=("chain", "draw"))
    return np.asarray(stacked.values, dtype=np.float64).reshape(-1)


def _draw_correlations(rho: np.ndarray, da: Any) -> dict[str, float]:
    stacked = da.stack(sample=("chain", "draw"))
    other = [dim for dim in stacked.dims if dim != "sample"]
    if other:
        stacked = stacked.transpose("sample", *other)
    values = np.asarray(stacked.values, dtype=np.float64)
    if values.ndim == 1:
        labels = ["value"]
        columns = values.reshape(-1, 1)
    else:
        labels = [str(label) for label in stacked.coords[other[-1]].values]
        columns = values.reshape(values.shape[0], -1)
    out: dict[str, float] = {}
    for index, label in enumerate(labels):
        column = columns[:, index]
        if float(np.std(column)) == 0.0 or float(np.std(rho)) == 0.0:
            out[label] = float("nan")
        else:
            out[label] = float(np.corrcoef(rho, column)[0, 1])
    return out
