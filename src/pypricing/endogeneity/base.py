"""Shared types for price-endogeneity corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel

AssumptionStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class Assumption:
    """One condition a method needs before its correction is trusted."""

    name: str
    status: AssumptionStatus
    detail: str


@dataclass(frozen=True)
class AssumptionReport:
    """Assumptions checked for one endogeneity method."""

    method: str
    assumptions: tuple[Assumption, ...]

    def verdict(self) -> str:
        """``ok``, ``weak``, or ``not identified``.

        A failed assumption means the correction is not identified. A warning
        means it can be estimated, but the estimate is fragile.
        """
        if any(item.status == "fail" for item in self.assumptions):
            return "not identified"
        if any(item.status == "warn" for item in self.assumptions):
            return "weak"
        return "ok"


@dataclass(frozen=True)
class QuickEstimate:
    """Point estimate from :meth:`EndogeneityMethod.quick_estimate`.

    ``elasticity`` is the pooled own-price slope. ``parameters`` holds the
    method's correction terms (``rho`` for instrumental variables).
    """

    method: str
    elasticity: float
    parameters: dict[str, float]


@runtime_checkable
class EndogeneityMethod(Protocol):
    """One way to correct a price that was set from the demand shock.

    ``prepare`` builds the method's regressors from a built model.
    ``check_assumptions`` says whether that correction is identified.
    ``quick_estimate`` is the fast point estimate, without MCMC.
    ``pymc_term`` is the term added to the demand mean while sampling.
    ``summarize_posterior`` reads the fitted correction parameters.
    """

    name: str

    def prepare(self, model: DemandModel) -> Any: ...

    def check_assumptions(self, model: DemandModel) -> AssumptionReport: ...

    def quick_estimate(self, model: DemandModel) -> QuickEstimate: ...

    def pymc_term(
        self,
        model_config: dict[str, Any] | None,
        data: Any,
        *,
        trend: Any,
        t0: Any,
        seasonality_components: tuple[str, ...] | None,
        time_effects: Any = None,
    ) -> Any: ...

    def summarize_posterior(
        self, model: DemandModel, *, hdi_prob: float = 0.9
    ) -> dict[str, Any]: ...
