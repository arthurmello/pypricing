"""Experimental latent instrumental variables: a mixture for the price driver."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pypricing.diagnostics import DiagnosticWarning
from pypricing.endogeneity.base import Assumption, AssumptionReport, QuickEstimate
from pypricing.endogeneity.design import exogenous_design, log_regular_price
from pypricing.model_components.priors import resolve_prior

if TYPE_CHECKING:
    from pypricing.data import PricePanelData
    from pypricing.model_components.time_terms import TimeEffectsKind, TrendKind
    from pypricing.models.basic import DemandModel

# Candidate mixture sizes. One normal is the benchmark they have to beat.
CLASS_COUNTS = (2, 3, 4)
KFOLDS = 5
MIN_OBS = 40
# A class this small, or with no within-class spread, is not a real instrument.
MIN_WEIGHT = 0.05
MIN_SCALE = 1e-3
# Elasticities that move by more than this across equally good class counts.
STABILITY_SPAN = 0.3
_EM_ITERATIONS = 150
_VARIANCE_FLOOR = 1e-6

EXPERIMENTAL_WARNING = (
    "Latent IV is experimental. It treats the unobserved driver of price as a "
    "discrete mixture, and that mixture has to be real for the correction to "
    "be identified."
)


@dataclass(frozen=True)
class _Mixture:
    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    responsibility: np.ndarray

    @property
    def min_weight(self) -> float:
        return float(self.weights.min())

    @property
    def max_scale(self) -> float:
        return float(self.scales.max())


@dataclass(frozen=True)
class _Candidate:
    n_classes: int
    elpd: float
    pointwise: np.ndarray
    elasticity: float | None
    mixture: _Mixture | None


@dataclass(frozen=True)
class LatentPrepared:
    """Mixture control and the leave-one-out comparison that chose it."""

    log_price: np.ndarray
    log_quantity: np.ndarray
    exogenous: np.ndarray
    price_residual: np.ndarray
    control: np.ndarray
    n_classes: int
    elpd_difference: float
    elpd_se: float
    min_weight: float
    max_scale: float
    elasticity_span: float
    failure: str | None


class LatentMethod:
    """Control function from a discrete unobserved driver of price.

    Experimental. Log regular price is residualized on the exogenous design.
    The residual is a Gaussian mixture with ordered class means. The control
    is the residual minus its fitted class mean, and demand gets
    ``rho_latent`` times that control. ``L`` is chosen by K-fold ELPD over
    2, 3, and 4 classes. The correction is not identified when a single
    normal fits as well, or when the classes collapse. It is weak when the
    elasticity moves across class counts that fit about as well.
    """

    name = "latent"
    experimental = True

    def prepare(self, model: DemandModel) -> LatentPrepared:
        """Mixture control for the class count that K-fold ELPD prefers."""
        self._require_built(model)
        log_price = log_regular_price(model)
        exogenous = exogenous_design(model)
        residual = _residual(log_price, exogenous)
        quantity = model.data[model.quantity_col].to_numpy(dtype=np.float64)
        log_quantity = np.log(np.maximum(quantity, model.quantity_floor))
        return _prepare_from_arrays(log_price, log_quantity, exogenous, residual)

    def check_assumptions(self, model: DemandModel) -> AssumptionReport:
        """Mixture against a normal, then stability of the elasticity across ``L``.

        K-fold ELPD that does not beat a single normal, or classes that
        collapse, is not identified. An elasticity that changes by more than
        0.3 across class counts within one SE of the best is ``weak``.
        """
        self._require_built(model)
        prepared = self.prepare(model)
        assumptions: list[Assumption] = []
        if prepared.failure is not None:
            detail = _failure_detail(prepared)
            assumptions.append(Assumption("mixture", "fail", detail))
            warnings.warn(detail, DiagnosticWarning, stacklevel=2)
            return AssumptionReport(method=self.name, assumptions=tuple(assumptions))

        assumptions.append(
            Assumption(
                "mixture",
                "pass",
                "A "
                f"{prepared.n_classes}-class mixture fits the price residual "
                f"better than a normal (ELPD difference {prepared.elpd_difference:.1f}, "
                f"SE {prepared.elpd_se:.1f}).",
            )
        )
        if prepared.elasticity_span > STABILITY_SPAN:
            detail = (
                "Elasticity changes by "
                f"{prepared.elasticity_span:.2f} across class counts that fit "
                f"about as well, above {STABILITY_SPAN:.1f}."
            )
            assumptions.append(Assumption("stability", "warn", detail))
            warnings.warn(detail, DiagnosticWarning, stacklevel=2)
        else:
            assumptions.append(
                Assumption(
                    "stability",
                    "pass",
                    "Elasticity changes by "
                    f"{prepared.elasticity_span:.2f} across class counts that "
                    "fit about as well.",
                )
            )
        return AssumptionReport(method=self.name, assumptions=tuple(assumptions))

    def quick_estimate(
        self, model: DemandModel, report: AssumptionReport | None = None
    ) -> QuickEstimate:
        """Pooled elasticity from log quantity on price and the mixture control.

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
            raise ValueError("Latent IV is not identified: " + " ".join(failed))
        prepared = self.prepare(model)
        design = np.column_stack(
            [prepared.exogenous, prepared.log_price, prepared.control]
        )
        coef = _lstsq(prepared.log_quantity, design)
        return QuickEstimate(
            method=self.name,
            elasticity=float(coef[-2]),
            parameters={"rho_latent": float(coef[-1])},
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
        """Mixture likelihood for the price residual, and ``rho_latent`` times it."""
        del data, trend, t0, seasonality_components, time_effects
        if model is None:
            raise RuntimeError("LatentMethod.pymc_term requires the demand model.")
        prepared = self.prepare(model)
        return _mixture_control(model_config, prepared)

    def summarize_posterior(
        self, model: DemandModel, *, hdi_prob: float = 0.9
    ) -> dict[str, Any]:
        """Posterior interval for ``rho_latent`` and the class count in the graph."""
        model._require_fitted()
        assert model.idata is not None
        post = model.idata.posterior
        if "rho_latent" not in post:
            raise ValueError(
                "Posterior has no rho_latent. Fit with endogeneity='latent' first."
            )
        rho = np.asarray(post["rho_latent"].values, dtype=np.float64).ravel()
        alpha = (1.0 - hdi_prob) / 2.0
        lo, hi = np.quantile(rho, [alpha, 1.0 - alpha])
        prepared = self.prepare(model)
        return {
            "rho_latent": {
                "mean": float(np.mean(rho)),
                "hdi": (float(lo), float(hi)),
                "hdi_includes_zero": bool(lo <= 0.0 <= hi),
            },
            "n_classes": prepared.n_classes,
        }

    @staticmethod
    def _require_built(model: DemandModel) -> None:
        if model.data is None or model.sku_levels_ is None:
            raise RuntimeError("Build the model before preparing latent IV.")


def _prepare_from_arrays(
    log_price: np.ndarray,
    log_quantity: np.ndarray,
    exogenous: np.ndarray,
    residual: np.ndarray,
) -> LatentPrepared:
    if residual.shape[0] < MIN_OBS:
        control = np.zeros_like(residual)
        return LatentPrepared(
            log_price=log_price,
            log_quantity=log_quantity,
            exogenous=exogenous,
            price_residual=residual,
            control=control,
            n_classes=CLASS_COUNTS[0],
            elpd_difference=0.0,
            elpd_se=0.0,
            min_weight=0.0,
            max_scale=0.0,
            elasticity_span=0.0,
            failure="too_few",
        )

    normal_pointwise = _kfold_loglik(residual, 1)
    candidates = [
        _full_candidate(residual, log_quantity, log_price, exogenous, n_classes)
        for n_classes in CLASS_COUNTS
    ]
    usable = [item for item in candidates if _separated(item)]
    pool = usable or candidates
    best = max(pool, key=lambda item: (item.elpd, -item.n_classes))
    difference = best.pointwise - normal_pointwise
    elpd_difference = float(difference.sum())
    elpd_se = float(np.std(difference, ddof=1) * np.sqrt(difference.size))
    assert best.mixture is not None
    competitive = [
        item.elasticity
        for item in usable
        if item.elasticity is not None and _within_one_se(item, best)
    ]
    span = 0.0 if len(competitive) < 2 else float(max(competitive) - min(competitive))
    if not usable:
        failure: str | None = "collapsed"
    elif elpd_difference <= elpd_se:
        failure = "normal"
    else:
        failure = None
    control = residual - best.mixture.responsibility @ best.mixture.means
    return LatentPrepared(
        log_price=log_price,
        log_quantity=log_quantity,
        exogenous=exogenous,
        price_residual=residual,
        control=control,
        n_classes=best.n_classes,
        elpd_difference=elpd_difference,
        elpd_se=elpd_se,
        min_weight=best.mixture.min_weight,
        max_scale=best.mixture.max_scale,
        elasticity_span=span,
        failure=failure,
    )


def _separated(item: _Candidate) -> bool:
    assert item.mixture is not None
    return item.mixture.min_weight >= MIN_WEIGHT and item.mixture.max_scale >= MIN_SCALE


def _within_one_se(item: _Candidate, best: _Candidate) -> bool:
    if item.n_classes == best.n_classes:
        return True
    gap = best.pointwise - item.pointwise
    se = float(np.std(gap, ddof=1) * np.sqrt(gap.size))
    return float(gap.sum()) <= se


def _full_candidate(
    residual: np.ndarray,
    log_quantity: np.ndarray,
    log_price: np.ndarray,
    exogenous: np.ndarray,
    n_classes: int,
) -> _Candidate:
    pointwise = _kfold_loglik(residual, n_classes)
    mixture = _fit_mixture(residual, n_classes)
    control = residual - mixture.responsibility @ mixture.means
    design = np.column_stack([exogenous, log_price, control])
    elasticity = float(_lstsq(log_quantity, design)[-2])
    return _Candidate(
        n_classes=n_classes,
        elpd=float(pointwise.sum()),
        pointwise=pointwise,
        elasticity=elasticity,
        mixture=mixture,
    )


def _failure_detail(prepared: LatentPrepared) -> str:
    if prepared.failure == "too_few":
        return (
            f"Not enough observations to compare price mixtures "
            f"(need at least {MIN_OBS})."
        )
    if prepared.failure == "collapsed":
        return (
            "Mixture classes collapse "
            f"(smallest weight {prepared.min_weight:.2f}); "
            "latent IV is not identified."
        )
    return (
        "A mixture does not fit the price residual better than a normal "
        f"(ELPD difference {prepared.elpd_difference:.1f}, "
        f"SE {prepared.elpd_se:.1f}); latent IV is not identified."
    )


def _mixture_control(
    model_config: dict[str, Any] | None, prepared: LatentPrepared
) -> Any:
    n_classes = prepared.n_classes
    residual = np.asarray(prepared.price_residual, dtype=np.float64)
    gaps = pm.HalfNormal("latent_gap", sigma=2.5, shape=n_classes - 1)
    free = pt.concatenate([pt.zeros(1), pt.cumsum(gaps)])
    weights = pm.Dirichlet("latent_pi", a=np.ones(n_classes))
    mu = free - pt.dot(weights, free)
    sigma = pm.HalfNormal("latent_sigma", sigma=1.0)
    pm.Mixture(
        "obs_latent_price",
        w=weights,
        comp_dists=[pm.Normal.dist(mu=mu[i], sigma=sigma) for i in range(n_classes)],
        observed=residual,
    )
    expected = _expected_class_mean(residual, mu, sigma, weights)
    rho = resolve_prior(
        model_config=model_config,
        param_name="rho_latent",
        default_dist=pm.Normal,
        default_kwargs={"mu": 0.0, "sigma": 1.0},
    )
    return rho * (pt.as_tensor_variable(residual) - expected)


def _expected_class_mean(
    residual: np.ndarray, mu: Any, sigma: Any, weights: Any
) -> Any:
    resid = pt.as_tensor_variable(residual)[:, None]
    log_comp = -0.5 * pt.square((resid - mu[None, :]) / sigma) - pt.log(sigma)
    log_tau = pt.log(weights)[None, :] + log_comp
    shifted = log_tau - pt.max(log_tau, axis=1, keepdims=True)
    tau = pt.exp(shifted)
    tau = tau / pt.sum(tau, axis=1, keepdims=True)
    return pt.sum(tau * mu[None, :], axis=1)


def _kfold_loglik(x: np.ndarray, n_classes: int) -> np.ndarray:
    n = x.shape[0]
    order = np.random.default_rng(0).permutation(n)
    pointwise = np.empty(n, dtype=np.float64)
    for test_idx in np.array_split(order, KFOLDS):
        train = np.ones(n, dtype=bool)
        train[test_idx] = False
        pointwise[test_idx] = _loglik(x[train], x[test_idx], n_classes)
    return pointwise


def _loglik(train: np.ndarray, test: np.ndarray, n_classes: int) -> np.ndarray:
    if n_classes == 1:
        scale = max(float(np.std(train)), MIN_SCALE)
        return _normal_logpdf(test, float(np.mean(train)), scale)
    mixture = _fit_mixture(train, n_classes)
    return _mixture_logpdf(test, mixture)


def _fit_mixture(x: np.ndarray, n_classes: int) -> _Mixture:
    x = np.asarray(x, dtype=np.float64)
    means = np.quantile(x, (np.arange(n_classes) + 0.5) / n_classes).astype(np.float64)
    for i in range(1, n_classes):
        if means[i] <= means[i - 1] + 1e-3:
            means[i] = means[i - 1] + 1e-3
    variance = np.full(n_classes, max(float(np.var(x)) / 2.0, _VARIANCE_FLOOR))
    log_weights = np.full(n_classes, -np.log(n_classes))
    for _ in range(_EM_ITERATIONS):
        responsibility = _responsibilities(x, means, variance, log_weights)
        counts = responsibility.sum(axis=0) + 1e-8
        updated = (responsibility * x[:, None]).sum(axis=0) / counts
        variance = (responsibility * (x[:, None] - updated) ** 2).sum(axis=0) / counts
        variance = np.maximum(variance, _VARIANCE_FLOOR)
        order = np.argsort(updated)
        shift = float(np.max(np.abs(updated - means)))
        means = updated[order]
        variance = variance[order]
        log_weights = np.log(counts[order] / counts.sum())
        if shift < 1e-6:
            break
    responsibility = _responsibilities(x, means, variance, log_weights)
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    return _Mixture(
        means=means,
        scales=np.sqrt(variance),
        weights=weights,
        responsibility=responsibility,
    )


def _responsibilities(
    x: np.ndarray, means: np.ndarray, variance: np.ndarray, log_weights: np.ndarray
) -> np.ndarray:
    log_prob = (
        log_weights
        - 0.5 * np.log(2.0 * np.pi * variance)
        - 0.5 * (x[:, None] - means) ** 2 / variance
    )
    log_prob -= log_prob.max(axis=1, keepdims=True)
    prob = np.exp(log_prob)
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-300)


def _mixture_logpdf(x: np.ndarray, mixture: _Mixture) -> np.ndarray:
    parts = [
        np.log(mixture.weights[i] + 1e-300)
        + _normal_logpdf(x, float(mixture.means[i]), float(mixture.scales[i]))
        for i in range(mixture.means.shape[0])
    ]
    stacked = np.column_stack(parts)
    shift = stacked.max(axis=1, keepdims=True)
    return shift.squeeze(axis=1) + np.log(np.exp(stacked - shift).sum(axis=1))


def _normal_logpdf(x: np.ndarray, mean: float, scale: float) -> np.ndarray:
    scale = max(scale, MIN_SCALE)
    return -0.5 * np.log(2.0 * np.pi * scale**2) - 0.5 * ((x - mean) / scale) ** 2


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return y - x @ _lstsq(y, x)


def _lstsq(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(x, np.asarray(y, dtype=np.float64), rcond=None)
    return np.asarray(coef, dtype=np.float64)
