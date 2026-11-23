"""Synthetic panel data for price–quantity (elasticity) experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union, overload

import numpy as np
import pandas as pd
from scipy.special import softplus

from pypricing.data import HierarchyIndex
from pypricing.model_components.cross_elasticity import (
    CrossElasticitySpec,
    enumerate_cross_pairs,
    parse_cross_elasticity_mode,
    validate_cross_elasticity_config,
)

RandomState = Union[int, np.random.Generator, None]

# Wider curvature draws so quadratic demo plots show a visible bend.
_QUADRATIC_CURVATURE_STD = 0.55
_QUADRATIC_CURVATURE_MIN_ABS = 0.12
_PANEL_NOISE_SIGMA = 0.12
_PANEL_SEASON_BETA = 0.15
# Gamma(2, scale=4) has mean 8 and positive skew. Discrete levels span the
# same price range as the normal random-walk's starting price.
_GAMMA_PRICE_SHAPE = 2.0
_GAMMA_PRICE_SCALE = 4.0
_DISCRETE_PRICE_MIN = 2.0
_DISCRETE_PRICE_MAX = 15.0
_PRICE_DISTRIBUTIONS = frozenset({"normal", "gamma", "discrete"})


@dataclass(frozen=True)
class MockDataTruth:
    """Ground-truth DGP parameters for a panel from :func:`generate_mock_data`.

    SKU-aligned arrays match labels ``sku_1``, … in ``sku_labels`` order.
    ``control_coefs``, when present, is one vector of length ``n_controls``
    shared by every SKU.
    """

    shape: str
    sku_labels: tuple[str, ...]
    alpha_sku: np.ndarray
    elasticity_sku: np.ndarray
    curvature_sku: np.ndarray | None
    noise_sigma: float
    control_coefs: np.ndarray | None
    gamma_pair: dict[tuple[int, int], float] | None
    volume_trend: float
    n_instruments: int = 0
    instrument_coef: float = 0.0
    endogeneity: float = 0.0
    price_distribution: str = "normal"
    n_price_points: int = 3
    heteroskedasticity: bool = False
    promo_elasticity: float | None = None
    lambda_period: np.ndarray | None = None
    common_shock_sigma: float = 0.0
    common_endogeneity: float = 0.0


@dataclass
class _SkuEffects:
    """Per-SKU latent effects drawn for one mock panel."""

    intercept: np.ndarray
    elasticity: np.ndarray
    curvature: np.ndarray | None
    hierarchy: HierarchyIndex | None


@dataclass
class _PanelSim:
    """Inputs for expanding drawn effects into long-format panel rows."""

    n_skus: int
    n_reg: int
    n_periods: int
    n_controls: int
    n_categories: int | None
    demand_shape: str
    effects: _SkuEffects
    cat_idx: np.ndarray | None
    region_labels: list[str] | None
    control_coefs: np.ndarray | None
    log_price_panel: np.ndarray
    gamma_map: dict[tuple[int, int], float] | None
    date_index: pd.DatetimeIndex | None
    period_season: np.ndarray
    annual_phase: np.ndarray
    include_seasonality: bool
    volume_trend: float
    n_instruments: int
    instrument_coef: float
    endogeneity: float
    common_endogeneity: float
    lambda_period: np.ndarray
    noise_sigma: float
    promotions: bool = False
    promo_frequency: float = 0.3
    promo_depth: float = 0.2
    promo_elasticity: float = -3.5
    price_distribution: str = "normal"
    n_price_points: int = 3
    heteroskedasticity: bool = False

    def lambda_at(self, region: int) -> np.ndarray:
        """Period shock for one region. 1-d when the shock is national."""
        if self.lambda_period.ndim == 2:
            return self.lambda_period[region]
        return self.lambda_period


def _as_rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


_VALID_DEMAND_SHAPES = frozenset({"log_log", "quadratic", "sigmoid"})


def _parse_price_distribution(price_distribution: str) -> str:
    key = price_distribution.strip().lower()
    if key not in _PRICE_DISTRIBUTIONS:
        allowed = ", ".join(sorted(_PRICE_DISTRIBUTIONS))
        raise ValueError(
            f"price_distribution {price_distribution!r} is not supported; "
            f"use one of: {allowed}"
        )
    return key


def _discrete_price_grid(n_points: int) -> np.ndarray:
    """``n_points`` price levels, equally spaced in log price from 2 to 15."""
    if n_points < 2:
        raise ValueError("n_price_points must be at least 2")
    return np.exp(
        np.linspace(
            np.log(_DISCRETE_PRICE_MIN),
            np.log(_DISCRETE_PRICE_MAX),
            n_points,
        )
    )


def _snap_log_price(log_price: np.ndarray, grid: np.ndarray) -> np.ndarray:
    log_grid = np.log(grid)
    nearest = np.abs(log_price[..., None] - log_grid).argmin(axis=-1)
    return np.log(grid[nearest])


def _parse_demand_shape(shape: str) -> str:
    key = shape.strip().lower()
    if key not in _VALID_DEMAND_SHAPES:
        allowed = ", ".join(sorted(_VALID_DEMAND_SHAPES))
        raise ValueError(
            f"demand shape {shape!r} is not supported; use one of: {allowed}"
        )
    return key


def _calendar_seasonality_index(
    start: pd.Timestamp, n_periods: int, freq: str
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    idx = pd.date_range(start=start, periods=n_periods, freq=freq)
    day_of_year = idx.dayofyear.to_numpy(dtype=float)
    phase = 2 * np.pi * (day_of_year - 1) / 365.25
    return idx, phase


def _decompose_leaf(leaf: int, dims: tuple[int, ...]) -> list[int]:
    rem = int(leaf)
    idxs: list[int] = []
    for d in reversed(dims):
        idxs.append(rem % d)
        rem //= d
    return list(reversed(idxs))


def _cumulative_level_codes(idxs: list[int], dims: tuple[int, ...]) -> list[int]:
    codes: list[int] = []
    for L in range(len(dims)):
        c = 0
        mul = 1
        for j in range(L, -1, -1):
            c += idxs[j] * mul
            mul *= dims[j]
        codes.append(c)
    return codes


def _sku_hierarchy_codes(
    n_skus: int, dims: tuple[int, ...]
) -> tuple[np.ndarray, tuple[int, ...]]:
    n_leaves = int(np.prod(dims))
    if n_leaves < 1:
        raise ValueError("hierarchy_levels product must be positive")
    n_levels = len(dims)
    n_groups_per_level = tuple(int(np.prod(dims[: L + 1])) for L in range(n_levels))
    mat = np.zeros((n_skus, n_levels), dtype=np.int64)
    for s in range(n_skus):
        leaf = s % n_leaves
        idxs = _decompose_leaf(leaf, dims)
        codes = _cumulative_level_codes(idxs, dims)
        mat[s, :] = codes
    return mat, n_groups_per_level


def _hierarchical_sum_per_sku(
    rng: np.random.Generator,
    n_skus: int,
    sku_codes: np.ndarray,
    n_groups_per_level: tuple[int, ...],
    mu: float,
    level_scales: np.ndarray,
    sku_scale: float,
) -> np.ndarray:
    n_levels = len(n_groups_per_level)
    level_effects: list[np.ndarray] = []
    for L, n_g in enumerate(n_groups_per_level):
        level_effects.append(rng.normal(0.0, float(level_scales[L]), size=n_g))
    out = np.empty(n_skus, dtype=np.float64)
    for s in range(n_skus):
        v = mu
        for L in range(n_levels):
            v += level_effects[L][sku_codes[s, L]]
        v += rng.normal(0.0, sku_scale)
        out[s] = v
    return out


def _validate_kwargs(
    n_periods: int,
    n_skus: int,
    n_categories: int | None,
    hierarchy_levels: tuple[int, ...] | None,
    n_regions: int | None,
    n_controls: int,
    *,
    cross_elasticity: str | None = None,
    cross_elasticity_group_level: int | None = None,
    n_instruments: int = 0,
) -> str | None:
    if n_periods <= 0:
        raise ValueError("n_periods must be positive")
    if n_skus <= 0:
        raise ValueError("n_skus must be positive")
    if hierarchy_levels is not None and n_categories is not None:
        raise ValueError("use only one of n_categories or hierarchy_levels, not both")
    if hierarchy_levels is not None:
        if len(hierarchy_levels) < 1:
            raise ValueError("hierarchy_levels must have at least one positive int")
        if any(h < 1 for h in hierarchy_levels):
            raise ValueError("each hierarchy_levels entry must be >= 1")
    if n_categories is not None:
        if n_categories < 1:
            raise ValueError("n_categories must be >= 1 when not None")
        if n_categories > n_skus:
            raise ValueError("n_categories cannot exceed n_skus")
    if n_regions is not None and n_regions < 1:
        raise ValueError("n_regions must be >= 1 when not None")
    if n_controls < 0:
        raise ValueError("n_controls must be non-negative")
    if n_instruments < 0:
        raise ValueError("n_instruments must be non-negative")

    ce = parse_cross_elasticity_mode(cross_elasticity)
    if ce is not None:
        gc = (
            tuple(f"category_{i + 1}" for i in range(len(hierarchy_levels)))
            if hierarchy_levels is not None
            else None
        )
        validate_cross_elasticity_config(
            cross_elasticity=CrossElasticitySpec(
                mode=ce,
                group_level=cross_elasticity_group_level,
            ),
            group_columns=gc,
        )
    return ce


def _draw_sku_effects(
    rng: np.random.Generator,
    *,
    n_skus: int,
    hierarchy_levels: tuple[int, ...] | None,
    demand_shape: str,
) -> _SkuEffects:
    """Draw per-SKU intercept, elasticity, optional curvature, and hierarchy."""
    if hierarchy_levels is not None:
        dims_h = tuple(int(h) for h in hierarchy_levels)
        sku_codes, n_groups_h = _sku_hierarchy_codes(n_skus, dims_h)
        hierarchy = HierarchyIndex(
            group_columns=tuple(f"category_{i + 1}" for i in range(len(dims_h))),
            sku_to_level_idx=sku_codes,
            n_groups_per_level=n_groups_h,
        )
        n_levels = len(dims_h)
        level_scale_a = rng.uniform(0.08, 0.28, size=n_levels)
        level_scale_e = rng.uniform(0.08, 0.28, size=n_levels)
        sku_scale_a = float(rng.uniform(0.05, 0.18))
        sku_scale_e = float(rng.uniform(0.05, 0.18))
        mu_alpha = float(rng.normal(5.0, 0.5))
        mu_elast = float(rng.uniform(-2.2, -0.4))
        intercept = _hierarchical_sum_per_sku(
            rng,
            n_skus,
            sku_codes,
            n_groups_h,
            mu_alpha,
            level_scale_a,
            sku_scale_a,
        )
        elasticity = _hierarchical_sum_per_sku(
            rng,
            n_skus,
            sku_codes,
            n_groups_h,
            mu_elast,
            level_scale_e,
            sku_scale_e,
        )
        curvature: np.ndarray | None = None
        if demand_shape == "quadratic":
            level_scale_k = rng.uniform(0.06, 0.22, size=n_levels)
            sku_scale_k = float(rng.uniform(0.04, 0.14))
            raw_k = _hierarchical_sum_per_sku(
                rng,
                n_skus,
                sku_codes,
                n_groups_h,
                0.0,
                level_scale_k,
                sku_scale_k,
            )
            curvature = np.sign(raw_k) * np.maximum(
                np.abs(raw_k), _QUADRATIC_CURVATURE_MIN_ABS
            )
        return _SkuEffects(
            intercept=intercept,
            elasticity=elasticity,
            curvature=curvature,
            hierarchy=hierarchy,
        )

    intercept = rng.normal(5.0, 0.5, size=n_skus)
    elasticity = rng.uniform(-2.2, -0.4, size=n_skus)
    curvature = None
    if demand_shape == "quadratic":
        curv = rng.normal(0.0, _QUADRATIC_CURVATURE_STD, size=n_skus)
        curvature = np.sign(curv) * np.maximum(
            np.abs(curv), _QUADRATIC_CURVATURE_MIN_ABS
        )
    return _SkuEffects(
        intercept=intercept,
        elasticity=elasticity,
        curvature=curvature,
        hierarchy=None,
    )


def _draw_cross_gamma_map(
    rng: np.random.Generator,
    *,
    mode: str,
    n_skus: int,
    hierarchy: HierarchyIndex | None,
    cross_elasticity_group_level: int | None,
) -> dict[tuple[int, int], float]:
    pairs = enumerate_cross_pairs(
        mode=mode,
        n_skus=n_skus,
        hierarchy=hierarchy,
        cross_elasticity_group_level=cross_elasticity_group_level,
    )
    mu_p = rng.normal(0.0, 0.04, size=pairs.n_pool)
    sig_p = rng.uniform(0.02, 0.06, size=pairs.n_pool)
    gamma_vec = mu_p[pairs.pair_pool] + sig_p[pairs.pair_pool] * rng.normal(
        size=pairs.n_pairs
    )
    return {
        (int(pairs.pair_from[p]), int(pairs.pair_to[p])): float(gamma_vec[p])
        for p in range(pairs.n_pairs)
    }


def _simulate_log_price_panel(
    rng: np.random.Generator,
    *,
    n_skus: int,
    n_reg: int,
    n_periods: int,
    price_shock_sigma: float = 0.03,
    price_distribution: str = "normal",
) -> np.ndarray:
    """Exogenous log price, before instruments and the demand shock.

    ``normal`` and ``discrete`` share the random-walk draws. ``discrete`` snaps
    to a grid later, after endogeneity is added. ``gamma`` draws iid prices and
    does not use ``price_shock_sigma``.
    """
    log_price_panel = np.zeros((n_skus, n_reg, n_periods), dtype=np.float64)
    for s in range(n_skus):
        for r in range(n_reg):
            if price_distribution == "gamma":
                price = rng.gamma(
                    _GAMMA_PRICE_SHAPE, _GAMMA_PRICE_SCALE, size=n_periods
                )
                log_price_panel[s, r, :] = np.log(price)
            else:
                base_price = rng.uniform(_DISCRETE_PRICE_MIN, _DISCRETE_PRICE_MAX)
                price_shocks = rng.normal(0, price_shock_sigma, size=n_periods).cumsum()
                log_price_panel[s, r, :] = np.log(base_price) + price_shocks
    return log_price_panel


def _mean_log_quantity(
    *,
    demand_shape: str,
    common: np.ndarray,
    log_price: np.ndarray,
    price: np.ndarray,
    elasticity: float,
    curvature: float | None,
    log_p_mid: float,
) -> np.ndarray:
    match demand_shape:
        case "log_log":
            return common + elasticity * log_price
        case "quadratic":
            if curvature is None:
                raise RuntimeError("quadratic demand requires curvature")
            beta1 = elasticity - 2.0 * curvature * log_p_mid
            return common + beta1 * log_price + curvature * (log_price**2)
        case "sigmoid":
            p_center = np.exp(log_p_mid)
            b = -2.0 * elasticity / p_center
            z = b * (price - p_center)
            return common - softplus(z)
        case _:
            raise ValueError(f"Unsupported demand shape: {demand_shape}")


def _draw_observed_log_prices(
    rng: np.random.Generator,
    sim: _PanelSim,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Observed prices after instruments and endogeneity, plus the shocks behind them.

    Drawn in the same SKU–region order as the quantity loop so a fixed seed keeps
    the same shocks. Cross-price terms use these log prices. With
    ``heteroskedasticity`` off and a normal or discrete distribution, the random
    draws match the historical generator; discrete prices are snapped afterwards,
    and the heteroskedastic shock is an extra draw only when that flag is on.
    """
    n_skus = sim.n_skus
    n_reg = sim.n_reg
    n_periods = sim.n_periods
    log_price_obs = np.empty((n_skus, n_reg, n_periods), dtype=np.float64)
    demand_shocks = np.empty((n_skus, n_reg, n_periods), dtype=np.float64)
    controls_panel = (
        np.empty((n_skus, n_reg, n_periods, sim.n_controls), dtype=np.float64)
        if sim.n_controls
        else None
    )
    instruments_panel = (
        np.empty((n_skus, n_reg, n_periods, sim.n_instruments), dtype=np.float64)
        if sim.n_instruments
        else None
    )
    hetero_panel = (
        np.empty((n_skus, n_reg, n_periods), dtype=np.float64)
        if sim.heteroskedasticity
        else None
    )

    for s in range(n_skus):
        for r in range(n_reg):
            log_price = np.asarray(sim.log_price_panel[s, r, :], dtype=np.float64).copy()
            demand_shock = rng.normal(0, sim.noise_sigma, size=n_periods)
            if sim.n_instruments:
                instruments = rng.normal(size=(n_periods, sim.n_instruments))
                log_price = log_price + sim.instrument_coef * instruments.sum(axis=1)
                assert instruments_panel is not None
                instruments_panel[s, r] = instruments
            if sim.endogeneity:
                log_price = log_price + sim.endogeneity * demand_shock
            log_price = log_price + sim.common_endogeneity * sim.lambda_at(r)
            if sim.n_controls:
                assert controls_panel is not None
                controls_panel[s, r] = rng.normal(size=(n_periods, sim.n_controls))
            if sim.heteroskedasticity:
                driver = rng.normal(size=n_periods)
                log_price = log_price + rng.normal(size=n_periods) * np.exp(
                    0.5 * driver
                )
                assert hetero_panel is not None
                hetero_panel[s, r] = driver
            log_price_obs[s, r] = log_price
            demand_shocks[s, r] = demand_shock
    if sim.price_distribution == "discrete":
        log_price_obs = _snap_log_price(
            log_price_obs, _discrete_price_grid(sim.n_price_points)
        )
    return (
        log_price_obs,
        np.exp(log_price_obs),
        demand_shocks,
        controls_panel,
        instruments_panel,
        hetero_panel,
    )


def _simulate_panel_rows(
    rng: np.random.Generator,
    sim: _PanelSim,
) -> list[dict]:
    rows: list[dict] = []
    beta_season = _PANEL_SEASON_BETA
    hierarchy = sim.effects.hierarchy
    (
        log_price_obs,
        price_obs,
        demand_shocks,
        controls_panel,
        instruments_panel,
        hetero_panel,
    ) = _draw_observed_log_prices(rng, sim)
    # Regular price is the random walk (plus instruments and endogeneity).
    # Promotions scale the paid price off that regular price.
    if sim.promotions:
        discount_panel = np.where(
            rng.random((sim.n_skus, sim.n_reg, sim.n_periods)) < sim.promo_frequency,
            sim.promo_depth,
            0.0,
        )
        paid_log_price = log_price_obs + np.log(1.0 - discount_panel)
    else:
        discount_panel = None
        paid_log_price = log_price_obs
    sku_log_p_mid = np.median(log_price_obs, axis=(1, 2))

    for s in range(sim.n_skus):
        for r in range(sim.n_reg):
            log_price = log_price_obs[s, r]
            price = price_obs[s, r]
            demand_shock = demand_shocks[s, r]
            instruments = None if instruments_panel is None else instruments_panel[s, r]
            hetero = None if hetero_panel is None else hetero_panel[s, r]

            season = (
                sim.period_season
                if sim.date_index is None
                else np.sin(sim.annual_phase)
            )
            season_term = beta_season * season if sim.include_seasonality else 0.0

            if (
                sim.n_controls
                and sim.control_coefs is not None
                and controls_panel is not None
            ):
                controls = controls_panel[s, r]
                ctrl_term = controls @ sim.control_coefs
            else:
                controls = None
                ctrl_term = 0.0

            if sim.date_index is not None and sim.volume_trend:
                t_years = np.asarray(
                    (sim.date_index - sim.date_index[0]) / pd.Timedelta(days=365.25),
                    dtype=np.float64,
                )
                trend_term = sim.volume_trend * t_years
            else:
                trend_term = 0.0

            common = (
                sim.effects.intercept[s]
                + season_term
                + ctrl_term
                + trend_term
                + demand_shock
                + sim.lambda_at(r)
            )
            mean_log_q = _mean_log_quantity(
                demand_shape=sim.demand_shape,
                common=common,
                log_price=log_price,
                price=price,
                elasticity=float(sim.effects.elasticity[s]),
                curvature=(
                    None
                    if sim.effects.curvature is None
                    else float(sim.effects.curvature[s])
                ),
                log_p_mid=float(sku_log_p_mid[s]),
            )
            if discount_panel is not None:
                mean_log_q = mean_log_q + sim.promo_elasticity * np.log(
                    1.0 - discount_panel[s, r]
                )

            if sim.gamma_map is not None:
                for t in range(sim.n_periods):
                    cross_sum = 0.0
                    for j in range(sim.n_skus):
                        if j == s:
                            continue
                        g = sim.gamma_map.get((s, j))
                        if g is not None:
                            cross_sum += g * paid_log_price[j, r, t]
                    mean_log_q[t] += cross_sum

            quantity = np.exp(mean_log_q)

            for t in range(sim.n_periods):
                if discount_panel is None:
                    paid_price = float(price[t])
                    paid_log = float(log_price[t])
                    discount_t = None
                else:
                    discount_t = float(discount_panel[s, r, t])
                    paid_price = float(price[t] * (1.0 - discount_t))
                    paid_log = float(np.log(paid_price))
                row: dict = {
                    "sku": f"sku_{s + 1}",
                    "period": sim.date_index[t] if sim.date_index is not None else t,
                    "price": paid_price,
                    "quantity": float(quantity[t]),
                    "log_price": paid_log,
                    "log_quantity": float(mean_log_q[t]),
                }
                if discount_t is not None:
                    row["discount_rate"] = discount_t
                if hierarchy is not None:
                    for L in range(hierarchy.n_levels):
                        row[f"category_{L + 1}"] = (
                            f"h{L + 1}_{int(hierarchy.sku_to_level_idx[s, L])}"
                        )
                elif sim.n_categories is not None and sim.cat_idx is not None:
                    row["category"] = f"category_{int(sim.cat_idx[s]) + 1}"
                if sim.region_labels is not None:
                    row["region"] = sim.region_labels[r]
                if controls is not None:
                    for k in range(sim.n_controls):
                        row[f"control_{k + 1}"] = float(controls[t, k])
                if instruments is not None:
                    for k in range(sim.n_instruments):
                        row[f"iv_{k + 1}"] = float(instruments[t, k])
                if hetero is not None:
                    row["hetero"] = float(hetero[t])
                rows.append(row)

    return rows


@overload
def generate_mock_data(
    n_periods: int,
    *,
    n_skus: int = 5,
    n_categories: int | None = None,
    hierarchy_levels: tuple[int, ...] | None = None,
    n_regions: int | None = None,
    n_controls: int = 0,
    random_state: RandomState = None,
    start_date: pd.Timestamp | str | None = None,
    freq: str = "W",
    shape: str = "log_log",
    cross_elasticity: str | None = None,
    cross_elasticity_group_level: int | None = None,
    include_seasonality: bool = False,
    price_shock_sigma: float = 0.03,
    volume_trend: float = 0.0,
    n_instruments: int = 0,
    instrument_coef: float = 0.5,
    endogeneity: float = 0.0,
    price_distribution: str = "normal",
    n_price_points: int = 3,
    heteroskedasticity: bool = False,
    common_shock_sigma: float = 0.0,
    common_endogeneity: float = 0.0,
    common_shock_by_region: bool = False,
    mundlak_coef: float = 0.0,
    noise_sigma: float | None = None,
    promotions: bool = False,
    promo_frequency: float = 0.3,
    promo_depth: float = 0.2,
    true_promo_elasticity: float | None = None,
    true_elasticity: float | None = None,
    return_truth: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def generate_mock_data(
    n_periods: int,
    *,
    n_skus: int = 5,
    n_categories: int | None = None,
    hierarchy_levels: tuple[int, ...] | None = None,
    n_regions: int | None = None,
    n_controls: int = 0,
    random_state: RandomState = None,
    start_date: pd.Timestamp | str | None = None,
    freq: str = "W",
    shape: str = "log_log",
    cross_elasticity: str | None = None,
    cross_elasticity_group_level: int | None = None,
    include_seasonality: bool = False,
    price_shock_sigma: float = 0.03,
    volume_trend: float = 0.0,
    n_instruments: int = 0,
    instrument_coef: float = 0.5,
    endogeneity: float = 0.0,
    price_distribution: str = "normal",
    n_price_points: int = 3,
    heteroskedasticity: bool = False,
    common_shock_sigma: float = 0.0,
    common_endogeneity: float = 0.0,
    common_shock_by_region: bool = False,
    mundlak_coef: float = 0.0,
    noise_sigma: float | None = None,
    promotions: bool = False,
    promo_frequency: float = 0.3,
    promo_depth: float = 0.2,
    true_promo_elasticity: float | None = None,
    true_elasticity: float | None = None,
    return_truth: Literal[True],
) -> tuple[pd.DataFrame, MockDataTruth]: ...


def generate_mock_data(
    n_periods: int,
    *,
    n_skus: int = 5,
    n_categories: int | None = None,
    hierarchy_levels: tuple[int, ...] | None = None,
    n_regions: int | None = None,
    n_controls: int = 0,
    random_state: RandomState = None,
    start_date: pd.Timestamp | str | None = None,
    freq: str = "W",
    shape: str = "log_log",
    cross_elasticity: str | None = None,
    cross_elasticity_group_level: int | None = None,
    include_seasonality: bool = False,
    price_shock_sigma: float = 0.03,
    volume_trend: float = 0.0,
    n_instruments: int = 0,
    instrument_coef: float = 0.5,
    endogeneity: float = 0.0,
    price_distribution: str = "normal",
    n_price_points: int = 3,
    heteroskedasticity: bool = False,
    common_shock_sigma: float = 0.0,
    common_endogeneity: float = 0.0,
    common_shock_by_region: bool = False,
    mundlak_coef: float = 0.0,
    noise_sigma: float | None = None,
    promotions: bool = False,
    promo_frequency: float = 0.3,
    promo_depth: float = 0.2,
    true_promo_elasticity: float | None = None,
    true_elasticity: float | None = None,
    return_truth: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, MockDataTruth]:
    """
    Build a long-format panel with neutral labels (sku_1, category_1, …).

    There are ``n_periods * n_skus`` rows when ``n_regions`` is omitted, or
    ``n_periods * n_skus * n_regions`` when regions are included (full factorial).

    Mean ``log_quantity`` follows one of:

    * ``"log_log"`` — ``LogLogDemandModel``: ``log Q ≈ α + ε log P + …``
    * ``"quadratic"`` — ``QuadraticLogDemandModel``: ``log Q ≈ α + β₁ log P + κ (log P)² + …``
      with ``β₁ = ε - 2 κ log P_mid`` and ``log P_mid`` the SKU's median observed
      log price, pooled across regions.
      Curvature ``κ`` uses a wider draw and a floor on ``|κ|`` so the parabolic term is easy to see.
    * ``"sigmoid"`` — ``SigmoidSaturationDemandModel``: ``log Q ≈ α - softplus(z) + …``
      with ``z = b (P - P_mid)``, ``P_mid = exp(SKU median log P)`` pooled across
      regions, and ``b = -2 ε / P_mid`` so elasticity at the center equals ``ε``.

    Common additions: controls, seasonality, Gaussian noise. Regions are separate
    price paths and cross-price cells, with no region intercept.

    Prices are partly set from the demand shock when ``endogeneity`` is not 0.
    ``price_distribution="normal"`` (the default) is a random walk on log price,
    one series per SKU–region path. ``"gamma"`` draws iid Gamma(2, scale=4)
    prices. ``"discrete"`` uses that same random walk, then snaps the price —
    after instruments and the demand shock — to ``n_price_points`` levels
    equally spaced in log price between 2 and 15. Optional ``iv_*`` columns
    shift log-price. ``heteroskedasticity=True`` adds a further log-price shock
    with standard deviation ``exp(0.5 * hetero)`` and writes the standard-normal
    driver as ``hetero``; that column does not enter the demand mean, and the
    flag off does not touch the random stream. Own-price and cross-price terms
    use the observed log prices. Positive ``endogeneity`` biases associational
    elasticity toward zero.

    Parameters
    ----------
    n_periods
        Time dimension length. ``period`` is ``0 … n_periods - 1`` unless
        ``start_date`` is set, in which case it is a datetime index.
    n_skus
        Number of SKUs.
    n_categories
        If ``None``, no ``category`` column (unless ``hierarchy_levels`` is set).
        Otherwise SKUs map to ``category_1``, … (requires ``n_categories <= n_skus``).
        Incompatible with ``hierarchy_levels``.
    hierarchy_levels
        If set, e.g. ``(3, 2)``, nested groups: ``category_1``, ``category_2``, …
        (coarse → fine) with a generative hierarchy for intercept, elasticity, and
        (for ``shape="quadratic"``) curvature. Incompatible with ``n_categories``.
    n_regions
        If ``None``, no ``region`` column and one slice per (period, sku).
        If ``>= 1``, include ``region_1`` … and expand to the full grid with
        one independent price path per (sku, region). Demand has no region
        intercept.
    n_controls
        Count of ``control_1`` … columns (standard normals). One coefficient per
        column, shared across SKUs, matching ``beta_control``.
    random_state
        Seed or ``numpy.random.Generator``.
    start_date
        If ``None``, integer ``period`` (``0 … n_periods - 1``). If set, ``period``
        is a calendar index with ``freq``, and seasonality uses day-of-year.
    freq
        Pandas offset alias when ``start_date`` is set (default ``\"W\"``).
    shape
        Demand mean shape: ``\"log_log\"``, ``\"quadratic\"``, or ``\"sigmoid\"`` only.
    cross_elasticity
        If set, ``\"all\"`` (directed cross-price terms for every SKU pair) or
        ``\"within_group\"`` (only pairs sharing the same group at
        ``cross_elasticity_group_level``); the latter requires ``hierarchy_levels``.
    cross_elasticity_group_level
        Required for ``\"within_group\"``: index into the hierarchy (``0`` = coarsest).
    include_seasonality
        If ``True``, add a sinusoidal term to mean log-quantity. Default
        ``False``. With ``start_date``, this is a yearly sine on day-of-year
        (matches ``seasonality="yearly"``). Without ``start_date``, it is a
        sine over integer ``0 … n_periods-1`` that the model cannot fit as
        Fourier.
    price_shock_sigma
        Std of per-period log-price innovations in the random walk (default ``0.03``).
        Larger values improve elasticity identification.
    volume_trend
        Shared annual log-quantity growth (e.g. ``0.1`` ≈ +10%/year). Requires
        ``start_date``. Stored on :class:`MockDataTruth` when ``return_truth=True``.
    n_instruments
        Count of ``iv_1`` … columns (standard normals). Each shifts log-price by
        ``instrument_coef`` (exclusion restriction: they do not enter demand).
    instrument_coef
        Shared first-stage slope on each instrument (default ``0.5``).
    endogeneity
        Loading of the demand shock onto log-price (default ``0.0`` = exogenous
        prices). Positive values bias associational elasticity toward zero.
    price_distribution
        Exogenous price draw: ``\"normal\"`` (default, log-price random walk),
        ``\"gamma\"`` (iid gamma prices), or ``\"discrete\"`` (the normal latent
        price snapped to a grid after the demand shock is added).
    n_price_points
        Number of levels when ``price_distribution="discrete"`` (default 3).
        Must be at least 2. Ignored otherwise.
    heteroskedasticity
        If ``True``, add a heteroskedastic log-price shock and a ``hetero``
        column (default ``False``). ``False`` leaves draws unchanged.
    common_shock_sigma
        Std of a period shock λ shared by every SKU (default ``0.0``, off).
        With ``common_shock_by_region``, one shock is drawn per region and period.
        ``0`` leaves the panel unchanged and does not touch the random stream.
    common_endogeneity
        Loading of λ onto log price (default ``0.0``). Log quantity always
        gets λ with coefficient 1. Positive values make high-demand periods
        high-price periods and bias elasticity toward zero when the model has
        no period effect.
    common_shock_by_region
        If ``True``, draw a separate λ per region (requires ``n_regions``).
        If ``False``, one national λ is shared by every region.
    mundlak_coef
        Shift each SKU intercept by this coefficient times the SKU's centered
        mean log price (default ``0.0``). Creates the correlation between
        baseline demand and price level that hierarchical intercepts shrink away.
    noise_sigma
        Std of the Gaussian demand shock (default ``0.12``).
    promotions
        If ``True``, mark a fraction ``promo_frequency`` of rows as temporary
        discounts of depth ``promo_depth`` (fraction off the regular price) and
        write ``discount_rate``. ``price`` is the price paid;
        ``regular = price / (1 - discount_rate)``. The promo term is log-linear:
        ``true_promo_elasticity * log(1 - discount)``. Cross-price terms use
        other SKUs' paid log prices. Fixed-amount discounts are not generated.
    promo_frequency
        Probability a row is on promo when ``promotions`` is set (default ``0.3``).
    promo_depth
        Discount fraction on promo rows (default ``0.2``, so 20% off).
    true_promo_elasticity
        Promo elasticity on every SKU when ``promotions`` is set. Default
        ``-3.5`` (short-term promo mean from Bijmolt et al. 2005).
    true_elasticity
        If set, every SKU's regular-price elasticity is this value instead of
        a random draw.
    return_truth
        If ``True``, also return a :class:`MockDataTruth` with per-SKU DGP
        parameters (and optional cross / control draws).

    Returns
    -------
    pandas.DataFrame or tuple
        Columns: ``sku``, ``period``, ``price``, ``quantity``, ``log_price``,
        ``log_quantity``, optional ``discount_rate`` (when ``promotions`` is
        set), ``category`` or ``category_1`` …, ``region``, ``control_*``,
        ``iv_*``, and ``hetero`` when ``heteroskedasticity`` is set. With
        ``return_truth=True``, ``(frame, truth)``.
    """
    ce = _validate_kwargs(
        n_periods,
        n_skus,
        n_categories,
        hierarchy_levels,
        n_regions,
        n_controls,
        cross_elasticity=cross_elasticity,
        cross_elasticity_group_level=cross_elasticity_group_level,
        n_instruments=n_instruments,
    )

    demand_shape = _parse_demand_shape(shape)
    price_distribution = _parse_price_distribution(price_distribution)
    if price_distribution == "discrete":
        _discrete_price_grid(n_price_points)
    if not isinstance(heteroskedasticity, bool):
        raise TypeError("heteroskedasticity must be True or False")

    if price_shock_sigma <= 0:
        raise ValueError("price_shock_sigma must be positive")
    if common_shock_sigma < 0:
        raise ValueError("common_shock_sigma must be non-negative")
    if common_shock_by_region and n_regions is None:
        raise ValueError("common_shock_by_region requires n_regions")
    if volume_trend != 0.0 and start_date is None:
        raise ValueError("volume_trend requires start_date")
    if noise_sigma is None:
        noise_sigma = _PANEL_NOISE_SIGMA
    elif noise_sigma <= 0:
        raise ValueError("noise_sigma must be positive")
    if promotions:
        if not 0.0 < promo_frequency <= 1.0:
            raise ValueError("promo_frequency must be in (0, 1]")
        if not 0.0 < promo_depth < 1.0:
            raise ValueError("promo_depth must be in (0, 1)")
        if true_promo_elasticity is None:
            true_promo_elasticity = -3.5

    rng = _as_rng(random_state)

    n_reg = n_regions if n_regions is not None else 1
    region_labels = (
        None if n_regions is None else [f"region_{r + 1}" for r in range(n_regions)]
    )

    if start_date is not None:
        date_index, annual_phase = _calendar_seasonality_index(
            pd.Timestamp(start_date), n_periods, freq
        )
    else:
        date_index = None
        annual_phase = np.empty(0)

    t_idx = np.arange(n_periods, dtype=float)
    period_season = np.sin(2 * np.pi * t_idx / max(n_periods, 1))

    effects = _draw_sku_effects(
        rng,
        n_skus=n_skus,
        hierarchy_levels=hierarchy_levels,
        demand_shape=demand_shape,
    )
    if true_elasticity is not None:
        effects.elasticity = np.full(n_skus, float(true_elasticity), dtype=np.float64)

    control_coefs = (
        rng.normal(0, 0.12, size=n_controls) if n_controls else None
    )

    cat_idx: np.ndarray | None = None
    if n_categories is not None:
        skus_per_cat = int(np.ceil(n_skus / n_categories))
        cat_idx = np.minimum(np.arange(n_skus) // skus_per_cat, n_categories - 1)

    log_price_panel = _simulate_log_price_panel(
        rng,
        n_skus=n_skus,
        n_reg=n_reg,
        n_periods=n_periods,
        price_shock_sigma=price_shock_sigma,
        price_distribution=price_distribution,
    )

    if common_shock_sigma == 0.0:
        shape = (n_reg, n_periods) if common_shock_by_region else (n_periods,)
        lambda_period = np.zeros(shape)
    elif common_shock_by_region:
        lambda_period = rng.normal(0.0, common_shock_sigma, size=(n_reg, n_periods))
    else:
        lambda_period = rng.normal(0.0, common_shock_sigma, size=n_periods)
    if mundlak_coef:
        sku_mean_log_p = log_price_panel.mean(axis=(1, 2))
        centered_log_p = sku_mean_log_p - float(sku_mean_log_p.mean())
        effects.intercept = np.asarray(effects.intercept, dtype=np.float64) + (
            float(mundlak_coef) * centered_log_p
        )
    gamma_map: dict[tuple[int, int], float] | None = None
    if ce is not None:
        gamma_map = _draw_cross_gamma_map(
            rng,
            mode=ce,
            n_skus=n_skus,
            hierarchy=effects.hierarchy,
            cross_elasticity_group_level=cross_elasticity_group_level,
        )

    rows = _simulate_panel_rows(
        rng,
        _PanelSim(
            n_skus=n_skus,
            n_reg=n_reg,
            n_periods=n_periods,
            n_controls=n_controls,
            n_categories=n_categories,
            demand_shape=demand_shape,
            effects=effects,
            cat_idx=cat_idx,
            region_labels=region_labels,
            control_coefs=control_coefs,
            log_price_panel=log_price_panel,
            gamma_map=gamma_map,
            date_index=date_index,
            period_season=period_season,
            annual_phase=annual_phase,
            include_seasonality=include_seasonality,
            volume_trend=float(volume_trend),
            n_instruments=int(n_instruments),
            instrument_coef=float(instrument_coef),
            endogeneity=float(endogeneity),
            common_endogeneity=float(common_endogeneity),
            price_distribution=price_distribution,
            n_price_points=int(n_price_points),
            heteroskedasticity=heteroskedasticity,
            lambda_period=lambda_period,
            noise_sigma=float(noise_sigma),
            promotions=bool(promotions),
            promo_frequency=float(promo_frequency),
            promo_depth=float(promo_depth),
            promo_elasticity=float(true_promo_elasticity or 0.0),
        ),
    )
    df = pd.DataFrame(rows)
    if not return_truth:
        return df

    sku_labels = tuple(f"sku_{s + 1}" for s in range(n_skus))
    truth = MockDataTruth(
        shape=demand_shape,
        sku_labels=sku_labels,
        alpha_sku=np.asarray(effects.intercept, dtype=np.float64).copy(),
        elasticity_sku=np.asarray(effects.elasticity, dtype=np.float64).copy(),
        curvature_sku=(
            None
            if effects.curvature is None
            else np.asarray(effects.curvature, dtype=np.float64).copy()
        ),
        noise_sigma=float(noise_sigma),
        control_coefs=(
            None
            if control_coefs is None
            else np.asarray(control_coefs, dtype=np.float64).copy()
        ),
        gamma_pair=None if gamma_map is None else dict(gamma_map),
        volume_trend=float(volume_trend),
        n_instruments=int(n_instruments),
        instrument_coef=float(instrument_coef),
        endogeneity=float(endogeneity),
        price_distribution=price_distribution,
        n_price_points=int(n_price_points),
        heteroskedasticity=heteroskedasticity,
        promo_elasticity=None if not promotions else float(true_promo_elasticity),
        lambda_period=np.asarray(lambda_period, dtype=np.float64).copy(),
        common_shock_sigma=float(common_shock_sigma),
        common_endogeneity=float(common_endogeneity),
    )
    return df, truth
