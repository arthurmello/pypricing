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

# Stronger nonlinearity for ``shape="quadratic"`` / ``"sigmoid"`` so demo plots show
# visible curvature (still the same functional form as the corresponding model).
_QUADRATIC_CURVATURE_STD = 0.55
_QUADRATIC_CURVATURE_MIN_ABS = 0.12
_SIGMOID_SLOPE_MULTIPLIER = 2.8
_PANEL_NOISE_SIGMA = 0.12
_PANEL_SEASON_BETA = 0.15


@dataclass(frozen=True)
class MockDataTruth:
    """Ground-truth DGP parameters for a panel from :func:`generate_mock_data`.

    SKU-aligned arrays match labels ``sku_1``, … in ``sku_labels`` order.
    """

    shape: str
    sku_labels: tuple[str, ...]
    alpha_sku: np.ndarray
    elasticity_sku: np.ndarray
    curvature_sku: np.ndarray | None
    noise_sigma: float
    control_coefs: np.ndarray | None
    gamma_pair: dict[tuple[int, int], float] | None


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
    n_regions: int | None
    n_controls: int
    n_categories: int | None
    demand_shape: str
    effects: _SkuEffects
    cat_idx: np.ndarray | None
    region_labels: list[str] | None
    region_shift: np.ndarray
    control_coefs: np.ndarray | None
    log_price_panel: np.ndarray
    gamma_map: dict[tuple[int, int], float] | None
    date_index: pd.DatetimeIndex | None
    period_season: np.ndarray
    annual_phase: np.ndarray
    include_seasonality: bool
    round_quantity: bool


def _as_rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


_VALID_DEMAND_SHAPES = frozenset({"log_log", "quadratic", "sigmoid"})


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
) -> np.ndarray:
    log_price_panel = np.zeros((n_skus, n_reg, n_periods), dtype=np.float64)
    for s in range(n_skus):
        for r in range(n_reg):
            base_price = rng.uniform(2.0, 15.0)
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
) -> np.ndarray:
    match demand_shape:
        case "log_log":
            return common + elasticity * log_price
        case "quadratic":
            if curvature is None:
                raise RuntimeError("quadratic demand requires curvature")
            log_p_mid = float(np.median(log_price))
            beta1 = elasticity - 2.0 * curvature * log_p_mid
            return common + beta1 * log_price + curvature * (log_price**2)
        case "sigmoid":
            # Same form as SigmoidSaturationDemandModel; multiplier steepens the bend.
            log_p_mid = float(np.median(log_price))
            p_center = np.exp(log_p_mid)
            b = _SIGMOID_SLOPE_MULTIPLIER * (-2.0 * elasticity / p_center)
            z = b * (price - p_center)
            return common - softplus(z)
        case _:
            raise ValueError(f"Unsupported demand shape: {demand_shape}")


def _simulate_panel_rows(
    rng: np.random.Generator,
    sim: _PanelSim,
) -> list[dict]:
    rows: list[dict] = []
    beta_season = _PANEL_SEASON_BETA
    sigma = _PANEL_NOISE_SIGMA
    hierarchy = sim.effects.hierarchy

    for s in range(sim.n_skus):
        for r in range(sim.n_reg):
            log_price = sim.log_price_panel[s, r, :]
            price = np.exp(log_price)

            season = (
                sim.period_season
                if sim.date_index is None
                else np.sin(sim.annual_phase)
            )
            season_term = beta_season * season if sim.include_seasonality else 0.0
            shift_r = sim.region_shift[r] if sim.n_regions is not None else 0.0

            if sim.n_controls and sim.control_coefs is not None:
                controls = rng.normal(size=(sim.n_periods, sim.n_controls))
                ctrl_term = controls @ sim.control_coefs[s]
            else:
                controls = None
                ctrl_term = 0.0

            common = (
                sim.effects.intercept[s]
                + shift_r
                + season_term
                + ctrl_term
                + rng.normal(0, sigma, size=sim.n_periods)
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
            )

            if sim.gamma_map is not None:
                for t in range(sim.n_periods):
                    cross_sum = 0.0
                    for j in range(sim.n_skus):
                        if j == s:
                            continue
                        g = sim.gamma_map.get((s, j))
                        if g is not None:
                            cross_sum += g * sim.log_price_panel[j, r, t]
                    mean_log_q[t] += cross_sum

            if sim.round_quantity:
                quantity = np.exp(mean_log_q).astype(int).clip(1)
                log_q = np.log(np.maximum(quantity.astype(np.float64), 1.0))
                price_out = np.round(price, 2)
                log_p_out = np.round(log_price, 4)
                log_q_out = np.round(log_q, 4)
            else:
                # Continuous / full-precision panel for recovery (model uses log(price)).
                quantity = np.exp(mean_log_q)
                log_q = mean_log_q
                price_out = price
                log_p_out = log_price
                log_q_out = log_q

            for t in range(sim.n_periods):
                q_t = float(quantity[t])
                row: dict = {
                    "sku": f"sku_{s + 1}",
                    "period": t,
                    "price": float(price_out[t]),
                    "quantity": int(q_t) if sim.round_quantity else q_t,
                    "log_price": float(log_p_out[t]),
                    "log_quantity": float(log_q_out[t]),
                }
                if sim.date_index is not None:
                    row["date"] = sim.date_index[t]
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
                        row[f"control_{k + 1}"] = round(float(controls[t, k]), 4)
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
    include_seasonality: bool = True,
    round_quantity: bool = True,
    price_shock_sigma: float = 0.03,
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
    include_seasonality: bool = True,
    round_quantity: bool = True,
    price_shock_sigma: float = 0.03,
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
    include_seasonality: bool = True,
    round_quantity: bool = True,
    price_shock_sigma: float = 0.03,
    return_truth: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, MockDataTruth]:
    """
    Build a long-format panel with neutral labels (sku_1, category_1, …).

    There are ``n_periods * n_skus`` rows when ``n_regions`` is omitted, or
    ``n_periods * n_skus * n_regions`` when regions are included (full factorial).

    Mean ``log_quantity`` (before integer rounding) follows one of:

    * ``"log_log"`` — ``LogLogDemandModel``: ``log Q ≈ α + ε log P + …``
    * ``"quadratic"`` — ``QuadraticLogDemandModel``: ``log Q ≈ α + β₁ log P + κ (log P)² + …``
      with ``β₁ = ε - 2 κ log P_mid`` and ``log P_mid`` the median log price on the path.
      Curvature ``κ`` uses a wider draw and a floor on ``|κ|`` so the parabolic term is easy to see.
    * ``"sigmoid"`` — ``SigmoidSaturationDemandModel``: ``log Q ≈ α - softplus(z) + …``
      with ``z = b (P - P_mid)``, ``P_mid = exp(median log P)``, and ``b`` proportional to
      ``-2 ε / P_mid`` (synthetic data applies a slope multiplier so the saturation bend is clearer).

    Common additions: region shift, controls, seasonality, Gaussian noise.

    Prices use a random walk on log scale (one series per SKU–region path).

    Parameters
    ----------
    n_periods
        Time dimension length. Integer ``period`` runs ``0 … n_periods - 1``.
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
        one independent price path per (sku, region).
    n_controls
        Count of ``control_1`` … columns (standard normals with random coefficients).
    random_state
        Seed or ``numpy.random.Generator``.
    start_date
        If ``None``, no ``date`` column; seasonality is a single cycle over
        ``period``. If set, builds a calendar index with ``freq`` and ``n_periods``,
        adds a ``date`` column, and uses annual (day-of-year) seasonality.
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
        If ``True`` (default), add a sinusoidal seasonality term to mean log-quantity.
        Set ``False`` for cleaner parameter-recovery experiments.
    round_quantity
        If ``True`` (default), store integer quantities (clipped at 1). Set ``False``
        to keep continuous quantity / ``log_quantity`` (useful for recovery tests).
    price_shock_sigma
        Std of per-period log-price innovations in the random walk (default ``0.03``).
        Larger values improve elasticity identification.
    return_truth
        If ``True``, also return a :class:`MockDataTruth` with per-SKU DGP
        parameters (and optional cross / control draws).

    Returns
    -------
    pandas.DataFrame or tuple
        Columns: ``sku``, ``period``, ``price``, ``quantity``, ``log_price``,
        ``log_quantity``, optional ``category`` or ``category_1`` …, ``region``,
        ``control_*``, optional ``date``. With ``return_truth=True``,
        ``(frame, truth)``.
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
    )

    demand_shape = _parse_demand_shape(shape)

    if price_shock_sigma <= 0:
        raise ValueError("price_shock_sigma must be positive")

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

    region_shift = (
        rng.normal(0, 0.15, size=n_regions) if n_regions is not None else np.zeros(1)
    )

    control_coefs = (
        rng.normal(0, 0.12, size=(n_skus, n_controls)) if n_controls else None
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
            n_regions=n_regions,
            n_controls=n_controls,
            n_categories=n_categories,
            demand_shape=demand_shape,
            effects=effects,
            cat_idx=cat_idx,
            region_labels=region_labels,
            region_shift=region_shift,
            control_coefs=control_coefs,
            log_price_panel=log_price_panel,
            gamma_map=gamma_map,
            date_index=date_index,
            period_season=period_season,
            annual_phase=annual_phase,
            include_seasonality=include_seasonality,
            round_quantity=round_quantity,
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
        noise_sigma=float(_PANEL_NOISE_SIGMA),
        control_coefs=(
            None
            if control_coefs is None
            else np.asarray(control_coefs, dtype=np.float64).copy()
        ),
        gamma_pair=None if gamma_map is None else dict(gamma_map),
    )
    return df, truth
