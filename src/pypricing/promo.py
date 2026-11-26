"""Promo planning on a fitted model: plan frames, scenario comparison, optimization.

Everything here works on posterior draws of *expected* quantity
(:meth:`DemandModel.expected_quantity_draws`), so scenarios are compared on the
same draws and probabilities such as "plan A beats the baseline" are exact for
the model.

Vocabulary
----------
plan frame
    Future rows (one per SKU, period and region) at regular price, built by
    :func:`plan_frame`. Rows flagged ``promo_window`` can receive a promo; the
    rest are a tail where post-promo lags play out.
unit
    The decision level: the combination of ``unit_cols`` (default the SKU, or
    e.g. ``("brand", "category")``). Every SKU in a unit gets the same promo.
option
    A ``(discount, promo_type)`` pair; ``discount = 0`` is "no promo".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pypricing.data.price_panel import parse_period_index
from pypricing.events import (
    BUILTIN_EVENTS,
    EVENT_PREFIX,
    _period_length,
    event_indicators,
)
from pypricing.model_components.promo_extras import NO_PROMO_TYPE

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel

WINDOW_COL = "promo_window"
OBJECTIVES = ("margin", "revenue", "units")


# ---------------------------------------------------------------- plan frames


def plan_frame(
    model: DemandModel,
    periods: int | Sequence[Any],
    *,
    history: pd.DataFrame | None = None,
    tail: int | None = None,
    unit_cost_col: str | None = "unit_cost",
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Future rows at regular price, one per SKU (and region) and period.

    Each SKU starts from its last observed row in ``history`` (default: the
    training data): regular price, unit cost, controls and group columns carry
    over. Event controls (``control_event_*``) are recomputed for the new
    dates. ``periods`` is a count of periods after the last training period
    (at the training spacing) or explicit dates. ``tail`` extra periods (default:
    the model's promo lags) follow with ``promo_window = False``. ``overrides``
    (keyed by SKU, and period when present) replaces any column, e.g. a
    forecast competitor price.
    """
    model._require_fitted()
    cols = model.panel_columns
    history = model.data if history is None else history
    if history is None:
        raise ValueError("No history: pass `history` or fit the model first")
    period_col = cols.period_col
    if period_col not in history.columns:
        raise ValueError(f"history needs period column {period_col!r}")
    discount_col = cols.discount_col

    index = parse_period_index(history[period_col], period_col=period_col)
    unique = index.unique().sort_values()
    step = _period_length(index) if len(unique) > 1 else pd.Timedelta(days=7)
    if isinstance(periods, int):
        if periods < 1:
            raise ValueError("periods must be >= 1")
        window = [unique[-1] + step * (k + 1) for k in range(periods)]
    else:
        window = list(
            parse_period_index(pd.Series(list(periods)), period_col=period_col)
        )
    if tail is None:
        tail = model.promo.lags if model.promo is not None else 0
    tail_periods = [window[-1] + step * (k + 1) for k in range(tail)]

    keys = [cols.sku_col]
    if cols.region_col is not None and cols.region_col in history.columns:
        keys.append(cols.region_col)
    hist = history.assign(_t=index.to_numpy()).sort_values("_t", kind="stable")
    last = hist.groupby(keys, sort=False).tail(1).drop(columns="_t")

    regular = last[cols.price_col].to_numpy(dtype=np.float64)
    if discount_col is not None and discount_col in last.columns:
        regular = regular / (1.0 - last[discount_col].to_numpy(dtype=np.float64))

    frames = []
    for when, in_window in [(p, True) for p in window] + [
        (p, False) for p in tail_periods
    ]:
        block = last.copy()
        block[period_col] = when
        block[cols.price_col] = regular
        if discount_col is not None:
            block[discount_col] = 0.0
        if model.promo is not None and model.promo.type_col is not None:
            block[model.promo.type_col] = NO_PROMO_TYPE
        block[WINDOW_COL] = in_window
        frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    if cols.quantity_col in out.columns:
        out[cols.quantity_col] = np.nan

    event_cols = [c for c in out.columns if c.startswith(EVENT_PREFIX)]
    if event_cols:
        names = [c[len(EVENT_PREFIX) :] for c in event_cols]
        builtin = [n for n in names if n in BUILTIN_EVENTS]
        flags = event_indicators(out[period_col], events=builtin, period_length=step)
        for col, name in zip(event_cols, names):
            out[col] = flags[name].to_numpy() if name in flags.columns else 0.0

    if overrides is not None:
        out = _apply_overrides(out, overrides, keys=keys, period_col=period_col)
    if unit_cost_col is not None and unit_cost_col not in out.columns:
        raise ValueError(f"history has no unit cost column {unit_cost_col!r}")
    return out


def _apply_overrides(
    frame: pd.DataFrame, overrides: pd.DataFrame, *, keys: list[str], period_col: str
) -> pd.DataFrame:
    on = [k for k in [*keys, period_col] if k in overrides.columns]
    if not on:
        raise ValueError(
            f"overrides need at least one key column of {keys + [period_col]}"
        )
    values = [c for c in overrides.columns if c not in on]
    merged = frame.merge(overrides, on=on, how="left", suffixes=("", "__new"))
    for c in values:
        new = f"{c}__new" if f"{c}__new" in merged.columns else c
        if new != c:
            merged[c] = merged[new].where(merged[new].notna(), merged[c])
            merged = merged.drop(columns=new)
    return merged


# ------------------------------------------------------------- evaluation


@dataclass
class Draws:
    """Per-draw, per-row outcomes, shape ``(n_draws, n_rows)``."""

    quantity: np.ndarray
    revenue: np.ndarray
    margin: np.ndarray | None
    discount_cost: np.ndarray

    def metric(self, objective: str) -> np.ndarray:
        if objective == "units":
            return self.quantity
        if objective == "revenue":
            return self.revenue
        if self.margin is None:
            raise ValueError("objective='margin' needs a unit cost column")
        return self.margin


def evaluate(
    model: DemandModel,
    frame: pd.DataFrame,
    *,
    unit_cost_col: str | None = "unit_cost",
) -> Draws:
    """Expected quantity, revenue, margin and promo cost per draw and row.

    Promo cost is the revenue given away on units sold: ``Q * regular * d``.
    """
    cols = model.panel_columns
    q = model.expected_quantity_draws(frame)
    q = q.reshape(-1, q.shape[-1])
    price = frame[cols.price_col].to_numpy(dtype=np.float64)
    d = (
        frame[cols.discount_col].to_numpy(dtype=np.float64)
        if cols.discount_col is not None
        else np.zeros(len(frame))
    )
    regular = price / (1.0 - d)
    margin = None
    if unit_cost_col is not None and unit_cost_col in frame.columns:
        cost = frame[unit_cost_col].to_numpy(dtype=np.float64)
        margin = q * (price - cost)[None, :]
    return Draws(
        quantity=q,
        revenue=q * price[None, :],
        margin=margin,
        discount_cost=q * (regular * d)[None, :],
    )


def apply_promos(
    model: DemandModel,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    unit_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Set discount (and promo type) on window rows of the assigned units.

    ``assignments`` has the unit columns, ``discount`` and optionally
    ``promo_type``; a ``period`` column limits a row to that period. Price is
    rescaled from the regular price so ``price = regular * (1 - discount)``.
    """
    cols = model.panel_columns
    if cols.discount_col is None:
        raise ValueError("The model has no discount column")
    unit_cols = list(unit_cols or [cols.sku_col])
    missing = [c for c in [*unit_cols, "discount"] if c not in assignments.columns]
    if missing:
        raise ValueError(f"assignments missing columns {missing}")
    d = assignments["discount"].to_numpy(dtype=np.float64)
    if ((d < 0) | (d >= 1)).any():
        raise ValueError("discount must be in [0, 1)")

    out = frame.copy()
    current = out[cols.discount_col].to_numpy(dtype=np.float64)
    regular = out[cols.price_col].to_numpy(dtype=np.float64) / (1.0 - current)
    window = (
        out[WINDOW_COL].to_numpy(dtype=bool)
        if WINDOW_COL in out.columns
        else np.ones(len(out), dtype=bool)
    )
    type_col = model.promo.type_col if model.promo is not None else None
    on = list(unit_cols)
    if cols.period_col in assignments.columns:
        on.append(cols.period_col)
    lookup = assignments.set_index(on)
    if not lookup.index.is_unique:
        raise ValueError("assignments must have one row per unit (and period)")
    row_keys = (
        pd.MultiIndex.from_frame(out[on]) if len(on) > 1 else pd.Index(out[on[0]])
    )
    pos = lookup.index.get_indexer(row_keys)
    hit = (pos >= 0) & window
    new_d = current.copy()
    new_d[hit] = lookup["discount"].to_numpy(dtype=np.float64)[pos[hit]]
    out[cols.discount_col] = new_d
    out[cols.price_col] = regular * (1.0 - new_d)
    if type_col is not None:
        types = out[type_col].astype(object).to_numpy().copy()
        if "promo_type" in lookup.columns:
            types[hit] = lookup["promo_type"].astype(object).to_numpy()[pos[hit]]
        types[new_d <= 0] = NO_PROMO_TYPE
        out[type_col] = types
    return out


def _hdi(x: np.ndarray, prob: float) -> tuple[float, float]:
    lo, hi = np.quantile(x, [(1 - prob) / 2, 1 - (1 - prob) / 2])
    return float(lo), float(hi)


def _check_objective(objective: str) -> None:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}; got {objective!r}")


@dataclass
class ScenarioComparison:
    """``summary``: one row per scenario. ``draws``: objective totals ``(n_draws, k)``."""

    summary: pd.DataFrame
    draws: pd.DataFrame


def compare_scenarios(
    model: DemandModel,
    base_frame: pd.DataFrame,
    scenarios: Mapping[str, pd.DataFrame],
    *,
    unit_cols: Sequence[str] | None = None,
    objective: str = "margin",
    unit_cost_col: str | None = "unit_cost",
    hdi_prob: float = 0.9,
) -> ScenarioComparison:
    """Compare promo scenarios against the no-promo baseline on the same draws.

    Each scenario is an assignments frame for :func:`apply_promos`. The summary
    has the posterior mean and HDI of the objective total, its increment over
    the baseline, ``prob_beats_baseline``, ``prob_best`` (across the scenarios,
    baseline included), and mean units, revenue, margin and promo cost.
    """
    _check_objective(objective)
    totals: dict[str, np.ndarray] = {}
    rows = []
    frames = {"baseline": base_frame}
    for name, assignments in scenarios.items():
        if name == "baseline":
            raise ValueError("'baseline' is reserved")
        frames[name] = apply_promos(model, base_frame, assignments, unit_cols=unit_cols)
    metrics: dict[str, Draws] = {}
    for name, frame in frames.items():
        draws = evaluate(model, frame, unit_cost_col=unit_cost_col)
        metrics[name] = draws
        totals[name] = draws.metric(objective).sum(axis=1)
    stacked = np.column_stack(list(totals.values()))
    best = np.argmax(stacked, axis=1)
    base = totals["baseline"]
    for k, (name, total) in enumerate(totals.items()):
        inc = total - base
        d = metrics[name]
        lo, hi = _hdi(total, hdi_prob)
        ilo, ihi = _hdi(inc, hdi_prob)
        rows.append(
            {
                "scenario": name,
                "objective": objective,
                "mean": float(total.mean()),
                "hdi_lower": lo,
                "hdi_upper": hi,
                "incremental_mean": float(inc.mean()),
                "incremental_hdi_lower": ilo,
                "incremental_hdi_upper": ihi,
                "prob_beats_baseline": float(np.mean(inc > 0))
                if name != "baseline"
                else np.nan,
                "prob_best": float(np.mean(best == k)),
                "units": float(d.quantity.sum(axis=1).mean()),
                "revenue": float(d.revenue.sum(axis=1).mean()),
                "margin": float(d.margin.sum(axis=1).mean())
                if d.margin is not None
                else np.nan,
                "promo_cost": float(d.discount_cost.sum(axis=1).mean()),
            }
        )
    summary = pd.DataFrame(rows).set_index("scenario")
    return ScenarioComparison(summary=summary, draws=pd.DataFrame(totals))


# ---------------------------------------------------------------- options


def _units(frame: pd.DataFrame, unit_cols: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in unit_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"plan frame missing unit columns {missing}")
    return frame[list(unit_cols)].drop_duplicates().reset_index(drop=True)


def _option_grid(
    discounts: Sequence[float], promo_types: Sequence[str] | None
) -> pd.DataFrame:
    depths = sorted({float(d) for d in discounts if d > 0})
    if any(d >= 1 or d < 0 for d in discounts):
        raise ValueError("discounts must be in [0, 1)")
    rows = [{"discount": 0.0, "promo_type": NO_PROMO_TYPE}]
    for d in depths:
        for t in promo_types or [None]:
            rows.append({"discount": d, "promo_type": t})
    return pd.DataFrame(rows)


def _row_unit_codes(
    frame: pd.DataFrame, units: pd.DataFrame, unit_cols: Sequence[str]
) -> np.ndarray:
    key = pd.MultiIndex.from_frame(units[list(unit_cols)])
    rows = pd.MultiIndex.from_frame(frame[list(unit_cols)])
    return key.get_indexer(rows)


def _is_coupled(model: DemandModel) -> bool:
    pressure = model.promo is not None and model.promo.pressure_level is not None
    return pressure or model.cross_elasticity is not None


def _observed_max_discount(
    model: DemandModel, units: pd.DataFrame, unit_cols: Sequence[str]
) -> np.ndarray:
    data = model.data
    disc = model.panel_columns.discount_col
    if data is None or disc is None or any(c not in data.columns for c in unit_cols):
        return np.full(len(units), np.inf)
    seen = data.groupby(list(unit_cols))[disc].max()
    key = (
        pd.MultiIndex.from_frame(units[list(unit_cols)])
        if len(unit_cols) > 1
        else units[unit_cols[0]]
    )
    return seen.reindex(key).fillna(0.0).to_numpy(dtype=np.float64)


@dataclass
class PromoOptions:
    """Every unit x option, with incremental outcomes vs no promo for that unit.

    ``table`` columns: unit columns, ``discount``, ``promo_type``,
    ``incremental_{objective}`` mean / HDI, ``prob_positive``,
    ``incremental_units``, ``incremental_revenue``, ``incremental_margin``,
    ``promo_cost`` (means) and ``extrapolated`` (deeper than any discount the
    unit had in the training data). ``draws[objective]`` is
    ``(n_draws, n_units, n_options)``.
    """

    table: pd.DataFrame
    units: pd.DataFrame
    options: pd.DataFrame
    draws: dict[str, np.ndarray] = field(repr=False)
    baseline: dict[str, float] = field(default_factory=dict)
    unit_cols: tuple[str, ...] = ()


def promo_options(
    model: DemandModel,
    base_frame: pd.DataFrame,
    *,
    discounts: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
    promo_types: Sequence[str] | None = None,
    unit_cols: Sequence[str] | None = None,
    objective: str = "margin",
    unit_cost_col: str | None = "unit_cost",
    hdi_prob: float = 0.9,
) -> PromoOptions:
    """Incremental outcome of each option for each unit, all else at regular price.

    Without coupling terms each option is evaluated for all units in one pass.
    With within-group pressure or cross-price terms each unit is evaluated on
    its own, so the numbers are the unit's marginal effect (its own lift minus
    what it takes from the others).
    """
    _check_objective(objective)
    cols = model.panel_columns
    unit_cols = tuple(unit_cols or (cols.sku_col,))
    type_col = model.promo.type_col if model.promo is not None else None
    if promo_types is not None and type_col is None:
        raise ValueError("promo_types needs a model fit with PromoSpec.type_col")
    if type_col is not None and promo_types is None:
        promo_types = list(model.promo_type_levels_ or [])[:1] or None
    units = _units(base_frame, unit_cols)
    options = _option_grid(discounts, promo_types)
    n_u, n_o = len(units), len(options)
    row_unit = _row_unit_codes(base_frame, units, unit_cols)

    base = evaluate(model, base_frame, unit_cost_col=unit_cost_col)
    names = ["quantity", "revenue", "discount_cost"] + (
        ["margin"] if base.margin is not None else []
    )
    base_tot = {n: getattr(base, n).sum(axis=1) for n in names}
    n_draws = base.quantity.shape[0]
    inc = {n: np.zeros((n_draws, n_u, n_o)) for n in names}

    def assign(option: pd.Series, which: np.ndarray) -> pd.DataFrame:
        a = units.iloc[which].copy()
        a["discount"] = option["discount"]
        if option["promo_type"] is not None:
            a["promo_type"] = option["promo_type"]
        return a

    for o in range(1, n_o):
        option = options.iloc[o]
        if not _is_coupled(model):
            frame = apply_promos(
                model, base_frame, assign(option, np.arange(n_u)), unit_cols=unit_cols
            )
            d = evaluate(model, frame, unit_cost_col=unit_cost_col)
            for n in names:
                delta = getattr(d, n) - getattr(base, n)
                for u in range(n_u):
                    inc[n][:, u, o] = delta[:, row_unit == u].sum(axis=1)
            continue
        for u in range(n_u):
            frame = apply_promos(
                model, base_frame, assign(option, np.array([u])), unit_cols=unit_cols
            )
            d = evaluate(model, frame, unit_cost_col=unit_cost_col)
            for n in names:
                inc[n][:, u, o] = getattr(d, n).sum(axis=1) - base_tot[n]

    obj_key = {"units": "quantity", "revenue": "revenue", "margin": "margin"}[objective]
    if obj_key not in inc:
        raise ValueError("objective='margin' needs a unit cost column")
    seen_max = _observed_max_discount(model, units, unit_cols)
    rows = []
    for u in range(n_u):
        for o in range(n_o):
            x = inc[obj_key][:, u, o]
            lo, hi = _hdi(x, hdi_prob)
            row = {c: units.iloc[u][c] for c in unit_cols}
            disc = float(options.iloc[o]["discount"])
            row.update(
                {
                    "discount": disc,
                    "promo_type": options.iloc[o]["promo_type"],
                    f"incremental_{objective}": float(x.mean()),
                    "hdi_lower": lo,
                    "hdi_upper": hi,
                    "prob_positive": float(np.mean(x > 0)) if o else np.nan,
                    "incremental_units": float(inc["quantity"][:, u, o].mean()),
                    "incremental_revenue": float(inc["revenue"][:, u, o].mean()),
                    "incremental_margin": float(inc["margin"][:, u, o].mean())
                    if "margin" in inc
                    else np.nan,
                    "promo_cost": float(inc["discount_cost"][:, u, o].mean()),
                    "extrapolated": bool(disc > seen_max[u] + 1e-9),
                }
            )
            rows.append(row)
    draws = {objective: inc[obj_key]}
    for n, key in (
        ("units", "quantity"),
        ("revenue", "revenue"),
        ("margin", "margin"),
        ("promo_cost", "discount_cost"),
    ):
        if key in inc:
            draws[n] = inc[key]
    baseline = {n: float(v.mean()) for n, v in base_tot.items()}
    return PromoOptions(
        table=pd.DataFrame(rows),
        units=units,
        options=options,
        draws=draws,
        baseline=baseline,
        unit_cols=unit_cols,
    )


# -------------------------------------------------------------- optimization


@dataclass
class PromoConstraints:
    """Limits on a promo plan. ``None`` means no limit.

    budget
        Maximum expected promo cost (revenue given away, ``Q * regular * d``).
    max_promos
        Maximum number of units on promo.
    max_promos_per
        Column name -> max units on promo per value of that column (e.g.
        ``{"category": 1}``). The column must be one of the unit columns.
    max_discount
        Deepest allowed discount, overall (float) or per unit-column value
        (``{"column": {value: max}}``).
    min_margin_rate
        Minimum total margin / revenue for the plan horizon.
    exclude_extrapolated
        Drop options deeper than any discount the unit had in the training data.
    """

    budget: float | None = None
    max_promos: int | None = None
    max_promos_per: Mapping[str, int] | None = None
    max_discount: float | Mapping[str, Mapping[Any, float]] | None = None
    min_margin_rate: float | None = None
    exclude_extrapolated: bool = False


@dataclass
class PromoPlan:
    """Result of :func:`optimize_promotions`.

    ``plan``: one row per unit with the chosen option and its expected
    incremental outcomes. ``comparison``: the best plan, the runner-up plans and
    the baseline compared on the same draws (``prob_best``,
    ``prob_beats_baseline``). ``options``: the full unit x option table.
    ``usage``: how much of each constraint the plan uses.
    """

    plan: pd.DataFrame
    comparison: pd.DataFrame
    options: pd.DataFrame
    usage: dict[str, Any]
    alternatives: list[pd.DataFrame]


def _allowed(opts: PromoOptions, constraints: PromoConstraints) -> np.ndarray:
    n_u, n_o = len(opts.units), len(opts.options)
    allowed = np.ones((n_u, n_o), dtype=bool)
    disc = opts.options["discount"].to_numpy(dtype=np.float64)
    md = constraints.max_discount
    if isinstance(md, (int, float)):
        allowed &= disc[None, :] <= float(md) + 1e-12
    elif isinstance(md, Mapping):
        for col, limits in md.items():
            if col not in opts.unit_cols:
                raise ValueError(f"max_discount column {col!r} is not a unit column")
            for u in range(n_u):
                cap = limits.get(opts.units.iloc[u][col])
                if cap is not None:
                    allowed[u] &= disc <= float(cap) + 1e-12
    if constraints.exclude_extrapolated:
        flags = opts.table["extrapolated"].to_numpy(dtype=bool).reshape(n_u, n_o)
        allowed &= ~flags
    allowed[:, 0] = True
    return allowed


def _solve_milp(
    value: np.ndarray,
    opts: PromoOptions,
    constraints: PromoConstraints,
    allowed: np.ndarray,
    exclude: list[np.ndarray],
) -> np.ndarray | None:
    from scipy.optimize import Bounds, LinearConstraint, milp

    n_u, n_o = value.shape
    n = n_u * n_o
    A, lo, hi = [], [], []
    for u in range(n_u):
        row = np.zeros(n)
        row[u * n_o : (u + 1) * n_o] = 1
        A.append(row)
        lo.append(1)
        hi.append(1)
    promo = np.tile(np.r_[0.0, np.ones(n_o - 1)], n_u)
    if constraints.budget is not None:
        A.append(opts.draws["promo_cost"].mean(axis=0).ravel())
        lo.append(-np.inf)
        hi.append(float(constraints.budget))
    if constraints.max_promos is not None:
        A.append(promo)
        lo.append(-np.inf)
        hi.append(float(constraints.max_promos))
    for col, cap in (constraints.max_promos_per or {}).items():
        if col not in opts.unit_cols:
            raise ValueError(f"max_promos_per column {col!r} is not a unit column")
        for val in opts.units[col].unique():
            mask = np.repeat((opts.units[col] == val).to_numpy(), n_o)
            A.append(promo * mask)
            lo.append(-np.inf)
            hi.append(float(cap))
    if constraints.min_margin_rate is not None:
        if "margin" not in opts.draws:
            raise ValueError("min_margin_rate needs a unit cost column")
        r = float(constraints.min_margin_rate)
        coef = (
            opts.draws["margin"].mean(axis=0) - r * opts.draws["revenue"].mean(axis=0)
        ).ravel()
        A.append(coef)
        lo.append(r * opts.baseline["revenue"] - opts.baseline["margin"])
        hi.append(np.inf)
    for chosen in exclude:
        A.append(chosen.ravel().astype(float))
        lo.append(-np.inf)
        hi.append(n_u - 1)
    ub = allowed.ravel().astype(float)
    res = milp(
        c=-value.ravel(),
        constraints=LinearConstraint(np.array(A), lo, hi),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), ub),
    )
    if res.x is None:
        return None
    return np.round(res.x).reshape(n_u, n_o).astype(bool)


def _assignments(opts: PromoOptions, choice: np.ndarray) -> pd.DataFrame:
    rows = []
    for u, o in enumerate(choice.argmax(axis=1)):
        row = {c: opts.units.iloc[u][c] for c in opts.unit_cols}
        row["discount"] = float(opts.options.iloc[o]["discount"])
        row["promo_type"] = opts.options.iloc[o]["promo_type"]
        rows.append(row)
    return pd.DataFrame(rows)


def _plan_value(
    model: DemandModel,
    base_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    unit_cols: Sequence[str],
    unit_cost_col: str | None,
) -> Draws:
    frame = apply_promos(model, base_frame, assignments, unit_cols=unit_cols)
    return evaluate(model, frame, unit_cost_col=unit_cost_col)


def _feasible_joint(
    d: Draws, choice: np.ndarray, opts: PromoOptions, c: PromoConstraints
) -> bool:
    if c.budget is not None and d.discount_cost.sum(axis=1).mean() > c.budget + 1e-9:
        return False
    if c.min_margin_rate is not None and d.margin is not None:
        if (
            d.margin.sum(axis=1).mean()
            < c.min_margin_rate * d.revenue.sum(axis=1).mean() - 1e-9
        ):
            return False
    on = choice[:, 1:].any(axis=1)
    if c.max_promos is not None and on.sum() > c.max_promos:
        return False
    for col, cap in (c.max_promos_per or {}).items():
        if (pd.Series(on).groupby(opts.units[col].to_numpy()).sum() > cap).any():
            return False
    return True


def _refine_coupled(
    model: DemandModel,
    base_frame: pd.DataFrame,
    opts: PromoOptions,
    choice: np.ndarray,
    allowed: np.ndarray,
    constraints: PromoConstraints,
    objective: str,
    unit_cost_col: str | None,
    max_rounds: int,
) -> np.ndarray:
    """Coordinate ascent on the joint objective, one unit at a time."""
    kw = dict(unit_cols=opts.unit_cols, unit_cost_col=unit_cost_col)
    best = _plan_value(model, base_frame, _assignments(opts, choice), **kw)
    best_val = best.metric(objective).sum(axis=1).mean()
    for _ in range(max_rounds):
        improved = False
        for u in range(len(opts.units)):
            current = int(choice[u].argmax())
            for o in np.flatnonzero(allowed[u]):
                if o == current:
                    continue
                trial = choice.copy()
                trial[u] = False
                trial[u, o] = True
                d = _plan_value(model, base_frame, _assignments(opts, trial), **kw)
                val = d.metric(objective).sum(axis=1).mean()
                if val > best_val + 1e-9 and _feasible_joint(
                    d, trial, opts, constraints
                ):
                    choice, best_val, current, improved = trial, val, int(o), True
        if not improved:
            break
    return choice


def _rank_by_joint_value(
    model: DemandModel,
    base_frame: pd.DataFrame,
    opts: PromoOptions,
    plans: list[np.ndarray],
    constraints: PromoConstraints,
    objective: str,
    unit_cost_col: str | None,
) -> list[np.ndarray]:
    """Order candidate plans by the joint simulation, feasible ones first.

    The MILP ranks plans on per-unit expected values; the joint simulation
    (same draws, all units together) is the number reported, so the plan
    called "best" must be the best on it.
    """
    scored = []
    for k, p in enumerate(plans):
        d = _plan_value(
            model,
            base_frame,
            _assignments(opts, p),
            unit_cols=opts.unit_cols,
            unit_cost_col=unit_cost_col,
        )
        feasible = _feasible_joint(d, p, opts, constraints)
        scored.append((not feasible, -d.metric(objective).sum(axis=1).mean(), k))
    return [plans[k] for _, _, k in sorted(scored)]


def optimize_promotions(
    model: DemandModel,
    base_frame: pd.DataFrame,
    *,
    discounts: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
    promo_types: Sequence[str] | None = None,
    unit_cols: Sequence[str] | None = None,
    objective: str = "margin",
    constraints: PromoConstraints | None = None,
    unit_cost_col: str | None = "unit_cost",
    n_alternatives: int = 4,
    hdi_prob: float = 0.9,
    max_refine_rounds: int = 3,
) -> PromoPlan:
    """Choose one option per unit to maximize the expected objective.

    Builds the unit x option table (:func:`promo_options`), solves the
    multiple-choice knapsack with the constraints as a MILP on expected values,
    then, when the model couples units (within-group pressure or cross-price),
    polishes the plan by coordinate ascent on the joint simulation. The next
    ``n_alternatives`` best plans (MILP with the earlier plans excluded) are
    compared with the best and the baseline on the same draws.
    """
    constraints = constraints or PromoConstraints()
    opts = promo_options(
        model,
        base_frame,
        discounts=discounts,
        promo_types=promo_types,
        unit_cols=unit_cols,
        objective=objective,
        unit_cost_col=unit_cost_col,
        hdi_prob=hdi_prob,
    )
    value = opts.draws[objective].mean(axis=0)
    allowed = _allowed(opts, constraints)
    plans: list[np.ndarray] = []
    for _ in range(1 + max(0, n_alternatives)):
        choice = _solve_milp(value, opts, constraints, allowed, plans)
        if choice is None:
            break
        plans.append(choice)
    if not plans:
        raise ValueError("No feasible plan under these constraints")
    if _is_coupled(model):
        refined = _refine_coupled(
            model,
            base_frame,
            opts,
            plans[0],
            allowed,
            constraints,
            objective,
            unit_cost_col,
            max_refine_rounds,
        )
        if not any((refined == p).all() for p in plans):
            plans.insert(0, refined)

    plans = _rank_by_joint_value(
        model, base_frame, opts, plans, constraints, objective, unit_cost_col
    )
    scenarios = {
        ("best" if k == 0 else f"alternative_{k}"): _assignments(opts, p)
        for k, p in enumerate(plans)
    }
    comparison = compare_scenarios(
        model,
        base_frame,
        scenarios,
        unit_cols=opts.unit_cols,
        objective=objective,
        unit_cost_col=unit_cost_col,
        hdi_prob=hdi_prob,
    ).summary

    best = plans[0]
    picked = best.argmax(axis=1)
    n_o = len(opts.options)
    table = opts.table
    plan = table.iloc[[u * n_o + o for u, o in enumerate(picked)]].reset_index(
        drop=True
    )
    d = _plan_value(
        model,
        base_frame,
        scenarios["best"],
        unit_cols=opts.unit_cols,
        unit_cost_col=unit_cost_col,
    )
    usage = {
        "promo_cost": float(d.discount_cost.sum(axis=1).mean()),
        "budget": constraints.budget,
        "n_promos": int((picked > 0).sum()),
        "max_promos": constraints.max_promos,
        "margin_rate": (
            float(d.margin.sum(axis=1).mean() / d.revenue.sum(axis=1).mean())
            if d.margin is not None
            else None
        ),
        "min_margin_rate": constraints.min_margin_rate,
        "refined_for_coupling": _is_coupled(model),
    }
    return PromoPlan(
        plan=plan,
        comparison=comparison,
        options=table,
        usage=usage,
        alternatives=list(scenarios.values())[1:],
    )


# ----------------------------------------------------------------- backtest


@dataclass
class PromoBacktest:
    """Out-of-sample check of a model on held-out periods.

    ``metrics``: rows ``overall``, ``promo`` and ``regular`` (test rows with and
    without a discount) with ``n``, ``wape`` (sum |error| / sum actual),
    ``bias`` (sum predicted / sum actual - 1), ``rmse_log`` and
    ``hdi_coverage``. ``lift``: per held-out promo row, the actual quantity, the
    prediction, and the predicted lift over the same row at regular price.
    ``predictions``: the test rows with predictions.
    """

    metrics: pd.DataFrame
    lift: pd.DataFrame
    predictions: pd.DataFrame
    n_train: int
    n_test: int


def promo_backtest(
    model: DemandModel,
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    hdi_prob: float = 0.9,
    random_seed: int | None = None,
    **fit_kwargs: Any,
) -> PromoBacktest:
    """Fit an unfitted copy on early periods and score the held-out ones.

    Predictions are made on the full frame and then restricted to the test
    rows, so lag and within-group pressure features see the real history.
    """
    cols = model.panel_columns
    work = model.clone()
    train, test = work.train_test_split(
        df, test_size=test_size, period_col=cols.period_col
    )
    if random_seed is not None:
        fit_kwargs.setdefault("random_seed", random_seed)
    work.fit(train, **fit_kwargs)

    pred = work.predict(df, hdi_prob=hdi_prob, random_seed=random_seed)
    is_test = df.index.isin(test.index)
    pred = pred.loc[is_test]
    y = pred[cols.quantity_col].to_numpy(dtype=np.float64)
    yhat = pred["quantity_mean"].to_numpy(dtype=np.float64)
    on_promo = (
        pred[cols.discount_col].to_numpy(dtype=np.float64) > 0
        if cols.discount_col is not None
        else np.zeros(len(pred), dtype=bool)
    )

    def metrics(mask: np.ndarray) -> dict[str, float]:
        if not mask.any():
            return {
                "n": 0,
                "wape": np.nan,
                "bias": np.nan,
                "rmse_log": np.nan,
                "hdi_coverage": np.nan,
            }
        a, p = y[mask], yhat[mask]
        floor = cols.quantity_floor
        lo = pred["quantity_hdi_lower"].to_numpy()[mask]
        hi = pred["quantity_hdi_upper"].to_numpy()[mask]
        return {
            "n": int(mask.sum()),
            "wape": float(np.abs(a - p).sum() / max(a.sum(), 1e-12)),
            "bias": float(p.sum() / max(a.sum(), 1e-12) - 1),
            "rmse_log": float(
                np.sqrt(
                    np.mean(
                        (
                            np.log(np.maximum(a, floor))
                            - pred["log_quantity_mean"].to_numpy()[mask]
                        )
                        ** 2
                    )
                )
            ),
            "hdi_coverage": float(np.mean((a >= lo) & (a <= hi))),
        }

    table = pd.DataFrame(
        {
            "overall": metrics(np.ones(len(pred), dtype=bool)),
            "promo": metrics(on_promo),
            "regular": metrics(~on_promo),
        }
    ).T

    lift = pd.DataFrame()
    if on_promo.any() and cols.discount_col is not None:
        full_regular = df.copy()
        d = full_regular[cols.discount_col].to_numpy(dtype=np.float64)
        regular_price = full_regular[cols.price_col].to_numpy(dtype=np.float64) / (
            1 - d
        )
        test_promo = is_test & (d > 0)
        full_regular.loc[test_promo, cols.discount_col] = 0.0
        full_regular.loc[test_promo, cols.price_col] = regular_price[test_promo]
        if work.promo is not None and work.promo.type_col is not None:
            full_regular.loc[test_promo, work.promo.type_col] = NO_PROMO_TYPE
        q_promo = work.expected_quantity_draws(df).reshape(-1, len(df))[:, test_promo]
        q_reg = work.expected_quantity_draws(full_regular).reshape(-1, len(df))[
            :, test_promo
        ]
        ratio = q_promo / q_reg
        keep = [
            c
            for c in [
                cols.sku_col,
                cols.period_col,
                cols.discount_col,
                cols.quantity_col,
            ]
            if c in df.columns
        ]
        lift = df.loc[test_promo, keep].copy()
        lift["predicted_quantity"] = q_promo.mean(axis=0)
        lift["predicted_lift"] = ratio.mean(axis=0) - 1
        lift["lift_hdi_lower"] = np.quantile(ratio, (1 - hdi_prob) / 2, axis=0) - 1
        lift["lift_hdi_upper"] = np.quantile(ratio, 1 - (1 - hdi_prob) / 2, axis=0) - 1
    return PromoBacktest(
        metrics=table, lift=lift, predictions=pred, n_train=len(train), n_test=len(test)
    )
