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
    # Future periods have no fitted period effect: reuse last year's, so
    # seasonality and events still shape the plan.
    period_effect = "seasonal" if model.time_effects is not None else "neutral"
    q = model.expected_quantity_draws(frame, period_effect=period_effect)
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
    if cols.period_col in assignments.columns and cols.period_col not in on:
        on.append(cols.period_col)
    if assignments.empty:
        return frame.copy()
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


def _units(
    frame: pd.DataFrame, unit_cols: Sequence[str], *, window_only: bool = False
) -> pd.DataFrame:
    missing = [c for c in unit_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"plan frame missing unit columns {missing}")
    rows = frame
    if window_only and WINDOW_COL in frame.columns:
        rows = frame[frame[WINDOW_COL].to_numpy(dtype=bool)]
    return rows[list(unit_cols)].drop_duplicates().reset_index(drop=True)


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
    """Do units interact (within-group pressure or cross-price terms)?"""
    pressure = model.promo is not None and model.promo.pressure_level is not None
    return pressure or model.cross_elasticity is not None


def _has_lags(model: DemandModel) -> bool:
    return model.promo is not None and model.promo.lags > 0


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


class _Evaluator:
    """Evaluate assignments on the smallest block of rows they can affect.

    Rows are split into independent blocks: with within-group pressure, a block
    is a pressure group (e.g. a category, all periods); otherwise a unit's own
    rows (all periods, so post-promo lags stay inside). With cross-price terms
    everything is one block. Changing one unit only needs its block re-scored.
    """

    METRICS = ("quantity", "revenue", "discount_cost", "margin")

    def __init__(
        self,
        model: DemandModel,
        frame: pd.DataFrame,
        units: pd.DataFrame,
        decision_cols: Sequence[str],
        level_cols: Sequence[str],
        unit_cost_col: str | None,
    ) -> None:
        self.model = model
        self.frame = frame.reset_index(drop=True)
        self.units = units
        self.decision_cols = list(decision_cols)
        self.unit_cost_col = unit_cost_col
        n = len(self.frame)
        if model.cross_elasticity is not None:
            block = np.zeros(n, dtype=np.int64)
        else:
            keys = list(level_cols)
            if model.promo is not None and model.promo.pressure_level is not None:
                groups = model.panel_columns.group_columns or ()
                gcol = groups[model.promo.pressure_level]
                if gcol not in self.frame.columns:
                    raise ValueError(
                        f"plan frame needs the pressure group column {gcol!r}"
                    )
                keys = [gcol]
            block = (
                self.frame.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()
            )
        unit_rows = _row_unit_codes(self.frame, units, decision_cols)
        unit_block = np.full(len(units), -1, dtype=np.int64)
        for u in range(len(units)):
            b = np.unique(block[unit_rows == u])
            if len(b) != 1:
                block[:] = 0  # a unit spans blocks: fall back to one block
                unit_block[:] = 0
                break
            unit_block[u] = b[0]
        self.block = block
        self.unit_block = unit_block
        self.block_rows = {b: np.flatnonzero(block == b) for b in np.unique(block)}
        self.block_frames = {
            b: self.frame.iloc[rows].reset_index(drop=True)
            for b, rows in self.block_rows.items()
        }
        self.base = {b: self._score(b, None) for b in self.block_rows}

    def _score(self, b: int, assignments: pd.DataFrame | None) -> dict[str, np.ndarray]:
        frame = self.block_frames[b]
        if assignments is not None and len(assignments):
            frame = apply_promos(
                self.model, frame, assignments, unit_cols=self.decision_cols
            )
        d = evaluate(self.model, frame, unit_cost_col=self.unit_cost_col)
        out = {
            "quantity": d.quantity.sum(axis=1),
            "revenue": d.revenue.sum(axis=1),
            "discount_cost": d.discount_cost.sum(axis=1),
        }
        if d.margin is not None:
            out["margin"] = d.margin.sum(axis=1)
        return out

    def units_in_block(self, b: int) -> np.ndarray:
        return np.flatnonzero(self.unit_block == b)

    def score_block(self, b: int, assignments: pd.DataFrame) -> dict[str, np.ndarray]:
        return self._score(b, assignments)

    def baseline_totals(self) -> dict[str, np.ndarray]:
        tot: dict[str, np.ndarray] = {}
        for scores in self.base.values():
            for k, v in scores.items():
                tot[k] = tot.get(k, 0) + v
        return tot


@dataclass
class PromoOptions:
    """Every unit x option, with incremental outcomes vs no promo for that unit.

    ``table`` columns: unit columns (plus the period with ``timing=True``),
    ``discount``, ``promo_type``, ``incremental_{objective}`` mean / HDI,
    ``prob_positive``, ``incremental_units``, ``incremental_revenue``,
    ``incremental_margin``, ``promo_cost`` (means) and ``extrapolated``
    (deeper than any discount the unit had in the training data).
    ``draws[objective]`` is ``(n_draws, n_units, n_options)``.
    """

    table: pd.DataFrame
    units: pd.DataFrame
    options: pd.DataFrame
    draws: dict[str, np.ndarray] = field(repr=False)
    baseline: dict[str, float] = field(default_factory=dict)
    unit_cols: tuple[str, ...] = ()
    level_cols: tuple[str, ...] = ()
    timing: bool = False
    evaluator: _Evaluator | None = field(default=None, repr=False)


def default_promo_type(model: DemandModel) -> str | None:
    """The most common promo type on discounted training rows: the plain discount.

    Used when the caller doesn't choose promo types, so the optimizer picks a
    discount depth only and never adds a mechanic (flyer, display, bundle)
    that has a cost the model does not know about.
    """
    if model.promo is None or model.promo.type_col is None or model.data is None:
        return None
    cols = model.panel_columns
    data = model.data
    on = data[data[cols.discount_col] > 0][model.promo.type_col].astype(str)
    on = on[on.isin(model.promo_type_levels_ or ())]
    if on.empty:
        levels = list(model.promo_type_levels_ or [])
        return levels[0] if levels else None
    return str(on.value_counts().index[0])


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
    timing: bool = False,
) -> PromoOptions:
    """Incremental outcome of each option for each unit, all else at regular price.

    ``promo_types=None`` on a model with promo types uses only the most common
    one (:func:`default_promo_type`), so only the depth is chosen.

    ``timing=False``: one decision per unit for the whole promo window.
    ``timing=True``: one decision per unit *and* window period, so the plan
    also says when to run each promo.

    Without coupling terms (and, with timing, without lags) each option is
    scored for all units in one pass. Otherwise each unit is scored on its own,
    on the block of rows it can affect, so the numbers are its marginal effect
    (own lift minus what it takes from the others and from its later weeks).
    """
    _check_objective(objective)
    cols = model.panel_columns
    level_cols = tuple(unit_cols or (cols.sku_col,))
    decision_cols = level_cols + ((cols.period_col,) if timing else ())
    type_col = model.promo.type_col if model.promo is not None else None
    if promo_types is not None and type_col is None:
        raise ValueError("promo_types needs a model fit with PromoSpec.type_col")
    if type_col is not None and promo_types is None:
        plain = default_promo_type(model)
        promo_types = [plain] if plain is not None else None
    frame = base_frame.reset_index(drop=True)
    units = _units(frame, decision_cols, window_only=timing)
    options = _option_grid(discounts, promo_types)
    n_u, n_o = len(units), len(options)

    def assign(option: pd.Series, which: np.ndarray) -> pd.DataFrame:
        a = units.iloc[which].copy()
        a["discount"] = option["discount"]
        if option["promo_type"] is not None:
            a["promo_type"] = option["promo_type"]
        return a

    names = ["quantity", "revenue", "discount_cost"]
    fast = not _is_coupled(model) and not (timing and _has_lags(model))
    evaluator = None
    if fast:
        base = evaluate(model, frame, unit_cost_col=unit_cost_col)
        if base.margin is not None:
            names.append("margin")
        base_tot = {n: getattr(base, n).sum(axis=1) for n in names}
        n_draws = base.quantity.shape[0]
        inc = {n: np.zeros((n_draws, n_u, n_o)) for n in names}
        row_unit = _row_unit_codes(frame, units, decision_cols)
        for o in range(1, n_o):
            promo = apply_promos(
                model,
                frame,
                assign(options.iloc[o], np.arange(n_u)),
                unit_cols=decision_cols,
            )
            d = evaluate(model, promo, unit_cost_col=unit_cost_col)
            for n in names:
                delta = getattr(d, n) - getattr(base, n)
                for u in range(n_u):
                    inc[n][:, u, o] = delta[:, row_unit == u].sum(axis=1)
    else:
        evaluator = _Evaluator(
            model, frame, units, decision_cols, level_cols, unit_cost_col
        )
        base_tot = evaluator.baseline_totals()
        if "margin" in base_tot:
            names.append("margin")
        n_draws = len(base_tot["quantity"])
        inc = {n: np.zeros((n_draws, n_u, n_o)) for n in names}
        for u in range(n_u):
            b = evaluator.unit_block[u]
            for o in range(1, n_o):
                scored = evaluator.score_block(
                    b, assign(options.iloc[o], np.array([u]))
                )
                for n in names:
                    inc[n][:, u, o] = scored[n] - evaluator.base[b][n]

    obj_key = {"units": "quantity", "revenue": "revenue", "margin": "margin"}[objective]
    if obj_key not in inc:
        raise ValueError("objective='margin' needs a unit cost column")
    seen_max = _observed_max_discount(model, units, level_cols)
    obj = inc[obj_key]
    lo = np.quantile(obj, (1 - hdi_prob) / 2, axis=0)
    hi = np.quantile(obj, 1 - (1 - hdi_prob) / 2, axis=0)
    mean = {n: v.mean(axis=0) for n, v in inc.items()}
    ppos = (obj > 0).mean(axis=0)
    table = pd.DataFrame(
        {c: np.repeat(units[c].to_numpy(), n_o) for c in decision_cols}
    )
    disc = np.tile(options["discount"].to_numpy(dtype=np.float64), n_u)
    table["discount"] = disc
    table["promo_type"] = np.tile(options["promo_type"].to_numpy(dtype=object), n_u)
    table[f"incremental_{objective}"] = mean[obj_key].ravel()
    table["hdi_lower"] = lo.ravel()
    table["hdi_upper"] = hi.ravel()
    pp = ppos.copy()
    pp[:, 0] = np.nan
    table["prob_positive"] = pp.ravel()
    table["incremental_units"] = mean["quantity"].ravel()
    table["incremental_revenue"] = mean["revenue"].ravel()
    table["incremental_margin"] = mean["margin"].ravel() if "margin" in mean else np.nan
    table["promo_cost"] = mean["discount_cost"].ravel()
    table["extrapolated"] = disc > np.repeat(seen_max, n_o) + 1e-9
    draws = {objective: inc[obj_key]}
    for n, key in (
        ("units", "quantity"),
        ("revenue", "revenue"),
        ("margin", "margin"),
        ("promo_cost", "discount_cost"),
    ):
        if key in inc:
            draws[n] = inc[key]
    baseline = {n: float(np.mean(v)) for n, v in base_tot.items()}
    return PromoOptions(
        table=table,
        units=units,
        options=options,
        draws=draws,
        baseline=baseline,
        unit_cols=decision_cols,
        level_cols=level_cols,
        timing=timing,
        evaluator=evaluator,
    )


# -------------------------------------------------------------- optimization


@dataclass
class PromoConstraints:
    """Limits on a promo plan. ``None`` means no limit.

    budget
        Maximum expected promo cost (revenue given away, ``Q * regular * d``).
    max_promos
        Maximum number of promos (units on promo; with timing, promo-periods).
    max_promos_per
        Column name -> max promos per value of that column (e.g.
        ``{"category": 1}``, or with timing ``{"period": 3}``). The column must
        be one of the unit columns (or the period column with timing).
    max_discount
        Deepest allowed discount, overall (float) or per unit-column value
        (``{"column": {value: max}}``).
    min_margin_rate
        Minimum total margin / revenue for the plan horizon.
    exclude_extrapolated
        Drop options deeper than any discount the unit had in the training data.
    max_promos_per_unit
        Timing only: most promo periods per unit in the window.
    min_gap
        Timing only: at least this many regular-price periods between two
        promos of the same unit.
    """

    budget: float | None = None
    max_promos: int | None = None
    max_promos_per: Mapping[str, int] | None = None
    max_discount: float | Mapping[str, Mapping[Any, float]] | None = None
    min_margin_rate: float | None = None
    exclude_extrapolated: bool = False
    max_promos_per_unit: int | None = None
    min_gap: int | None = None


@dataclass
class PromoPlan:
    """Result of :func:`optimize_promotions`.

    ``plan``: one row per unit (with timing: per unit and period) with the
    chosen option and its expected incremental outcomes. ``comparison``: the
    best plan, the runner-up plans and the baseline compared on the same draws
    (``prob_best``, ``prob_beats_baseline``). ``options``: the full unit x
    option table. ``usage``: how much of each constraint the plan uses.
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


def _level_groups(opts: PromoOptions) -> np.ndarray:
    """Code of each decision unit's level unit (unit without the period)."""
    return opts.units.groupby(list(opts.level_cols), sort=False).ngroup().to_numpy()


def _period_order(opts: PromoOptions, period_col: str) -> np.ndarray:
    periods = pd.Series(opts.units[period_col])
    return periods.rank(method="dense").astype(int).to_numpy() - 1


def _count_rows(
    opts: PromoOptions, constraints: PromoConstraints, period_col: str
) -> list[tuple[np.ndarray, float]]:
    """(mask over units, cap) for every promo-count constraint."""
    rows: list[tuple[np.ndarray, float]] = []
    n_u = len(opts.units)
    if constraints.max_promos is not None:
        rows.append((np.ones(n_u, dtype=bool), float(constraints.max_promos)))
    for col, cap in (constraints.max_promos_per or {}).items():
        if col not in opts.unit_cols:
            raise ValueError(f"max_promos_per column {col!r} is not a unit column")
        for val in opts.units[col].unique():
            rows.append(((opts.units[col] == val).to_numpy(), float(cap)))
    if (
        constraints.max_promos_per_unit is not None or constraints.min_gap
    ) and not opts.timing:
        raise ValueError("max_promos_per_unit and min_gap need timing=True")
    if opts.timing:
        level = _level_groups(opts)
        order = _period_order(opts, period_col)
        if constraints.max_promos_per_unit is not None:
            for g in np.unique(level):
                rows.append((level == g, float(constraints.max_promos_per_unit)))
        gap = int(constraints.min_gap or 0)
        if gap > 0:
            for g in np.unique(level):
                in_g = level == g
                for start in range(order.max() + 1):
                    window = in_g & (order >= start) & (order <= start + gap)
                    if window.sum() > 1:
                        rows.append((window, 1.0))
    return rows


def _solve_milp(
    value: np.ndarray,
    opts: PromoOptions,
    constraints: PromoConstraints,
    allowed: np.ndarray,
    exclude: list[np.ndarray],
    period_col: str,
) -> np.ndarray | None:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n_u, n_o = value.shape
    n = n_u * n_o
    count_rows = _count_rows(opts, constraints, period_col)
    n_rows = n_u + len(count_rows) + len(exclude) + 2
    A = lil_matrix((n_rows, n))
    lo, hi = [], []
    r = 0
    for u in range(n_u):
        A[r, u * n_o : (u + 1) * n_o] = 1
        lo.append(1)
        hi.append(1)
        r += 1
    for mask, cap in count_rows:
        for u in np.flatnonzero(mask):
            A[r, u * n_o + 1 : (u + 1) * n_o] = 1
        lo.append(-np.inf)
        hi.append(cap)
        r += 1
    if constraints.budget is not None:
        A[r, :] = opts.draws["promo_cost"].mean(axis=0).ravel()
        lo.append(-np.inf)
        hi.append(float(constraints.budget))
        r += 1
    if constraints.min_margin_rate is not None:
        if "margin" not in opts.draws:
            raise ValueError("min_margin_rate needs a unit cost column")
        rate = float(constraints.min_margin_rate)
        coef = (
            opts.draws["margin"].mean(axis=0)
            - rate * opts.draws["revenue"].mean(axis=0)
        ).ravel()
        A[r, :] = coef
        lo.append(rate * opts.baseline["revenue"] - opts.baseline["margin"])
        hi.append(np.inf)
        r += 1
    for chosen in exclude:
        A[r, :] = chosen.ravel().astype(float)
        lo.append(-np.inf)
        hi.append(n_u - 1)
        r += 1
    A = A[:r].tocsr()
    res = milp(
        c=-value.ravel(),
        constraints=LinearConstraint(A, lo, hi),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), allowed.ravel().astype(float)),
        options={"time_limit": 60},
    )
    if res.x is None:
        return None
    return np.round(res.x).reshape(n_u, n_o).astype(bool)


def _assignments(
    opts: PromoOptions, choice: np.ndarray, *, only_promos: bool = False
) -> pd.DataFrame:
    picked = choice.argmax(axis=1)
    out = opts.units.copy()
    out["discount"] = opts.options["discount"].to_numpy(dtype=np.float64)[picked]
    out["promo_type"] = opts.options["promo_type"].to_numpy(dtype=object)[picked]
    if only_promos:
        out = out[out["discount"] > 0]
    return out.reset_index(drop=True)


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


def _counts_ok(
    choice: np.ndarray, opts: PromoOptions, c: PromoConstraints, period_col: str
) -> bool:
    on = choice[:, 1:].any(axis=1)
    return all(
        on[mask].sum() <= cap + 1e-9 for mask, cap in _count_rows(opts, c, period_col)
    )


def _totals_ok(tot: Mapping[str, float], c: PromoConstraints) -> bool:
    if c.budget is not None and tot["discount_cost"] > c.budget + 1e-9:
        return False
    if c.min_margin_rate is not None and "margin" in tot:
        if tot["margin"] < c.min_margin_rate * tot["revenue"] - 1e-9:
            return False
    return True


def _draw_totals(d: Draws) -> dict[str, float]:
    out = {
        "quantity": float(d.quantity.sum(axis=1).mean()),
        "revenue": float(d.revenue.sum(axis=1).mean()),
        "discount_cost": float(d.discount_cost.sum(axis=1).mean()),
    }
    if d.margin is not None:
        out["margin"] = float(d.margin.sum(axis=1).mean())
    return out


def _refine_coupled(
    opts: PromoOptions,
    choice: np.ndarray,
    allowed: np.ndarray,
    constraints: PromoConstraints,
    objective: str,
    max_rounds: int,
    period_col: str,
) -> np.ndarray:
    """Coordinate ascent on the joint objective, one unit at a time.

    Only the changed unit's block is re-scored, so a pass costs one block
    evaluation per unit and option.
    """
    ev = opts.evaluator
    assert ev is not None
    key = {"units": "quantity", "revenue": "revenue", "margin": "margin"}[objective]

    def block_means(b: int, ch: np.ndarray) -> dict[str, float]:
        idx = ev.units_in_block(b)
        sub = _assignments_subset(opts, ch, idx)
        scored = ev.score_block(b, sub) if len(sub) else ev.base[b]
        return {k: float(v.mean()) for k, v in scored.items()}

    blocks = {int(b): block_means(int(b), choice) for b in np.unique(ev.unit_block)}
    fixed = {k: 0.0 for k in next(iter(blocks.values()))}
    for b, scores in ev.base.items():
        if b not in blocks:
            for k, v in scores.items():
                fixed[k] += float(v.mean())

    def total(bl: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
        tot = dict(fixed)
        for scores in bl.values():
            for k, v in scores.items():
                tot[k] += v
        return tot

    best_tot = total(blocks)
    for _ in range(max_rounds):
        improved = False
        for u in range(len(opts.units)):
            b = int(ev.unit_block[u])
            current = int(choice[u].argmax())
            for o in np.flatnonzero(allowed[u]):
                if o == current:
                    continue
                trial = choice.copy()
                trial[u] = False
                trial[u, o] = True
                if not _counts_ok(trial, opts, constraints, period_col):
                    continue
                trial_blocks = dict(blocks)
                trial_blocks[b] = block_means(b, trial)
                tot = total(trial_blocks)
                if tot[key] > best_tot[key] + 1e-9 and _totals_ok(tot, constraints):
                    choice, blocks, best_tot, current, improved = (
                        trial,
                        trial_blocks,
                        tot,
                        int(o),
                        True,
                    )
        if not improved:
            break
    return choice


def _assignments_subset(
    opts: PromoOptions, choice: np.ndarray, idx: np.ndarray
) -> pd.DataFrame:
    a = _assignments(opts, choice)
    a = a.iloc[idx]
    return a[a["discount"] > 0].reset_index(drop=True)


def _rank_by_joint_value(
    model: DemandModel,
    base_frame: pd.DataFrame,
    opts: PromoOptions,
    plans: list[np.ndarray],
    constraints: PromoConstraints,
    objective: str,
    unit_cost_col: str | None,
    period_col: str,
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
            _assignments(opts, p, only_promos=True),
            unit_cols=opts.unit_cols,
            unit_cost_col=unit_cost_col,
        )
        tot = _draw_totals(d)
        feasible = _totals_ok(tot, constraints) and _counts_ok(
            p, opts, constraints, period_col
        )
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
    timing: bool = False,
) -> PromoPlan:
    """Choose one option per unit (and, with ``timing``, per period) to
    maximize the expected objective.

    Builds the unit x option table (:func:`promo_options`), solves the
    multiple-choice knapsack with the constraints as a MILP on expected values,
    then, when units interact (within-group pressure, cross-price, or post-promo
    lags across periods with timing), polishes the plan by coordinate ascent on
    the joint simulation. The next ``n_alternatives`` best plans (MILP with the
    earlier plans excluded) are compared with the best and the baseline on the
    same draws.
    """
    constraints = constraints or PromoConstraints()
    period_col = model.panel_columns.period_col
    base_frame = base_frame.reset_index(drop=True)
    opts = promo_options(
        model,
        base_frame,
        discounts=discounts,
        promo_types=promo_types,
        unit_cols=unit_cols,
        objective=objective,
        unit_cost_col=unit_cost_col,
        hdi_prob=hdi_prob,
        timing=timing,
    )
    value = opts.draws[objective].mean(axis=0)
    allowed = _allowed(opts, constraints)
    plans: list[np.ndarray] = []
    for _ in range(1 + max(0, n_alternatives)):
        choice = _solve_milp(value, opts, constraints, allowed, plans, period_col)
        if choice is None:
            break
        plans.append(choice)
    if not plans:
        raise ValueError("No feasible plan under these constraints")
    coupled = opts.evaluator is not None
    if coupled:
        refined = _refine_coupled(
            opts,
            plans[0],
            allowed,
            constraints,
            objective,
            max_refine_rounds,
            period_col,
        )
        if not any((refined == p).all() for p in plans):
            plans.insert(0, refined)

    plans = _rank_by_joint_value(
        model,
        base_frame,
        opts,
        plans,
        constraints,
        objective,
        unit_cost_col,
        period_col,
    )
    scenarios = {
        ("best" if k == 0 else f"alternative_{k}"): _assignments(
            opts, p, only_promos=True
        )
        for k, p in enumerate(plans)
    }
    comparison = compare_scenarios(
        model,
        base_frame,
        {k: v for k, v in scenarios.items()},
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
    tot = _draw_totals(d)
    usage = {
        "promo_cost": tot["discount_cost"],
        "budget": constraints.budget,
        "n_promos": int((picked > 0).sum()),
        "max_promos": constraints.max_promos,
        "margin_rate": tot["margin"] / tot["revenue"] if "margin" in tot else None,
        "min_margin_rate": constraints.min_margin_rate,
        "refined_for_coupling": coupled,
        "timing": timing,
    }
    return PromoPlan(
        plan=plan,
        comparison=comparison,
        options=table,
        usage=usage,
        alternatives=[_assignments(opts, p) for p in plans[1:]],
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
