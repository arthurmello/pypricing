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
from dataclasses import dataclass
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
