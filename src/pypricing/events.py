"""Retail event dummies (Black Friday, Christmas, ...) as control columns."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pypricing.data.price_panel import parse_period_index
from pypricing.synthetic_promo import black_friday

BUILTIN_EVENTS = ("black_friday", "cyber_monday", "christmas", "new_year")
EVENT_PREFIX = "control_event_"

DateSpan = tuple[pd.Timestamp, pd.Timestamp]


def _builtin_spans(name: str, years: Iterable[int]) -> list[DateSpan]:
    spans: list[DateSpan] = []
    for y in years:
        if name == "black_friday":
            d = black_friday(y)
            spans.append((d, d))
        elif name == "cyber_monday":
            d = black_friday(y) + pd.Timedelta(days=3)
            spans.append((d, d))
        elif name == "christmas":
            spans.append((pd.Timestamp(y, 12, 15), pd.Timestamp(y, 12, 24)))
        elif name == "new_year":
            spans.append((pd.Timestamp(y, 1, 1), pd.Timestamp(y, 1, 1)))
        else:
            raise ValueError(
                f"Unknown built-in event {name!r}; use one of {BUILTIN_EVENTS}"
            )
    return spans


def _custom_spans(dates: Sequence[Any]) -> list[DateSpan]:
    spans: list[DateSpan] = []
    for item in dates:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            start, end = pd.Timestamp(item[0]), pd.Timestamp(item[1])
        else:
            start = end = pd.Timestamp(item)
        if end < start:
            raise ValueError(f"Event span ends before it starts: {item!r}")
        spans.append((start.normalize(), end.normalize()))
    return spans


def _period_length(index: pd.DatetimeIndex) -> pd.Timedelta:
    unique = index.unique().sort_values()
    if len(unique) < 2:
        return pd.Timedelta(days=1)
    gaps = np.diff(unique.to_numpy()).astype("timedelta64[s]").astype(np.int64)
    return pd.Timedelta(seconds=float(np.median(gaps)))


def event_indicators(
    periods: pd.Series | pd.DatetimeIndex,
    *,
    events: Sequence[str] = BUILTIN_EVENTS,
    custom: Mapping[str, Sequence[Any]] | None = None,
    period_length: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """One 0/1 column per event: does the period overlap the event?

    A period runs from its timestamp for ``period_length`` (inferred from the
    median spacing: a week for weekly data). ``custom`` maps a name to dates
    or ``(start, end)`` spans, inclusive.
    """
    index = parse_period_index(pd.Series(periods), period_col="period")
    length = period_length or _period_length(index)
    starts = index.normalize()
    ends = starts + length - pd.Timedelta(days=1)
    ends = ends.where(ends >= starts, starts)
    years = range(int(starts.min().year) - 1, int(ends.max().year) + 2)

    spans: dict[str, list[DateSpan]] = {
        name: _builtin_spans(name, years) for name in events
    }
    for name, dates in (custom or {}).items():
        spans[name] = _custom_spans(dates)

    out = {}
    s = starts.to_numpy()
    e = ends.to_numpy()
    for name, items in spans.items():
        active = np.zeros(len(index), dtype=np.float64)
        for start, end in items:
            active = np.maximum(
                active, (s <= end.to_datetime64()) & (e >= start.to_datetime64())
            )
        out[name] = active
    return pd.DataFrame(out, index=pd.RangeIndex(len(index)))


def add_event_columns(
    df: pd.DataFrame,
    *,
    period_col: str = "period",
    events: Sequence[str] = BUILTIN_EVENTS,
    custom: Mapping[str, Sequence[Any]] | None = None,
    prefix: str = EVENT_PREFIX,
    drop_constant: bool = True,
    period_length: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Copy of ``df`` with ``{prefix}{event}`` 0/1 columns.

    The default prefix makes them auto-detected control columns. With
    ``drop_constant`` an event that never (or always) occurs in ``df`` is left
    out, since its effect would not be identified.
    """
    if period_col not in df.columns:
        raise ValueError(f"Missing period column: {period_col!r}")
    flags = event_indicators(
        df[period_col], events=events, custom=custom, period_length=period_length
    )
    out = df.copy()
    for name in flags.columns:
        column = flags[name].to_numpy()
        if drop_constant and column.min() == column.max():
            continue
        out[f"{prefix}{name}"] = column
    return out
