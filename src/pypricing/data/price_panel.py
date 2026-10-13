"""Validated tabular panel for elasticity models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from pypricing.data.index import CrossPairIndex, HierarchyIndex

_NUMERIC_PERIOD_MSG = (
    "{period_col!r} is numeric; map it to timestamps before using trend or "
    "seasonality (integer periods are not calendar dates)."
)


def _as_naive_datetime_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


def _is_datetime_like(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        pd.api.types.is_datetime64_any_dtype(dtype)
        or isinstance(dtype, pd.DatetimeTZDtype)
        or isinstance(dtype, pd.PeriodDtype)
    )


def parse_period_index(
    values: pd.Series | pd.Index | np.ndarray | list[Any],
    *,
    period_col: str = "period",
) -> pd.DatetimeIndex:
    """Convert a period column to naive UTC timestamps.

    Accepts datetime-like values and strings that all parse. Rejects numeric
    values (including numeric object columns) so integer period indices are
    not silently treated as Unix times.
    """
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError(f"NaN/NaT in {period_col!r} are not allowed")

    if isinstance(series.dtype, pd.PeriodDtype):
        series = series.dt.to_timestamp()

    if _is_datetime_like(series):
        return _as_naive_datetime_index(pd.DatetimeIndex(series))

    if pd.api.types.is_numeric_dtype(series):
        raise ValueError(_NUMERIC_PERIOD_MSG.format(period_col=period_col))

    as_numeric = pd.to_numeric(series, errors="coerce")
    if as_numeric.notna().all():
        raise ValueError(_NUMERIC_PERIOD_MSG.format(period_col=period_col))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        examples = series[parsed.isna()].head(3).tolist()
        raise ValueError(
            f"{period_col!r} could not be parsed as dates (examples: {examples}). "
            "Pass a datetime column."
        )
    return _as_naive_datetime_index(pd.DatetimeIndex(parsed))


def period_to_t_years(
    period_index: pd.DatetimeIndex,
    t0: pd.Timestamp,
) -> np.ndarray:
    """Years since ``t0`` for each timestamp (365.25-day year)."""
    t0 = pd.Timestamp(t0)
    if t0.tz is not None:
        t0 = t0.tz_convert("UTC").tz_localize(None)
    delta_days = (period_index - t0) / pd.Timedelta(days=1)
    return np.asarray(delta_days, dtype=np.float64) / 365.25


@dataclass(frozen=True)
class PanelColumns:
    """Column mapping and panel-build knobs for long-format price data."""

    sku_col: str = "sku"
    price_col: str = "price"
    discount_col: str | None = None
    """Optional column of discounts as a fraction of the regular price.

    ``None`` (default) keeps a single price elasticity. ``0.2`` means 20% off
    the regular price; ``price`` remains the price actually paid. Regular price
    is ``price / (1 - discount)``. Fixed-amount discounts and multibuys must be
    converted to this fraction before fitting.
    """
    quantity_col: str = "quantity"
    quantity_floor: float = 1.0
    control_columns: tuple[str, ...] | None = None
    iv_columns: tuple[str, ...] | None = None
    """Instrument columns for control-function IV on the log-demand models.

    ``None`` (default) auto-detects columns named ``iv_*``. An empty tuple
    disables IV even if such columns exist.
    """
    group_columns: tuple[str, ...] | None = None
    period_col: str = "period"
    region_col: str | None = None
    floor_censoring_warn_threshold: float | None = 0.5
    """Warn when a SKU has this fraction (or more) of observations at/below
    ``quantity_floor`` -- such a SKU carries little to no price-response
    signal, even though nothing in ``run_diagnostics()`` otherwise flags it.
    Set to ``None`` to disable the check."""


def _auto_prefix_columns(df: pd.DataFrame, prefix: str) -> tuple[str, ...]:
    return tuple(c for c in df.columns if c.startswith(prefix))


def _resolve_feature_columns(
    df: pd.DataFrame,
    specified: tuple[str, ...] | None,
    *,
    auto_prefix: str,
    label: str,
    reserved: set[str],
) -> tuple[str, ...]:
    if specified is None:
        cols = _auto_prefix_columns(df, auto_prefix)
    else:
        cols = tuple(specified)
        if len(set(cols)) != len(cols):
            raise ValueError(f"{label} must not contain duplicates")
        bad = set(cols) - set(df.columns)
        if bad:
            raise ValueError(f"{label} not in frame: {sorted(bad)}")
    overlap = set(cols) & reserved
    if overlap:
        raise ValueError(f"{label} must not overlap sku/price/quantity: {overlap}")
    return cols


def decompose_log_price(
    price: np.ndarray,
    discount: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return log paid price, log regular price, and the log discount index.

    Regular price is ``price / (1 - discount)``. With ``discount is None`` the
    log regular price is the log paid price (the same array) and the index is
    ``None``.
    """
    log_paid = np.log(np.asarray(price, dtype=np.float64))
    if discount is None:
        return log_paid, log_paid, None
    log_index = np.log(1.0 - np.asarray(discount, dtype=np.float64))
    return log_paid, log_paid - log_index, log_index


def skus_without_variation(
    values: np.ndarray,
    obs_sku_idx: np.ndarray,
    sku_levels: pd.Index,
    *,
    atol: float = 1e-8,
) -> list[Any]:
    """SKU labels whose ``values`` do not change across observations."""
    values = np.asarray(values, dtype=np.float64)
    found: list[Any] = []
    for i, sku in enumerate(sku_levels):
        column = values[np.asarray(obs_sku_idx) == i]
        if column.size == 0:
            continue
        if float(np.max(column) - np.min(column)) <= atol:
            found.append(sku)
    return found


def _example_rows(df: pd.DataFrame, mask: np.ndarray, columns: list[str]) -> str:
    show = [c for c in columns if c in df.columns]
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))[:3]
    if len(idx) == 0:
        return df.loc[:, show].head(3).to_string(index=False)
    return df.iloc[idx].loc[:, show].to_string(index=False)


def read_discount_fraction(
    df: pd.DataFrame,
    *,
    discount_col: str,
    sku_col: str,
    price_col: str,
) -> np.ndarray:
    """Numeric discount fractions in ``[0, 1)``, with example rows on failure."""
    if discount_col not in df.columns:
        raise ValueError(f"Missing discount column: {discount_col!r}")
    series = df[discount_col]
    example_cols = [sku_col, price_col, discount_col]
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"{discount_col!r} must be numeric (a fraction of the regular price, "
            f"0.2 = 20% off). Example rows:\n"
            f"{_example_rows(df, np.ones(len(df), dtype=bool), example_cols)}"
        )
    values = series.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = ~np.isfinite(values)
        raise ValueError(
            f"NaN in {discount_col!r} are not allowed. Example rows:\n"
            f"{_example_rows(df, bad, example_cols)}"
        )
    if (values < 0).any():
        raise ValueError(
            f"{discount_col!r} must be >= 0 (fraction of the regular price). "
            f"Example rows:\n{_example_rows(df, values < 0, example_cols)}"
        )
    if (values >= 1).any():
        raise ValueError(
            f"{discount_col!r} must be < 1 (fraction of the regular price). "
            f"Example rows:\n{_example_rows(df, values >= 1, example_cols)}"
        )
    return values


def apply_discount(
    df: pd.DataFrame,
    discount: float | np.ndarray,
    *,
    panel_columns: PanelColumns | None = None,
) -> pd.DataFrame:
    """Set a discount and scale ``price`` off the implied regular price.

    Regular price is ``price / (1 - discount)`` when the discount column is
    already present, and the current ``price`` otherwise (no promo yet). The
    returned frame keeps ``price == regular * (1 - new_discount)``.

    ``discount`` is a fraction of the regular price (``0.2`` = 20% off) and
    must lie in ``[0, 1)``. A scalar applies to every row. When
    ``panel_columns.discount_col`` is unset, the column written is
    ``discount_rate``.
    """
    cols = panel_columns or PanelColumns()
    discount_col = cols.discount_col or "discount_rate"
    price_col = cols.price_col
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col!r}")

    out = df.copy()
    new_discount = np.asarray(discount, dtype=np.float64)
    if new_discount.ndim == 0:
        new_discount = np.full(len(out), float(new_discount))
    elif new_discount.shape != (len(out),):
        raise ValueError(
            f"discount length {new_discount.shape[0]} does not match frame length {len(out)}"
        )
    if not np.isfinite(new_discount).all() or (new_discount < 0).any() or (new_discount >= 1).any():
        raise ValueError("discount must be finite and in [0, 1)")

    price = out[price_col].to_numpy(dtype=np.float64)
    if discount_col in out.columns:
        current = out[discount_col].to_numpy(dtype=np.float64)
        if not np.isfinite(current).all() or (current < 0).any() or (current >= 1).any():
            raise ValueError(
                f"Existing {discount_col!r} must be finite and in [0, 1) "
                "so the regular price is defined"
            )
        regular = price / (1.0 - current)
    else:
        regular = price
    out[price_col] = regular * (1.0 - new_discount)
    out[discount_col] = new_discount
    return out


def _read_panel_discount(
    df: pd.DataFrame,
    cols: PanelColumns,
    *,
    control_columns: tuple[str, ...],
    iv_columns: tuple[str, ...],
    group_columns: tuple[str, ...] | None,
) -> np.ndarray | None:
    discount_col = cols.discount_col
    if discount_col is None:
        return None
    if discount_col not in df.columns:
        raise ValueError(f"Missing discount column: {discount_col!r}")
    overlaps: list[str] = []
    if discount_col in control_columns:
        overlaps.append("control_columns")
    if discount_col in iv_columns:
        overlaps.append("iv_columns")
    if group_columns is not None and discount_col in group_columns:
        overlaps.append("group_columns")
    if overlaps:
        raise ValueError(
            f"discount_col {discount_col!r} must not overlap {', '.join(overlaps)}"
        )
    return read_discount_fraction(
        df,
        discount_col=discount_col,
        sku_col=cols.sku_col,
        price_col=cols.price_col,
    )


def floor_censored_fraction(
    df: pd.DataFrame,
    *,
    panel_columns: PanelColumns | None = None,
) -> pd.Series:
    """Per-SKU fraction of observations at or below ``quantity_floor``.

    A SKU whose quantity sits at the floor for most of its history carries
    little to no price-response signal in this data -- ``elasticity_sku`` for
    that SKU is effectively unidentified, even though nothing in
    ``run_diagnostics()`` (divergences, r-hat) will flag it, since the fit
    itself has no way to see that the censoring happened.
    """
    cols = panel_columns or PanelColumns()
    at_floor = df[cols.quantity_col] <= cols.quantity_floor
    return at_floor.groupby(df[cols.sku_col]).mean()


def _build_hierarchy_index(
    df: pd.DataFrame,
    sku_col: str,
    group_columns: tuple[str, ...],
    sku_levels: pd.Index,
) -> HierarchyIndex:
    """One row per SKU in ``sku_levels`` order; group index matrix (n_skus, n_levels)."""
    dup = df.groupby(sku_col)[list(group_columns)].nunique()
    if (dup > 1).any().any():
        raise ValueError(
            "group_columns must be constant within each SKU; conflicting values found"
        )
    one = df.drop_duplicates(subset=[sku_col], keep="first")
    one = one.set_index(sku_col).reindex(sku_levels)
    if one[list(group_columns)].isna().any().any():
        raise ValueError("Missing group column values for some SKU levels")

    sku_to_level_idx = np.zeros((len(sku_levels), len(group_columns)), dtype=np.int64)
    n_groups: list[int] = []
    for i, col in enumerate(group_columns):
        cat = pd.Categorical(one[col])
        sku_to_level_idx[:, i] = cat.codes.astype(np.int64)
        n_groups.append(int(cat.categories.size))
    return HierarchyIndex(
        group_columns=group_columns,
        sku_to_level_idx=sku_to_level_idx,
        n_groups_per_level=tuple(n_groups),
    )


def _pair_lookup_map(
    pair_from: np.ndarray, pair_to: np.ndarray
) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for p, (i, j) in enumerate(zip(pair_from.tolist(), pair_to.tolist())):
        out[(int(i), int(j))] = p
    return out


def _cell_log_prices_and_rows(
    grp: pd.DataFrame,
    df: pd.DataFrame,
    *,
    sku_col: str,
    price_col: str,
    sku_to_idx: dict[Any, int],
    n_skus: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one market cell; return ``(log_p_arr, row_indices)`` of length ``n_skus``."""
    log_p_arr = np.empty(n_skus, dtype=np.float64)
    row_indices = np.empty(n_skus, dtype=np.int64)
    seen = np.zeros(n_skus, dtype=bool)
    for ix in grp.index:
        row = df.loc[ix]
        sku = row[sku_col]
        si = sku_to_idx[sku]
        if seen[si]:
            raise ValueError(
                f"Duplicate (cell, sku) for cross-elasticity: sku index {si}"
            )
        seen[si] = True
        p = float(row[price_col])
        if p <= 0:
            raise ValueError("price must be positive for cross-elasticity")
        log_p_arr[si] = float(np.log(p))
        row_indices[si] = int(df.index.get_loc(ix))

    if not seen.all():
        raise ValueError("Missing SKUs in a market cell for cross-elasticity")
    return log_p_arr, row_indices


def _fill_cross_matrix_for_cell(
    X: np.ndarray,
    *,
    log_p_arr: np.ndarray,
    row_indices: np.ndarray,
    n_skus: int,
    pair_map: dict[tuple[int, int], int],
) -> None:
    """Scatter competitor log prices into ``X`` for this cell's rows."""
    for si in range(n_skus):
        r = int(row_indices[si])
        for j in range(n_skus):
            if j == si:
                continue
            pidx = pair_map.get((si, j))
            if pidx is not None:
                X[r, pidx] = log_p_arr[j]


def _assert_cross_matrix_focal_rows(
    X: np.ndarray,
    *,
    obs_sku_idx: np.ndarray,
    pair_from: np.ndarray,
) -> None:
    """Columns must be zero on rows whose focal SKU is not ``pair_from[p]``."""
    n_obs, n_pairs = X.shape
    for r in range(n_obs):
        si = int(obs_sku_idx[r])
        for p in range(n_pairs):
            if pair_from[p] != si and X[r, p] != 0.0:
                raise RuntimeError("internal: cross matrix inconsistency")


@dataclass
class PricePanelData:
    """
    Long-format panel prepared for PyMC: one row per observation.

    Expects **price** and **quantity** in natural units; builds ``log_price`` and
    ``log_quantity`` internally for the log-log likelihood.

    ``log_price`` is the price actually paid (also the competitor price in
    cross-price terms). When ``discount_col`` is set, ``log_regular_price`` is
    ``log(price) - log(1 - discount)`` and ``log_discount_index`` is
    ``log(1 - discount)``. Without a discount column those are the paid log
    price and ``None``.
    """

    log_price: np.ndarray
    log_quantity: np.ndarray
    obs_sku_idx: np.ndarray
    n_skus: int
    sku_levels: pd.Index
    control_matrix: np.ndarray
    control_names: tuple[str, ...]
    iv_matrix: np.ndarray
    iv_names: tuple[str, ...]
    n_obs: int
    log_regular_price: np.ndarray
    """Log regular price. Equal to ``log_price`` when there is no discount column."""
    log_discount_index: np.ndarray | None = None
    """``log(1 - discount)``; ``None`` when the panel has no discount column."""
    hierarchy: HierarchyIndex | None = None
    """Present when ``PanelColumns.group_columns`` were set at build time."""
    cross_pair_log_price_matrix: np.ndarray | None = None
    """Shape ``(n_obs, n_cross_pairs)``: log competitor prices for directed cross pairs."""
    cross_pairs: CrossPairIndex | None = None
    """Directed pair / pool index when cross-elasticity is enabled."""
    period_index: pd.DatetimeIndex | None = None
    """Row-aligned timestamps when ``period_col`` is datetime (or parsed)."""

    @property
    def n_iv(self) -> int:
        return int(self.iv_matrix.shape[1])

    @property
    def n_cross_pairs(self) -> int:
        return 0 if self.cross_pairs is None else self.cross_pairs.n_pairs

    @property
    def n_cross_pool(self) -> int:
        return 0 if self.cross_pairs is None else self.cross_pairs.n_pool

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        panel_columns: PanelColumns | None = None,
        parse_period: bool = False,
    ) -> PricePanelData:
        """
        Parameters
        ----------
        panel_columns
            Column mapping and build knobs. When ``None``, uses ``PanelColumns()``
            defaults. ``quantity_floor`` is applied as
            ``log(max(quantity, quantity_floor))``. ``group_columns`` are optional
            hierarchy columns, coarse → fine, constant within each SKU.
            ``iv_columns`` are optional instruments (``None`` auto-detects ``iv_*``).
            ``discount_col``, when set, is a fraction of the regular price in
            ``[0, 1)`` (``0.2`` = 20% off). ``price`` stays the price paid.
        parse_period
            If ``True``, require ``period_col`` and convert it with
            :func:`parse_period_index`. If ``False`` (default), attach
            ``period_index`` only when the column is already datetime-like.
        """
        cols = panel_columns or PanelColumns()
        sku_col = cols.sku_col
        price_col = cols.price_col
        quantity_col = cols.quantity_col
        quantity_floor = cols.quantity_floor
        group_columns = cols.group_columns

        if quantity_floor <= 0:
            raise ValueError("quantity_floor must be positive")

        reserved = {sku_col, price_col, quantity_col}
        control_columns = _resolve_feature_columns(
            df,
            cols.control_columns,
            auto_prefix="control_",
            label="control_columns",
            reserved=reserved,
        )
        iv_columns = _resolve_feature_columns(
            df,
            cols.iv_columns,
            auto_prefix="iv_",
            label="iv_columns",
            reserved=reserved,
        )
        overlap_iv_ctrl = set(iv_columns) & set(control_columns)
        if overlap_iv_ctrl:
            raise ValueError(
                f"iv_columns must not overlap control_columns: {sorted(overlap_iv_ctrl)}"
            )

        if group_columns is not None:
            group_columns = tuple(group_columns)
            if len(group_columns) == 0:
                group_columns = None
            elif len(set(group_columns)) != len(group_columns):
                raise ValueError("group_columns must not contain duplicates")

        base = (sku_col, price_col, quantity_col)
        missing = set(base) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        extra = [*control_columns, *iv_columns]
        if group_columns is not None:
            bad_g = set(group_columns) - set(df.columns)
            if bad_g:
                raise ValueError(f"group_columns not in frame: {sorted(bad_g)}")
            overlap = set(group_columns) & {sku_col, price_col, quantity_col}
            if overlap:
                raise ValueError(
                    f"group_columns must not overlap sku/price/quantity: {overlap}"
                )
            overlap_g_iv = set(group_columns) & set(iv_columns)
            if overlap_g_iv:
                raise ValueError(
                    f"iv_columns must not overlap group_columns: {sorted(overlap_g_iv)}"
                )
            extra = [*group_columns, *extra]

        discount = _read_panel_discount(
            df,
            cols,
            control_columns=control_columns,
            iv_columns=iv_columns,
            group_columns=group_columns,
        )

        if cols.period_col in df.columns:
            dup_key = [sku_col, cols.period_col]
            if cols.region_col is not None and cols.region_col in df.columns:
                dup_key.append(cols.region_col)
            dup_counts = df.groupby(dup_key, sort=False).size()
            n_dup_groups = int((dup_counts > 1).sum())
            if n_dup_groups:
                keys = ", ".join(dup_key)
                warnings.warn(
                    f"Found {n_dup_groups} ({keys}) combination(s) "
                    "with more than one row; they will be fit as independent "
                    "observations. If this is unintentional (e.g. a duplicated "
                    "join), deduplicate before fitting.",
                    stacklevel=2,
                )

        sub = df[[*base, *extra]].copy()
        if sub[[sku_col, price_col, quantity_col]].isna().any().any():
            raise ValueError(
                "NaN in sku / price / quantity are not allowed after column selection"
            )

        price = sub[price_col].to_numpy(dtype=np.float64)
        quantity = sub[quantity_col].to_numpy(dtype=np.float64)
        if (price <= 0).any():
            raise ValueError("price must be strictly positive for log transform")
        if (quantity < 0).any():
            raise ValueError("quantity must be non-negative")
        log_price, log_regular_price, log_discount_index = decompose_log_price(
            price, discount
        )
        log_quantity = np.log(np.maximum(quantity, quantity_floor))

        if len(control_columns) and sub[list(control_columns)].isna().any().any():
            raise ValueError("NaN in control columns are not allowed")
        if len(iv_columns) and sub[list(iv_columns)].isna().any().any():
            raise ValueError("NaN in instrument columns are not allowed")
        if group_columns is not None and sub[list(group_columns)].isna().any().any():
            raise ValueError("NaN in group_columns are not allowed")

        cat = pd.Categorical(sub[sku_col])
        obs_sku_idx = cat.codes.astype(np.int64)
        n_skus = int(cat.categories.size)
        if n_skus < 1:
            raise ValueError("No SKU levels found")

        sku_levels = pd.Index(cat.categories)
        hierarchy: HierarchyIndex | None = None
        if group_columns is not None:
            hierarchy = _build_hierarchy_index(sub, sku_col, group_columns, sku_levels)

        if len(control_columns):
            control_matrix = sub[list(control_columns)].to_numpy(dtype=np.float64)
        else:
            control_matrix = np.empty((len(sub), 0), dtype=np.float64)

        if len(iv_columns):
            iv_matrix = sub[list(iv_columns)].to_numpy(dtype=np.float64)
        else:
            iv_matrix = np.empty((len(sub), 0), dtype=np.float64)

        period_index: pd.DatetimeIndex | None = None
        period_col = cols.period_col
        if parse_period:
            if period_col not in df.columns:
                raise ValueError(f"Missing period column: {period_col!r}")
            period_index = parse_period_index(
                df[period_col], period_col=period_col
            )
        elif period_col in df.columns and _is_datetime_like(df[period_col]):
            period_index = parse_period_index(
                df[period_col], period_col=period_col
            )

        return cls(
            log_price=log_price,
            log_quantity=log_quantity,
            obs_sku_idx=obs_sku_idx,
            n_skus=n_skus,
            sku_levels=sku_levels,
            control_matrix=control_matrix,
            control_names=tuple(control_columns),
            iv_matrix=iv_matrix,
            iv_names=tuple(iv_columns),
            n_obs=int(len(sub)),
            log_regular_price=log_regular_price,
            log_discount_index=log_discount_index,
            hierarchy=hierarchy,
            period_index=period_index,
        )

    @staticmethod
    def build_cross_log_price_matrix(
        df: pd.DataFrame,
        *,
        panel_columns: PanelColumns,
        sku_levels: pd.Index,
        pair_from_idx: np.ndarray,
        pair_to_idx: np.ndarray,
        obs_sku_idx: np.ndarray,
    ) -> np.ndarray:
        """
        Rows align with ``df`` row order. Column ``p`` is nonzero on rows whose focal SKU
        is ``pair_from[p]``, with value log price of ``pair_to[p]`` in the same market cell.
        """
        sku_col = panel_columns.sku_col
        price_col = panel_columns.price_col
        period_col = panel_columns.period_col
        region_col = panel_columns.region_col

        n_obs = len(df)
        n_skus = len(sku_levels)
        n_pairs = len(pair_from_idx)
        pair_map = _pair_lookup_map(pair_from_idx, pair_to_idx)
        sku_to_idx = {sku_levels[i]: i for i in range(n_skus)}

        if period_col not in df.columns:
            raise ValueError(f"Missing period_col {period_col!r} for cross-elasticity")
        if region_col is not None and region_col not in df.columns:
            raise ValueError(f"Missing region_col {region_col!r} for cross-elasticity")

        cell_cols = [period_col] + ([region_col] if region_col is not None else [])
        X = np.zeros((n_obs, n_pairs), dtype=np.float64)

        for _cell_key, grp in df.groupby(cell_cols, sort=False):
            if len(grp) != n_skus:
                raise ValueError(
                    "cross-elasticity requires a balanced panel: each market cell must "
                    f"have exactly one row per SKU; expected {n_skus} rows, got {len(grp)}"
                )
            log_p_arr, row_indices = _cell_log_prices_and_rows(
                grp,
                df,
                sku_col=sku_col,
                price_col=price_col,
                sku_to_idx=sku_to_idx,
                n_skus=n_skus,
            )
            _fill_cross_matrix_for_cell(
                X,
                log_p_arr=log_p_arr,
                row_indices=row_indices,
                n_skus=n_skus,
                pair_map=pair_map,
            )

        _assert_cross_matrix_focal_rows(
            X, obs_sku_idx=obs_sku_idx, pair_from=pair_from_idx
        )
        return X

    def with_cross_elasticity(
        self,
        df: pd.DataFrame,
        *,
        panel_columns: PanelColumns,
        cross_pairs: CrossPairIndex,
    ) -> PricePanelData:
        cross_pair_log_price_matrix = self.build_cross_log_price_matrix(
            df=df,
            panel_columns=panel_columns,
            sku_levels=self.sku_levels,
            pair_from_idx=cross_pairs.pair_from,
            pair_to_idx=cross_pairs.pair_to,
            obs_sku_idx=self.obs_sku_idx,
        )
        if int(cross_pair_log_price_matrix.shape[1]) != cross_pairs.n_pairs:
            raise RuntimeError(
                "cross matrix width does not match CrossPairIndex.n_pairs"
            )

        return replace(
            self,
            cross_pair_log_price_matrix=cross_pair_log_price_matrix,
            cross_pairs=cross_pairs,
        )

    def coords(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sku": np.arange(self.n_skus),
            "obs": np.arange(self.n_obs),
        }
        if self.hierarchy is not None:
            for i, n in enumerate(self.hierarchy.n_groups_per_level):
                out[f"group_{i}"] = np.arange(n)
        if self.n_cross_pairs > 0:
            out["cross_pair"] = np.arange(self.n_cross_pairs, dtype=np.int64)
            out["cross_pool"] = np.arange(self.n_cross_pool, dtype=np.int64)
        if self.n_iv > 0:
            out["iv"] = np.arange(self.n_iv, dtype=np.int64)
        return out
