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
    quantity_col: str = "quantity"
    quantity_floor: float = 1.0
    control_columns: tuple[str, ...] | None = None
    group_columns: tuple[str, ...] | None = None
    period_col: str = "period"
    region_col: str | None = None
    floor_censoring_warn_threshold: float | None = 0.5
    """Warn when a SKU has this fraction (or more) of observations at/below
    ``quantity_floor`` -- such a SKU carries little to no price-response
    signal, even though nothing in ``run_diagnostics()`` otherwise flags it.
    Set to ``None`` to disable the check."""


def _auto_control_columns(df: pd.DataFrame) -> tuple[str, ...]:
    return tuple(c for c in df.columns if c.startswith("control_"))


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
    """

    log_price: np.ndarray
    log_quantity: np.ndarray
    obs_sku_idx: np.ndarray
    n_skus: int
    sku_levels: pd.Index
    control_matrix: np.ndarray
    control_names: tuple[str, ...]
    n_obs: int
    hierarchy: HierarchyIndex | None = None
    """Present when ``PanelColumns.group_columns`` were set at build time."""
    cross_pair_log_price_matrix: np.ndarray | None = None
    """Shape ``(n_obs, n_cross_pairs)``: log competitor prices for directed cross pairs."""
    cross_pairs: CrossPairIndex | None = None
    """Directed pair / pool index when cross-elasticity is enabled."""
    period_index: pd.DatetimeIndex | None = None
    """Row-aligned timestamps when ``period_col`` is datetime (or parsed)."""

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
        control_columns = cols.control_columns
        group_columns = cols.group_columns

        if quantity_floor <= 0:
            raise ValueError("quantity_floor must be positive")

        if control_columns is None:
            control_columns = _auto_control_columns(df)
        else:
            bad = set(control_columns) - set(df.columns)
            if bad:
                raise ValueError(f"control_columns not in frame: {sorted(bad)}")
            overlap_c = set(control_columns) & {sku_col, price_col, quantity_col}
            if overlap_c:
                raise ValueError(
                    f"control_columns must not overlap sku/price/quantity: {overlap_c}"
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

        extra = list(control_columns)
        if group_columns is not None:
            bad_g = set(group_columns) - set(df.columns)
            if bad_g:
                raise ValueError(f"group_columns not in frame: {sorted(bad_g)}")
            overlap = set(group_columns) & {sku_col, price_col, quantity_col}
            if overlap:
                raise ValueError(
                    f"group_columns must not overlap sku/price/quantity: {overlap}"
                )
            extra = [*group_columns, *extra]

        if cols.period_col in df.columns:
            dup_counts = df.groupby([sku_col, cols.period_col]).size()
            n_dup_groups = int((dup_counts > 1).sum())
            if n_dup_groups:
                warnings.warn(
                    f"Found {n_dup_groups} ({sku_col}, {cols.period_col}) combination(s) "
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
        log_price = np.log(price)
        log_quantity = np.log(np.maximum(quantity, quantity_floor))

        if len(control_columns) and sub[list(control_columns)].isna().any().any():
            raise ValueError("NaN in control columns are not allowed")
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
            n_obs=int(len(sub)),
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
        return out
