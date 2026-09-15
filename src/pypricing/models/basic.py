from __future__ import annotations

import json
import warnings
from abc import abstractmethod
from importlib.metadata import PackageNotFoundError, version as pkg_version
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
import pytensor.tensor as pt
from pymc_marketing.model_builder import ModelBuilder

from pypricing.model_components.cross_elasticity import (
    CrossElasticitySpec,
    build_gamma_pair,
    enumerate_cross_pairs,
    validate_cross_elasticity_config,
)
from pypricing.data import CrossPairIndex, PanelColumns, PricePanelData
from pypricing.data.price_panel import (
    floor_censored_fraction,
    parse_period_index,
    period_to_t_years,
)
from pypricing.model_components.time_terms import (
    TrendKind,
    fourier_features,
    infer_panel_frequency,
    normalize_seasonality,
    seasonality_components_for_frequency,
)

try:
    _PKG_VERSION = pkg_version("pypricing")
except PackageNotFoundError:  # pragma: no cover
    _PKG_VERSION = "0.0.1"


def _seasonality_to_json(
    value: Literal["auto"] | tuple[str, ...] | None,
) -> str | list[str] | None:
    if value is None or value == "auto":
        return value
    return list(value)


class DemandModel(ModelBuilder):
    """Base class for Bayesian log-demand models on long-format price panels.

    Subclasses define the own-price mean curve; this base handles fit/predict,
    optional controls, hierarchical SKU effects via ``PanelColumns.group_columns``,
    optional linear time trend, optional calendar seasonality, optional
    cross-price terms, plotting helpers, and per-SKU revenue optimization.

    Parameters
    ----------
    panel_columns
        Column names and panel knobs (SKU / price / quantity, controls,
        instruments, hierarchy, period / region). Defaults to ``PanelColumns()``.
        ``iv_columns`` (or auto-detected ``iv_*`` columns) enable control-function
        IV on the log-demand model classes.
    cross_elasticity
        If set, add directed cross-price effects (``CrossElasticitySpec``). Requires
        a balanced market cell per ``period`` (and ``region`` when used).
    trend
        Linear time trend in log-quantity, in years since the earliest training
        date. ``None`` (default) omits the term. ``"shared"`` is one slope;
        ``"sku"`` is a per-SKU slope pooled toward a shared mean (and through
        ``group_columns`` when set). Requires a datetime ``period`` column.
    seasonality
        Shared Fourier seasonality in log-quantity. ``None`` (default) omits it.
        ``"yearly"`` / ``"weekly"`` or a sequence of those add the corresponding
        harmonics. ``"auto"`` picks components from the panel's date spacing
        (daily → yearly+weekly; weekly/monthly → yearly). Requires datetime
        ``period``.
    model_config
        Prior overrides keyed by parameter name, each
        ``{"dist": <PyMC dist>, "kwargs": {...}}``.
    sampler_config
        Optional defaults forwarded to PyMC sampling.

    Notes
    -----
    Observation model is ``log Q ~ Normal(mu, sigma)`` with
    ``log Q = log(max(quantity, quantity_floor))``. See the example notebooks for
    workflows; use :class:`LogLogDemandModel`, :class:`QuadraticLogDemandModel`, or
    :class:`SigmoidSaturationDemandModel` for concrete curves.
    """

    _model_type = "DemandModel"
    version = _PKG_VERSION

    def __init__(
        self,
        *,
        panel_columns: PanelColumns | None = None,
        cross_elasticity: CrossElasticitySpec | None = None,
        trend: TrendKind | None = None,
        seasonality: str | Sequence[str] | None = None,
        model_config: dict[str, Any] | None = None,
        sampler_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_config=model_config, sampler_config=sampler_config)
        self.panel_columns = panel_columns or PanelColumns()
        self.cross_elasticity = cross_elasticity
        self.trend = trend
        self.seasonality = normalize_seasonality(seasonality)
        self.sku_levels_: pd.Index | None = None
        self.control_names_: tuple[str, ...] = ()
        self.iv_names_: tuple[str, ...] = ()
        self.log_price_midpoint_sku_: np.ndarray | None = None
        self.data: pd.DataFrame | None = None
        self.cross_pairs_: CrossPairIndex | None = None
        self.t0_: pd.Timestamp | None = None
        self.seasonality_: tuple[str, ...] | None = None
        self._validate_configuration()

    @property
    def model_name(self) -> str:
        return self._model_type

    @property
    def sku_col(self) -> str:
        return self.panel_columns.sku_col

    @property
    def price_col(self) -> str:
        return self.panel_columns.price_col

    @property
    def quantity_col(self) -> str:
        return self.panel_columns.quantity_col

    @property
    def quantity_floor(self) -> float:
        return self.panel_columns.quantity_floor

    @property
    def control_columns(self) -> tuple[str, ...] | None:
        return self.panel_columns.control_columns

    @property
    def iv_columns(self) -> tuple[str, ...] | None:
        return self.panel_columns.iv_columns

    @property
    def group_columns(self) -> tuple[str, ...] | None:
        return self.panel_columns.group_columns

    @property
    def period_col(self) -> str:
        return self.panel_columns.period_col

    @property
    def region_col(self) -> str | None:
        return self.panel_columns.region_col

    @abstractmethod
    def _build_pymc_model(self, data: PricePanelData) -> pm.Model:
        pass

    @abstractmethod
    def compute_mu_from_posterior(
        self,
        *,
        posterior: xr.Dataset,
        log_price: np.ndarray,
        obs_sku_idx: np.ndarray,
        X_control: np.ndarray | None,
        X_cross: np.ndarray | None = None,
        t_years: np.ndarray | None = None,
        X_season: np.ndarray | None = None,
    ) -> np.ndarray:
        pass

    def _uses_calendar(self) -> bool:
        return self.trend is not None or self.seasonality is not None

    def _validate_configuration(self) -> None:
        if self.trend not in (None, "sku", "shared"):
            raise ValueError(
                f"trend must be None, 'sku', or 'shared'; got {self.trend!r}"
            )
        validate_cross_elasticity_config(
            cross_elasticity=self.cross_elasticity,
            group_columns=self.panel_columns.group_columns,
        )

    @property
    def default_model_config(self) -> dict:
        return {}

    @property
    def default_sampler_config(self) -> dict:
        return {}

    @property
    def _serializable_model_config(self) -> dict:
        return self.model_config

    def create_idata_attrs(self) -> dict[str, str]:
        attrs = super().create_idata_attrs()
        cols = self.panel_columns
        ce = self.cross_elasticity
        attrs.update(
            {
                "panel_columns": json.dumps(
                    {
                        "sku_col": cols.sku_col,
                        "price_col": cols.price_col,
                        "quantity_col": cols.quantity_col,
                        "quantity_floor": cols.quantity_floor,
                        "control_columns": cols.control_columns,
                        "iv_columns": cols.iv_columns,
                        "group_columns": cols.group_columns,
                        "period_col": cols.period_col,
                        "region_col": cols.region_col,
                        "floor_censoring_warn_threshold": cols.floor_censoring_warn_threshold,
                    }
                ),
                "cross_elasticity": json.dumps(
                    None
                    if ce is None
                    else {"mode": ce.mode, "group_level": ce.group_level}
                ),
                "trend": json.dumps(self.trend),
                "seasonality": json.dumps(_seasonality_to_json(self.seasonality)),
            }
        )
        return attrs

    @classmethod
    def attrs_to_init_kwargs(cls, attrs) -> dict[str, Any]:
        kwargs = super().attrs_to_init_kwargs(attrs)
        ce_raw = json.loads(attrs["cross_elasticity"])
        cross_elasticity = (
            None
            if ce_raw is None
            else CrossElasticitySpec(
                mode=ce_raw["mode"],
                group_level=ce_raw.get("group_level"),
            )
        )

        pc = json.loads(attrs["panel_columns"])
        control_raw = pc.get("control_columns")
        iv_raw = pc.get("iv_columns")
        group_raw = pc.get("group_columns")
        panel_columns = PanelColumns(
            sku_col=pc["sku_col"],
            price_col=pc["price_col"],
            quantity_col=pc["quantity_col"],
            quantity_floor=float(pc["quantity_floor"]),
            control_columns=tuple(control_raw) if control_raw is not None else None,
            iv_columns=tuple(iv_raw) if iv_raw is not None else None,
            group_columns=tuple(group_raw) if group_raw is not None else None,
            period_col=pc.get("period_col", "period"),
            region_col=pc.get("region_col"),
            floor_censoring_warn_threshold=pc.get(
                "floor_censoring_warn_threshold", 0.5
            ),
        )

        trend = json.loads(attrs["trend"]) if "trend" in attrs else None
        seasonality = (
            json.loads(attrs["seasonality"]) if "seasonality" in attrs else None
        )
        kwargs.update(
            {
                "panel_columns": panel_columns,
                "cross_elasticity": cross_elasticity,
                "trend": trend,
                "seasonality": seasonality,
            }
        )
        return kwargs

    def _add_fit_data_group(self, data: pd.DataFrame) -> None:
        fit_data = data.to_xarray()
        if self.idata is None:
            raise RuntimeError("No idata available to attach fit_data.")
        if "fit_data" in self.idata:
            del self.idata.fit_data
        with np.errstate(all="ignore"):
            self.idata.add_groups(fit_data=fit_data)

    def _cross_term(self, data: PricePanelData) -> pt.TensorVariable:
        cross_term = 0.0
        if (
            data.cross_pairs is not None
            and data.cross_pair_log_price_matrix is not None
        ):
            X_cross = pt.constant(data.cross_pair_log_price_matrix)
            gamma_pair = build_gamma_pair(
                self.model_config,
                pair_pool_idx=data.cross_pairs.pair_pool,
            )
            cross_term = pt.dot(X_cross, gamma_pair)
        return cross_term

    def build_model(self, data: pd.DataFrame) -> None:
        self._validate_configuration()
        cols = self.panel_columns
        if self.cross_elasticity is not None:
            if cols.period_col not in data.columns:
                raise ValueError(
                    f"cross_elasticity requires column period_col={cols.period_col!r}"
                )
            if cols.region_col is not None and cols.region_col not in data.columns:
                raise ValueError(
                    f"cross_elasticity requires column region_col={cols.region_col!r}"
                )

        panel = PricePanelData.from_frame(
            data,
            panel_columns=cols,
            parse_period=self._uses_calendar(),
        )
        if self.trend is not None:
            if panel.period_index is None:
                raise RuntimeError("internal: period_index missing with trend enabled")
            self.t0_ = pd.Timestamp(panel.period_index.min())
        else:
            self.t0_ = None
        self.seasonality_ = self._resolve_seasonality(panel.period_index)

        threshold = cols.floor_censoring_warn_threshold
        if threshold is not None:
            floor_frac = floor_censored_fraction(data, panel_columns=cols)
            censored = floor_frac[floor_frac >= threshold]
            if len(censored):
                warnings.warn(
                    f"{len(censored)} SKU(s) have quantity at/below quantity_floor "
                    f"for >= {threshold:.0%} of their observations "
                    f"({sorted(censored.index.tolist())}); elasticity for these "
                    "SKUs is likely unidentified from this data. See "
                    "run_diagnostics()['floor_censored_skus'] for the exact "
                    "fractions.",
                    stacklevel=2,
                )

        if self.cross_elasticity is not None:
            cross_pairs = enumerate_cross_pairs(
                mode=self.cross_elasticity.mode,
                n_skus=panel.n_skus,
                hierarchy=panel.hierarchy,
                cross_elasticity_group_level=self.cross_elasticity.group_level,
            )
            panel = panel.with_cross_elasticity(
                df=data,
                panel_columns=cols,
                cross_pairs=cross_pairs,
            )
            self.cross_pairs_ = cross_pairs
        else:
            self.cross_pairs_ = None

        mid = np.empty(panel.n_skus, dtype=np.float64)
        for sku_idx in range(panel.n_skus):
            mask = panel.obs_sku_idx == sku_idx
            mid[sku_idx] = float(np.median(panel.log_price[mask]))
        self.log_price_midpoint_sku_ = mid

        self.model = self._build_pymc_model(panel)
        self.sku_levels_ = panel.sku_levels
        self.control_names_ = panel.control_names
        self.iv_names_ = panel.iv_names
        self.data = data.copy()

    def build_from_idata(self, idata: az.InferenceData) -> None:
        if not hasattr(idata, "fit_data"):
            raise ValueError("Loaded idata is missing fit_data group.")
        data = idata.fit_data.to_dataframe().reset_index(drop=True)
        self.build_model(data)

    def fit(self, data: pd.DataFrame, **sample_kwargs: Any) -> az.InferenceData:
        self.build_model(data)
        sampler_kwargs = dict(self.sampler_config)
        sampler_kwargs.update(sample_kwargs)
        with self.model:
            self.idata = pm.sample(**sampler_kwargs)
        self._add_fit_data_group(data)
        self.set_idata_attrs(self.idata)
        return self.idata

    def _require_fitted(self) -> None:
        if (
            self.idata is None
            or "posterior" not in self.idata
            or self.sku_levels_ is None
        ):
            raise RuntimeError("Model has not been fit yet; call fit() first.")

    def fit_summary(
        self,
        var_names: list[str] | None = None,
        **summary_kwargs: Any,
    ) -> pd.DataFrame:
        self._require_fitted()
        assert self.idata is not None
        if var_names is None:
            candidates = [
                "alpha_sku",
                "elasticity_sku",
                "curvature_sku",
                "sigma",
                "beta_control",
                "mu_trend",
                "trend_sku",
                "beta_season",
                "rho",
                "pi",
                "sigma_price",
                "alpha_price_sku",
                "beta_control_price",
            ]
            candidates.append("gamma_pair")
            var_names = [v for v in candidates if v in self.idata.posterior]
        return az.summary(self.idata, var_names=var_names, **summary_kwargs)

    def run_diagnostics(
        self, *, floor_censoring_threshold: float | None = None
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        floor_censoring_threshold
            Overrides ``panel_columns.floor_censoring_warn_threshold`` for this
            call only, without needing to refit. ``None`` (default) uses the
            value configured on the model.
        """
        self._require_fitted()
        assert self.idata is not None
        out: dict[str, Any] = {}
        if "sample_stats" in self.idata and "diverging" in self.idata.sample_stats:
            out["n_divergent"] = int(self.idata.sample_stats["diverging"].sum().item())
        else:
            out["n_divergent"] = None
        try:
            summary = self.fit_summary()
            out["max_rhat"] = (
                float(summary["r_hat"].max())
                if "r_hat" in summary.columns and len(summary)
                else None
            )
        except Exception:
            out["max_rhat"] = None

        threshold = (
            floor_censoring_threshold
            if floor_censoring_threshold is not None
            else self.panel_columns.floor_censoring_warn_threshold
        )
        if threshold is not None:
            assert self.data is not None
            floor_frac = floor_censored_fraction(self.data, panel_columns=self.panel_columns)
            out["floor_censored_skus"] = {
                sku: float(frac) for sku, frac in floor_frac.items() if frac >= threshold
            }
        else:
            out["floor_censored_skus"] = {}
        out.update(self._iv_diagnostics())
        return out

    def _iv_diagnostics(self, *, hdi_prob: float = 0.9) -> dict[str, Any]:
        """Weak-IV and endogeneity flags from the control-function posterior."""
        assert self.idata is not None
        post = self.idata.posterior
        if "rho" not in post and "pi" not in post:
            return {}

        out: dict[str, Any] = {}
        alpha = (1.0 - hdi_prob) / 2.0
        if "rho" in post:
            rho = np.asarray(post["rho"].values, dtype=np.float64).ravel()
            out["rho_mean"] = float(np.mean(rho))
            lo, hi = np.quantile(rho, [alpha, 1.0 - alpha])
            out["rho_hdi"] = (float(lo), float(hi))
            out["rho_hdi_includes_zero"] = bool(lo <= 0.0 <= hi)
        if "pi" in post:
            pi = np.asarray(post["pi"].values, dtype=np.float64)
            if pi.ndim == 2:
                pi = pi[:, :, None]
            flat = pi.reshape(-1, pi.shape[-1])
            lo = np.quantile(flat, alpha, axis=0)
            hi = np.quantile(flat, 1.0 - alpha, axis=0)
            includes_zero = (lo <= 0.0) & (hi >= 0.0)
            names = list(self.iv_names_) or [
                f"pi[{i}]" for i in range(flat.shape[1])
            ]
            out["weak_iv_instruments"] = [
                names[i] for i in range(len(includes_zero)) if bool(includes_zero[i])
            ]
            out["weak_iv"] = bool(np.all(includes_zero))
        return out

    def graphviz(self):
        self._require_fitted()
        if not hasattr(self, "model"):
            raise RuntimeError(
                "graphviz() requires the in-memory PyMC model; refit after load()."
            )
        try:
            import graphviz  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "graphviz() requires the optional 'graphviz' extra. "
                "Install with: pip install 'pypricing[graphviz]' "
                "(also needs the system Graphviz binaries)."
            ) from exc
        return pm.model_to_graphviz(self.model)

    def plot_elasticity_posterior(self, *, hdi_prob: float = 0.9, ax=None):
        from pypricing.plotting import plot_elasticity_posterior as _plot

        return _plot(self, hdi_prob=hdi_prob, ax=ax)

    def plot_cross_price_effects_heatmap(
        self,
        *,
        agg: Literal["mean", "median"] = "mean",
        cmap: str = "coolwarm",
        ax=None,
    ):
        from pypricing.plotting import plot_cross_price_effects_heatmap as _plot

        return _plot(self, agg=agg, cmap=cmap, ax=ax)

    def plot_response_curve(
        self,
        *,
        sku: Any,
        price_grid: Iterable[float],
        controls: dict[str, float] | None = None,
        at_period: pd.Timestamp | str | None = None,
        hdi_prob: float = 0.9,
        ax=None,
    ):
        from pypricing.plotting import plot_response_curve as _plot

        return _plot(
            self,
            sku=sku,
            price_grid=price_grid,
            controls=controls,
            at_period=at_period,
            hdi_prob=hdi_prob,
            ax=ax,
        )

    def plot_local_elasticity_vs_price(
        self,
        *,
        sku: Any,
        price_grid: Iterable[float],
        controls: dict[str, float] | None = None,
        at_period: pd.Timestamp | str | None = None,
        hdi_prob: float = 0.9,
        fd_step: float | None = None,
        ax=None,
    ):
        from pypricing.plotting import plot_local_elasticity_vs_price as _plot

        return _plot(
            self,
            sku=sku,
            price_grid=price_grid,
            controls=controls,
            at_period=at_period,
            hdi_prob=hdi_prob,
            fd_step=fd_step,
            ax=ax,
        )

    def plot_revenue_vs_price(
        self,
        *,
        sku: Any,
        price_grid: Iterable[float],
        controls: dict[str, float] | None = None,
        at_period: pd.Timestamp | str | None = None,
        hdi_prob: float = 0.9,
        random_seed: int | None = None,
        ax=None,
    ):
        from pypricing.plotting import plot_revenue_vs_price as _plot

        return _plot(
            self,
            sku=sku,
            price_grid=price_grid,
            controls=controls,
            at_period=at_period,
            hdi_prob=hdi_prob,
            random_seed=random_seed,
            ax=ax,
        )

    def plot_optimization_summary(
        self,
        opt_df: pd.DataFrame,
        *,
        reference_prices: Any = None,
        controls_df: pd.DataFrame | None = None,
        include_revenue_comparison: bool = True,
        ax=None,
    ):
        from pypricing.plotting import plot_optimization_summary as _plot

        return _plot(
            self,
            opt_df,
            reference_prices=reference_prices,
            controls_df=controls_df,
            include_revenue_comparison=include_revenue_comparison,
            ax=ax,
        )

    def plot_posterior_predictive_calibration(
        self,
        *,
        df: pd.DataFrame | None = None,
        hdi_prob: float = 0.94,
        style: Literal["series", "calibration"] = "series",
        random_seed: int | None = None,
        ax=None,
    ):
        from pypricing.plotting import (
            plot_posterior_predictive_calibration as _plot,
        )

        return _plot(
            self,
            df=df,
            hdi_prob=hdi_prob,
            style=style,
            random_seed=random_seed,
            ax=ax,
        )

    def _build_prediction_features(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        self._require_fitted()
        assert self.sku_levels_ is not None
        if self.panel_columns.sku_col not in df.columns:
            raise ValueError(f"Missing required column: {self.panel_columns.sku_col!r}")
        if self.panel_columns.price_col not in df.columns:
            raise ValueError(
                f"Missing required column: {self.panel_columns.price_col!r}"
            )

        sku_values = df[self.panel_columns.sku_col]
        unknown_mask = ~sku_values.isin(self.sku_levels_)
        if bool(unknown_mask.any()):
            unknown = pd.unique(sku_values[unknown_mask]).tolist()
            raise ValueError(
                f"Unknown {self.panel_columns.sku_col} values in prediction data: {unknown}"
            )

        cat = pd.Categorical(sku_values, categories=self.sku_levels_)
        obs_sku_idx = cat.codes.astype(np.int64)
        price = df[self.panel_columns.price_col].to_numpy(dtype=np.float64)
        if (price <= 0).any():
            raise ValueError("price must be strictly positive for log transform")
        log_price = np.log(price)

        if self.control_names_:
            missing = [c for c in self.control_names_ if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing control columns in prediction data: {missing}"
                )
            X = df.loc[:, list(self.control_names_)].to_numpy(dtype=np.float64)
            if np.isnan(X).any():
                raise ValueError("NaN in control columns are not allowed")
            return log_price, obs_sku_idx, X
        return log_price, obs_sku_idx, None

    def _resolve_seasonality(
        self, period_index: pd.DatetimeIndex | None
    ) -> tuple[str, ...] | None:
        spec = self.seasonality
        if spec is None:
            return None
        if spec == "auto":
            if period_index is None:
                raise RuntimeError("internal: period_index missing with seasonality")
            freq = infer_panel_frequency(period_index)
            return seasonality_components_for_frequency(freq, period_index)
        return spec

    def default_counterfactual_period(self) -> pd.Timestamp:
        """Last training timestamp; used when holding trend/seasonality fixed."""
        if not self._uses_calendar():
            raise RuntimeError(
                "default_counterfactual_period requires trend or seasonality"
            )
        if self.data is not None and self.period_col in self.data.columns:
            idx = parse_period_index(
                self.data[self.period_col], period_col=self.period_col
            )
            return pd.Timestamp(idx.max())
        if self.t0_ is not None:
            return self.t0_
        raise RuntimeError("No training periods available for a counterfactual date")

    def _period_index_for_frame(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        period_col = self.period_col
        if period_col not in df.columns:
            raise ValueError(
                f"Model was fit with trend or seasonality; prediction data needs "
                f"datetime column {period_col!r}."
            )
        return parse_period_index(df[period_col], period_col=period_col)

    def _t_years_for_frame(self, df: pd.DataFrame) -> np.ndarray | None:
        if self.trend is None:
            return None
        if self.t0_ is None:
            raise RuntimeError("Model is missing t0_; fit the model first.")
        idx = self._period_index_for_frame(df)
        return period_to_t_years(idx, self.t0_)

    def _t_years_for_counterfactual(
        self,
        n: int,
        at_period: pd.Timestamp | str | None = None,
    ) -> np.ndarray | None:
        if self.trend is None:
            return None
        if self.t0_ is None:
            raise RuntimeError("Model is missing t0_; fit the model first.")
        ts = (
            pd.Timestamp(at_period)
            if at_period is not None
            else self.default_counterfactual_period()
        )
        t = period_to_t_years(pd.DatetimeIndex([ts]), self.t0_)[0]
        return np.full(n, t, dtype=np.float64)

    def _season_features_for_index(
        self, period_index: pd.DatetimeIndex
    ) -> np.ndarray | None:
        if not self.seasonality_:
            return None
        X, _names = fourier_features(period_index, self.seasonality_)
        return X

    def _season_features_for_frame(self, df: pd.DataFrame) -> np.ndarray | None:
        if not self.seasonality_:
            return None
        return self._season_features_for_index(self._period_index_for_frame(df))

    def _season_features_for_counterfactual(
        self,
        n: int,
        at_period: pd.Timestamp | str | None = None,
    ) -> np.ndarray | None:
        if not self.seasonality_:
            return None
        ts = (
            pd.Timestamp(at_period)
            if at_period is not None
            else self.default_counterfactual_period()
        )
        row = self._season_features_for_index(pd.DatetimeIndex([ts]))
        assert row is not None
        return np.repeat(row, n, axis=0)

    def _cross_matrix_for_frame(
        self, df: pd.DataFrame, obs_sku_idx: np.ndarray
    ) -> np.ndarray | None:
        if (
            self.cross_elasticity is None
            or self.cross_pairs_ is None
            or self.sku_levels_ is None
        ):
            return None
        return PricePanelData.build_cross_log_price_matrix(
            df=df,
            panel_columns=self.panel_columns,
            sku_levels=self.sku_levels_,
            pair_from_idx=self.cross_pairs_.pair_from,
            pair_to_idx=self.cross_pairs_.pair_to,
            obs_sku_idx=obs_sku_idx,
        )

    def sample_posterior_predictive(
        self, df: pd.DataFrame, *, random_seed: int | None = None
    ) -> xr.Dataset:
        self._require_fitted()
        assert self.idata is not None

        log_price, obs_sku_idx, X_control = self._build_prediction_features(df)
        X_cross = self._cross_matrix_for_frame(df, obs_sku_idx)
        t_years = self._t_years_for_frame(df)
        X_season = self._season_features_for_frame(df)
        post = self.idata.posterior
        sigma = post["sigma"].values
        mu = self.compute_mu_from_posterior(
            posterior=post,
            log_price=log_price,
            obs_sku_idx=obs_sku_idx,
            X_control=X_control,
            X_cross=X_cross,
            t_years=t_years,
            X_season=X_season,
        )

        rng = np.random.default_rng(random_seed)
        noise = rng.normal(size=mu.shape) * sigma[:, :, None]
        log_quantity = mu + noise
        quantity = np.exp(log_quantity)
        n_obs = log_price.shape[0]
        obs = np.arange(n_obs, dtype=np.int64)

        return xr.Dataset(
            data_vars={
                "log_quantity": (("chain", "draw", "obs"), log_quantity),
                "quantity": (("chain", "draw", "obs"), quantity),
            },
            coords={
                "chain": post.coords["chain"].values,
                "draw": post.coords["draw"].values,
                "obs": obs,
            },
            attrs={"pypricing_version": "mvp", "model_name": self.model_name},
        )

    def predict(
        self,
        df: pd.DataFrame,
        *,
        hdi_prob: float = 0.94,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        ds = self.sample_posterior_predictive(df, random_seed=random_seed)
        q = ds["quantity"]
        mean = q.mean(dim=("chain", "draw")).to_numpy()
        hdi = az.hdi(q, hdi_prob=hdi_prob)
        hdi_q = hdi["quantity"] if isinstance(hdi, xr.Dataset) else hdi
        lower = hdi_q.sel(hdi="lower").to_numpy()
        upper = hdi_q.sel(hdi="higher").to_numpy()

        out = df.copy()
        out["quantity_mean"] = mean
        out["quantity_hdi_lower"] = lower
        out["quantity_hdi_upper"] = upper
        return out

    def train_test_split(
        self,
        df: pd.DataFrame,
        *,
        test_size: float = 0.2,
        period_col: str = "period",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1 (exclusive)")
        if period_col not in df.columns:
            raise ValueError(f"Missing period column: {period_col!r}")
        periods = pd.unique(df[period_col])
        if len(periods) < 2:
            raise ValueError("Need at least 2 unique periods for a train/test split")
        try:
            periods_sorted = np.sort(periods)
        except Exception:
            periods_sorted = np.array(sorted(periods.tolist()), dtype=object)
        n_test_periods = max(1, int(np.ceil(test_size * len(periods_sorted))))
        test_periods = set(periods_sorted[-n_test_periods:].tolist())
        is_test = df[period_col].isin(test_periods)
        return df.loc[~is_test].copy(), df.loc[is_test].copy()

    def fit_train_test(
        self,
        df: pd.DataFrame,
        *,
        test_size: float = 0.2,
        period_col: str = "period",
        hdi_prob: float = 0.94,
        random_seed: int | None = None,
        **fit_sample_kwargs: Any,
    ) -> dict[str, Any]:
        if self.panel_columns.quantity_col not in df.columns:
            raise ValueError(
                f"Missing required column for evaluation: {self.panel_columns.quantity_col!r}"
            )
        train_df, test_df = self.train_test_split(
            df, test_size=test_size, period_col=period_col
        )
        if random_seed is not None and "random_seed" not in fit_sample_kwargs:
            fit_sample_kwargs["random_seed"] = random_seed
        self.fit(train_df, **fit_sample_kwargs)
        preds = self.predict(test_df, hdi_prob=hdi_prob, random_seed=random_seed)

        y_true = test_df[self.panel_columns.quantity_col].to_numpy(dtype=np.float64)
        y_pred = preds["quantity_mean"].to_numpy(dtype=np.float64)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        lower = preds["quantity_hdi_lower"].to_numpy(dtype=np.float64)
        upper = preds["quantity_hdi_upper"].to_numpy(dtype=np.float64)
        hdi_coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        return {
            "rmse": rmse,
            "hdi_coverage": hdi_coverage,
            "test_predictions": preds,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }

    def optimize_prices(
        self,
        price_bounds: dict[Any, tuple[float, float]] | pd.Series,
        *,
        controls_df: pd.DataFrame | None = None,
        minimize_options: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        from pypricing.optimizer import optimize_prices as _optimize_prices

        self._require_fitted()
        return _optimize_prices(
            model=self,
            price_bounds=price_bounds,
            controls_df=controls_df,
            minimize_options=minimize_options,
        )

    def quantity_multiplier_summary(
        self,
        *,
        price_multiplier: float,
        hdi_prob: float = 0.9,
        return_draws: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, xr.DataArray]:
        from pypricing.posterior import (
            quantity_multiplier_posterior as _quantity_multiplier_posterior,
            summarize_quantity_multiplier_by_sku as _quantity_multiplier_summary,
        )

        self._require_fitted()
        summary = _quantity_multiplier_summary(
            model=self,
            price_multiplier=price_multiplier,
            hdi_prob=hdi_prob,
        )
        if not return_draws:
            return summary
        draws = _quantity_multiplier_posterior(
            model=self,
            price_multiplier=price_multiplier,
        )
        return summary, draws
