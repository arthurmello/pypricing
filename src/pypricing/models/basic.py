from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
import pytensor.tensor as pt

from pypricing.arviz_compat import add_dataset_group, hdi_bounds, read_netcdf, summary
from pypricing.model_components.cross_elasticity import (
    CrossElasticitySpec,
    build_gamma_pair,
    enumerate_cross_pairs,
    validate_cross_elasticity_config,
)
from pypricing.data import CrossPairIndex, PanelColumns, PricePanelData
from pypricing.diagnostics import DiagnosticWarning
from pypricing.endogeneity import (
    EndogeneityMethod,
    get_endogeneity_method,
    parse_endogeneity,
)
from pypricing.endogeneity.latent import EXPERIMENTAL_WARNING
from pypricing.data.price_panel import (
    decompose_log_price,
    floor_censored_fraction,
    parse_period_index,
    period_to_t_years,
    read_discount_fraction,
    skus_without_variation,
)
from pypricing.model_components.priors import resolve_prior
from pypricing.model_components.promo_extras import (
    ExtraFeatures,
    PromoSpec,
    build_extra_features,
    extra_mu_term,
)
from pypricing.model_components.time_terms import (
    TimeEffectsKind,
    TrendKind,
    fourier_features,
    infer_panel_frequency,
    normalize_seasonality,
    seasonality_components_for_frequency,
)

if TYPE_CHECKING:
    from pypricing.diagnostics.report import DiagnosticsReport

try:
    _PKG_VERSION = pkg_version("pypricing")
except PackageNotFoundError:  # pragma: no cover
    _PKG_VERSION = "0.0.1"



class DifferentModelError(Exception):
    """Loaded InferenceData does not match this model class or configuration."""


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {
            key: value
            for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith("_")
        }
    return str(obj)


def _seasonality_to_json(
    value: Literal["auto"] | tuple[str, ...] | None,
) -> str | list[str] | None:
    if value is None or value == "auto":
        return value
    return list(value)


def _as_label_index(levels: np.ndarray) -> pd.Index:
    index = pd.Index(levels)
    if pd.api.types.is_datetime64_any_dtype(index):
        return pd.DatetimeIndex(index)
    return index


def _label_codes(levels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``-1`` where ``values`` is not a training label."""
    index = _as_label_index(levels)
    keys = pd.Index(values)
    if isinstance(index, pd.DatetimeIndex):
        keys = pd.DatetimeIndex(pd.to_datetime(keys))
    return index.get_indexer(keys).astype(np.int64)


def _one_label_code(levels: np.ndarray, label: Any) -> int:
    code = int(_label_codes(levels, np.asarray([label]))[0])
    if code < 0:
        raise ValueError(
            f"period_effect label {label!r} is not a training period"
        )
    return code


def _unknown_row_groups(
    codes: np.ndarray, values: np.ndarray | None
) -> list[np.ndarray]:
    """Row positions whose period is not in the training sample, one array per label."""
    unknown = np.flatnonzero(codes < 0)
    if unknown.size == 0:
        return []
    if values is None:
        return [unknown]
    keys = pd.Series(np.asarray(values)[unknown])
    return [
        unknown[np.asarray(locator, dtype=np.int64)]
        for locator in keys.groupby(keys, sort=False).indices.values()
    ]


class DemandModel:
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
        IV when ``endogeneity`` is left unset. After ``build_model``, the active
        correction is ``endogeneity_method_``. ``discount_col``, when set, splits
        own-price response into a regular-price elasticity (``elasticity_sku``,
        the curve at regular price) and a log-linear promo elasticity
        (``promo_elasticity_sku``). ``predict`` then requires that column;
        ``price`` stays the price actually paid.
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
        harmonics.         ``"auto"`` picks components from the panel's date spacing
        (daily → yearly+weekly; weekly/monthly → yearly). Requires datetime
        ``period``.
    time_effects
        Free period effects in log-quantity. ``None`` (default) omits them.
        ``"period"`` is one zero-sum effect per period, shared by every SKU.
        ``"region_period"`` adds a zero-sum region-by-period deviation and
        requires ``panel_columns.region_col``. Cannot be combined with
        ``trend`` or ``seasonality`` (a free period effect absorbs both).
        Every period cell needs at least two SKUs.
    mundlak
        If ``True``, the SKU-intercept prior mean includes
        ``gamma_mundlak`` times that SKU's centered mean log regular price.
        Removes the shrinkage bias that appears when baseline demand is
        correlated with a SKU's price level. Default ``False``.
    promo
        Extra promo terms (:class:`PromoSpec`): curved depth response,
        promo-mechanic effects, post-promo lags and within-group
        cannibalization (``pressure_level`` needs ``group_columns``).
        Requires ``panel_columns.discount_col``. ``None`` (default) keeps the
        log-linear promo term only.
    varying_controls
        Control columns whose slope varies by SKU: the shared ``beta_control``
        plus a zero-mean deviation pooled through ``group_columns``
        (``beta_control_dev_sku``). Each must be one of the model's controls.
        Use it for competitor prices with category- or brand-specific effects.
    endogeneity
        Correction for a price set from the demand shock. ``None`` (default)
        uses control-function IV when instrument columns are present, and no
        correction otherwise. ``"copula"`` adds a Gaussian-copula control from
        the price residual; it is identified only when that residual is not
        normal. ``"lewbel"`` builds instruments from a heteroskedastic price
        residual and estimates them with the IV control function. It is not
        identified when that residual is homoskedastic, or when every exogenous
        regressor is constant. ``"latent"`` is experimental: it treats the
        unobserved driver of price as a discrete mixture and is not identified
        when a single normal fits that residual as well. ``"iv"`` requires
        instruments. ``compare_endogeneity`` runs the fast check for each
        method. On
        ``QuadraticLogDemandModel`` and ``SigmoidSaturationDemandModel``,
        setting this warns: a control function in a nonlinear curve needs a
        stronger assumption than in a linear one. ``predict`` and
        ``optimize_prices`` use the structural curve, with the control at 0.
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
        time_effects: TimeEffectsKind | None = None,
        mundlak: bool = False,
        endogeneity: str | None = None,
        promo: PromoSpec | None = None,
        varying_controls: Sequence[str] | None = None,
        model_config: dict[str, Any] | None = None,
        sampler_config: dict[str, Any] | None = None,
    ) -> None:
        self.model_config = self.default_model_config | (model_config or {})
        self.sampler_config = self.default_sampler_config | (sampler_config or {})
        self.idata: az.InferenceData | None = None
        self.panel_columns = panel_columns or PanelColumns()
        self.cross_elasticity = cross_elasticity
        self.trend = trend
        self.seasonality = normalize_seasonality(seasonality)
        self.time_effects = time_effects
        self.mundlak = bool(mundlak)
        self.endogeneity = parse_endogeneity(endogeneity)
        self.promo = promo
        self.varying_controls = (
            tuple(varying_controls) if varying_controls else None
        )
        self.varying_control_idx_: tuple[int, ...] = ()
        self.pressure_group_codes_: np.ndarray | None = None
        self._fit_extra_: ExtraFeatures | None = None
        self.promo_type_levels_: tuple[str, ...] | None = None
        self.sku_levels_: pd.Index | None = None
        self.control_names_: tuple[str, ...] = ()
        self.iv_names_: tuple[str, ...] = ()
        self.endogeneity_method_: EndogeneityMethod | None = None
        self.log_price_midpoint_sku_: np.ndarray | None = None
        self.sku_mean_log_price_: np.ndarray | None = None
        self.period_levels_: np.ndarray | None = None
        self.region_levels_: np.ndarray | None = None
        self.data: pd.DataFrame | None = None
        self.cross_pairs_: CrossPairIndex | None = None
        self.t0_: pd.Timestamp | None = None
        self.seasonality_: tuple[str, ...] | None = None
        self.no_discount_variation_skus_: list[Any] = []
        self.no_regular_price_variation_skus_: list[Any] = []
        self.fit_kwargs_: dict[str, Any] = {}
        self._validate_configuration()
        if self.endogeneity is not None:
            self._warn_nonlinear_endogeneity(stacklevel=2)
        if self.endogeneity == "latent":
            self._warn_experimental_latent(stacklevel=2)

    def clone(self, **panel_overrides: Any) -> Self:
        """Unfitted copy with the same configuration.

        ``panel_overrides`` replace fields of ``panel_columns``. Once fitted,
        ``control_columns`` is pinned to ``control_names_`` unless overridden,
        so auto-detection can't change the controls between fits.
        """
        if self.sku_levels_ is not None:
            panel_overrides.setdefault("control_columns", self.control_names_)
        return type(self)(
            panel_columns=dataclasses.replace(self.panel_columns, **panel_overrides),
            cross_elasticity=self.cross_elasticity,
            trend=self.trend,
            seasonality=self.seasonality,
            time_effects=self.time_effects,
            mundlak=self.mundlak,
            endogeneity=self.endogeneity,
            promo=self.promo,
            varying_controls=self.varying_controls,
            model_config=self.model_config,
            sampler_config=self.sampler_config,
        )

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
        log_discount_index: np.ndarray | None = None,
        lambda_obs: np.ndarray | None = None,
        extra: ExtraFeatures | None = None,
    ) -> np.ndarray:
        """Mean log-quantity. ``log_price`` is log regular price when discounts are used."""
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
        if self.time_effects not in (None, "period", "region_period"):
            raise ValueError(
                "time_effects must be None, 'period', or 'region_period'; "
                f"got {self.time_effects!r}"
            )
        if self.time_effects is not None and (
            self.trend is not None or self.seasonality is not None
        ):
            raise ValueError(
                "time_effects cannot be combined with trend or seasonality; "
                "a free period effect absorbs both"
            )
        if self.time_effects == "region_period" and self.panel_columns.region_col is None:
            raise ValueError(
                "time_effects='region_period' requires panel_columns.region_col"
            )
        if self.promo is not None and self.panel_columns.discount_col is None:
            raise ValueError("promo requires panel_columns.discount_col")
        if self.promo is not None and self.promo.pressure_level is not None:
            groups = self.panel_columns.group_columns or ()
            if self.promo.pressure_level >= len(groups):
                raise ValueError(
                    "PromoSpec.pressure_level indexes group_columns; got "
                    f"{self.promo.pressure_level} with group_columns={groups!r}"
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

    @property
    def id(self) -> str:
        hasher = hashlib.sha256()
        config_json = json.dumps(
            self._serializable_model_config,
            sort_keys=True,
            default=_json_default,
        )
        hasher.update(config_json.encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        return hasher.hexdigest()[:16]

    def create_idata_attrs(self) -> dict[str, str]:
        cols = self.panel_columns
        ce = self.cross_elasticity
        return {
            "id": self.id,
            "model_type": self._model_type,
            "version": self.version,
            "sampler_config": json.dumps(self.sampler_config, default=_json_default),
            "model_config": json.dumps(
                self._serializable_model_config, default=_json_default
            ),
            "panel_columns": json.dumps(
                {
                    "sku_col": cols.sku_col,
                    "price_col": cols.price_col,
                    "discount_col": cols.discount_col,
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
            "time_effects": json.dumps(self.time_effects),
            "mundlak": json.dumps(self.mundlak),
            "endogeneity": json.dumps(self.endogeneity),
            "promo": json.dumps(None if self.promo is None else self.promo.to_json()),
            "varying_controls": json.dumps(self.varying_controls),
        }

    def set_idata_attrs(
        self, idata: az.InferenceData | None = None
    ) -> az.InferenceData:
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs = self.create_idata_attrs()
        return idata

    def save(self, fname: str, **kwargs: Any) -> None:
        if self.idata is None or "posterior" not in self.idata:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")
        self.idata.to_netcdf(str(Path(fname)), **kwargs)

    @classmethod
    def _model_config_formatting(cls, model_config: dict) -> dict:
        for value in model_config.values():
            if not isinstance(value, dict):
                continue
            for sub_key, sub_value in value.items():
                if not isinstance(sub_value, list):
                    continue
                if sub_key == "dims":
                    value[sub_key] = tuple(sub_value)
                else:
                    value[sub_key] = np.array(sub_value)
        return model_config

    @classmethod
    def attrs_to_init_kwargs(cls, attrs) -> dict[str, Any]:
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
            discount_col=pc.get("discount_col"),
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
        time_effects = (
            json.loads(attrs["time_effects"]) if "time_effects" in attrs else None
        )
        mundlak = json.loads(attrs["mundlak"]) if "mundlak" in attrs else False
        endogeneity = (
            json.loads(attrs["endogeneity"]) if "endogeneity" in attrs else None
        )
        promo = PromoSpec.from_json(
            json.loads(attrs["promo"]) if "promo" in attrs else None
        )
        varying_raw = (
            json.loads(attrs["varying_controls"])
            if "varying_controls" in attrs
            else None
        )
        return {
            "promo": promo,
            "varying_controls": varying_raw,
            "model_config": cls._model_config_formatting(
                json.loads(attrs["model_config"])
            ),
            "sampler_config": json.loads(attrs["sampler_config"]),
            "panel_columns": panel_columns,
            "cross_elasticity": cross_elasticity,
            "trend": trend,
            "seasonality": seasonality,
            "time_effects": time_effects,
            "mundlak": mundlak,
            "endogeneity": endogeneity,
        }

    @classmethod
    def load(cls, fname: str, check: bool = True) -> Self:
        idata = read_netcdf(fname)
        try:
            return cls.load_from_idata(idata, check=check)
        except DifferentModelError as exc:
            raise DifferentModelError(
                f"The file {fname!r} does not contain InferenceData for "
                f"{cls._model_type!r}"
            ) from exc

    @classmethod
    def load_from_idata(
        cls, idata: az.InferenceData, check: bool = True
    ) -> Self:
        model = cls(**cls.attrs_to_init_kwargs(idata.attrs))
        model.idata = idata
        model.build_from_idata(idata)
        if not check:
            return model
        if model.version != idata.attrs["version"]:
            raise DifferentModelError(
                f"The model version ({idata.attrs['version']}) in the "
                f"InferenceData does not match ({model.version})."
            )
        if model.id != idata.attrs["id"]:
            raise DifferentModelError(
                "The model id in the InferenceData does not match the model id."
            )
        return model

    def _add_fit_data_group(self, data: pd.DataFrame) -> None:
        fit_data = data.to_xarray()
        if self.idata is None:
            raise RuntimeError("No idata available to attach fit_data.")
        add_dataset_group(self.idata, "fit_data", fit_data)

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

    def _endogeneity_term(self, data: PricePanelData) -> Any:
        """Demand-mean term from the active endogeneity method, or 0."""
        method = self.endogeneity_method_
        if method is None:
            return 0.0
        return method.pymc_term(
            self.model_config,
            data,
            trend=self.trend,
            t0=self.t0_,
            seasonality_components=self.seasonality_,
            time_effects=self.time_effects,
            model=self,
        )

    def _warn_nonlinear_endogeneity(self, *, stacklevel: int) -> None:
        """Control functions in a nonlinear curve need a stronger assumption."""
        if self.model_name not in {"quadratic", "sigmoid"}:
            return
        warnings.warn(
            "A control function in a nonlinear demand curve needs a stronger "
            "assumption than in a linear one: the demand shock has to enter in "
            "a way that stays valid along the whole curve. "
            f"{type(self).__name__} with endogeneity set relies on that.",
            DiagnosticWarning,
            stacklevel=stacklevel,
        )

    def _warn_experimental_latent(self, *, stacklevel: int) -> None:
        """Latent IV is experimental, whichever curve it is attached to."""
        warnings.warn(EXPERIMENTAL_WARNING, DiagnosticWarning, stacklevel=stacklevel)

    def _resolve_endogeneity(
        self, name: str | None, n_iv: int
    ) -> EndogeneityMethod | None:
        if name is None:
            return get_endogeneity_method("iv") if n_iv else None
        if name == "iv" and n_iv == 0:
            raise ValueError(
                "endogeneity='iv' requires instruments. Set "
                "PanelColumns.iv_columns or add iv_* columns."
            )
        return get_endogeneity_method(name)

    def endogeneity_check(
        self,
        data: pd.DataFrame,
        *,
        method: str = "copula",
    ) -> dict[str, Any]:
        """Fast copula, Lewbel, latent IV, or IV check: assumptions and a point estimate, no MCMC.

        Builds the model with ``method``. When the verdict is ``not identified``
        the result has no ``estimate``. A ``weak`` verdict still includes one.
        """
        if self.endogeneity is None:
            chosen = parse_endogeneity(method)
            if chosen is not None:
                self._warn_nonlinear_endogeneity(stacklevel=2)
            if chosen == "latent":
                self._warn_experimental_latent(stacklevel=2)
        self.build_model(data, endogeneity=method)
        impl = self.endogeneity_method_
        if impl is None:
            raise ValueError(
                "No endogeneity method to check. Pass method='copula', 'lewbel', "
                "'latent', or 'iv'."
            )
        report = impl.check_assumptions(self)
        result: dict[str, Any] = {
            "method": impl.name,
            "verdict": report.verdict(),
            "assumptions": report,
        }
        if report.verdict() != "not identified":
            result["estimate"] = impl.quick_estimate(self, report)
        return result

    def compare_endogeneity(
        self,
        data: pd.DataFrame,
        *,
        methods: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Fast comparison of endogeneity corrections. No MCMC.

        One row per method, plus an ``ols`` row for the uncorrected regression.
        ``methods`` defaults to copula, latent IV, and Lewbel, and includes IV
        when the frame has instruments. At least two methods are required. IV
        with no instruments is a ``not identified`` row.

        The call does not change this model. Fit the method you keep. If that
        fit's correction interval includes 0, there is no evidence of
        endogeneity: use the uncorrected model, because the correction adds
        noise.
        """
        from pypricing.endogeneity.compare import comparison_frame

        return comparison_frame(self, data, methods=methods)

    def build_model(
        self, data: pd.DataFrame, *, endogeneity: str | None = None
    ) -> None:
        self._validate_configuration()
        self._validate_time_effect_cells(data)
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

        self._warn_unidentified_prices(panel)

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
        means = np.empty(panel.n_skus, dtype=np.float64)
        for sku_idx in range(panel.n_skus):
            mask = panel.obs_sku_idx == sku_idx
            column = panel.log_regular_price[mask]
            mid[sku_idx] = float(np.median(column))
            means[sku_idx] = float(np.mean(column))
        self.log_price_midpoint_sku_ = mid
        self.sku_mean_log_price_ = means if self.mundlak else None
        self.period_levels_ = panel.period_levels
        self.region_levels_ = panel.region_levels

        chosen = (
            self.endogeneity if endogeneity is None else parse_endogeneity(endogeneity)
        )
        self.sku_levels_ = panel.sku_levels
        self.control_names_ = panel.control_names
        self.iv_names_ = panel.iv_names
        self.data = data.copy()
        self.endogeneity_method_ = self._resolve_endogeneity(chosen, panel.n_iv)
        self.promo_type_levels_ = None
        self.varying_control_idx_ = ()
        if self.varying_controls:
            missing = [c for c in self.varying_controls if c not in panel.control_names]
            if missing:
                raise ValueError(
                    f"varying_controls {missing} are not control columns of the model "
                    f"({list(panel.control_names)})"
                )
            self.varying_control_idx_ = tuple(
                panel.control_names.index(c) for c in self.varying_controls
            )
        self.pressure_group_codes_ = None
        if self.promo is not None and self.promo.pressure_level is not None:
            assert panel.hierarchy is not None
            self.pressure_group_codes_ = np.asarray(
                panel.hierarchy.sku_to_level_idx[:, self.promo.pressure_level],
                dtype=np.int64,
            )
        self._fit_extra_ = self._extra_features(
            data,
            obs_sku_idx=panel.obs_sku_idx,
            log_discount_index=panel.log_discount_index,
            X_control=panel.control_matrix,
        )
        if self._fit_extra_ is not None and self._fit_extra_.type_onehot is not None:
            self.promo_type_levels_ = self._fit_extra_.type_levels
        self.model = self._build_pymc_model(panel)

    def _extra_features(
        self,
        df: pd.DataFrame,
        *,
        obs_sku_idx: np.ndarray,
        log_discount_index: np.ndarray | None,
        X_control: np.ndarray | None,
    ) -> ExtraFeatures | None:
        """Row-aligned features for the extra terms, or ``None``."""
        if self.promo is None and not self.varying_control_idx_:
            return None
        return build_extra_features(
            spec=self.promo,
            log_discount_index=log_discount_index,
            n_obs=len(obs_sku_idx),
            df=df,
            type_levels=self.promo_type_levels_,
            panel_columns=self.panel_columns,
            X_control=X_control,
            varying_idx=self.varying_control_idx_,
            varying_names=self.varying_controls or (),
            row_group=(
                None
                if self.pressure_group_codes_ is None
                else self.pressure_group_codes_[obs_sku_idx]
            ),
        )

    def _extra_for_regular_price(
        self, X_control: np.ndarray | None
    ) -> ExtraFeatures | None:
        """Extras for rows at regular price (no promo): only varying controls."""
        if not self.varying_control_idx_ or X_control is None:
            return None
        return build_extra_features(
            spec=None,
            log_discount_index=None,
            n_obs=X_control.shape[0],
            X_control=X_control,
            varying_idx=self.varying_control_idx_,
            varying_names=self.varying_controls or (),
        )

    def _extra_term(self, data: PricePanelData) -> Any:
        """PyMC term for the extra promo / control features, or 0."""
        return extra_mu_term(self.model_config, data, self._fit_extra_)

    def _warn_unidentified_prices(self, panel: PricePanelData) -> None:
        regular = skus_without_variation(
            panel.log_regular_price, panel.obs_sku_idx, panel.sku_levels
        )
        if panel.log_discount_index is None:
            discount: list[Any] = []
        else:
            discount = skus_without_variation(
                panel.log_discount_index, panel.obs_sku_idx, panel.sku_levels
            )
        self.no_regular_price_variation_skus_ = regular
        self.no_discount_variation_skus_ = discount
        if discount:
            assert panel.log_discount_index is not None
            all_zero = True
            for i, sku in enumerate(panel.sku_levels):
                if sku not in discount:
                    continue
                column = panel.log_discount_index[panel.obs_sku_idx == i]
                if float(np.max(np.abs(column))) > 1e-8:
                    all_zero = False
                    break
            reason = (
                "discount = 0 on every row" if all_zero else "no discount variation"
            )
            warnings.warn(
                f"SKU(s) {discount} have {reason}; promo_elasticity_sku is "
                "prior-only for them.",
                stacklevel=3,
            )
        if regular:
            warnings.warn(
                f"SKU(s) {regular} have a regular price that never changes; "
                "elasticity_sku is prior-only for them. Use group_columns to pool "
                "price variation across SKUs.",
                stacklevel=3,
            )

    def build_from_idata(self, idata: az.InferenceData) -> None:
        if not hasattr(idata, "fit_data"):
            raise ValueError("Loaded idata is missing fit_data group.")
        data = idata.fit_data.to_dataframe().reset_index(drop=True)
        self.build_model(data)

    def fit(self, data: pd.DataFrame, **sample_kwargs: Any) -> az.InferenceData:
        self.build_model(data)
        if (
            self.endogeneity_method_ is not None
            and self.endogeneity_method_.name != "iv"
        ):
            self.endogeneity_method_.check_assumptions(self)
        self.fit_kwargs_ = dict(sample_kwargs)
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
                "promo_elasticity_sku",
                "curvature_sku",
                "sigma",
                "beta_control",
                "mu_trend",
                "trend_sku",
                "beta_season",
                "rho",
                "rho_copula",
                "rho_latent",
                "pi",
                "sigma_price",
                "alpha_price_sku",
                "beta_control_price",
                "lambda_period",
                "sigma_lambda_period",
                "lambda_region_period",
                "sigma_lambda_region_period",
                "gamma_mundlak",
            ]
            candidates.append("gamma_pair")
            var_names = [v for v in candidates if v in self.idata.posterior]
        return summary(self.idata, var_names=var_names, **summary_kwargs)

    def run_diagnostics(
        self,
        *,
        fit: bool = True,
        benchmarks: bool = True,
        sensitivity: bool = True,
        instruments: bool = True,
        falsification: bool = False,
        compare_ols: bool = False,
        hdi_prob: float = 0.94,
        floor_censoring_threshold: float | None = None,
    ) -> DiagnosticsReport:
        """Sampler health plus the selected model checks, as a printable report.

        Parameters
        ----------
        fit, benchmarks, sensitivity, instruments
            Toggle the fast checks (on by default). ``instruments`` is skipped
            when the model has none.
        falsification, compare_ols
            Toggle the checks that refit the model (off by default).
        hdi_prob
            Interval width for the fit coverage and falsification checks.
        floor_censoring_threshold
            Overrides ``panel_columns.floor_censoring_warn_threshold`` for this
            call only, without needing to refit. ``None`` (default) uses the
            value configured on the model.
        """
        from pypricing.diagnostics.report import run_diagnostics as _run_diagnostics

        return _run_diagnostics(
            self,
            self._sampler_diagnostics(floor_censoring_threshold),
            fit=fit,
            benchmarks=benchmarks,
            sensitivity=sensitivity,
            instruments=instruments,
            falsification=falsification,
            compare_ols=compare_ols,
            hdi_prob=hdi_prob,
        )

    def _sampler_diagnostics(
        self, floor_censoring_threshold: float | None = None
    ) -> dict[str, Any]:
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
        out["no_discount_variation_skus"] = list(self.no_discount_variation_skus_)
        out["no_regular_price_variation_skus"] = list(
            self.no_regular_price_variation_skus_
        )
        from pypricing.diagnostics.identification import iv_diagnostics

        out.update(iv_diagnostics(self))
        if self.time_effects is not None and self.data is not None:
            from pypricing.diagnostics.identification import (
                WITHIN_PERIOD_PRICE_SHARE_MIN,
                within_period_price_share,
            )

            share = within_period_price_share(self)
            out["within_period_price_share"] = share
            out["weak_within_period_price"] = bool(
                share < WITHIN_PERIOD_PRICE_SHARE_MIN
            )
            if out["weak_within_period_price"]:
                from pypricing.diagnostics import DiagnosticWarning

                warnings.warn(
                    f"Only {share:.0%} of log-price variance is within period "
                    f"(below {WITHIN_PERIOD_PRICE_SHARE_MIN:.0%}); elasticity "
                    "rests on little identifying variation.",
                    DiagnosticWarning,
                    stacklevel=3,
                )
        return out

    def graphviz(self):
        self._require_fitted()
        if not hasattr(self, "model"):
            raise RuntimeError(
                "graphviz() requires the in-memory PyMC model; refit after load()."
            )
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
        period_effect: Any = "neutral",
        hdi_prob: float = 0.9,
        random_seed: int | None = None,
        ax=None,
    ):
        from pypricing.plotting import plot_response_curve as _plot

        return _plot(
            self,
            sku=sku,
            price_grid=price_grid,
            controls=controls,
            at_period=at_period,
            period_effect=period_effect,
            hdi_prob=hdi_prob,
            random_seed=random_seed,
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
        period_effect: Any = "neutral",
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
            period_effect=period_effect,
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
        at_period: pd.Timestamp | str | None = None,
        period_effect: Any = "neutral",
        random_seed: int | None = None,
        include_revenue_comparison: bool = True,
        ax=None,
    ):
        from pypricing.plotting import plot_optimization_summary as _plot

        return _plot(
            self,
            opt_df,
            reference_prices=reference_prices,
            controls_df=controls_df,
            at_period=at_period,
            period_effect=period_effect,
            random_seed=random_seed,
            include_revenue_comparison=include_revenue_comparison,
            ax=ax,
        )

    def plot_posterior_predictive_calibration(
        self,
        *,
        df: pd.DataFrame | None = None,
        hdi_prob: float = 0.94,
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
            random_seed=random_seed,
            ax=ax,
        )

    def _build_prediction_features(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Log regular price, SKU index, controls, and log discount index.

        ``price`` is the price actually paid. When the model was fit with
        ``discount_col``, that column is required.
        """
        self._require_fitted()
        assert self.sku_levels_ is not None
        if self.panel_columns.sku_col not in df.columns:
            raise ValueError(f"Missing required column: {self.panel_columns.sku_col!r}")
        if self.panel_columns.price_col not in df.columns:
            raise ValueError(
                f"Missing required column: {self.panel_columns.price_col!r}"
            )
        discount_col = self.panel_columns.discount_col
        if discount_col is not None and discount_col not in df.columns:
            raise ValueError(
                f"Model was fit with discount column {discount_col!r}; "
                "prediction data must include it. `price` is the price actually paid."
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
        discount = None
        if discount_col is not None:
            discount = read_discount_fraction(
                df,
                discount_col=discount_col,
                sku_col=self.panel_columns.sku_col,
                price_col=self.panel_columns.price_col,
            )
        _log_paid, log_regular, log_discount_index = decompose_log_price(price, discount)

        if self.control_names_:
            missing = [c for c in self.control_names_ if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing control columns in prediction data: {missing}"
                )
            X = df.loc[:, list(self.control_names_)].to_numpy(dtype=np.float64)
            if np.isnan(X).any():
                raise ValueError("NaN in control columns are not allowed")
            return log_regular, obs_sku_idx, X, log_discount_index
        return log_regular, obs_sku_idx, None, log_discount_index

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

    def _validate_time_effect_cells(self, data: pd.DataFrame) -> None:
        if self.time_effects is None:
            return
        period_col = self.period_col
        if period_col not in data.columns:
            raise ValueError(f"time_effects requires column {period_col!r}")
        keys = [period_col]
        if self.time_effects == "region_period":
            region_col = self.region_col
            if region_col is None or region_col not in data.columns:
                raise ValueError(
                    "time_effects='region_period' requires column "
                    f"{region_col!r}"
                )
            keys.append(region_col)
        counts = data.groupby(keys, sort=False)[self.sku_col].nunique()
        thin = counts[counts < 2]
        if len(thin):
            raise ValueError(
                "time_effects needs at least 2 SKUs in every period cell so "
                "the period effect is separate from price; "
                f"{len(thin)} cell(s) have fewer"
            )

    def _pymc_coords(self, data: PricePanelData) -> dict[str, Any]:
        coords = data.coords()
        if self.time_effects is None:
            coords.pop("period", None)
            coords.pop("region", None)
        elif self.time_effects == "period":
            coords.pop("region", None)
        return coords

    def _mundlak_mean_shift(self) -> Any:
        """``gamma_mundlak * centered mean log price``, or ``None`` when off.

        Call inside an active ``pm.Model``. The shift is the prior mean of
        ``alpha_sku`` only.
        """
        if not self.mundlak:
            return None
        if self.sku_mean_log_price_ is None:
            raise RuntimeError("Missing sku_mean_log_price_; fit the model first.")
        centered = np.asarray(self.sku_mean_log_price_, dtype=np.float64)
        centered = centered - float(centered.mean())
        gamma = resolve_prior(
            model_config=self.model_config,
            param_name="gamma_mundlak",
            default_dist=pm.Normal,
            default_kwargs={"mu": 0.0, "sigma": 2.0},
        )
        return gamma * pt.constant(centered)

    def _period_effect_draws(
        self,
        n_obs: int,
        *,
        periods: np.ndarray | None = None,
        regions: np.ndarray | None = None,
        period_effect: Any = "neutral",
        random_seed: int | None = None,
    ) -> np.ndarray | None:
        """Period-effect contribution, shape ``(chain, draw, n_obs)``.

        Training periods use the fitted λ. Other periods follow ``period_effect``:
        ``"neutral"`` is 0, ``"draw"`` is ``N(0, sigma_lambda)`` per posterior
        draw and per distinct future period, and any other value reuses that
        training period's λ.
        """
        if self.time_effects is None:
            return None
        self._require_fitted()
        assert self.idata is not None and self.period_levels_ is not None
        post = self.idata.posterior
        lam = np.asarray(post["lambda_period"].values, dtype=np.float64)
        n_chain, n_draw, _n_period = lam.shape
        if periods is None:
            codes = np.full(n_obs, -1, dtype=np.int64)
        else:
            codes = _label_codes(self.period_levels_, periods)
        out = np.zeros((n_chain, n_draw, n_obs), dtype=np.float64)
        known = codes >= 0
        if known.any():
            out[:, :, known] = lam[:, :, codes[known]]

        groups = _unknown_row_groups(codes, periods)
        rng = np.random.default_rng(random_seed)
        if groups and period_effect == "draw":
            sigma = np.asarray(post["sigma_lambda_period"].values, dtype=np.float64)
            for rows in groups:
                shock = rng.normal(size=(n_chain, n_draw)) * sigma
                out[:, :, rows] = shock[:, :, None]
        elif groups and period_effect != "neutral":
            code = _one_label_code(self.period_levels_, period_effect)
            for rows in groups:
                out[:, :, rows] = lam[:, :, code][:, :, None]

        if self.time_effects != "region_period":
            return out
        if regions is None:
            raise ValueError(
                "time_effects='region_period' requires a region column "
                "on prediction data"
            )
        if self.region_levels_ is None or "lambda_region_period" not in post:
            raise RuntimeError("Fitted model is missing region-period effects.")
        region_codes = _label_codes(self.region_levels_, regions)
        if (region_codes < 0).any():
            bad = pd.unique(np.asarray(regions)[region_codes < 0]).tolist()
            raise ValueError(f"Unknown region values in prediction data: {bad}")
        lam_region = np.asarray(post["lambda_region_period"].values, dtype=np.float64)
        if known.any():
            out[:, :, known] = out[:, :, known] + lam_region[
                :, :, region_codes[known], codes[known]
            ]
        if not groups:
            return out
        if period_effect == "neutral":
            return out
        if period_effect == "draw":
            sigma_region = np.asarray(
                post["sigma_lambda_region_period"].values, dtype=np.float64
            )
            for rows in groups:
                for region_code in np.unique(region_codes[rows]):
                    mask = region_codes[rows] == region_code
                    shock = rng.normal(size=(n_chain, n_draw)) * sigma_region
                    out[:, :, rows[mask]] = out[:, :, rows[mask]] + shock[:, :, None]
            return out
        code = _one_label_code(self.period_levels_, period_effect)
        unknown = np.flatnonzero(codes < 0)
        out[:, :, unknown] = out[:, :, unknown] + lam_region[
            :, :, region_codes[unknown], code
        ]
        return out

    def _prediction_period_values(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.time_effects is None:
            return None, None
        periods = (
            df[self.period_col].to_numpy()
            if self.period_col in df.columns
            else None
        )
        if self.time_effects != "region_period":
            return periods, None
        region_col = self.region_col
        if region_col is None or region_col not in df.columns:
            raise ValueError(
                "time_effects='region_period' requires a region column "
                f"{region_col!r} on prediction data"
            )
        return periods, df[region_col].to_numpy()

    def sample_posterior_predictive(
        self,
        df: pd.DataFrame,
        *,
        random_seed: int | None = None,
        period_effect: Any = "neutral",
    ) -> xr.Dataset:
        self._require_fitted()
        assert self.idata is not None

        log_price, obs_sku_idx, X_control, log_discount_index = (
            self._build_prediction_features(df)
        )
        X_cross = self._cross_matrix_for_frame(df, obs_sku_idx)
        t_years = self._t_years_for_frame(df)
        X_season = self._season_features_for_frame(df)
        periods, regions = self._prediction_period_values(df)
        lambda_obs = self._period_effect_draws(
            log_price.shape[0],
            periods=periods,
            regions=regions,
            period_effect=period_effect,
            random_seed=random_seed,
        )
        extra = self._extra_features(
            df,
            obs_sku_idx=obs_sku_idx,
            log_discount_index=log_discount_index,
            X_control=X_control,
        )
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
            log_discount_index=log_discount_index,
            lambda_obs=lambda_obs,
            extra=extra,
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
        period_effect: Any = "neutral",
    ) -> pd.DataFrame:
        """Predict quantity. ``period_effect`` applies only to periods that were
        not in the training sample: ``"neutral"`` (default) uses λ = 0,
        ``"draw"`` draws λ ~ N(0, sigma_lambda) per posterior draw, and a
        training period label reuses that period's fitted λ.
        """
        ds = self.sample_posterior_predictive(
            df, random_seed=random_seed, period_effect=period_effect
        )
        q = ds["quantity"]
        mean = q.mean(dim=("chain", "draw")).to_numpy()
        lower, upper = hdi_bounds(q, hdi_prob)

        out = df.copy()
        out["quantity_mean"] = mean
        out["log_quantity_mean"] = (
            ds["log_quantity"].mean(dim=("chain", "draw")).to_numpy()
        )
        out["quantity_hdi_lower"] = np.asarray(lower)
        out["quantity_hdi_upper"] = np.asarray(upper)
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
        # Keep the original period values. np.sort(...).tolist() turns datetimes
        # into integer nanoseconds, and isin then matches nothing.
        try:
            periods_sorted = pd.Index(periods).sort_values()
        except TypeError:
            periods_sorted = pd.Index(sorted(periods.tolist()))
        n_test_periods = max(1, int(np.ceil(test_size * len(periods_sorted))))
        is_test = df[period_col].isin(periods_sorted[-n_test_periods:])
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
        from pypricing.diagnostics.fit import fit_metrics

        preds = self.predict(test_df, hdi_prob=hdi_prob, random_seed=random_seed)
        overall = fit_metrics(
            preds,
            quantity_col=self.quantity_col,
            quantity_floor=self.quantity_floor,
        ).loc["overall"]
        return {
            "rmse": float(overall["rmse"]),
            "rmse_log": float(overall["rmse_log"]),
            "bias_log": float(overall["bias_log"]),
            "hdi_coverage": float(overall["hdi_coverage"]),
            "test_predictions": preds,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }

    def check_fit(
        self,
        df: pd.DataFrame | None = None,
        *,
        by: str | None = None,
        hdi_prob: float = 0.94,
        min_rows: int = 10,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        from pypricing.diagnostics.fit import check_fit as _check_fit

        return _check_fit(
            self,
            df,
            by=by,
            hdi_prob=hdi_prob,
            min_rows=min_rows,
            random_seed=random_seed,
        )

    def check_benchmarks(
        self, *, extreme_threshold: float = 0.99
    ) -> dict[str, pd.DataFrame | None]:
        from pypricing.diagnostics.benchmarks import (
            check_benchmarks as _check_benchmarks,
        )

        return _check_benchmarks(self, extreme_threshold=extreme_threshold)

    def check_falsification(
        self,
        df: pd.DataFrame | None = None,
        *,
        hdi_prob: float = 0.94,
        **sample_kwargs: Any,
    ) -> dict[str, Any]:
        from pypricing.diagnostics.falsification import (
            check_falsification as _check_falsification,
        )

        return _check_falsification(self, df, hdi_prob=hdi_prob, **sample_kwargs)

    def check_instruments(
        self,
        *,
        hdi_prob: float = 0.9,
        compare_ols: bool = True,
        **sample_kwargs: Any,
    ) -> dict[str, Any]:
        from pypricing.diagnostics.identification import (
            check_instruments as _check_instruments,
        )

        return _check_instruments(
            self, hdi_prob=hdi_prob, compare_ols=compare_ols, **sample_kwargs
        )

    def check_sensitivity(
        self, *, q: float = 1.0, alpha: float = 0.05
    ) -> pd.DataFrame:
        from pypricing.diagnostics.sensitivity import (
            check_sensitivity as _check_sensitivity,
        )

        return _check_sensitivity(self, q=q, alpha=alpha)

    def optimize_prices(
        self,
        price_bounds: dict[Any, tuple[float, float]] | pd.Series,
        *,
        controls_df: pd.DataFrame | None = None,
        at_period: pd.Timestamp | str | None = None,
        period_effect: Any = "neutral",
        random_seed: int | None = None,
        minimize_options: dict[str, Any] | None = None,
        objective: str = "revenue",
        unit_cost: dict[Any, float] | pd.Series | None = None,
    ) -> pd.DataFrame:
        from pypricing.optimizer import optimize_prices as _optimize_prices

        self._require_fitted()
        return _optimize_prices(
            model=self,
            price_bounds=price_bounds,
            controls_df=controls_df,
            at_period=at_period,
            period_effect=period_effect,
            random_seed=random_seed,
            minimize_options=minimize_options,
            objective=objective,
            unit_cost=unit_cost,
        )

    def elasticities(self, *, hdi_prob: float = 0.9) -> pd.DataFrame:
        """Posterior mean and HDI of own-price elasticity, one row per SKU label."""
        from pypricing.posterior import elasticities as _elasticities

        return _elasticities(self, hdi_prob=hdi_prob)

    def quantity_multiplier_summary(
        self,
        *,
        price_multiplier: float,
        hdi_prob: float = 0.9,
        return_draws: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, xr.DataArray]:
        """Quantity change from scaling regular price by ``price_multiplier``.

        Log-log only. The discount fraction is held fixed; ``elasticity_sku``
        is the regular-price elasticity.
        """
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
