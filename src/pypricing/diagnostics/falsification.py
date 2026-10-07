from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np
import pandas as pd

from pypricing.data import PanelColumns

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel

LEAD_COL = "log_price_lead"


def add_lead_log_price(
    df: pd.DataFrame, *, panel_columns: PanelColumns
) -> pd.DataFrame:
    """Add next period's log price, centered on each SKU's mean log price.

    The lead is taken within SKU (and region, when set). Rows without a next
    period -- the last period, or before a gap in the SKU's history -- are
    dropped, so the result only contains rows with a valid lead.
    """
    cols = panel_columns
    if cols.period_col not in df.columns:
        raise ValueError(f"Missing period column: {cols.period_col!r}")
    if LEAD_COL in df.columns:
        raise ValueError(f"Column {LEAD_COL!r} already exists in the frame.")

    keys = [cols.sku_col]
    if cols.region_col is not None and cols.region_col in df.columns:
        keys.append(cols.region_col)

    periods = np.sort(pd.unique(df[cols.period_col]))
    period_pos = pd.Series(np.arange(len(periods)), index=periods)

    out = df.copy()
    out["_pos"] = out[cols.period_col].map(period_pos).to_numpy()
    out["_log_p"] = np.log(out[cols.price_col].to_numpy(dtype=float))
    out = out.sort_values([*keys, "_pos"])

    grouped = out.groupby(keys, observed=True, sort=False)
    next_pos = grouped["_pos"].shift(-1)
    next_log_p = grouped["_log_p"].shift(-1)
    sku_mean = out.groupby(cols.sku_col, observed=True)["_log_p"].transform("mean")

    out[LEAD_COL] = (next_log_p - sku_mean).where(next_pos == out["_pos"] + 1)
    out = out.dropna(subset=[LEAD_COL]).drop(columns=["_pos", "_log_p"])
    return out.sort_index()


def _clone(model: DemandModel, control_columns: tuple[str, ...]) -> DemandModel:
    return type(model)(
        panel_columns=dataclasses.replace(
            model.panel_columns, control_columns=control_columns
        ),
        cross_elasticity=model.cross_elasticity,
        trend=model.trend,
        seasonality=model.seasonality,
        model_config=model.model_config,
        sampler_config=model.sampler_config,
    )


def _elasticity_means(model: DemandModel) -> pd.Series:
    draws = model.idata.posterior["elasticity_sku"]
    return pd.Series(
        draws.mean(dim=("chain", "draw")).to_numpy(),
        index=pd.Index(model.sku_levels_, name=model.sku_col),
    )


def check_falsification(
    model: DemandModel,
    df: pd.DataFrame | None = None,
    *,
    hdi_prob: float = 0.94,
    **sample_kwargs: Any,
) -> dict[str, Any]:
    """Placebo test: next period's price shouldn't explain today's demand.

    Refits two copies of ``model`` on the same rows (those with a valid next
    period): a baseline, and one with the centered lead log price as an extra
    control. Warns when the lead coefficient's HDI excludes zero.

    ``sample_kwargs`` default to those passed to the model's last ``fit()``;
    any given here override them.

    A nonzero lead effect suggests an unmeasured driver of both price and
    demand, but forward-looking shoppers (waiting for a promotion, stocking up
    before a price rise) produce one too -- treat it as a flag to investigate.
    """
    model._require_fitted()
    sample_kwargs = {**model.fit_kwargs_, **sample_kwargs}
    work = model.data if df is None else df
    trimmed = add_lead_log_price(work, panel_columns=model.panel_columns)

    baseline = _clone(model, model.control_names_)
    baseline.fit(trimmed, **sample_kwargs)
    with_lead = _clone(model, (*model.control_names_, LEAD_COL))
    with_lead.fit(trimmed, **sample_kwargs)

    lead_draws = np.asarray(
        with_lead.idata.posterior["beta_control"].values[..., -1]
    ).ravel()
    lo, hi = (float(x) for x in az.hdi(lead_draws, hdi_prob=hdi_prob))
    excludes_zero = not (lo <= 0.0 <= hi)
    if excludes_zero:
        warnings.warn(
            f"Next period's price predicts today's demand (lead coefficient "
            f"{hdi_prob:.0%} HDI [{lo:.3f}, {hi:.3f}] excludes 0). Check for "
            "unmeasured drivers of both price and demand, or forward-looking "
            "shoppers.",
            stacklevel=3,
        )

    elasticity = pd.DataFrame(
        {
            "baseline": _elasticity_means(baseline),
            "with_lead": _elasticity_means(with_lead),
        }
    )
    elasticity["shift"] = elasticity["with_lead"] - elasticity["baseline"]
    return {
        "lead_coef_mean": float(lead_draws.mean()),
        "lead_coef_hdi": (lo, hi),
        "hdi_excludes_zero": excludes_zero,
        "n": len(trimmed),
        "elasticity": elasticity,
    }
