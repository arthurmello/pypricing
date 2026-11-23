"""Side-by-side fast checks for the endogeneity corrections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from pypricing.data import PricePanelData
from pypricing.endogeneity import ENDOGENEITY_METHODS, parse_endogeneity
from pypricing.endogeneity.base import AssumptionReport
from pypricing.endogeneity.design import exogenous_design, log_regular_price

if TYPE_CHECKING:
    from pypricing.models.basic import DemandModel

_NO_INSTRUMENTS = (
    "No instruments. Set PanelColumns.iv_columns or add iv_* columns."
)


def comparison_frame(
    model: DemandModel,
    data: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Verdict and pooled elasticity for each method, plus the uncorrected fit.

    The caller's model is left unchanged. IV with no instruments is a row,
    not an error.
    """
    names = _method_names(model, data, methods)
    rows: list[dict[str, object]] = []
    ols: float | None = None
    for name in names:
        if name == "iv" and _n_instruments(model, data) == 0:
            rows.append(
                {
                    "method": name,
                    "verdict": "not identified",
                    "elasticity": np.nan,
                    "note": _NO_INSTRUMENTS,
                }
            )
            continue
        probe = model.clone()
        result = probe.endogeneity_check(data, method=name)
        estimate = result.get("estimate")
        rows.append(
            {
                "method": name,
                "verdict": result["verdict"],
                "elasticity": np.nan if estimate is None else float(estimate.elasticity),
                "note": _note(result["assumptions"]),
            }
        )
        if ols is None:
            ols = _ols_elasticity(probe)
    if ols is None:
        raise RuntimeError("internal: comparison produced no uncorrected elasticity")
    rows.insert(
        0,
        {
            "method": "ols",
            "verdict": "uncorrected",
            "elasticity": ols,
            "note": "no correction",
        },
    )
    frame = pd.DataFrame(rows)
    return frame.set_index("method")


def _method_names(
    model: DemandModel,
    data: pd.DataFrame,
    methods: Sequence[str] | None,
) -> tuple[str, ...]:
    if methods is None:
        n_iv = _n_instruments(model, data)
        names = tuple(name for name in ENDOGENEITY_METHODS if name != "iv" or n_iv > 0)
    else:
        if isinstance(methods, str):
            raise TypeError("methods must be a sequence of method names, not a string")
        parsed: list[str] = []
        for name in methods:
            key = parse_endogeneity(name)
            if key is None:
                raise ValueError("methods must be endogeneity method names")
            parsed.append(key)
        names = tuple(parsed)
    if len(names) < 2:
        raise ValueError("compare_endogeneity needs at least two methods.")
    if len(set(names)) != len(names):
        raise ValueError("methods must not contain duplicates.")
    return names


def _n_instruments(model: DemandModel, data: pd.DataFrame) -> int:
    panel = PricePanelData.from_frame(
        data,
        panel_columns=model.panel_columns,
        parse_period=model._uses_calendar(),
    )
    return panel.n_iv


def _note(report: AssumptionReport) -> str:
    flagged = [item.detail for item in report.assumptions if item.status != "pass"]
    if flagged:
        return " ".join(flagged)
    return "identified"


def _ols_elasticity(model: DemandModel) -> float:
    log_price = log_regular_price(model)
    exogenous = exogenous_design(model)
    assert model.data is not None
    quantity = model.data[model.quantity_col].to_numpy(dtype=np.float64)
    log_quantity = np.log(np.maximum(quantity, model.quantity_floor))
    design = np.column_stack([exogenous, log_price])
    coef, _, _, _ = np.linalg.lstsq(design, log_quantity, rcond=None)
    return float(coef[-1])
