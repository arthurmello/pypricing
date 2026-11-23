"""Corrections for a price set from the same shock that moves demand."""

from pypricing.endogeneity.base import (
    Assumption,
    AssumptionReport,
    EndogeneityMethod,
    QuickEstimate,
)
from pypricing.endogeneity.copula import CopulaMethod
from pypricing.endogeneity.design import exogenous_design
from pypricing.endogeneity.iv import IVMethod
from pypricing.endogeneity.latent import LatentMethod
from pypricing.endogeneity.lewbel import LewbelMethod

ENDOGENEITY_METHODS = ("copula", "iv", "latent", "lewbel")

__all__ = [
    "Assumption",
    "AssumptionReport",
    "CopulaMethod",
    "EndogeneityMethod",
    "IVMethod",
    "LatentMethod",
    "LewbelMethod",
    "QuickEstimate",
    "exogenous_design",
    "get_endogeneity_method",
    "parse_endogeneity",
]


def parse_endogeneity(value: str | None) -> str | None:
    """``None``, ``"copula"``, ``"iv"``, ``"latent"``, or ``"lewbel"``."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("endogeneity must be a string or None")
    key = value.strip().lower()
    allowed = ", ".join(ENDOGENEITY_METHODS)
    if key not in ENDOGENEITY_METHODS:
        raise ValueError(
            f"endogeneity {value!r} is not supported; use one of: {allowed}"
        )
    return key


def get_endogeneity_method(name: str) -> EndogeneityMethod:
    """Fresh method instance for ``"copula"``, ``"iv"``, ``"latent"``, or ``"lewbel"``."""
    key = parse_endogeneity(name)
    if key == "copula":
        return CopulaMethod()
    if key == "iv":
        return IVMethod()
    if key == "latent":
        return LatentMethod()
    if key == "lewbel":
        return LewbelMethod()
    raise ValueError("endogeneity method name is required")
