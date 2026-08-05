from importlib.metadata import PackageNotFoundError, version

from pypricing.data import (
    CrossPairIndex,
    HierarchyIndex,
    PanelColumns,
    PricePanelData,
    floor_censored_fraction,
)
from pypricing.model_components.cross_elasticity import CrossElasticitySpec
from pypricing.models import (
    DemandModel,
    LogLogDemandModel,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
)
from pypricing.optimizer import optimize_prices
from pypricing.synthetic_data import generate_mock_data

try:
    __version__ = version("pypricing")
except PackageNotFoundError:  # pragma: no cover - editable/src path without install
    __version__ = "0.0.1"

__all__ = [
    "__version__",
    "CrossElasticitySpec",
    "CrossPairIndex",
    "DemandModel",
    "HierarchyIndex",
    "PanelColumns",
    "PricePanelData",
    "LogLogDemandModel",
    "QuadraticLogDemandModel",
    "SigmoidSaturationDemandModel",
    "floor_censored_fraction",
    "generate_mock_data",
    "optimize_prices",
]
