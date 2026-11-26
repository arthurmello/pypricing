from importlib.metadata import PackageNotFoundError, version

from pypricing.data import (
    CrossPairIndex,
    HierarchyIndex,
    PanelColumns,
    PricePanelData,
    apply_discount,
    floor_censored_fraction,
)
from pypricing.events import add_event_columns, event_indicators
from pypricing.model_components.cross_elasticity import CrossElasticitySpec
from pypricing.model_components.promo_extras import PromoSpec
from pypricing.models import (
    DemandModel,
    LogLogDemandModel,
    QuadraticLogDemandModel,
    SigmoidSaturationDemandModel,
)
from pypricing.optimizer import optimize_prices
from pypricing.promo import (
    PromoConstraints,
    compare_scenarios,
    optimize_promotions,
    plan_frame,
    promo_backtest,
    promo_options,
)
from pypricing.synthetic_data import generate_mock_data
from pypricing.synthetic_promo import generate_promo_panel

try:
    __version__ = version("pypricing")
except PackageNotFoundError:  # pragma: no cover - editable/src path without install
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "CrossElasticitySpec",
    "CrossPairIndex",
    "DemandModel",
    "HierarchyIndex",
    "PanelColumns",
    "PricePanelData",
    "PromoConstraints",
    "PromoSpec",
    "LogLogDemandModel",
    "QuadraticLogDemandModel",
    "SigmoidSaturationDemandModel",
    "add_event_columns",
    "apply_discount",
    "event_indicators",
    "floor_censored_fraction",
    "generate_mock_data",
    "generate_promo_panel",
    "optimize_prices",
    "optimize_promotions",
    "plan_frame",
    "promo_options",
    "promo_backtest",
    "compare_scenarios",
]
