from __future__ import annotations


import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from pypricing.data import PricePanelData
from pypricing.models.basic import DemandModel
from pypricing.model_components.sku_effects import get_sku_effect
from pypricing.model_components.global_terms import get_sigma, get_controls_term
from pypricing.model_components.posterior_mu import add_controls_and_cross
from pypricing.model_components.time_terms import get_trend_term


class LogLogDemandModel(DemandModel):
    """Constant-elasticity log-log demand.

    Mean log-quantity:

    ``mu = alpha_sku + elasticity_sku * log(P) + controls + cross + trend``

    ``elasticity_sku`` is own-price elasticity (e.g. ``-1.5`` ≈ 1% price up → 1.5%
    quantity down in expectation). Best as a simple constant-elasticity baseline.
    Quantity-multiplier helpers apply only to this model.
    """

    @property
    def model_name(self) -> str:
        return "log_log"

    def _build_pymc_model(self, data: PricePanelData) -> pm.Model:
        coords = data.coords()
        obs_sku = pt.constant(data.obs_sku_idx)
        log_p = pt.constant(data.log_price)

        with pm.Model(coords=coords) as model:
            alpha_sku = get_sku_effect(
                "alpha",
                data,
                self.model_config,
                mu_default_mu=6.0,
                mu_default_sigma=2.0,
            )
            elasticity_sku = get_sku_effect(
                "elasticity",
                data,
                self.model_config,
                mu_default_mu=-1.0,
                mu_default_sigma=2.0,
            )

            sigma = get_sigma(self.model_config)

            controls_term = get_controls_term(self.model_config, data)
            trend_term = get_trend_term(
                self.model_config, data, trend=self.trend, t0=self.t0_
            )

            mu = (
                alpha_sku[obs_sku]
                + elasticity_sku[obs_sku] * log_p
                + controls_term
                + self._cross_term(data)
                + trend_term
            )
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data.log_quantity)
        return model

    def compute_mu_from_posterior(
        self,
        *,
        posterior: xr.Dataset,
        log_price: np.ndarray,
        obs_sku_idx: np.ndarray,
        X_control: np.ndarray | None,
        X_cross: np.ndarray | None = None,
        t_years: np.ndarray | None = None,
    ) -> np.ndarray:
        alpha = posterior["alpha_sku"].values
        elasticity = posterior["elasticity_sku"].values
        mu = (
            alpha[:, :, obs_sku_idx]
            + elasticity[:, :, obs_sku_idx] * log_price[None, None, :]
        )
        return add_controls_and_cross(
            mu=mu,
            posterior=posterior,
            X_control=X_control,
            X_cross=X_cross,
            t_years=t_years,
            obs_sku_idx=obs_sku_idx,
        )
