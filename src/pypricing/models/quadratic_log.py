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


class QuadraticLogDemandModel(DemandModel):
    """Log-demand with quadratic curvature in log-price.

    Mean log-quantity:

    ``mu = alpha_sku + beta1_sku * log(P) + curvature_sku * log(P)^2 + ...``

    Reparameterized so ``elasticity_sku`` equals local elasticity at the per-SKU
    training median log-price (``log_price_midpoint_sku_``). Use when elasticity
    should change with price level (premium vs discount regimes).
    """

    @property
    def model_name(self) -> str:
        return "quadratic"

    def _build_pymc_model(self, data: PricePanelData) -> pm.Model:
        if self.log_price_midpoint_sku_ is None:
            raise RuntimeError("Missing log_price_midpoint_sku_; fit the model first.")

        coords = data.coords()
        obs_sku = pt.constant(data.obs_sku_idx)
        log_p = pt.constant(data.log_price)
        log_p_mid_sku = pt.constant(
            np.asarray(self.log_price_midpoint_sku_, dtype=float)
        )

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
            curvature_sku = get_sku_effect(
                "curvature",
                data,
                self.model_config,
                mu_default_mu=0.0,
                mu_default_sigma=0.2,
                level_scale_default_sigma=0.2,
                sku_scale_default_sigma=0.2,
            )
            sigma = get_sigma(self.model_config)

            controls_term = get_controls_term(self.model_config, data)
            trend_term = get_trend_term(
                self.model_config, data, trend=self.trend, t0=self.t0_
            )

            beta1_sku = elasticity_sku - 2.0 * curvature_sku * log_p_mid_sku

            mu = (
                alpha_sku[obs_sku]
                + beta1_sku[obs_sku] * log_p
                + curvature_sku[obs_sku] * (log_p**2)
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
        if self.log_price_midpoint_sku_ is None:
            raise RuntimeError("Missing log_price_midpoint_sku_; fit the model first.")
        if "curvature_sku" not in posterior:
            raise RuntimeError("Missing 'curvature_sku' posterior for quadratic model.")

        alpha = posterior["alpha_sku"].values
        elasticity = posterior["elasticity_sku"].values
        curvature = posterior["curvature_sku"].values

        alpha_obs = alpha[:, :, obs_sku_idx]
        elasticity_obs = elasticity[:, :, obs_sku_idx]
        curvature_obs = curvature[:, :, obs_sku_idx]
        log_price_obs = log_price[None, None, :]
        log_p_mid_obs = np.asarray(self.log_price_midpoint_sku_, dtype=np.float64)[
            obs_sku_idx
        ]

        beta1_obs = elasticity_obs - 2.0 * curvature_obs * log_p_mid_obs[None, None, :]
        mu = alpha_obs + beta1_obs * log_price_obs + curvature_obs * (log_price_obs**2)
        return add_controls_and_cross(
            mu=mu,
            posterior=posterior,
            X_control=X_control,
            X_cross=X_cross,
            t_years=t_years,
            obs_sku_idx=obs_sku_idx,
        )
