from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from pypricing.data import PricePanelData
from scipy.special import softplus

from pypricing.models.basic import DemandModel
from pypricing.model_components.priors import resolve_prior
from pypricing.model_components.sku_effects import get_sku_effect
from pypricing.model_components.global_terms import get_sigma, get_controls_term
from pypricing.model_components.posterior_mu import add_controls_and_cross
from pypricing.model_components.time_terms import get_season_term, get_trend_term


class SigmoidSaturationDemandModel(DemandModel):
    """Saturating demand in level price via softplus.

    Mean log-quantity:

    ``mu = alpha_sku - softplus(b_sku * (P - P_center_sku)) + controls + cross + trend + season``

    with ``b_sku = -2 * elasticity_sku / P_center_sku`` so elasticity at the center
    equals ``elasticity_sku``. ``log_price_center_sku`` is learned; its prior mean is
    the per-SKU training median log-price. Prefer when response flattens at extreme
    prices.
    """

    @property
    def model_name(self) -> str:
        return "sigmoid"

    def _build_pymc_model(self, data: PricePanelData) -> pm.Model:
        if self.log_price_midpoint_sku_ is None:
            raise RuntimeError("Missing log_price_midpoint_sku_; fit the model first.")

        coords = data.coords()
        obs_sku = pt.constant(data.obs_sku_idx)
        log_p = pt.constant(data.log_price)
        price = pt.exp(log_p)
        log_p_mid_sku = pt.constant(
            np.asarray(self.log_price_midpoint_sku_, dtype=float)
        )

        with pm.Model(coords=coords) as model:
            # Learn the per-SKU center price; anchor it with a prior centered at the
            # empirical (data-derived) midpoint price.
            log_price_center_sku = resolve_prior(
                model_config=self.model_config,
                param_name="log_price_center_sku",
                default_dist=pm.Normal,
                default_kwargs={"mu": log_p_mid_sku, "sigma": 0.5},
                dims="sku",
            )
            price_center_sku = pt.exp(log_price_center_sku)

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
            season_term = get_season_term(
                self.model_config, data, components=self.seasonality_
            )

            b_sku = -2.0 * elasticity_sku / price_center_sku
            z = b_sku[obs_sku] * (price - price_center_sku[obs_sku])

            mu = (
                alpha_sku[obs_sku]
                - pt.softplus(z)
                + controls_term
                + self._cross_term(data)
                + trend_term
                + season_term
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
        X_season: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.log_price_midpoint_sku_ is None:
            raise RuntimeError("Missing log_price_midpoint_sku_; fit the model first.")

        alpha = posterior["alpha_sku"].values
        elasticity = posterior["elasticity_sku"].values
        alpha_obs = alpha[:, :, obs_sku_idx]
        elasticity_obs = elasticity[:, :, obs_sku_idx]
        p_obs = np.exp(log_price)[None, None, :]
        if "log_price_center_sku" in posterior:
            log_price_center_sku = posterior["log_price_center_sku"].values
            p_center_obs = np.exp(log_price_center_sku[:, :, obs_sku_idx])
        else:
            # Backward compatibility for previously-saved models.
            p_mid_obs = np.exp(
                np.asarray(self.log_price_midpoint_sku_, dtype=np.float64)
            )[obs_sku_idx]
            p_center_obs = p_mid_obs[None, None, :]

        b_obs = -2.0 * elasticity_obs / p_center_obs
        z = b_obs * (p_obs - p_center_obs)
        mu = alpha_obs - softplus(z)
        return add_controls_and_cross(
            mu=mu,
            posterior=posterior,
            X_control=X_control,
            X_cross=X_cross,
            t_years=t_years,
            obs_sku_idx=obs_sku_idx,
            X_season=X_season,
        )
