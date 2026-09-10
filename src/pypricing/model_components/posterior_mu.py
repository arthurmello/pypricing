import numpy as np
import xarray as xr


def add_trend(
    mu: np.ndarray,
    posterior: xr.Dataset,
    t_years: np.ndarray | None,
    obs_sku_idx: np.ndarray,
) -> np.ndarray:
    if t_years is None:
        return mu
    t = t_years[None, None, :]
    if "trend_sku" in posterior:
        trend = posterior["trend_sku"].values
        return mu + trend[:, :, obs_sku_idx] * t
    if "mu_trend" in posterior:
        mu_trend = posterior["mu_trend"].values
        return mu + mu_trend[:, :, None] * t
    return mu


def add_controls_and_cross(
    mu: np.ndarray,
    posterior: xr.Dataset,
    X_control: np.ndarray | None,
    X_cross: np.ndarray | None,
    *,
    t_years: np.ndarray | None = None,
    obs_sku_idx: np.ndarray | None = None,
) -> np.ndarray:
    if X_control is not None:
        beta = posterior["beta_control"].values
        mu = mu + np.einsum("cdk,ok->cdo", beta, X_control, optimize=True)
    if X_cross is not None and "gamma_pair" in posterior:
        gamma = posterior["gamma_pair"].values
        mu = mu + np.einsum("cdp,op->cdo", gamma, X_cross, optimize=True)
    if t_years is not None:
        if obs_sku_idx is None:
            raise ValueError("obs_sku_idx is required when t_years is provided")
        mu = add_trend(mu, posterior, t_years, obs_sku_idx)
    return mu
