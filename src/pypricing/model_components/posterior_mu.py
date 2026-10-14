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


def add_season(
    mu: np.ndarray,
    posterior: xr.Dataset,
    X_season: np.ndarray | None,
) -> np.ndarray:
    if X_season is None or "beta_season" not in posterior:
        return mu
    if X_season.shape[1] == 0:
        return mu
    beta = posterior["beta_season"].values
    return mu + np.einsum("cdk,ok->cdo", beta, X_season, optimize=True)


def add_promo(
    mu: np.ndarray,
    posterior: xr.Dataset,
    log_discount_index: np.ndarray | None,
    obs_sku_idx: np.ndarray,
) -> np.ndarray:
    """Add ``promo_elasticity_sku * log(1 - d)``. No-op without a discount index."""
    if log_discount_index is None or "promo_elasticity_sku" not in posterior:
        return mu
    promo = posterior["promo_elasticity_sku"].values
    index = np.asarray(log_discount_index, dtype=np.float64)
    return mu + promo[:, :, obs_sku_idx] * index[None, None, :]


def add_controls_and_cross(
    mu: np.ndarray,
    posterior: xr.Dataset,
    X_control: np.ndarray | None,
    X_cross: np.ndarray | None,
    *,
    t_years: np.ndarray | None = None,
    obs_sku_idx: np.ndarray | None = None,
    X_season: np.ndarray | None = None,
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
    return add_season(mu, posterior, X_season)
