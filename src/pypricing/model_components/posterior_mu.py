import numpy as np
import xarray as xr


def add_controls_and_cross(
    mu: np.ndarray,
    posterior: xr.Dataset,
    X_control: np.ndarray | None,
    X_cross: np.ndarray | None,
) -> np.ndarray:
    if X_control is not None:
        beta = posterior["beta_control"].values
        mu = mu + np.einsum("cdk,ok->cdo", beta, X_control, optimize=True)
    if X_cross is not None and "gamma_pair" in posterior:
        gamma = posterior["gamma_pair"].values
        mu = mu + np.einsum("cdp,op->cdo", gamma, X_cross, optimize=True)
    return mu
