# Priors

Override priors by passing `model_config` to each model class constructor. Each
entry uses:

```python
{
    "dist": <PyMC distribution constructor>,
    "kwargs": { ... },
}
```

Example:

```python
import pymc as pm
from pypricing import QuadraticLogDemandModel

model = QuadraticLogDemandModel(
    model_config={
        "alpha_sku": {"dist": pm.Normal, "kwargs": {"mu": 0.0, "sigma": 1.0}},
        "elasticity_sku": {"dist": pm.Normal, "kwargs": {"mu": -1.0, "sigma": 0.7}},
        "sigma": {"dist": pm.HalfNormal, "kwargs": {"sigma": 0.3}},
        # "curvature_sku": ... (only QuadraticLogDemandModel)
        # "beta_control": ...  (only when you have controls)
    }
)
```

## Defaults

- `alpha_sku ~ Normal(mu=6, sigma=2)`
- `elasticity_sku ~ Normal(mu=-1, sigma=2)`
- `sigma ~ HalfNormal(sigma=0.5)`
- `beta_control ~ Normal(mu=0, sigma=0.5)` (if controls exist)
- `pi ~ Normal(mu=0, sigma=1)` / `rho ~ Normal(mu=0, sigma=1)` /
  `sigma_price ~ HalfNormal(sigma=0.5)` (if instruments exist)
- `mu_trend ~ Normal(mu=0, sigma=0.05)` (if `trend` is set)
- `beta_season ~ Normal(mu=0, sigma=0.5)` (if `seasonality` is set)
- `curvature_sku ~ Normal(mu=0, sigma=0.2)` (quadratic only)
- `log_price_center_sku ~ Normal(mu=log_price_midpoint_sku_, sigma=0.5)`
  (sigmoid only)
