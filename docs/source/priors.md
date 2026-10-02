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
- `elasticity_sku ~ Normal(mu=-2.6, sigma=1.5)`
- `sigma ~ HalfNormal(sigma=0.5)`
- `beta_control ~ Normal(mu=0, sigma=0.5)` (if controls exist)
- `pi ~ Normal(mu=0, sigma=1)` / `rho ~ Normal(mu=0, sigma=1)` /
  `sigma_price ~ HalfNormal(sigma=0.5)` (if instruments exist)
- `mu_trend ~ Normal(mu=0, sigma=0.05)` (if `trend` is set)
- `beta_season ~ Normal(mu=0, sigma=0.5)` (if `seasonality` is set)
- `curvature_sku ~ Normal(mu=0, sigma=0.2)` (quadratic only)
- `log_price_center_sku ~ Normal(mu=log_price_midpoint_sku_, sigma=0.5)`
  (sigmoid only)

With `group_columns`, the elasticity location prior is on the hierarchy mean
(`mu_elasticity ~ Normal(mu=-2.6, sigma=1.5)`). Per-SKU and per-group scales
stay `HalfNormal(sigma=1)`. On the quadratic and sigmoid curves this prior is
the local elasticity at the SKU's reference price.

## Why -2.6

The default is the mean elasticity in Bijmolt, van Heerde, and Pieters (2005),
"New empirical generalizations on the determinants of price elasticity,"
*Journal of Marketing Research*, pp. 145–146, 149: mean −2.62 across 1,851
estimates (median −2.22, SD 2.21). 81% of those estimates fall in [−4, 0].
`sigma=1.5` is tighter than that cross-study SD, so SKUs with little price
variation shrink toward that bulk instead of toward −1.

On a hierarchical panel the location prior is `mu_elasticity`, with the same
`kwargs`.
