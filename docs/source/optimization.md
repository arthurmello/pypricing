# Price optimization

{func}`~pypricing.optimize_prices` (also `model.optimize_prices(...)`) chooses
one price per SKU that maximizes the posterior mean of **expected** revenue
`price * exp(μ + σ²/2)`. `μ` is the demand-curve mean log-quantity and `σ` is
the residual scale, so `exp(μ + σ²/2)` is expected quantity. That is the same
target `predict` reports as `quantity_mean`.

```python
from pypricing import LogLogDemandModel, generate_mock_data

df = generate_mock_data(
    n_periods=20, n_skus=5, n_controls=1, include_seasonality=False, random_state=0
)
model = LogLogDemandModel()
model.fit(df, draws=500, tune=500, chains=2, random_seed=0)

price_bounds = {
    sku: (float(g["price"].min() * 0.8), float(g["price"].max() * 1.3))
    for sku, g in df.groupby("sku")
}
controls_df = df.groupby("sku", as_index=False)[model.control_names_].median()
opt = model.optimize_prices(price_bounds=price_bounds, controls_df=controls_df)
print(opt.head())
```

## Caveats

- Optimization is **per-SKU and independent**. Each SKU is a 1D search on
  log-price: a grid over the bounds, then a local polish around the best grid
  point.
- **Not supported** when `cross_elasticity` is enabled (revenue then depends on
  the full market cell).
- For {class}`~pypricing.LogLogDemandModel`, expected revenue is proportional to
  $p^{1+\varepsilon}$ on each posterior draw. That is monotone in price unless
  $\varepsilon = -1$, so the optimum is an endpoint of `price_bounds`. A
  posterior that straddles $-1$ is U-shaped in price; the optimum is still the
  better endpoint. An interior maximum needs a price-dependent elasticity
  ({class}`~pypricing.QuadraticLogDemandModel` or
  {class}`~pypricing.SigmoidSaturationDemandModel`).
- If the model was fit with controls, pass `controls_df` with one row per SKU.
- With `trend` / `seasonality`, the calendar is held at the last training date
  unless you pass `at_period=`. Pass that same value to
  `plot_optimization_summary` so the reference-revenue bars use that date.

See the [quickstart notebook](notebooks/quickstart) for
`plot_revenue_vs_price` and `plot_optimization_summary`.
