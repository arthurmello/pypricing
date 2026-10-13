# Promotions

A single `price` column mixes two different responses. Shoppers react more to a
temporary discount than to a change in the regular price. Bijmolt, van Heerde,
and Pieters (2005), Table 2 Panel B (p. 150), report a short-term promo
elasticity of about −3.63 against −2.36 for the actual price. If most of the
price variation in a panel is promotional, one elasticity attributes that
stronger response to the regular price, and `optimize_prices()` can recommend
regular prices that are too low.

The fix follows the paper: split price into the regular price and a price
index `actual / regular`.

## Data contract

Set `PanelColumns(discount_col=...)`. `price` stays the price actually paid.
The discount column is a fraction of the **regular** price: `0.2` means 20%
off, `0` means no promo. Values must be numeric, finite, and in `[0, 1)`.

```text
regular price = price / (1 - discount)
log regular   = log(price) - log(1 - discount)
discount index = log(1 - discount)    # 0 off promo, negative on promo
```

Fixed-amount discounts and multibuys have to be converted to an effective
percent off the regular price before fitting. A regular price of 20 with 5 off
is `discount_rate = 0.25` and `price = 15`.

`log_price` in the panel stays the paid price. Cross-price terms still use
other SKUs' paid log prices.

With `discount_col=None` the model is unchanged.

## Model

On every curve:

$$
\mu = f_{\mathrm{sku}}(\log R) + \beta^{\mathrm{promo}}_{\mathrm{sku}} \log(1 - d) + \text{controls} + \text{cross} + \text{trend} + \text{season}
$$

`f_sku` is that model's existing curve, evaluated at the regular price `R`
(the sigmoid's price level is the regular price). `elasticity_sku` is the
regular-price elasticity. The promo term is log-linear in every model:
discounts sit in a narrow band, so curvature in the promo response is not
identifiable. `promo_elasticity_sku` pools through `group_columns`. Its default
prior is `Normal(-3.5, 1.5)`.

## What is identified

`build_model` warns, and `run_diagnostics()` lists:

- `no_discount_variation_skus` — discount never moves (often 0 on every row).
  `promo_elasticity_sku` is prior-only for that SKU.
- `no_regular_price_variation_skus` — the regular price never changes.
  `elasticity_sku` is prior-only. This is common. `group_columns` pools
  regular-price variation across SKUs in the same group.

## Prediction and regular prices

`predict` / `sample_posterior_predictive` require the discount column when the
model was fit with one. `price` is still the price paid.

`apply_discount(df, discount)` writes the discount and scales `price` from the
implied regular price, so the two columns stay consistent.

`optimize_prices` sets the discount to 0 and searches over the regular price.
`quantity_multiplier_summary` scales the regular price and holds the discount
fraction fixed.

With instruments, the control function instruments **log regular price**. The
discount is treated as exogenous and is not instrumented. That assumes
temporary discounts are not set in response to the demand shock. See
[Identification](identification.md).

## Example

The [promotions notebook](notebooks/promotions) runs this end to end.

```python
from pypricing import (
    LogLogDemandModel,
    PanelColumns,
    apply_discount,
    generate_mock_data,
)

df = generate_mock_data(
    n_periods=40,
    n_skus=4,
    promotions=True,
    promo_frequency=0.3,
    promo_depth=0.2,
    include_seasonality=False,
    random_state=0,
)
model = LogLogDemandModel(
    panel_columns=PanelColumns(discount_col="discount_rate", quantity_floor=1e-6)
)
model.fit(df, draws=500, tune=500, chains=2, random_seed=0)

scenario = apply_discount(df.drop(columns=["quantity"]), 0.2)
print(model.predict(scenario, random_seed=0).head())
```
