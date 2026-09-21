# Prediction, evaluation, and persistence

## Prediction

- `sample_posterior_predictive(df)` returns an `xarray.Dataset` with posterior
  draws for `log_quantity` and `quantity`.
- `predict(df)` returns a DataFrame with `quantity_mean`, `quantity_hdi_lower`,
  and `quantity_hdi_upper`.

Prediction requires:

- `sku` and `price`
- all control columns used during fit (if any)
- datetime `period` if the model was fit with `trend` or `seasonality`
- `quantity` is **not** required

Unknown SKUs at prediction time raise an error (no cold-start handling yet).

If the model was fit with instruments, the control-function residual is set to 0.
`predict`, posterior predictive calibration, and `fit_train_test` all use that
structural curve. In-sample HDI coverage is coverage of the structural curve,
not of the observation model fitted during estimation, which still includes the
residual.

## Train/test evaluation (time-aware)

`fit_train_test(df, period_col="period", test_size=0.2, ...)`:

- holds out the **last** fraction of unique periods (avoids time leakage)
- fits on train, predicts on test
- returns `rmse` on `quantity_mean`, `hdi_coverage` (fraction of true
  quantities inside the predicted HDI), and `test_predictions`

## Save / load

`<ModelClass>.save(path)` writes the model `InferenceData` to NetCDF (`.nc`)
with model attrs.

`<ModelClass>.load(path)` restores from that artifact and rebuilds the PyMC
model.

## Quantity-multiplier helpers

`pypricing.posterior.summarize_quantity_multiplier_one_sku(...)` converts
posterior elasticity draws into a posterior over relative quantity change for a
price multiplier $m$ via $m^{\varepsilon}$.

For all SKUs at once:

- `model.quantity_multiplier_summary(price_multiplier=..., hdi_prob=...)`
  — one row per SKU with mean and HDI bounds
- `return_draws=True` also returns an `xarray` with dims
  `("chain", "draw", "sku")`

Only valid for {class}`~pypricing.LogLogDemandModel` (constant elasticity).
