# Demand curves

At a high level pypricing fits **log-demand** with Gaussian noise:

$$
\log Q \sim \mathcal{N}(\mu, \sigma)
$$

where $\mu$ is a per-SKU demand curve plus optional controls, trend, and
seasonality. Pick the curve via the model class:

- {class}`~pypricing.LogLogDemandModel`
- {class}`~pypricing.QuadraticLogDemandModel`
- {class}`~pypricing.SigmoidSaturationDemandModel`

## Log-log (constant elasticity)

$$
\mu = \alpha_{\mathrm{sku}} + \varepsilon_{\mathrm{sku}} \log P + X\beta + \text{trend} + \text{season}
$$

$\varepsilon_{\mathrm{sku}}$ is own-price elasticity. Example: $\varepsilon=-1.5$
implies a 1% price increase → about 1.5% quantity decrease (locally / in
expectation).

**Best when** you want a simple constant-elasticity approximation.

## Quadratic (price-dependent elasticity)

$$
\mu = \alpha_{\mathrm{sku}} + \beta_{1,\mathrm{sku}}\log P + c_{\mathrm{sku}} (\log P)^2 + X\beta + \text{trend} + \text{season}
$$

Elasticity at a **per-SKU midpoint price** is constrained to equal
`elasticity_sku`. The midpoint is the training-data **median** log-price per
SKU.

**Best when** elasticity changes with price level (e.g. premium vs discount
regimes).

## Sigmoid (saturating response)

$$
\mu = \alpha_{\mathrm{sku}} - \mathrm{softplus}\!\big(b_{\mathrm{sku}}(P - P_{\mathrm{center,sku}})\big) + X\beta + \text{trend} + \text{season}
$$

The per-SKU center $P_{\mathrm{center,sku}} = \exp(\texttt{log\_price\_center\_sku})$
is learned. Its prior is centered at the empirical per-SKU median log-price
(`log_price_midpoint_sku_`), with default `sigma=0.5`. The curve is
parameterized so elasticity at the center equals `elasticity_sku`
(`b_sku = -2 * elasticity_sku / price_center_sku`).

**Best when** response flattens at extreme prices.

## Controls

If the frame has `control_*` columns (or you pass `control_columns=(...)`), the
model includes a **shared** linear term $X\beta$: one global `beta_control`
across SKUs. The same controls must be provided at prediction time.
