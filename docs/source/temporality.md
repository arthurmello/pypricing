# Trend and seasonality

`period` is the market-cell clock. Leave `trend` / `seasonality` as `None`
(default) and integer periods still work. Turning either on requires a
**datetime** `period` column.

```python
from pypricing import LogLogDemandModel, generate_mock_data

df = generate_mock_data(
    n_periods=2 * 52,
    n_skus=6,
    start_date="2020-01-06",  # calendar period
    freq="W",
    include_seasonality=True,  # yearly sine on day-of-year
    volume_trend=0.08,         # shared annual log-growth; needs start_date
    random_state=0,
)
model = LogLogDemandModel(trend="shared", seasonality="auto")
model.fit(df, draws=500, tune=500, chains=2, random_seed=0)
```

## Trend

- `"shared"`: one slope `mu_trend * t`
- `"sku"`: per-SKU slope pooled toward that mean

$t$ is years since the first training date.

## Seasonality

Shared Fourier terms (not per-SKU):

- `"yearly"` / `"weekly"`, or a sequence of those
- `"auto"` picks from the panel grain (daily → yearly+weekly; weekly/monthly →
  yearly)

## Prediction and optimization

Plots and {func}`~pypricing.optimize_prices` hold the calendar at the **last
training date** (`at_period=` to override). Prediction needs a datetime
`period` column when trend or seasonality was fit.

## Which time term

Use one, not both. A free effect on every period already contains a trend and a
seasonal wave, so the model rejects `time_effects` together with `trend` or
`seasonality`.

- **Estimating the elasticity.** Use `time_effects="period"` (or
  `"region_period"` when shocks differ by region). A one-week shock shared by
  every SKU, such as weather or a competitor campaign, biases the elasticity if
  only a smooth trend and seasonality are in the model. The trend and the
  seasonal shape remain inside the fitted period effects for the training
  weeks.
- **Pricing a future date.** Use `trend` and `seasonality`. They carry the
  slope and the seasonal shape into a date the model has not seen. A period
  effect does not: a future period is 0, a random draw, or a copy of one past
  period. Mark a known holiday as its own `control_*` column (`1` on those
  weeks, `0` elsewhere) so the spike sits on top of the smooth curve.

## Period fixed effects

Trend and Fourier seasonality are smooth. A shock that hits every SKU in one
period (a holiday, a heat wave, a competitor campaign) needs a free effect
instead:

```python
model = LogLogDemandModel(time_effects="period")
# or, with PanelColumns(region_col="region"):
model = LogLogDemandModel(time_effects="region_period")
```

`lambda_period` is a zero-sum effect shared by every SKU in the period.
`region_period` adds `lambda_region_period` on top, zero-sum across both
region and period. Do not combine either with `trend` or `seasonality`: the
free effect absorbs both, and the model rejects that combination. Each period
cell needs at least two SKUs, or the effect is not separate from price.

Prediction uses the fitted λ on training periods. For any other period,
`period_effect=` chooses:

- `"neutral"` (default): λ = 0, the average period
- `"draw"`: λ ~ N(0, sigma_lambda) on each posterior draw
- a training period label: reuse that period's λ ("price it like last Black Friday")

`optimize_prices` and the price-curve plots take the same argument.

## Mundlak term

Hierarchical SKU intercepts shrink toward a common mean. If high-price SKUs
also have higher baseline demand, that shrinkage loads the correlation onto
the elasticity and pulls it toward zero. `mundlak=True` puts
`gamma_mundlak * (SKU mean log price − average)` into the intercept's prior
mean. It does not replace instruments for shocks that move one SKU in one period.

## Mock data caveat

`generate_mock_data(..., include_seasonality=True)` **without** `start_date`
still adds a sine over integer `0 … n_periods-1`. That is not calendar
seasonality, and the model cannot fit it as Fourier (ints are rejected). Pass
`include_seasonality=False` unless you use a dated panel.
