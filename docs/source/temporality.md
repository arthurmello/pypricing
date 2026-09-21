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
    round_quantity=False,
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

## Mock data caveat

`generate_mock_data(..., include_seasonality=True)` **without** `start_date`
still adds a sine over integer `0 … n_periods-1`. That is not calendar
seasonality, and the model cannot fit it as Fourier (ints are rejected). Pass
`include_seasonality=False` unless you use a dated panel.
