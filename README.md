# pypricing

Bayesian log-demand pricing analytics with PyMC for long-format panels (one row
per SKU–time observation): own-price elasticity, optional controls, instruments,
hierarchy, trend/seasonality, and counterfactual prediction.

## Install

```bash
pip install pypricing
```

`model.graphviz()` also needs the [system Graphviz](https://graphviz.org/download/)
binaries (e.g. `brew install graphviz` on macOS).

Full docs (guides + API + notebooks):
[pypricing.readthedocs.io](https://pypricing.readthedocs.io)

## Quickstart

```python
import numpy as np

from pypricing import LogLogDemandModel, generate_mock_data

df = generate_mock_data(
    n_periods=20,
    n_skus=5,
    n_controls=2,
    include_seasonality=False,
    random_state=0,
)

model = LogLogDemandModel()
model.fit(df, draws=1000, tune=1000, chains=4, random_seed=42)

df_scenario = df.drop(columns=["quantity"]).copy()
df_scenario["price"] = df_scenario["price"] * 1.05
print(model.predict(df_scenario, hdi_prob=0.9, random_seed=123).head())
```

## Data contract

`fit(df)` expects level-scale columns:

- **Required:** `sku`, `price` (strictly positive), `quantity` (non-negative)
- **Optional:** `control_*`, `iv_*`, hierarchy via `PanelColumns(group_columns=...)`,
  `period` / `region` (datetime `period` required for `trend` / `seasonality`)

Column names are configurable via `PanelColumns`. Internally the model uses
`log(price)` and `log(max(quantity, quantity_floor))` (default floor `1.0`).

## When to use pypricing (and when not to use it)
pypricing assumes products have posted prices that changed over time, and you observe the units sold at each price.

Retail, consumer goods and e-commerce are good use cases: many products, regular sales volume, frequent price or promo changes.

Bad use cases include one-off prices, very low volume, or sales limited by capacity (hotels, airlines).