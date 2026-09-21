# pypricing

Bayesian log-demand pricing analytics with PyMC for long-format panels (one row
per SKU–time observation): own-price elasticity, optional controls, instruments,
hierarchy, trend/seasonality, and counterfactual prediction.

## Install

```bash
pip install pypricing
```

Optional model-graph rendering (also needs the
[system Graphviz](https://graphviz.org/download/) binaries):

```bash
pip install 'pypricing[graphviz]'
```

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

## When this is not identified

If prices respond to the same shocks that move demand, a regression of quantity
on price mixes the demand slope with that feedback. More regressors do not fix
it. You need exogenous price variation: a randomized experiment, cost /
commodity shocks, or other valid instruments (`iv_*`). See
[Identification](https://pypricing.readthedocs.io/en/latest/identification.html).