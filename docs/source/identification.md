# Identification

When pricing decisions respond to the same factors that drive demand,
observational elasticity estimates can be biased. That cannot be diagnosed from
observed price and quantity alone.

## The problem

A regression of quantity on price recovers a mix of:

1. the demand slope you want (how quantity responds to price), and
2. feedback from pricing rules that react to demand shocks (promotions timed to
   soft weeks, list prices set from forecasts, etc.).

Adding more controls only helps if those controls absorb the shared shock.
Usually they do not. You need **exogenous** price variation.

## What to do

Use one of:

- a randomized experiment (price or promo assignment),
- cost / commodity / wholesale shocks that move price but not demand directly,
- other valid instruments ($Z$) with the same exclusion story.

More regressors are not a substitute.

## Instruments in pypricing

Pass `iv_*` columns (or `PanelColumns(iv_columns=...)`). An empty
`iv_columns=()` turns IV off even if `iv_*` columns exist. Columns must be
numeric with no NaNs.

```python
from pypricing import LogLogDemandModel, PanelColumns, generate_mock_data

df = generate_mock_data(
    n_periods=40,
    n_skus=4,
    n_instruments=1,      # writes iv_1
    endogeneity=1.0,      # demand shock also moves price
    include_seasonality=False,
    random_state=0,
)
model = LogLogDemandModel()  # auto-detects iv_*
# or: LogLogDemandModel(panel_columns=PanelColumns(iv_columns=("iv_1",)))
model.fit(df, draws=500, tune=500, chains=2, random_seed=0)
print(model.run_diagnostics())  # rho, first_stage_f, weak_iv
```

### Estimator (control function)

Log-log (quadratic and sigmoid use the same first-stage residual on their own
mean curves):

- **Price:**
  $\log P = \alpha^P_{\mathrm{sku}} + Z\pi + X\gamma^P + \text{trend/season} + v$
- **Demand:**
  $\log Q = \alpha_{\mathrm{sku}} + \varepsilon_{\mathrm{sku}}\log P + X\beta + \rho v + \ldots$

$Z$ is excluded from demand. $\rho$ away from 0 is evidence that price is
endogenous. `run_diagnostics()["first_stage_f"]` is the partial F for the
instruments in the price equation, after SKU intercepts, controls, trend, and
seasonality. `weak_iv` is true when that F is below 10, the usual rule of
thumb for one endogenous price. It is not a Stock–Yogo critical value.

{meth}`~pypricing.DemandModel.predict` and
{func}`~pypricing.optimize_prices` use the structural demand curve (control
residual set to 0). The IV correction is for **estimation** only.

### Exclusion

A valid instrument must move price without moving demand directly. Cost /
commodity shocks usually qualify; lagged sales usually do not. The library
**cannot** check the exclusion restriction — that is a modeling assumption.
