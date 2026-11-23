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

Prefer variation that is not set from demand:

- a randomized experiment (price or promo assignment),
- cost / commodity / wholesale shocks that move price but not demand directly,
- other valid instruments ($Z$) with the same exclusion story.

More regressors are not a substitute. If you don't have an instrument,
[copula, Lewbel, and latent IV](#without-an-external-instrument) replace it
with a distributional assumption. `compare_endogeneity` checks them side by
side before you fit one.

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
print(model.check_instruments())  # first-stage F, Sargan test, rho, IV vs OLS
```

### Estimator (control function)

Log-log (quadratic and sigmoid use the same first-stage residual on their own
mean curves):

- **Price:**
  $\log P = \alpha^P_{\mathrm{sku}} + Z\pi + X\gamma^P + \text{trend/season} + v$
- **Demand:**
  $\log Q = \alpha_{\mathrm{sku}} + \varepsilon_{\mathrm{sku}}\log P + X\beta + \rho v + \ldots$

$Z$ is excluded from demand. $\rho$ away from 0 is evidence that price is
endogenous. An interval that includes 0 is no evidence of endogeneity, and
the instrumented elasticities are noisier, so drop the instruments.
`check_instruments()["first_stage"]["f"]` is the partial F for the
instruments in the price equation, after SKU intercepts, controls, trend, and
seasonality. `r2_without` and `r2_with` are the R² of that equation without
and with the instruments. `weak` is true when F is below 10, a rule of thumb
for one endogenous price, not a guarantee the instruments are valid and not a
Stock–Yogo critical value. A weak instrument can bias the corrected
elasticities by more than fitting without instruments. A large F does not
show that the instrument affects demand only through price. See
[Diagnostics](diagnostics.md#instruments) for how to read all the instrument
checks.

With a discount column the first stage is for **log regular price**,
$\log R = \log P - \log(1-d)$, not the price actually paid. The discount
index $\log(1-d)$ is treated as exogenous: it is not instrumented, and
`promo_elasticity_sku` is estimated from discount variation directly. That
assumes temporary discounts are not set in response to the demand shock.

{meth}`~pypricing.DemandModel.predict` and
{func}`~pypricing.optimize_prices` use the structural demand curve (control
residual set to 0). The IV correction is for **estimation** only.

### Period effects

A demand shock common to every SKU in a period is not fixed by an instrument
for the SKU-level shock. `time_effects="period"` (or `"region_period"`) absorbs
it. The control-function first stage gets its own period effects, so the
residual is price variation left inside the period.

What remains for the elasticity is within-period price variation, after SKU
and period means. `run_diagnostics` reports that share as
`within_period_price_share` and warns when it is below 10%: the elasticity is
then resting on very little variation. SKU-by-period shocks that price reacts
to are still an instruments problem; period effects do not replace them.

### Exclusion

A valid instrument must move price without moving demand directly. Cost /
commodity shocks usually qualify; lagged sales usually do not. The library
**cannot** check the exclusion restriction — that is a modeling assumption.

## Without an external instrument

Copula, Lewbel, and latent IV correct the same bias without an `iv_*` column.
Each one assumes something about the price distribution instead of an excluded
instrument. `compare_endogeneity` runs the fast check for every method and
does not sample. It leaves the model unchanged.

```python
from pypricing import LogLogDemandModel, generate_mock_data

df = generate_mock_data(
    n_periods=300,
    n_skus=1,
    price_distribution="gamma",
    endogeneity=1.5,
    noise_sigma=0.35,
    price_shock_sigma=0.02,
    include_seasonality=False,
    random_state=0,
)
table = LogLogDemandModel().compare_endogeneity(df)
```

The first row is the uncorrected regression (`ols`). Each other row is a
method's verdict and pooled elasticity:

| Verdict | Meaning |
| --- | --- |
| `ok` | The method's assumptions hold. The elasticity is the fast estimate. |
| `weak` | It can be estimated, but the result is fragile. |
| `not identified` | Do not use it. There is no elasticity in that row. |
| `uncorrected` | The OLS row. No endogeneity correction. |

With no instrument columns the default methods are copula, latent IV, and
Lewbel. IV is added when the frame has instruments. Pass `methods=` to choose
the list; it needs at least two names. Asking for IV with no instruments
returns a `not identified` row.

Fit the method you keep, for example `LogLogDemandModel(endogeneity="copula")`.
`predict` and `optimize_prices` use the structural curve, with the control at 0.

### Which method

**IV** (`endogeneity="iv"`). You have a valid excluded instrument. It is the
control function above. No instruments, or instruments with no remaining
variation, is not identified. A first-stage F below 10 is weak.

**Copula** (`endogeneity="copula"`). The price residual, after the exogenous
regressors, is not normal. The control is the normal score of that residual's
rank. A residual consistent with a normal distribution is not identified, so
the method does not apply to a normal price. On a quadratic or sigmoid curve,
a correlation above 0.8 between the control and the curve's bend means you
cannot separate endogeneity from the shape of the curve. After the fit, the
summary reports the posterior correlation of `rho_copula` with `curvature_sku`
or `log_price_center_sku`.

**Lewbel** (`endogeneity="lewbel"`). Some exogenous regressor drives the
variance of the price residual. The generated instruments are that regressor,
demeaned, times the residual, and estimation reuses the IV control function.
The regressor has to be in the model (a control, or more than one SKU). A
homoskedastic residual, or a design where every exogenous regressor is
constant, is not identified. On the mock data, pass the `hetero` column as a
control.

**Latent IV** (`endogeneity="latent"`). Experimental. An unobserved discrete
class shifts price and is independent of the demand shock; the shared shock
stays in the continuous part of price. The residual is a Gaussian mixture with
ordered class means, and the number of classes is chosen by K-fold ELPD among
2, 3, and 4. A single normal that fits as well, or classes that collapse, is
not identified. The elasticity moving by more than 0.3 across class counts
that fit about as well is weak. Setting it warns that the method is
experimental. The generator's discrete prices are a different case: there the
price point itself moves with the demand shock.

### Warnings

On `QuadraticLogDemandModel` and `SigmoidSaturationDemandModel`, setting any
of these warns. A control function in a nonlinear curve needs a stronger
assumption than in a linear one: the demand shock has to enter in a way that
stays valid along the whole curve.

### If there is no evidence of endogeneity

The fast check says whether a method is identified. It does not say whether
price was endogenous. After you fit the method you chose, read the correction
parameter: `rho` for IV and Lewbel, `rho_copula` for the copula, `rho_latent`
for latent IV. An interval that includes 0 is no evidence of endogeneity. Use
the uncorrected model. The correction adds noise.
