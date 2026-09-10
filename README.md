# pypricing

Bayesian own-price elasticity estimation with PyMC for long-format panels (one row per SKU–time observation).

This is currently an MVP for log-demand panels (per-SKU intercept + per-SKU elasticity) with:

- optional **shared control** regressors (`control_`* columns)
- optional **hierarchical / partial pooling** across group columns
- optional **linear time trend** (`trend="shared"` or `"sku"`)
- optional **calendar seasonality** (`seasonality="yearly"` / `"weekly"` / `"auto"`)
- optional **cross-price elasticities** (`CrossElasticitySpec`)
- simple diagnostics + plotting helpers
- posterior predictive simulation for counterfactual price scenarios
- per-SKU **revenue price optimization** (`optimize_prices`)

## Install

```bash
pip install pypricing
```

Optional model-graph rendering (also needs the [system Graphviz](https://graphviz.org/download/) binaries):

```bash
pip install 'pypricing[graphviz]'
```

Docs: [pypricing.readthedocs.io](https://pypricing.readthedocs.io)

### Development

From a clone of this repository:

```bash
uv sync --extra dev
uv run pytest
```

For local docs builds: `uv sync --extra docs`.

## Quickstart

```python
import numpy as np

from pypricing import LogLogDemandModel, generate_mock_data

# 1) Example data (replace with your own panel)
df = generate_mock_data(
    n_periods=20,
    n_skus=5,
    n_controls=2,              # creates control_1, control_2
    include_seasonality=False, # no calendar terms in this example
    random_state=0,
)

# 2) Fit
model = LogLogDemandModel()
idata = model.fit(
    df,
    draws=1000,
    tune=1000,
    chains=4,
    random_seed=42,
)

print(model.run_diagnostics())
print(model.fit_summary().head())

# 3) Counterfactual prediction (quantity column not required)
df_scenario = df.drop(columns=["quantity"]).copy()
df_scenario["price"] = df_scenario["price"] * 1.05  # +5% price scenario

pred = model.predict(df_scenario, hdi_prob=0.9, random_seed=123)
print(pred[["sku", "period", "price", "quantity_mean", "quantity_hdi_lower", "quantity_hdi_upper"]].head())

# 4) Plots
_ = model.plot_elasticity_posterior(hdi_prob=0.9)

sku0 = model.sku_levels_[0]
price_grid = np.linspace(df["price"].min(), df["price"].max(), 25)
controls = {c: float(df.loc[df["sku"] == sku0, c].iloc[0]) for c in model.control_names_}
_ = model.plot_response_curve(sku=sku0, price_grid=price_grid, controls=controls, hdi_prob=0.9)
```

## Data format

`LogLogDemandModel.fit(df)` expects a pandas DataFrame with **level-scale** columns:

- **Required**
  - `sku` (configurable via `sku_col`)
  - `price` (configurable via `price_col`) — must be **strictly positive**
  - `quantity` (configurable via `quantity_col`) — must be **non-negative**
- **Optional**
  - `control_`* columns (or pass an explicit `control_columns=(...)` via `PanelColumns`) — must be numeric, no NaNs
  - hierarchy columns (e.g. `category_1`, `category_2`) via `PanelColumns(group_columns=...)`
  - `period` (and often `region`) for `fit_train_test()` / cross-elasticity market cells.
    Integer or string keys are fine. **Datetime `period` is required** if you enable
    `trend` or `seasonality` (integer indexes are not treated as Unix times).

Internally the model works on logs:

- `log_price = log(price)`
- `log_quantity = log(max(quantity, quantity_floor))`

The default `quantity_floor=1.0` allows zero quantities without \log(0).

## What model is being fit?

At a high level this package fits **log-demand** with Gaussian noise:


\log Q \sim \mathcal{N}(\mu, \sigma)


Where \mu is a per-SKU demand curve plus optional controls, trend, and seasonality.

### Demand curve model classes

Pick the model class directly:

- `LogLogDemandModel` (default/simple)
- `QuadraticLogDemandModel`
- `SigmoidSaturationDemandModel`

#### `log_log` (default)

Constant elasticity log-log:


$\mu = \alpha_{\text{sku}} + \epsilon_{\text{sku}} \log P + X\beta + \text{trend} + \text{season}$


- **Interpretation**: \epsilon_{\text{sku}} is own-price elasticity.
  - Example: \epsilon=-1.5 implies a 1% price increase → ~1.5% quantity decrease (locally / in expectation).
- **Best when**: you want a simple constant-elasticity approximation.

#### `quadratic`

Allows elasticity to vary with price (curvature in log-price):


$\mu = \alpha_{\text{sku}} + \beta_{1,\text{sku}}\log P + c_{\text{sku}} (\log P)^2 + X\beta + \text{trend} + \text{season}$


This parameterization enforces that the elasticity at a **per-SKU midpoint price** equals `elasticity_sku`.
The midpoint is computed from the training data as the **median** log-price per SKU.

- **Best when**: elasticity changes with price level (e.g., premium vs discount regimes).

#### `sigmoid`

Saturating response curve in **level price** (softplus / log-sigmoid form):


\mu = \alpha_{\text{sku}} - \mathrm{softplus}(b_{\text{sku}}(P - P_{\text{center,sku}})) + X\beta + \text{trend} + \text{season}


The per-SKU **center** `P_{\text{center,sku}} = \exp(\texttt{log\_price\_center\_sku})` is a **learned** parameter.
Its prior is centered at the empirical per-SKU median log-price from training data (`log_price_midpoint_sku_`), with default `sigma=0.5`.
The curve is parameterized so the elasticity at `P_{\text{center,sku}}` equals `elasticity_sku` (via `b_sku = -2 * elasticity_sku / price_center_sku`).

- **Best when**: response “flattens out” at extreme prices (a simple saturation behavior).

### Trend and seasonality

`period` is the market-cell clock. Leave `trend` / `seasonality` as `None` (the
default) and integer periods still work. Turning either on requires a **datetime**
`period` column.

```python
from pypricing import LogLogDemandModel, generate_mock_data

df = generate_mock_data(
    n_periods=2 * 52,
    n_skus=6,
    start_date="2020-01-06",  # writes a calendar into period
    freq="W",
    include_seasonality=True,  # yearly sine on day-of-year
    volume_trend=0.08,         # shared annual log-growth; needs start_date
    round_quantity=False,
    random_state=0,
)
model = LogLogDemandModel(trend="shared", seasonality="auto")
model.fit(df, draws=500, tune=500, chains=2, random_seed=0)
```

- **`trend`**: `"shared"` is one slope `mu_trend * t`; `"sku"` is a per-SKU slope
  pooled toward that mean (`t` is years since the first training date).
- **`seasonality`**: shared Fourier terms (not per-SKU). `"yearly"` / `"weekly"`
  or a sequence of those. `"auto"` picks from the panel grain (daily →
  yearly+weekly; weekly/monthly → yearly).
- Plots and `optimize_prices` hold the calendar at the **last training date**
  (`at_period=` to override). Prediction needs a datetime `period` column.

`generate_mock_data(..., include_seasonality=True)` **without** `start_date`
still adds a sine over integer `0 … n_periods-1`. That is not calendar
seasonality, and the model cannot fit it as Fourier (ints are rejected). Pass
`include_seasonality=False` unless you are using a dated panel.

### Controls (`control_`*)

If your frame contains `control_*` columns (or you pass `control_columns=(...)`), the model includes a **shared** linear term X\beta:

- one global coefficient vector `beta_control` shared across all SKUs
- controls must also be provided at prediction time

### Priors and customization (`model_config`)

You can override priors by passing `model_config` to each model class constructor.
Each entry uses:

```python
{
    "dist": <PyMC distribution constructor>,
    "kwargs": { ... },
}
```

Example:

```python
import pymc as pm
from pypricing import QuadraticLogDemandModel

model = QuadraticLogDemandModel(
    model_config={
        "alpha_sku": {"dist": pm.Normal, "kwargs": {"mu": 0.0, "sigma": 1.0}},
        "elasticity_sku": {"dist": pm.Normal, "kwargs": {"mu": -1.0, "sigma": 0.7}},
        "sigma": {"dist": pm.HalfNormal, "kwargs": {"sigma": 0.3}},
        # "curvature_sku": ... (only used by QuadraticLogDemandModel)
        # "beta_control": ...  (only used when you have controls)
    }
)
```

Defaults today:

- `alpha_sku ~ Normal(mu=6, sigma=2)`
- `elasticity_sku ~ Normal(mu=-1, sigma=2)`
- `sigma ~ HalfNormal(sigma=0.5)`
- `beta_control ~ Normal(mu=0, sigma=0.5)` (if controls exist)
- `mu_trend ~ Normal(mu=0, sigma=0.05)` (if `trend` is set)
- `beta_season ~ Normal(mu=0, sigma=0.5)` (if `seasonality` is set)
- `curvature_sku ~ Normal(mu=0, sigma=0.2)` (only for `quadratic`)
- `log_price_center_sku ~ Normal(mu=log_price_midpoint_sku_, sigma=0.5)` (only for `sigmoid`)

## Prediction

- `sample_posterior_predictive(df)` returns an `xarray.Dataset` with posterior draws for `log_quantity` and `quantity`.
- `predict(df)` returns a DataFrame with:
  - `quantity_mean`
  - `quantity_hdi_lower`
  - `quantity_hdi_upper`

Prediction requires:

- `sku` and `price`
- all control columns used during fit (if any)
- datetime `period` if the model was fit with `trend` or `seasonality`
- `quantity` is **not** required

Unknown SKUs at prediction time raise an error (no cold-start handling yet).

## Train/test evaluation (time-aware)

`fit_train_test(df, period_col="period", test_size=0.2, ...)`:

- holds out the **last** fraction of unique periods (to avoid time leakage)
- fits on train, predicts on test
- returns:
  - `rmse` on `quantity_mean`
  - `hdi_coverage`: fraction of true quantities inside the predicted HDI
  - `test_predictions`: the prediction frame

## Save / load

`<ModelClass>.save(path)` saves the model `InferenceData` to NetCDF (`.nc`) with model attrs.

`<ModelClass>.load(path)` restores the model from that NetCDF artifact and rebuilds the PyMC model.

## Interpretation helper

`pypricing.posterior.summarize_quantity_multiplier_one_sku(...)` converts posterior elasticity draws into a posterior over relative quantity change for a price multiplier m via m^{\epsilon}.

For all SKUs at once, use the model method:

- `model.quantity_multiplier_summary(price_multiplier=..., hdi_prob=...)`
- returns one row per SKU with mean and HDI bounds of the quantity multiplier
- optionally, `model.quantity_multiplier_summary(..., return_draws=True)` returns
  `(summary_df, draws_da)` where `draws_da` has dims `("chain", "draw", "sku")`

This is only valid for `LogLogDemandModel` (constant elasticity).

## Price optimization

`optimize_prices` (also available as `model.optimize_prices(...)`) chooses one price per SKU that maximizes the posterior **mean** of revenue `price * exp(μ)`, where `μ` is the demand-curve mean log-quantity (same convention as prediction; residual `σ` is not folded into `exp(μ)`).

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
controls_df = (
    df.groupby("sku", as_index=False)[model.control_names_].median()
)
opt = model.optimize_prices(price_bounds=price_bounds, controls_df=controls_df)
print(opt.head())
```

Caveats:

- Optimization is **per-SKU and independent** (1D SciPy search on log-price within bounds).
- **Not supported** when `cross_elasticity` is enabled (revenue then depends on the full market cell).
- For `LogLogDemandModel`, expected revenue is proportional to `p^(1+ε)` draw-wise; if elasticity is roughly constant and `|ε| ≠ 1`, the optimum often sits on a **bound**.
- If the model was fit with controls, pass `controls_df` with one row per SKU.

See `docs/source/notebooks/quickstart.ipynb` for plots (`plot_revenue_vs_price`, `plot_optimization_summary`).

## Hierarchy and cross-elasticity

Pass column mapping via `PanelColumns`:

```python
from pypricing import CrossElasticitySpec, LogLogDemandModel, PanelColumns

model = LogLogDemandModel(
    panel_columns=PanelColumns(group_columns=("category_1", "category_2")),
    cross_elasticity=CrossElasticitySpec(mode="within_group", group_level=0),
)
```

Omit `cross_elasticity` for own-price only. Use `mode="all"` for every directed SKU pair (no `group_level`).

## What is NOT implemented (yet)

- Cold-start prediction for unseen SKUs
- Joint / cross-aware price optimization (use counterfactual prediction on a full market cell instead)

## Documentation

Hosted docs (API + notebooks): [https://pypricing.readthedocs.io](https://pypricing.readthedocs.io)

To build HTML locally (Sphinx + notebooks, no notebook re-execution):

```bash
uv sync --extra docs
cd docs && uv run make html
# open docs/build/html/index.html
```

## Development

```bash
uv sync --extra dev
uv run pytest                 # unit + integration (default)
uv run pytest -m unit         # fast only
uv run pytest -m integration  # short MCMC smoke
uv run pytest -m recovery     # parameter recovery (slower)
uv run ruff check src tests
```

