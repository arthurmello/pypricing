# pypricing

Bayesian log-demand pricing analytics with PyMC for long-format panels.

```bash
pip install pypricing
```

```python
from pypricing import LogLogDemandModel, generate_mock_data

df = generate_mock_data(
    n_periods=20, n_skus=5, n_controls=2, include_seasonality=False, random_state=0
)
model = LogLogDemandModel()
model.fit(df, draws=500, tune=500, chains=2, random_seed=0)
print(model.fit_summary().head())
```

## Guides

```{toctree}
:maxdepth: 1

models
temporality
identification
priors
workflows
optimization
hierarchy
```

## Notebooks and API

```{toctree}
:maxdepth: 1

notebooks/index
api/index
```

Contributors: clone the repo and use `uv sync --extra dev` (add `--extra docs`
for Sphinx).
