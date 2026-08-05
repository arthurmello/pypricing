# pypricing

Bayesian own-price (and optional cross-price) elasticity estimation with PyMC
for long-format panels.

```bash
pip install pypricing
```

```python
from pypricing import LogLogDemandModel, generate_mock_data

df = generate_mock_data(n_periods=20, n_skus=5, n_controls=2, random_state=0)
model = LogLogDemandModel()
model.fit(df, draws=500, tune=500, chains=2, random_seed=0)
print(model.fit_summary().head())
```

Contributors: clone the repo and use `uv sync --extra dev` (add `--extra docs` for Sphinx).

See the [quickstart notebook](notebooks/quickstart) for an end-to-end tour,
or the [API reference](api/index) for the public surface.

```{toctree}
:maxdepth: 2
:hidden:

notebooks/index
api/index
```
