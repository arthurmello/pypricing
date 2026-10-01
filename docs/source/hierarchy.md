# Hierarchy and cross-elasticity

Pass column mapping via {class}`~pypricing.PanelColumns`:

```python
from pypricing import CrossElasticitySpec, LogLogDemandModel, PanelColumns

model = LogLogDemandModel(
    panel_columns=PanelColumns(group_columns=("category_1", "category_2")),
    cross_elasticity=CrossElasticitySpec(mode="within_group", group_level=0),
)
model.fit(df, draws=500, tune=500, chains=2, target_accept=0.95)
```

Omit `cross_elasticity` for own-price only. Use `mode="all"` for every directed
SKU pair (no `group_level`).

Hierarchy columns enable partial pooling of SKU intercepts / elasticities
toward group means. Cross-price terms need a balanced market cell per `period`
(and `region` when used).

Group scales and cross-price coefficients add curved regions to the posterior.
Pass `target_accept=0.95` to `fit` so NUTS uses a smaller step size. PyMC's
default is 0.8. `0.9` still leaves a noticeable share of divergent draws on
these models.

See the [hierarchical](notebooks/hierarchical) and
[cross-elasticity](notebooks/cross_elasticity) notebooks.
