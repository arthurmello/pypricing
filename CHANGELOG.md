# Changelog

## Unreleased

### Changed

- The posterior `sku` coordinate is the SKU label, in `sku_levels_` order.
  Integer codes stay internal (`obs_sku_idx`). `sel(sku=0)` no longer selects
  the first SKU; use the label, or `isel(sku=0)`. `model.elasticities()`
  returns the posterior mean and HDI indexed by those labels.
