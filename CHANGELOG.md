# Changelog

## Unreleased

### Added

- Promo planning (`pypricing.promo`): `plan_frame`, `compare_scenarios`,
  `promo_options`, `optimize_promotions` (MILP with budget, promo-count,
  per-group, max-discount and min-margin-rate constraints; coupling-aware
  refinement) and `promo_backtest`.
- `PromoSpec`: quadratic discount-depth response, promo-mechanic effects
  (`type_col`), post-promo lags and within-group cannibalization
  (`pressure_level`).
- `varying_controls`: per-SKU control slopes pooled through `group_columns`
  (competitor prices with category-specific effects).
- `add_event_columns` / `event_indicators`: retail event dummies.
- `aggregate_panel`: roll SKUs up to brand x category with fixed weights.
- `optimize_prices(objective="margin", unit_cost=...)`.
- `DemandModel.expected_quantity_draws`.
- `center_price=True` for log-log models (faster sampling).
- `generate_promo_panel`: synthetic retail promo data with known truth.
- Optional `fast` extra (nutpie).

### Changed

- The posterior `sku` coordinate is the SKU label, in `sku_levels_` order.
  Integer codes stay internal (`obs_sku_idx`). `sel(sku=0)` no longer selects
  the first SKU; use the label, or `isel(sku=0)`. `model.elasticities()`
  returns the posterior mean and HDI indexed by those labels.
