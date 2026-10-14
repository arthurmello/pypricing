# Diagnostics

A model that samples cleanly can still be wrong. These checks ask whether the
fit is good where it matters, whether the elasticities are plausible, and how
much the estimates depend on assumptions you cannot test directly. None of
them proves a model right; each one can flag a reason to distrust it.

## Running everything

```python
report = model.run_diagnostics()
print(report)
```

```text
sampler        OK       0 divergences, max r-hat 1.010
fit            OK       94% HDI coverage 0.95, rmse_log 0.11, bias_log +0.00
benchmarks     OK       own elasticities in [-1.69, -0.71]
sensitivity    OK       pooled RV 0.52 (at alpha: 0.37); weakest sku_3 (0.00)
instruments    SKIPPED  model has no instruments
falsification  SKIPPED  refits the model; pass falsification=True
```

A check shows `WARN` when it finds an issue, and the messages are listed under
`Flags:`. Full results are attributes of the report:

```python
report.fit             # DataFrame, see "Model fit"
report.sensitivity     # DataFrame, see "Sensitivity"
report.flags           # {"fit": [...], "benchmarks": [...], ...}
report.ok              # True when nothing was flagged
report["n_divergent"]  # sampler results, as before
```

Each check can be switched on or off. `falsification` and `compare_ols` refit
the model (reusing the sampler settings of the last `fit()`), so they are off
by default:

```python
model.run_diagnostics(falsification=True, compare_ols=True)
```

`run_diagnostics` uses default settings. To change them, call the individual
`check_*` methods described below.

| Check | Method | Cost | Key number |
| --- | --- | --- | --- |
| Model fit | `check_fit` | instant | `hdi_coverage` near `hdi_prob`, in every bucket |
| Benchmarks | `check_benchmarks` | instant | `share_less_elastic` below 0.99, elasticity below 0 |
| Falsification | `check_falsification` | 2 refits | lead coefficient HDI includes 0 |
| Instruments | `check_instruments` | 1 refit (instant with `compare_ols=False`) | first-stage F above 10, Sargan p above 0.05 |
| Sensitivity | `check_sensitivity` | instant | `rv_qa`, compared with your controls |

## Model fit

**Question:** does the model predict well across the whole price range, or only
in the middle?

```python
model.check_fit()                  # in sample, bucketed by price
model.check_fit(test_df)           # out of sample
model.check_fit(by="control_1")    # bucketed by another column, e.g. media
```

Rows are ranked within each SKU and split into thirds: `low`, `mid`, and
`high`. Ranks are always relative to the training data, so on a test frame
"high" means high compared with the prices the model learned from. The result
has one `overall` row plus one row per bucket:

- `hdi_coverage`: share of observed quantities inside the predicted interval.
  It should be close to `hdi_prob` (0.94 by default).
- `rmse_log`: typical relative error (0.10 is roughly 10%).
- `bias_log`: average of log(observed) minus log(predicted). Positive means the
  model under-predicts, negative means it over-predicts.
- `rmse`: error in quantity units. Dominated by high-volume rows, so compare it
  across buckets with care.

**How to read it:** good overall numbers with poor coverage or a strong bias in
the `low` or `high` bucket mean the demand curve has the wrong shape at the
extremes. That is where price recommendations usually land, so try a
different curve (quadratic or sigmoid) before optimizing. Buckets with fewer
than 10 rows show `NaN` metrics.

## Benchmarks

**Question:** are the elasticities plausible compared with published
estimates and basic theory?

```python
out = model.check_benchmarks()
out["own"]     # one row per SKU
out["cross"]   # one row per cross-price pair, or None
```

Own-price elasticities are compared with the meta-analysis of Bijmolt, van
Heerde and Pieters (2005): about 1,850 estimates, averaging −2.6, with 81%
between −4 and 0.

- `share_less_elastic`: share of published estimates closer to zero than this
  SKU. 0.99 means more elastic than 99% of them. Values of 0.99 or more raise a
  warning (`extreme_threshold`).
- `prob_positive`: posterior probability that the elasticity is positive. A
  positive mean raises a warning, since demand should fall as price rises.

**How to read it:** an extreme or positive elasticity usually points to a data
or identification problem (promotions timed to demand, too little price
variation) rather than an unusual product.

Cross-price effects (`gamma_mean`, `prob_positive`) get no warning. Substitutes
should be positive and complements negative, and the model cannot know which
pairs are which, so check them against what you know about the products.
Diversion ratios against second-choice survey data (Conlon and Mortimer, 2021)
are not implemented.

## Falsification

**Question:** does next period's price "predict" today's demand? It shouldn't:
shoppers cannot react to a price that does not exist yet.

```python
out = model.check_falsification()
out["lead_coef_hdi"], out["hdi_excludes_zero"]
out["elasticity"]   # baseline vs with_lead, per SKU
```

The model is refit twice on the same rows: once as is, and once with next
period's price as an extra control. Rows without a next period (the last
period, or before a gap) are dropped.

**How to read it:** an HDI that includes 0 is the expected result. If it
excludes 0, something you aren't measuring may drive both price and sales,
such as prices set from demand forecasts. Forward-looking shoppers (waiting for
a promotion, stocking up before a price rise) produce the same signal, so treat
it as a flag to investigate. The `elasticity` table shows whether adding the
lead moves the elasticities: large shifts make the problem more serious.

## Instruments

**Question:** if the model uses instruments, are they strong, do they agree
with each other, and do they change the answer? See
[Identification](identification.md) for how instruments work.

```python
out = model.check_instruments()                   # refits once without instruments
out = model.check_instruments(compare_ols=False)  # instant, no IV vs OLS comparison
```

- `first_stage`: how strongly the instruments move price, after the other
  covariates. An F below 10 means weak instruments: the estimates will be noisy
  and pulled toward the uncorrected ones.
- `overid`: the Sargan test, available with two or more instruments. Each
  instrument implies its own elasticity; a p-value below 0.05 means they
  disagree, so at least one may affect demand through something other than
  price.
- `rho`: the endogeneity term. An HDI excluding 0 means price responds to
  demand shocks, so the correction matters.
- `elasticity`: elasticities with instruments (`iv`) and from a refit without
  them (`ols`). Large shifts mean the instruments change the answer, which is
  reassuring only if the first two checks pass.

None of these checks whether the instruments are valid in the first place
(that they affect demand only through price). That remains an assumption.

## Sensitivity

**Question:** how strong would an unmeasured confounder have to be to erase
the price effect? This gives a robustness number rather than a yes/no answer
(Cinelli and Hazlett, 2020).

```python
model.check_sensitivity()
model.check_sensitivity(q=0.5)   # strength needed to halve the effect
```

One row per SKU plus a `pooled` row:

- `rv`: robustness value. A confounder explaining this share of the leftover
  variation in both price and demand would bring the estimate to zero (or
  reduce it by a fraction `q`). Higher is more robust.
- `rv_qa`: the same, but only until the estimate is no longer significant at
  `alpha` (0.05). Usually the more useful of the two.
- `partial_r2`: share of the leftover demand variation that price explains.
  A confounder must explain at least this much to erase the estimate.

**How to read it:** there is no universal cutoff. Ask whether a confounder as
strong as `rv_qa` is plausible, for example compared with how much your
strongest control explains. An `rv_qa` of 0 means the estimate is already not
significant.

The numbers come from a plain linear regression of log quantity on log price,
fitted separately per SKU. The model instead shares information across SKUs,
so for SKUs with little price variation the two can disagree, and those SKUs
get low robustness values. With instruments, use the
[instrument checks](#instruments) instead.
