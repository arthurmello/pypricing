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
  covariates. `f` is the partial F. `r2_without` is the R² of log price on
  those covariates alone; `r2_with` adds the instruments. The gap is the price
  variation the instruments add. An F below 10 is a rule of thumb for weak
  instruments, not proof they are valid: the corrected elasticities can be
  more biased than a fit without instruments, so reconsider or drop them.
  A large F still does not show that an instrument affects demand only through
  price.
- `overid`: the Sargan test, available with two or more instruments. Each
  instrument implies its own elasticity; a p-value below 0.05 means they
  disagree, so at least one may affect demand through something other than
  price.
- `rho`: the endogeneity term. An HDI excluding 0 means price responds to
  demand shocks, so the correction matters. An HDI that includes 0 is no
  evidence of endogeneity; the instrumented elasticities are noisier, so
  drop the instruments. The same decision applies to copula, Lewbel, and
  latent IV: if `rho_copula` or `rho_latent` includes 0, use the uncorrected
  model. See [Identification](identification.md#without-an-external-instrument).
- `elasticity`: elasticities with instruments (`iv`) and from a refit without
  them (`ols`). Large shifts mean the instruments change the answer, which is
  reassuring only if the first two checks pass.

None of these checks whether the instruments are valid in the first place
(that they affect demand only through price). That remains an assumption.
What to write down, including when the correction is not IV, is in
[Reporting endogeneity](#reporting-endogeneity).

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

## Reporting endogeneity

A clean sample is not a report. Write three things. The first two are about
the category and the design. The third is the correction, and the library
supplies its numbers.

### Sources, and the controls that address them

Name the ways price in this category is set from demand, and the term that
absorbs each one. Typical sources, and what in the model can take them:

- A promotion timed to a soft week. If the discount is not set from the demand
  shock, it stays in `promo_elasticity_sku` and out of the instrumented price.
  If it is set from the shock, it is part of the endogenous price.
- A list price set from a forecast of demand. That is the shock the correction
  is for. A control absorbs it only when the control is the forecast, or
  something that moves with it.
- A shock shared by every SKU in a period. `time_effects="period"` (or
  `"region_period"`) absorbs it. It does not absorb a shock that hits one SKU
  and that SKU's price.
- A SKU whose baseline demand sits at a different price level from the others.
  `mundlak=True` puts that SKU's mean log price into the intercept, so the
  elasticity is not standing in for the baseline.
- Anything you measured that moves both price and demand (distribution, cost,
  season, a campaign). Those are `control_*` columns, or
  `PanelColumns.control_columns`.

Say which of these you included, and which source each one is there for.

### What remains unaddressed

Say what those controls do not absorb. A control only helps if it actually
moves with the shared shock. What is usually left:

- A SKU-and-period shock that price still reacts to. Period effects do not
  remove it. That is what an instrument, or one of the other corrections, is
  for.
- The exclusion restriction: an instrument, a Lewbel regressor, or a latent
  class has to move price without moving demand on its own. The library cannot
  test that.
- On a quadratic or sigmoid curve, the control function assumes the demand
  shock stays valid along the whole curve, not only as a shift next to a
  linear price. Fitting one of those curves with `endogeneity` set warns about
  this.

Two checks speak to the leftover, and neither shows it is gone.
`check_falsification` asks whether next period's price predicts today's
demand. `check_sensitivity` asks how strong an unmeasured confounder would
have to be, compared with your strongest control (`rv_qa`). Report them as
bounds, not as a pass.

### The correction, and its checks

Name the method you fit and why. `compare_endogeneity` is the fast check: a
row that says `not identified` is a method you did not use, and the note says
why. Then report the numbers for the method you kept. If the correction's
interval includes 0, say there is no evidence of endogeneity and that the
uncorrected model is the one you use.

**IV.** After `fit`, `check_instruments` returns the three tests:

- Instrument strength: `first_stage["f"]`, and `r2_without` against `r2_with`.
  F below 10 is weak. Say so, and do not prefer the corrected elasticities
  over the uncorrected ones on the strength of a weak instrument.
- Overidentification: `overid["p_value"]` when `overid["applicable"]` is true,
  which requires two or more instruments. A p-value below 0.05 means the
  instruments disagree. With one instrument, say the test does not apply.
- The endogeneity test: `rho["hdi"]`. An interval that excludes 0 is evidence
  that price responds to the demand shock. An interval that includes 0 is no
  evidence of endogeneity.

**Lewbel.** From `endogeneity_check(df, method="lewbel")`, report the
`heteroskedasticity` assumption (the Breusch-Pagan p-value) and the
`relevance` assumption (the first-stage F of the generated instruments). After
the fit, `summarize_posterior` gives `rho`, and that is the endogeneity test.
Do not report a Sargan p-value. `check_instruments` only sees `iv_*` columns,
not the generated instruments.

**Copula.** From `endogeneity_check(df, method="copula")`, report the
`non_normal_price` assumption. Its detail is the normality p-value of the
price residual. The correction is not identified when that residual is
consistent with a normal distribution. On a quadratic or sigmoid curve, also
report the `nonlinear_terms` assumption: a correlation above 0.8 means the
control is hard to tell apart from the bend of the curve. After the fit,
`summarize_posterior` gives `rho_copula` and, on those curves, the posterior
correlation with `curvature_sku` or `log_price_center_sku`.

**Latent IV.** Say that the method is experimental. From
`endogeneity_check(df, method="latent")`, report the `mixture` assumption:
the ELPD difference against a single normal, and its standard error. A normal
that fits as well means the correction is not identified. If the verdict is
`weak`, report the `stability` assumption: the elasticity moved across class
counts that fit about as well. After the fit, `summarize_posterior` gives
`rho_latent` and `n_classes`.

```python
checked = model.endogeneity_check(df, method="copula")
for item in checked["assumptions"].assumptions:
    print(item.name, item.status, item.detail)

summary = model.endogeneity_method_.summarize_posterior(model)
```
