# StatFuzz v0.1 Design Notes

## 1. Product question

StatFuzz is not a replacement for SciPy or statsmodels. It is a testing layer around statistical procedures.

The core question is:

> Under a controlled data-generating process, does a statistical procedure exhibit the property we expect at a finite sample size?

v0.1 answers that question only for Welch's two-sample t-test and empirical Type-I error.

## 2. Core abstractions

### DataGenerator

A DGP owns the assumptions about how observations are generated. It must expose a deterministic `sample(rng, n)` method and a readable name.

### Statistical method

A method maps generated data to a statistic, p-value, interval, prediction, or other output. v0.1 implements Welch's two-sided t-test directly and cross-checks it against SciPy.

### Metric / property

A metric aggregates repeated method outputs into a property such as Type-I error, coverage, bias, power, or calibration. v0.1 implements empirical Type-I error and its Monte Carlo standard error.

### Result

`StressTestResult` is intentionally data-oriented. Later report renderers should consume this object rather than making simulation code responsible for presentation.

## 3. Statistical contract for Type-I error

A Type-I error experiment is meaningful only when the simulated data satisfy the null hypothesis.

For Welch's test in v0.1, the target null is equality of population means. Therefore each built-in DGP exposes an explicit arithmetic mean. The shifted log-normal generator subtracts the raw log-normal theoretical mean before adding the requested target mean.

This avoids a common simulation mistake: changing skewness or scale while accidentally changing the null quantity being tested.

## 4. Monte Carlo uncertainty

For `B` independent simulation replicates and empirical rejection rate `p_hat`, v0.1 reports

`MCSE = sqrt(p_hat * (1 - p_hat) / B)`.

The `OUTSIDE_TOLERANCE` label is not a formal inferential decision. It only means that the observed empirical property differs from the nominal target by more than a user-chosen engineering tolerance.

Future versions should support uncertainty-aware criteria explicitly.

## 5. Reproducibility

Every public simulation API accepts a seed. A seed fixes the pseudo-random experiment so examples, bug reports, and CI failures can be reproduced.

For scientific reporting, users should still re-run important findings with larger simulation budgets and independent seeds.

## 6. Why automatic search is not in v0.1

Searching many DGP parameters and reporting only the worst case introduces a selection problem: the same Monte Carlo noise used to search can exaggerate the apparent severity of the discovered counterexample.

v0.2/v0.3 should therefore distinguish:

1. a **search budget** used to locate candidate failure regions; and
2. an independent **validation budget** used to re-estimate the final candidate.

This search/validation split is part of the intended statistical design, not an implementation detail.

## 7. Planned evolution

```text
v0.1  DGP -> simulate -> method -> metric -> result
v0.2  + parameter spaces + search tables
v0.3  + candidate counterexample discovery + hold-out validation
v0.4  + counterexample shrinking
v0.5  + failure maps + reports
v0.6  + CI-friendly statistical assertions
```
