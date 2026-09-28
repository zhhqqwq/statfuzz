# StatCI

StatCI turns an existing StatFuzz statistical result into a continuous-integration assertion.

The first v0.6 rule is deliberately simple and explicit:

```text
PASS  iff  abs(observed - target) <= tolerance
```

This is an engineering tolerance check. It is **not** a new hypothesis test.

Monte Carlo standard error (MCSE) is retained in the evidence record, but it does not silently widen or narrow the user-specified tolerance.

## Pytest-style assertion

```python
from statfuzz import assert_property

assert_property(
    result,
    property="type1_error",
    target=0.05,
    tolerance=0.01,
)
```

On PASS, `assert_property` returns a `StatCIResult`.

On FAIL, it raises `StatisticalAssertionError`, which subclasses Python's `AssertionError`. Pytest therefore reports it as a normal failed assertion.

The error message includes the observed value, target, signed deviation, absolute deviation, tolerance, MCSE, simulation budget, and seed.

## Non-raising checks

CI integrations often need to collect several checks before deciding the process exit status. Use `check_property` for that:

```python
from statfuzz import check_property

ci_result = check_property(
    result,
    property="type1_error",
    target=0.05,
    tolerance=0.01,
)

print(ci_result.status)
```

`check_property` returns either `PASS` or `FAIL` without raising for statistical failure.

Configuration errors still raise normally. For example, the requested property must match `StressTestResult.metric`.

## Reusable assertion definitions

```python
from statfuzz import StatisticalAssertion

type1_error_contract = StatisticalAssertion(
    property="type1_error",
    target=0.05,
    tolerance=0.01,
)

result_a = type1_error_contract.evaluate(experiment_a)
result_b = type1_error_contract.evaluate(experiment_b)
```

This separates the statistical contract from any one experiment.

## Machine-readable result

Every check produces a `StatCIResult` with schema version `1.0`.

```python
ci_result.write_json("statci-result.json")
```

The JSON includes:

```text
schema_version
property
target
tolerance
observed
deviation
absolute_deviation
passed
status
evidence
  method
  metric
  dgp1 / dgp2
  n1 / n2
  simulations
  seed
  mcse
```

Serialization is deterministic for the same result object: keys are sorted and no wall-clock timestamp is injected.

## Why MCSE is evidence rather than the rule

Suppose the assertion is:

```text
target = 0.05
tolerance = 0.01
```

and the observed empirical Type-I error is `0.061`.

The first StatCI contract fails because:

```text
abs(0.061 - 0.05) = 0.011 > 0.01
```

That decision is the same whether MCSE is tiny or large. MCSE is still reported so reviewers can judge Monte Carlo precision.

Future StatCI rules may explicitly model uncertainty, but they must be opt-in rules with separate semantics rather than hidden changes to `tolerance`.

## Current scope

Implemented in this node:

- `StatisticalAssertion`;
- `StatCIResult`;
- `check_property(...)`;
- `assert_property(...)`;
- `StatisticalAssertionError`;
- deterministic JSON output.

Deferred to later v0.6 nodes:

- badge/status artifacts;
- statistical regression comparison against a baseline report.


## Aggregating multiple checks

Use `StatCISuiteResult` to preserve a set of individual `StatCIResult` values and derive one overall CI status:

```python
from statfuzz import StatCISuiteResult, check_property

suite = StatCISuiteResult.from_results(
    [
        check_property(
            result_a,
            property="type1_error",
            target=0.05,
            tolerance=0.01,
        ),
        check_property(
            result_b,
            property="type1_error",
            target=0.05,
            tolerance=0.01,
        ),
    ],
    name="Core statistical checks",
)
```

The suite is PASS only when every child result passes.

It exposes:

```text
suite.total
suite.passed_count
suite.failed_count
suite.passed
suite.status
```

and deterministic machine-readable output:

```python
suite.write_json("statci-suite.json")
```

The suite JSON embeds the complete child StatCIResult payloads rather than reducing them to booleans.

## GitHub Actions job summary

```python
from statfuzz import write_github_summary

write_github_summary(suite)
```

Inside GitHub Actions, `write_github_summary` reads `GITHUB_STEP_SUMMARY` and **appends** a Markdown block.

The summary includes:

- overall PASS / FAIL;
- passed / failed / total counts;
- a table of every check;
- observed value;
- target;
- tolerance;
- signed deviation;
- MCSE;
- simulation budget;
- seed;
- a focused list of failed checks.

When running outside GitHub Actions, an explicit path can be supplied:

```python
write_github_summary(suite, "statci-summary.md")
```

## Recommended CI order

Write evidence before making the process fail:

```python
from statfuzz import assert_suite, write_github_summary

write_github_summary(suite)
suite.write_json("statci-suite.json")
assert_suite(suite)
```

If the suite failed, `assert_suite` raises `StatCISuiteError`, which subclasses `AssertionError` and carries the full suite as `error.suite`.

This ordering means the GitHub job summary has already been written when the final assertion marks the step as failed.

A minimal Actions step can therefore look like:

```yaml
- name: Run statistical CI checks
  run: python examples/statci_suite.py
```

In a production repository, the script would normally save the suite JSON to a known artifact path before calling `assert_suite`.

## Current v0.6 status

Implemented:

- single statistical assertions;
- machine-readable StatCIResult;
- StatCISuiteResult aggregation;
- deterministic suite JSON;
- GitHub Actions Markdown summaries;
- overall suite PASS / FAIL assertion.

Still deferred:

- statistical regression comparison against a baseline;
- more advanced uncertainty-aware assertion rules.


## Status artifact and badge JSON

A full `StatCISuiteResult` is useful for debugging and audit trails, but many CI consumers only need a small overall status document.

```python
from statfuzz import StatCIStatusArtifact

status = StatCIStatusArtifact.from_suite(suite)

status.write_json("statci-status.json")
status.write_badge_json("statci-badge.json")
```

The lightweight status JSON contains:

```text
schema_version
suite_name
status
passed
checks
  total
  passed
  failed
badge
```

It intentionally does not duplicate the complete child `StatCIResult` payloads already stored in `statci-suite.json`.

### Shields endpoint badge payload

`statci-badge.json` follows the Shields endpoint-badge response shape:

```json
{
  "schemaVersion": 1,
  "label": "StatCI",
  "message": "PASS · 3/3",
  "color": "brightgreen"
}
```

A failing suite uses a red badge and a message such as `FAIL · 2/3`.

A custom label can be supplied:

```python
status = StatCIStatusArtifact.from_suite(
    suite,
    badge_label="statistical CI",
)
```

The badge file itself is only JSON. To render it as a live endpoint badge, publish it at a stable public URL that the badge service can fetch. A private GitHub Actions artifact is useful for workflow/download purposes, but it is not by itself a public badge endpoint.

## Uploading CI artifacts

Generate evidence before the final gate:

```python
suite.write_json("statci-suite.json")

status = StatCIStatusArtifact.from_suite(suite)
status.write_json("statci-status.json")
status.write_badge_json("statci-badge.json")

write_github_summary(suite)
assert_suite(suite)
```

Then upload the files even when the statistical gate fails:

```yaml
- name: Run StatCI
  run: python path/to/statci_checks.py

- name: Upload StatCI artifacts
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: statci
    path: |
      statci-suite.json
      statci-status.json
      statci-badge.json
```

Because the files are written before `assert_suite`, the `if: always()` upload step can still preserve them after a failing StatCI run.

## Current v0.6 status

Implemented:

- single statistical assertions;
- machine-readable `StatCIResult`;
- `StatCISuiteResult` aggregation;
- deterministic suite JSON;
- GitHub Actions Markdown summaries;
- lightweight status artifact;
- Shields endpoint-compatible badge JSON;
- overall suite PASS / FAIL assertion.

Still deferred:

- statistical regression comparison against a baseline;
- more advanced uncertainty-aware assertion rules.


## Baseline statistical regression comparison

A CI run may still PASS its absolute tolerance contract while becoming meaningfully worse than a previously accepted baseline. StatCI therefore supports a second, separate question:

> Did the current statistical behavior deteriorate beyond both an engineering worsening threshold and the Monte Carlo uncertainty guard?

This comparison does **not** call every raw Monte Carlo difference a regression.

### Matching rules

Baseline and current checks are matched by a stable comparison key containing:

```text
property
target
tolerance
method
metric
dgp1
dgp2
n1
n2
```

The key intentionally excludes:

```text
seed
simulation budget
observed estimate
MCSE
```

because those are run-specific evidence rather than experiment identity.

By default matching is strict. A missing baseline check, a new current check, or a duplicate comparison key is treated as a configuration error rather than silently paired or ignored.

```python
from statfuzz import StatCISuiteResult

baseline = StatCISuiteResult.read_json("baseline-statci-suite.json")
current = StatCISuiteResult.read_json("current-statci-suite.json")
```

Suite JSON is validated while loading: schema version, child result consistency, PASS/FAIL, deviations, seeds, and counts must agree with the serialized evidence.

### What counts as worsening?

For one matched check, StatCI compares distance from the assertion target:

```text
worsening
=
current.absolute_deviation
-
baseline.absolute_deviation
```

This matters because an observed estimate can move numerically while actually moving **closer** to the target.

A regression is declared only when:

```text
worsening
>
minimum_worsening
+
uncertainty_multiplier × uncertainty_scale
```

The boundary is strict: equality does not count as a regression.

### Conservative uncertainty mode

The default policy is:

```python
from statfuzz import RegressionPolicy

policy = RegressionPolicy(
    minimum_worsening=0.0,
    uncertainty_multiplier=2.0,
    uncertainty_mode="conservative",
)
```

The conservative uncertainty scale is:

```text
baseline.mcse + current.mcse
```

This is deliberately called an **uncertainty scale**, not the MCSE of the difference. It avoids assuming the baseline and current Monte Carlo estimates are independent.

The default multiplier of 2.0 is an engineering guard band. StatFuzz does not present it as a formal 95% hypothesis test or confidence interval.

### Independent-stream mode

When baseline and current were generated with known, distinct random seeds, an independent-stream approximation can be requested:

```python
policy = RegressionPolicy(
    uncertainty_mode="independent",
    uncertainty_multiplier=2.0,
)
```

The uncertainty scale then becomes:

```text
sqrt(baseline.mcse^2 + current.mcse^2)
```

StatFuzz requires both seeds to be known and different before using this mode. Distinct seeds are an explicit engineering assumption about separate Monte Carlo streams; the mode should not be used when covariance between runs is intentionally introduced.

### PASS→FAIL is metadata, not an override

A current result can cross the absolute assertion boundary while the baseline-to-current worsening remains small relative to Monte Carlo uncertainty.

StatCI records:

```text
pass_to_fail
fail_to_pass
```

separately from the regression decision.

That means a PASS→FAIL transition is visible in the report, but the baseline regression gate still follows the explicit regression policy instead of silently bypassing the uncertainty guard.

### Regression suite

```python
from statfuzz import compare_suites

regression = compare_suites(
    baseline,
    current,
    policy=policy,
    name="PR statistical regression",
)
```

Each matched comparison is classified as one of:

```text
IMPROVED
STABLE
WITHIN_GUARD
REGRESSION
```

The aggregate `StatCIRegressionSuiteResult` exposes:

```text
status
passed
total
regression_count
pass_to_fail_count
improvement_count
missing_current
new_current
```

and deterministic JSON:

```python
regression.write_json("statci-regression.json")
```

### GitHub Actions summary and gate

```python
from statfuzz import (
    assert_regression_suite,
    write_regression_summary,
)

write_regression_summary(regression)
regression.write_json("statci-regression.json")
assert_regression_suite(regression)
```

The Markdown summary includes the baseline/current absolute deviations, worsening, uncertainty scale, guard threshold, direction, and PASS→FAIL transitions.

As with the ordinary StatCI suite, write the summary and JSON before calling the final assertion so regression evidence remains available even when CI fails.

### Current v0.6 status

Implemented:

- single statistical assertions;
- machine-readable `StatCIResult`;
- `StatCISuiteResult` aggregation;
- deterministic suite JSON;
- GitHub Actions Markdown summaries;
- lightweight status and badge artifacts;
- uncertainty-aware baseline regression comparison;
- regression JSON / GitHub summary / CI gate.

The main v0.6 roadmap is now complete. Future extensions may add richer uncertainty-aware assertion families, baseline management workflows, or repeated-run models.
