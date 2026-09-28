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
