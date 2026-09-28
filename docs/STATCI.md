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

- GitHub Actions job summaries;
- aggregate suites of multiple assertions;
- badge/status artifacts;
- statistical regression comparison against a baseline report.
