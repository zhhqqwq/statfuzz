# Counterexample shrinking

## Goal

Search and validation can identify a parameter point with unusual finite-sample behavior, but the first discovered point may be unnecessarily complicated.

Shrinking asks:

> Can we move the discovered point toward a simpler description while preserving an explicit failure criterion?

StatFuzz v0.4 implements deterministic greedy shrinking over user-declared simplicity levels.

## Explicit simplicity levels

StatFuzz does not guess that a numerically smaller value is always simpler.

Instead, the user declares the order explicitly:

```python
from statfuzz.search import ShrinkDimension, ShrinkPlan

plan = ShrinkPlan(
    (
        ShrinkDimension("n", (4, 8, 12, 20)),
        ShrinkDimension("sigma", (0.4, 0.8, 1.0, 1.4)),
    )
)
```

Each tuple is ordered from **simplest to most complex**.

This makes the interpretation domain-specific. For sample size, smaller values may be simpler. For another parameter, the simplest value may instead be a canonical baseline such as zero, one, or a model-specific reference point.

## Failure criteria

The default criterion is:

```python
OutsideToleranceCriterion()
```

It preserves the condition represented by `StressTestResult.status == "OUTSIDE_TOLERANCE"`.

A more explicit alternative is an objective threshold:

```python
ObjectiveThresholdCriterion(
    objective="absolute_deviation",
    minimum_score=0.02,
)
```

This accepts a proposal only when its re-evaluated objective score remains at or above the threshold.

## Algorithm

Given a starting ParameterPoint:

1. Re-evaluate the starting point with the shrink-stage root seed.
2. Reject shrinking if the starting point no longer satisfies the failure criterion.
3. Visit shrink dimensions in the explicit plan order.
4. For each dimension, try simpler levels from simplest upward.
5. Re-simulate every proposed full ParameterPoint.
6. Accept the first proposal that still satisfies the failure criterion.
7. Repeat passes until no dimension can be simplified further.

Every accepted move strictly decreases the plan-relative complexity score, so the process terminates.

The algorithm is greedy and deterministic. It produces a locally simplified point under the declared plan, evaluation budget, seed, and criterion. StatFuzz does **not** claim mathematical minimality.

## Reproducibility

A proposal receives a deterministic seed derived from:

```text
shrink root seed + canonical full ParameterPoint
```

The same proposal therefore receives the same Monte Carlo random stream if encountered again in the same shrink run.

The evaluator must:

- use the provided seed;
- use the provided simulation budget;
- return both values accurately in StressTestResult.

StatFuzz checks these contracts.

## Trace

`CounterexampleShrinkResult.steps` stores every accepted and rejected proposal.

Each ShrinkStep includes:

```text
attempt
pass index
parameter
from value
to value
full proposed ParameterPoint
seed
simulation budget
empirical estimate
MCSE
criterion
accepted / rejected
```

The trace is intended to make the simplification process auditable rather than presenting only the final point.

## Current scope

The first shrinking layer works on scalar parameters already represented inside ParameterPoint. This covers sample sizes and numeric distribution parameters, and can also represent discrete scalar choices when the user supplies an ordering.

Automatic **distribution-family simplification**—for example replacing a mixture distribution with a single-component family—is not implemented yet. That requires a richer representation than scalar ParameterPoint replacement and remains a later v0.4 milestone.
