# Search design

## Why ParameterSpace does not simulate data

ParameterSpace represents only a finite Cartesian product of named candidate values.

It intentionally knows nothing about data-generating processes, sample sizes, statistical methods, metrics, or Monte Carlo simulation. The statistical experiment is supplied through an evaluator:

```python
def evaluate(point, seed):
    params = point.as_dict()
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=params["sigma"]),
        n1=params["n"],
        n2=params["n"],
        simulations=2_000,
        seed=seed,
    )
```

This keeps the search layer reusable for future coverage, bias, power, calibration, regression, bootstrap, and causal-inference experiments.

## Deterministic point order

Parameter names are sorted canonically and candidate order within each axis is preserved. Constructing the same logical mapping with a different key insertion order therefore produces the same point order.

## Deterministic per-point seeds

A grid-level root seed is combined with the canonical JSON representation of each ParameterPoint using BLAKE2b.

This means:

1. the same root seed and same point produce the same child seed;
2. different points receive distinct deterministic streams in ordinary use;
3. reordering the grid does not silently change the random stream assigned to an existing point.

The evaluator is required to pass this child seed into stress_test.

## Search result versus statistical confirmation

grid_search returns every evaluated point and can rank them by

```text
absolute_deviation = abs(empirical - nominal)
```

This ranking is exploratory. Searching many points and selecting the largest observed deviation can select Monte Carlo noise as well as genuine finite-sample behavior.

A later validation layer (Issue #3) will therefore re-run a selected candidate with an independent validation seed and a separate, usually larger, simulation budget. Search and validation estimates should be reported separately.


## Independent hold-out validation

A point selected because it looked extreme during the search stage has been selected using the same Monte Carlo noise that produced its estimate. StatFuzz therefore treats the search estimate as exploratory.

The validation API re-runs the exact same ParameterPoint with a different root seed and a new deterministic per-point child seed:

```python
from statfuzz.search import validate_candidate

validated = validate_candidate(
    search=search,
    evaluate=validation_evaluate,
    validation_root_seed=2026,
)
```

The validation evaluator is separate from the search evaluator on purpose. This lets the validation stage use a larger simulation budget without modifying the search record.

A CandidateValidationResult stores both stages:

```text
candidate point
search root seed
validation root seed
search child seed
validation child seed
search simulation budget
validation simulation budget
search empirical estimate + MCSE
validation empirical estimate + MCSE
```

The API rejects reuse of the same root seed. It also verifies that the validation evaluator passes the provided child seed through to the underlying stress test.

### Interpretation

Independent validation reduces the risk that a candidate looks extreme only because it was selected as the maximum over noisy search estimates. It does not by itself prove a universal mathematical counterexample. Important findings should still be interpreted in terms of the chosen DGP family, parameter space, simulation budget, uncertainty, and failure criterion.

A future find_counterexample API will combine candidate discovery and this validation stage while keeping the two estimates distinct.


## High-level counterexample discovery

The low-level pieces can be composed manually:

```text
ParameterSpace
    |
    v
grid_search
    |
    v
ranked candidate
    |
    v
validate_candidate
```

For the common case, `find_counterexample` now performs that orchestration while preserving both underlying result objects:

```python
from statfuzz.search import find_counterexample

discovery = find_counterexample(
    space=space,
    search_evaluate=search_evaluate,
    validation_evaluate=validation_evaluate,
    search_root_seed=42,
    validation_root_seed=2026,
)
```

The function currently uses the documented `absolute_deviation` objective, selects rank 0 from the complete grid search, and independently validates that exact ParameterPoint.

The returned `CounterexampleDiscoveryResult` contains:

- the full `GridSearchResult`, including every evaluated point;
- the `CandidateValidationResult`;
- direct access to the selected point;
- direct access to the search-stage and validation-stage StressTestResult objects.

This high-level API is intentionally an orchestration layer. It does not hide the search table, merge the two Monte Carlo estimates, or convert a simulation result into a mathematical proof.

### Current interpretation boundary

At this stage, "counterexample" means:

> a parameter point within the explicitly searched DGP family that was selected by the search objective and then re-estimated using an independent validation random stream.

It does not mean that StatFuzz has proved the statistical method fails for every related distribution, every nearby parameter point, or an unrestricted population class. Shrinking, broader search strategies, and richer uncertainty reporting are later roadmap items.
