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

grid_search returns every evaluated point and ranks through a SearchObjective.

Built-in objectives are:

```text
absolute_deviation = abs(empirical - nominal)
positive_deviation = empirical - nominal
negative_deviation = nominal - empirical
```

Custom objectives can implement a name and score(StressTestResult) method. Larger objective scores always rank first. Every search row records both the objective name and its score.

This ranking is exploratory. Searching many points and selecting the largest observed deviation can select Monte Carlo noise as well as genuine finite-sample behavior.

The validation layer therefore re-runs a selected candidate with an independent validation seed and a separate, usually larger, simulation budget. Search and validation estimates are reported separately.


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

The high-level find_counterexample API combines candidate discovery and this validation stage while keeping the two estimates distinct.


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
from statfuzz.search import DiscoveryBudget, find_counterexample

budget = DiscoveryBudget(
    search_simulations=2_000,
    validation_simulations=20_000,
)

discovery = find_counterexample(
    space=space,
    search_evaluate=search_evaluate,
    validation_evaluate=validation_evaluate,
    budget=budget,
    search_root_seed=42,
    validation_root_seed=2026,
    objective="absolute_deviation",
)
```

The function accepts any supported SearchObjective, selects rank 0 from the complete grid search under that objective, and independently validates that exact ParameterPoint. Its DiscoveryBudget is executable: search and validation simulation counts are passed into the corresponding evaluators, and StatFuzz checks that each returned StressTestResult reports the requested simulation count.

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


## Discovery budgets

`DiscoveryBudget` makes the two Monte Carlo stages explicit:

```python
from statfuzz.search import DiscoveryBudget

budget = DiscoveryBudget(
    search_simulations=2_000,
    validation_simulations=20_000,
)
```

High-level discovery evaluators receive three arguments:

```python
def search_evaluate(point, seed, simulations):
    return stress_test(
        ...,
        simulations=simulations,
        seed=seed,
    )
```

The same contract applies to the validation evaluator. StatFuzz rejects a result whose `StressTestResult.simulations` differs from the requested budget. This prevents documentation or metadata from claiming one Monte Carlo budget while the evaluator silently runs another.


## Random search without replacement

`random_search` targets finite ParameterSpace objects that are too large to evaluate exhaustively.

```python
from statfuzz.search import random_search

search = random_search(
    space=space,
    evaluate=evaluate,
    draws=100,
    seed=42,
    objective="absolute_deviation",
)
```

It has four important properties:

1. **No replacement.** A ParameterPoint is evaluated at most once within one random-search run.
2. **Seed reproducibility.** The root seed deterministically controls the sampled flat indices.
3. **Stable per-point evaluation seeds.** Once a point is selected, its evaluation seed is derived from the root seed and canonical point JSON, using the same rule as grid search.
4. **No full-space materialization.** ParameterSpace.point_at(index) converts a flat Cartesian-product index to a ParameterPoint in O(number_of_parameters) time.

The sampler uses a sparse partial Fisher-Yates shuffle. It stores only O(draws) swap entries instead of allocating an array of size len(space). The current unbiased index generator uses 64-bit draws, so spaces larger than 2**64 points are rejected explicitly.

`RandomSearchResult` extends the common `SearchResult` interface and adds:

```text
sampled_indices
space_size
draws
coverage_fraction
```

Because both GridSearchResult and RandomSearchResult share SearchResult, `validate_candidate` accepts either one.

### Random search in high-level discovery

`find_counterexample` remains exhaustive by default. Pass `search_draws` to switch its search stage to random search while keeping the same SearchObjective and DiscoveryBudget:

```python
discovery = find_counterexample(
    space=space,
    search_evaluate=search_evaluate,
    validation_evaluate=validation_evaluate,
    budget=budget,
    search_root_seed=42,
    validation_root_seed=2026,
    objective="absolute_deviation",
    search_draws=100,
)
```

The search simulation budget still applies to every sampled point, and the independent validation budget still applies only to the selected candidate.
