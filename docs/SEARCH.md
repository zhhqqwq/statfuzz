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
