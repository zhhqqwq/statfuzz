# Reporting

StatFuzz v0.5 uses one report model for machine-readable JSON, two-dimensional failure maps, and standalone HTML.

## Architecture

```text
SearchResult / Validation / Shrink results
                 |
                 v
          StatFuzzReport
          /      |      \
         /       |       \
      JSON   FailureMap   HTML
```

The report layer never reruns simulations. It freezes existing result objects into a deterministic snapshot.

## Build a report

```python
from statfuzz.report import build_report, failure_map_2d

failure_map = failure_map_2d(
    search=search,
    x_parameter="n",
    y_parameter="sigma",
)

report = build_report(
    title="Welch t-test stress report",
    search=search,
    validation=validation,
    scalar_shrink=scalar_shrink,
    family_shrink=family_shrink,
    failure_map=failure_map,
)
```

All optional sections are omitted as `null` when they are not supplied.

## Stable JSON schema

Every report contains a `schema_version`. The current schema version is `1.2`.

```python
report.write_json("statfuzz-report.json")
```

The JSON serializer:

- uses deterministic key ordering;
- does not add a wall-clock timestamp;
- rejects NaN/Infinity through the underlying result contracts;
- stores search strategy, root seed, objective, parameters, seeds, estimates, MCSE, status, validation, shrinking traces, and failure-map data;
- stores each DGP's human-readable name separately from its structured machine identity (family + full-precision scalar parameters).

This makes identical result objects produce identical JSON content.

## Failure maps

A 2D failure map is a view of existing search records.

```python
failure_map = failure_map_2d(
    search=search,
    x_parameter="n",
    y_parameter="sigma",
)
```

For search spaces with more than two parameters, all remaining dimensions must be fixed explicitly:

```python
failure_map = failure_map_2d(
    search=search,
    x_parameter="n",
    y_parameter="sigma",
    fixed={"variance_ratio": 2.0},
)
```

This prevents accidental projection of multiple statistical experiments onto the same 2D coordinate.

Each covered cell stores:

```text
x / y values
full parameter point
objective score
empirical estimate
nominal target
deviation
MCSE
simulation budget
seed
status
```

The uncertainty overlay is currently MCSE. HTML reports display MCSE directly in each cell rather than implying a stronger interval interpretation.

Random-search maps may be sparse. Unobserved cells remain explicit `null` values in JSON and appear as `Missing` in HTML.

## Standalone HTML

```python
from statfuzz.report import write_html

write_html(report, "statfuzz-report.html")
```

The renderer has no additional plotting or frontend dependency. It includes:

- report/search summary;
- independent validation comparison;
- 2D failure-map table;
- MCSE uncertainty in every covered map cell;
- scalar and family shrink traces;
- complete search-record table.

User-provided strings are HTML-escaped.

## Interpretation

A failure map visualizes the experiments that were actually evaluated. It does not fill missing random-search cells by interpolation.

Likewise, the report preserves the existing distinction between exploratory search and independent validation. Rendering a candidate in a polished HTML page does not strengthen the underlying statistical evidence.


## Search multiplicity metadata

Every report now includes a `search.multiplicity` snapshot for the top-ranked search candidate.

It records the number of evaluated selection opportunities, tie-aware empirical rank information, objective-score distribution summaries, and the fraction of evaluated points outside the engineering tolerance.

The HTML renderer shows this information in a dedicated **Search Multiplicity** section and explicitly labels it exploratory. The empirical percentile and upper-tail fraction are not p-values.

When a `CandidateValidationResult` is supplied to `build_report`, the report also contains `selection_effect` with the search objective score, independent validation objective score, their gap, both MCSE values, both simulation budgets, and both stage statuses.

A positive search-minus-validation gap is evidence that the independent re-run was less extreme on the chosen objective. The report does not label that gap as an unbiased winner's-curse estimate.


## Structured DGP identity

Report schema 1.2 adds `dgp1_identity` and `dgp2_identity` to every stress-test
snapshot. Display names such as `Normal(mean=0, sd=1)` remain for readability, but
machine consumers should use the structured identity when deciding whether two
experiments are the same.

For example, `sd=1.0000001` and `sd=1.0000002` may round to the same display name,
but their identity parameter values remain distinct in JSON.
