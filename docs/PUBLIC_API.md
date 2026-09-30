# Public API

StatFuzz is a pre-1.0 package. This document defines the supported import surface for the
first public release candidate.

## What counts as public?

A name is part of the supported Python API when it is exported through the `__all__`
of one of the documented namespaces below.

Direct imports from implementation modules such as
`statfuzz.report.model`, `statfuzz.search.grid`, or `statfuzz.statci.regression`
may work, but are not compatibility promises unless the same name is also exported
from a documented public namespace.

This keeps implementation modules free to evolve while making the user-facing surface
auditable.

## `statfuzz`

The package root intentionally contains only the core simulation API and common StatCI
workflow.

Core:

- `__version__`
- `BootstrapCoverageProgress`
- `BootstrapCoverageResult`
- `bootstrap_mean_coverage`
- `stress_test`
- `StressTestResult`
- `MeanEqualityNull`
- `MeanNullCheck`
- `MeanTarget`
- `MeanTargetCheck`

Common StatCI workflow:

- `StatisticalAssertion`
- `StatisticalAssertionError`
- `StatCIResult`
- `StatCISuiteResult`
- `StatCISuiteError`
- `StatCIStatusArtifact`
- `RegressionPolicy`
- `StatCIRegressionError`
- `check_property`
- `assert_property`
- `assert_suite`
- `compare_suites`
- `assert_regression_suite`
- `write_github_summary`
- `write_regression_summary`

Advanced schema constants, pairwise regression result types, comparison keys, and pure
Markdown renderers live in `statfuzz.statci` instead of being duplicated at the package
root.

## `statfuzz.dgp`

The built-in DGP namespace is public:

- `DataGenerator`
- `DGPIdentity`
- `Normal`
- `LogNormal`
- `StudentT`
- `MixtureNormal`

The mean-preserving semantics documented for these generators are part of their public
behavior. `DGPIdentity` is the supported structured identity type for custom generators
that need stable machine-readable matching across runs.

## `statfuzz.methods`

The methods namespace exposes stable statistical-method contracts and reference
operations:

- `BootstrapMeanPercentile`
- `bootstrap_mean_percentile_child_seed`
- `bootstrap_mean_percentile_child_rng`
- `bootstrap_mean_percentile_interval`
- `welch_ttest_pvalue`
- `welch_ttest_pvalues_batch`

`BootstrapMeanPercentile` is an immutable method configuration. Its canonical
machine identity includes the fixed method name and semantics version together
with `resamples`, `interval_level`, and `quantile_method`.

## `statfuzz.search`

The search namespace is public. It contains the finite parameter-space model, search
strategies, discovery workflow, validation, shrinking, objectives, and multiplicity
diagnostics.

The supported names are the values of `statfuzz.search.__all__`. Important entry points
include:

- `ParameterPoint`, `ParameterSpace`
- `grid_search`, `random_search`
- `SearchResult`, `GridSearchResult`, `RandomSearchResult`
- `DiscoveryBudget`, `find_counterexample`
- `validate_candidate`
- `shrink_counterexample`, `shrink_dgp_family`
- `summarize_search_multiplicity`, `summarize_selection_effect`

Result/trace dataclasses exported by this namespace are also public because users need
them to inspect and serialize a search workflow.

## `statfuzz.report`

The report namespace deliberately exposes workflow-level objects only:

- `REPORT_SCHEMA_VERSION`
- `StatFuzzReport`
- `FailureMap2D`
- `build_report`
- `failure_map_2d`
- `render_html`
- `write_html`

Snapshot implementation dataclasses such as `StressTestSnapshot` and
`SearchRecordSnapshot` are internal details. The machine-readable report schema is the
compatibility boundary for serialized reports.

## `statfuzz.statci`

The StatCI namespace is the full advanced API. In addition to the common names re-exported
at `statfuzz`, it exposes:

- schema-version constants;
- `StatCIComparisonKey`;
- `StatCIRegressionResult` and `StatCIRegressionSuiteResult`;
- `compare_results`;
- `render_github_summary` and `render_regression_summary`.

Use this namespace when building integrations rather than relying on private module paths.

## Serialization schemas

Package version and serialization schema versions are deliberately separate.

A package patch release may preserve the Python API while incrementing a JSON report
schema if a machine-readable format needs an incompatible change. Consumers should read
the schema-version field rather than infer it from `statfuzz.__version__`.

Current schema constants are exposed from their owning namespaces.

## Pre-1.0 compatibility promise

StatFuzz follows a SemVer-inspired pre-1.0 policy:

- patch releases should not intentionally break names documented here;
- minor releases may make documented breaking changes while the API is still alpha;
- reasonable deprecation periods are preferred when practical;
- all intentional public-API changes must appear in `CHANGELOG.md`.

The exact exported-name sets are covered by `tests/test_public_api.py`, so accidental
growth or removal of the public surface fails CI.


## Type-I null contract

For the current Welch Type-I error experiment, built-in DGPs expose a finite
`population_mean` and StatFuzz verifies equality before simulation.

Custom DGPs may also expose `population_mean`. If they cannot, callers must pass
`MeanEqualityNull(mean=..., note=...)` to make the unverified analytical
assumption explicit. The resulting `MeanNullCheck` is stored in
`StressTestResult`, StatCI evidence, and reports.


## Mean-target truth contract

Mean-coverage experiments use a truth contract that is separate from the
Type-I null contract.

Built-in DGPs expose a finite `population_mean`, which can be verified
automatically. Custom DGPs that do not expose a population mean must be paired
with an explicit `MeanTarget(mean=..., note=...)`. The resolved
`MeanTargetCheck` records whether the truth came from a verified DGP
population mean or from an explicit caller declaration.


## Scalar bootstrap percentile reference oracle

`bootstrap_mean_percentile_interval(sample, rng, method)` is the scalar
reference implementation for the first bootstrap method. It consumes a caller
provided NumPy `Generator` and, for each configured bootstrap replicate,
performs exactly one `rng.integers(0, n, size=n)` replacement-index draw,
computes that resampled mean, and only after all replicate means are complete
evaluates the two percentile quantiles using the method's locked
`quantile_method="linear"`.

The function does not derive seeds or create child generators. RNG ownership
and child-stream derivation are separate concerns for a later phase.


## Bootstrap child RNG derivation

`bootstrap_mean_percentile_child_seed(root_seed, logical_outer_index)`
deterministically maps one experiment root seed and one logical outer replicate
index to a bootstrap child seed. The mapping uses a versioned canonical JSON
payload containing:

- domain = `statfuzz.bootstrap_mean_percentile.child_rng`;
- RNG semantics version = `1`;
- root seed;
- logical outer index.

The payload is hashed with BLAKE2b using a 128-bit digest and fixed
personalization `statfuzz-bsprng1`. The digest is interpreted as an unsigned
big-endian integer.

`bootstrap_mean_percentile_child_rng(...)` always constructs
`numpy.random.Generator(numpy.random.PCG64(child_seed))`; it never calls
`default_rng()`.

This makes child-stream assignment a pure function of root seed and logical
outer index. Request order and consumption of one child stream cannot perturb
another child stream.


## Bootstrap coverage result contract

`BootstrapCoverageResult` is the public aggregate result contract for the
future percentile-bootstrap mean coverage executor. It satisfies the same
`StatisticalPropertyResult` shape used by search and shrinking while keeping
coverage-specific evidence explicit:

- the configured `BootstrapMeanPercentile` method;
- resolved `MeanTargetCheck`;
- coverage count and Bernoulli-rate evidence;
- bootstrap interval level as the nominal property target;
- Monte Carlo evidence confidence level and Wilson interval;
- DGP identity, sample size, root seed, and engineering tolerance.

Phase C1 does not yet provide a public outer simulation function. The
single-replicate event helper remains an implementation-level reference
operation until the coverage executor is introduced.


## Bootstrap mean coverage executor

`bootstrap_mean_coverage(...)` supports outer execution batching through
`batch_size` (default `64`). Batching changes scheduling only; bootstrap
resampling remains scalar and there is no bootstrap vectorization.

For logical outer replicate `i`, execution semantics remain:

```text
outer Generator(PCG64(root_seed))
-> dgp.sample(...)
-> deterministic bootstrap child_rng(i)
-> scalar percentile interval oracle
-> inclusive coverage event
-> cumulative coverage_count
```

The outer data Generator is explicitly `PCG64` and is separate from every
bootstrap child Generator. Therefore changing bootstrap resample count cannot
change the outer DGP sample sequence.

The executor resolves `MeanTarget` and validates `batch_size` before the
first draw, then returns a `BootstrapCoverageResult` backed by generic
binomial-rate Monte Carlo evidence.

Within each outer batch, samples are generated one-by-one in logical-index
order and then evaluated one-by-one in that same order. The tested
`batch_size=1/2/7/64/>simulations` cases are exactly equivalent to the C2
scalar oracle in result, coverage-event sequence, outer sample sequence, and
final outer RNG state.


## Bootstrap coverage progress callback

`bootstrap_mean_coverage(..., progress_callback=...)` emits immutable
`BootstrapCoverageProgress` snapshots only after a complete outer batch has
been evaluated and its coverage count committed.

Each snapshot contains:

- `completed`: cumulative committed outer replicates;
- `total`: total requested outer simulations;
- `covered`: cumulative covered intervals;
- `empirical`: exactly `covered / completed`.

There is no event at zero. Callback return values are ignored. A callback
exception propagates immediately after the completed batch has been committed,
and no progress event is emitted for a batch whose coverage evaluation fails
before completion.

Progress observation is execution-only: enabling or disabling the callback does
not alter the statistical result, outer sample sequence, coverage-event
sequence, or outer RNG state.


## Bootstrap coverage in search and validation

The existing search APIs accept any result satisfying
`StatisticalPropertyResult`, including `BootstrapCoverageResult`.
No bootstrap-specific search API is required.

A caller may therefore use `bootstrap_mean_coverage(...)` inside:

- `grid_search(...)`;
- `random_search(...)`;
- `validate_candidate(...)`;
- `find_counterexample(...)`.

Search point seeds continue to be derived solely from the search root seed and
canonical `ParameterPoint`. Independent validation uses the same selected
point with a child seed derived from a distinct validation root seed. Ranking
continues to use the existing generic objective functions over
`nominal / empirical / deviation`.

Phase E1 does not change shrinking, report, or StatCI behavior.


## Bootstrap coverage parameter shrinking

The existing `shrink_counterexample(...)` workflow accepts
`BootstrapCoverageResult` through the common `StatisticalPropertyResult`
contract. No bootstrap-specific shrinking API is required.

Each start/proposal point receives the existing deterministic shrink seed
derived from the shrink root seed and the complete canonical
`ParameterPoint`. The default `OutsideToleranceCriterion` accepts a simpler
proposal only when its bootstrap coverage result remains outside tolerance.

Phase E2 validates a real percentile-bootstrap coverage shrink trace with both
accepted and rejected proposals and a deterministic plan-local minimal final
point. Repeated evaluation of the same rejected point receives the same seed
and reproduces the same `BootstrapCoverageResult`.

Family shrinking, reports, and StatCI remain unchanged.


## Bootstrap coverage family shrinking

The existing `shrink_dgp_family(...)` workflow accepts
`BootstrapCoverageResult` through the common `StatisticalPropertyResult`
contract. No bootstrap-specific family-shrinking API is required.

Each canonical `FamilyPoint` receives the existing deterministic family seed
derived from the family-shrink root seed and the complete serialized family
point. Candidates are evaluated from the simplest family upward, and the first
candidate that still satisfies the failure criterion is accepted.

Phase E3 validates a real percentile-bootstrap coverage path in which the
simplest Normal candidate returns inside tolerance and is rejected, while the
next LogNormal candidate remains outside tolerance and is accepted. This makes
LogNormal the simplest still-failing family in the explicit canonical plan.

Report and StatCI behavior remain unchanged.


## Bootstrap coverage reports

`build_report(...)` now freezes report results through an internal
property-generic snapshot layer. Existing Type-I results continue to serialize
through the unchanged `StressTestSnapshot` shape, while
`BootstrapCoverageResult` uses a dedicated coverage snapshot.

For Bootstrap coverage, report result objects include:

- method and metric;
- DGP display name and structured identity;
- resolved `MeanTargetCheck`;
- coverage count;
- bootstrap method machine identity;
- Monte Carlo evidence interval;
- `n`, simulations, seed, nominal coverage, empirical coverage, MCSE,
  tolerance, deviation, and status.

The same snapshot path is used by search records, independent validation,
parameter shrinking, family shrinking, and failure maps.

HTML rendering keeps the existing Type-I table/summary vocabulary unchanged.
Coverage reports use coverage-specific labels such as `Covered`,
`MC evidence interval`, and `Mean target`.

`REPORT_SCHEMA_VERSION` remains `1.3`; existing Type-I JSON result objects
retain their prior field names and shape. StatCI is not changed by this report
integration.


## Bootstrap coverage in StatCI assertions

StatCI Phase 1 allows `check_property(...)`, `assert_property(...)`, and
`StatisticalAssertion.evaluate(...)` to consume a real
`BootstrapCoverageResult` through the shared `StatisticalPropertyResult`
contract.

The assertion rule is unchanged:

`abs(observed - target) <= tolerance`

For a Bootstrap coverage assertion, `observed` is empirical coverage and MCSE
is retained as evidence rather than folded into the engineering threshold.

Current-run Bootstrap StatCI evidence records:

- method and coverage metric;
- DGP display name and stable `DGPIdentity`;
- sample size, simulations, and seed;
- MCSE;
- resolved `MeanTargetCheck`;
- coverage count;
- full `BootstrapMeanPercentile` machine identity;
- Monte Carlo evidence confidence level, method, and bounds.

Existing Type-I StatCI serialization remains unchanged, including
`STATCI_SCHEMA_VERSION = "1.2"`, its legacy evidence object, and GitHub summary
format.

Bootstrap baseline loading and regression comparison are intentionally deferred
to a later StatCI phase.


## Bootstrap StatCI regression identity and baseline loading

StatCI Phase 2A defines a stable Bootstrap coverage comparison identity without
enabling regression judgments.

`BootstrapCoverageComparisonKey` is available from `statfuzz.statci`.
Its identity includes:

- property name, assertion target, and assertion tolerance;
- method and metric;
- stable `DGPIdentity`;
- complete resolved `MeanTargetCheck`;
- complete `BootstrapMeanPercentile` machine identity;
- sample size `n`.

The comparison identity deliberately excludes Monte Carlo realization controls
and evidence budget:

- seed;
- simulations;
- observed coverage;
- MCSE;
- coverage count;
- Monte Carlo evidence confidence level and interval bounds.

This allows independent or higher-budget baseline/current runs to represent the
same statistical assertion.

Bootstrap coverage `StatCIResult` JSON can now be strictly reconstructed by
`StatCIResult.from_dict(...)`; `StatCISuiteResult.from_json(...)` therefore
also supports persisted Bootstrap coverage baselines. The coverage loader
strictly validates the evidence shape, DGP identity, target check, bootstrap
method identity, coverage count, evidence interval, and assertion invariants.

Existing Type-I baseline loading remains unchanged and
`STATCI_SCHEMA_VERSION` remains `1.2`.

Phase 2A explicitly does not enable `compare_results(...)` or
`compare_suites(...)` for Bootstrap coverage. Those calls reject coverage
results until Phase 2B defines the regression/worsening semantics.


## Bootstrap StatCI single-result regression comparison

StatCI Phase 2B enables `compare_results(...)` for exactly matched Bootstrap
coverage checks while keeping suite matching deferred.

Coverage comparison first constructs the Phase 2A
`BootstrapCoverageComparisonKey` for baseline and current results. The
comparison proceeds only when those keys are exactly equal.

After identity matching, Bootstrap coverage reuses the existing
`RegressionPolicy` semantics without a coverage-specific worsening formula:

`worsening = current.absolute_deviation - baseline.absolute_deviation`

For conservative uncertainty mode:

`uncertainty_scale = baseline.mcse + current.mcse`

For independent uncertainty mode:

`uncertainty_scale = sqrt(baseline.mcse**2 + current.mcse**2)`

Independent mode continues to require known, distinct baseline/current seeds.

A regression is reported only when worsening is strictly greater than:

`minimum_worsening + uncertainty_multiplier * uncertainty_scale`

PASS-to-FAIL transitions are recorded but do not bypass the uncertainty guard.

`compare_suites(...)` remains unavailable for Bootstrap coverage in this
phase; suite matching and regression-summary integration remain deferred.
