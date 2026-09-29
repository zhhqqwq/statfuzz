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
