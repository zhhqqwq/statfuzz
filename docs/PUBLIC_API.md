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
