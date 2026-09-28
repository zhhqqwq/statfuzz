# Changelog

All notable changes to StatFuzz will be documented in this file.

The project is pre-1.0. See [docs/VERSIONING.md](docs/VERSIONING.md) for the compatibility
policy.

## [Unreleased]

### Added

- Added reproducible batched Monte Carlo execution with scalar sample generation and
  vectorized SciPy t-tail evaluation. The default batch size is 64, while
  `batch_size=1` preserves the scalar reference path exactly.
- Added strict fixed-seed tests requiring batch sizes 1, 2, 7, 64, and larger than the
  simulation budget to preserve results, logical sample order, and final RNG state.
- Added a reproducible Monte Carlo performance-baseline harness and GitHub Actions
  workflow covering built-in DGPs, sample sizes, simulation budgets, and component
  microbenchmarks before any batching optimization.
- Documented the future batching invariant that changing execution batch size must not
  change logical replicate random streams or the final fixed-seed result.
- Added an explicit equal-means null contract for Type-I error experiments. Built-in
  DGP population means are verified before simulation; custom DGPs without a
  `population_mean` require `MeanEqualityNull`.
- Added rejection counts and configurable-confidence Wilson binomial intervals while
  retaining MCSE and the existing tolerance rule.

### Changed

- StatCI JSON schema is now 1.2 and records null verification plus binomial interval
  evidence; schema 1.0 and 1.1 payloads remain readable.
- StatCI regression schema is now 1.2 and includes the null hypothesis in experiment
  matching.
- Report JSON schema is now 1.3 and includes null verification, rejection counts, and
  confidence intervals.

### Fixed

- Reject non-finite built-in DGP parameters, generated samples, Welch intermediates, and
  invalid p-values instead of silently counting invalid simulations as non-rejections.
- Preserve structured DGP family/parameter identity so rounded human-readable names
  cannot cause false StatCI baseline matches.

## [0.1.0] - 2026-09-28

### Added

- Reproducible Monte Carlo stress testing for Welch's two-sample t-test Type-I error.
- Mean-preserving Normal, LogNormal, Student-t, and two-component Normal mixture DGPs.
- MCSE reporting and deterministic random seeds.
- Finite `ParameterSpace`, exhaustive grid search, and reproducible random search.
- Named and custom search objectives.
- Multiplicity-aware exploratory search summaries and search-to-validation diagnostics.
- Independent candidate validation with separate search and validation simulation budgets.
- High-level `find_counterexample(...)` discovery workflow.
- Scalar and cross-DGP-family counterexample shrinking with reproducible traces.
- Two-dimensional failure maps with MCSE uncertainty.
- Deterministic report JSON and standalone HTML output.
- StatCI property assertions, suite aggregation, GitHub Actions summaries, status/badge
  artifacts, and uncertainty-aware baseline regression comparison.
- Python 3.10, 3.11, and 3.12 CI coverage.

### Changed

- Hardened the public import surface before the first public package release.
- Single-sourced the package version and expanded release/package metadata.
- Added wheel/sdist build, clean-install, and installed-distribution quality gates.
- Added public API, versioning, and release-process documentation.

### Statistical interpretation

- Search-selected extrema are treated as exploratory findings.
- Independent validation is retained separately from search estimates.
- Multiplicity ranks are descriptive and are not presented as corrected p-values.
- Regression comparison uses explicit Monte Carlo uncertainty guards rather than raw
  baseline/current differences.
