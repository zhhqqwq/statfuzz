# Changelog

All notable changes to StatFuzz will be documented in this file.

The project is pre-1.0. See [docs/VERSIONING.md](docs/VERSIONING.md) for the compatibility
policy.

## [Unreleased]

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
