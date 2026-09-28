# StatFuzz 0.1.0 — Release Candidate Notes

StatFuzz 0.1.0 is the first planned public alpha release.

It turns statistical robustness checks into a reproducible workflow:

```text
DGP
 -> Monte Carlo stress test
 -> parameter search
 -> independent validation
 -> counterexample shrinking
 -> failure maps / reports
 -> StatCI gate
 -> baseline regression comparison
```

## Highlights

### Statistical stress testing

The initial engine evaluates empirical Type-I error for the two-sided Welch t-test and
reports Monte Carlo standard error. Built-in DGPs include Normal, shifted LogNormal,
Student-t, and a two-component Normal mixture while preserving the target arithmetic mean
needed by the null experiment.

### Reproducible counterexample discovery

Finite parameter spaces can be searched exhaustively or sampled without replacement.
Search-selected candidates can be independently re-simulated with a separate validation
budget and random stream.

Search multiplicity is reported explicitly: the tool records how many candidates were
evaluated, tie-aware empirical rank information, objective-score quantiles, and
search-to-validation objective changes without pretending those quantities are corrected
p-values.

### Counterexample shrinking

Validated candidates can be simplified over user-declared scalar parameter levels and
explicit DGP-family hierarchies. Every proposal is re-simulated and the accepted/rejected
trace is retained.

### Reports

Search, validation, shrinking, and failure-map data can be frozen into deterministic JSON
and standalone HTML reports. Sparse random-search map cells remain explicit rather than
being interpolated.

### StatCI

StatCI converts statistical properties into CI contracts, aggregates multiple checks,
writes GitHub Actions summaries, emits machine-readable status/badge artifacts, and can
compare a current suite against a historical baseline using an explicit Monte Carlo
uncertainty guard.

## Installation target

The package requires Python 3.10 or newer and has only two runtime dependencies:

- NumPy
- SciPy

The release CI builds both a wheel and source distribution and tests a clean installation
of the built artifacts before publication.

## Alpha-status caveat

This is a pre-1.0 alpha API. The supported import surface is documented in
`docs/PUBLIC_API.md`. Minor 0.x releases may contain documented breaking changes while
the design matures; patch releases should preserve documented public API.

## Release publication

These are candidate notes only. The hardening PR does **not** create a tag, GitHub Release,
or PyPI publication. Publication happens only after the release quality gate is green on
the final merge commit.
