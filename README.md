# StatFuzz

[CI](https://github.com/zhhqqwq/statfuzz/actions/workflows/ci.yml) · [Documentation](docs/PUBLIC_API.md) · [Examples](examples) · [简体中文](README.zh-CN.md)

Monte Carlo stress testing for statistical methods.

StatFuzz is a Python library for studying how statistical methods behave under
specified data-generating conditions. It searches finite parameter spaces for
departures from a target property, re-evaluates selected candidates with a fresh
random stream, and records the estimates, Monte Carlo uncertainty, and seeds.

The built-in experiment currently measures the Type I error rate of Welch's
two-sample t-test. Use it to explore how sample size and distribution shape affect
the rejection rate under an equal-means null hypothesis, or to track statistical
behavior across code changes.

Version 0.1.0 is alpha software. Public interfaces may change in minor releases;
see the [versioning policy](docs/VERSIONING.md).

## Installation

Requires Python 3.10 or later. NumPy and SciPy are installed as dependencies.
The CI test matrix covers Python 3.10, 3.11, and 3.12.

Install from a source checkout:

```bash
git clone https://github.com/zhhqqwq/statfuzz.git
cd statfuzz
python -m pip install .
```

## Quick start

Estimate the rejection rate using two independent samples from the same shifted
lognormal distribution:

```python
from statfuzz import stress_test
from statfuzz.dgp import LogNormal

result = stress_test(
    method="welch_ttest",
    metric="type1_error",
    dgp=LogNormal(sigma=1.4, mean=0.0),
    n1=12,
    n2=12,
    simulations=10_000,
    alpha=0.05,
    tolerance=0.01,
    seed=42,
)

print(result)
```

The result reports the rejection count, empirical rejection rate, Monte Carlo
standard error (MCSE), a 95% Wilson binomial interval, and the deviation from
`alpha`. A result passes when the absolute deviation is at most `tolerance`;
the interval is evidence and does not silently change that engineering rule.

`LogNormal(mean=0.0)` shifts the draws to have population arithmetic mean zero.
Built-in DGPs expose their population mean, so Type-I error experiments verify the
equal-means null **before sampling**. If a custom DGP cannot expose
`population_mean`, supply an explicit `MeanEqualityNull` declaration instead.

## Search, validate, and report

This complete example evaluates nine parameter combinations with 2,000
simulations each, then re-evaluates the candidate with the largest absolute
deviation using 20,000 simulations and a separate seed.

```python
from statfuzz import stress_test
from statfuzz.dgp import LogNormal
from statfuzz.report import build_report, failure_map_2d, write_html
from statfuzz.search import DiscoveryBudget, ParameterSpace, find_counterexample

space = ParameterSpace({
    "n": [8, 12, 20],
    "sigma": [0.6, 1.0, 1.4],
})


def evaluate(point, seed, simulations):
    return stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=LogNormal(sigma=point["sigma"], mean=0.0),
        n1=point["n"],
        n2=point["n"],
        simulations=simulations,
        alpha=0.05,
        tolerance=0.01,
        seed=seed,
    )


discovery = find_counterexample(
    space=space,
    search_evaluate=evaluate,
    validation_evaluate=evaluate,
    budget=DiscoveryBudget(
        search_simulations=2_000,
        validation_simulations=20_000,
    ),
    search_root_seed=42,
    validation_root_seed=2026,
    objective="absolute_deviation",
)

print(discovery.point)
print(discovery.validation_result)

report = build_report(
    title="Welch t-test under shifted lognormal sampling",
    search=discovery.search,
    validation=discovery.validation,
    failure_map=failure_map_2d(
        search=discovery.search,
        x_parameter="n",
        y_parameter="sigma",
    ),
)
report.write_json("statfuzz-report.json")
write_html(report, "statfuzz-report.html")
```

Open `statfuzz-report.html` in a browser to inspect the search, validation result,
and two-dimensional failure map. The JSON and HTML exports use the same result
snapshot.

`find_counterexample` returns the highest-ranked candidate even if its validation
result is within tolerance. Inspect `discovery.validation_result.status` before
describing it as a failure.

For larger finite spaces, set `search_draws` on `find_counterexample` to sample
without replacement. The search layer calls the supplied evaluator with a
parameter point, seed, and simulation budget; the evaluator defines the
statistical experiment.

## Statistical checks in CI

Using `result` from the quick-start example:

```python
from statfuzz import check_property

check = check_property(
    result,
    property="type1_error",
    target=0.05,
    tolerance=0.01,
)
print(check.status)
check.write_json("statci-result.json")
```

Use `assert_property` with the same arguments to raise
`StatisticalAssertionError` when the check fails. It inherits from
`AssertionError` and carries the full result.

StatCI also aggregates checks into suites, writes GitHub Actions summaries, and
exports status and badge JSON. Baseline comparisons match experiment identities
and check whether the absolute deviation from the target has worsened beyond
both the configured engineering threshold and MCSE guard.
See the [StatCI guide](docs/STATCI.md) for policies and CI examples.

## Supported features

| Area | Current support |
| --- | --- |
| Statistical experiment | Welch's two-sided t-test; verified equal-means Type I error, rejection counts, MCSE, and Wilson interval |
| Data generation | Normal, shifted lognormal, Student-t, and two-component normal mixtures |
| Search | Finite Cartesian parameter spaces; grid search and random sampling without replacement |
| Validation | Separate search and validation budgets, deterministic child seeds, and separate estimates |
| Simplification | Greedy parameter and distribution-family shrinking with an explicit ordering and recorded trace |
| Reporting | Search-selection diagnostics, 2D failure maps, JSON, and standalone HTML |
| StatCI | Assertions, suites, GitHub Actions summaries, status/badge artifacts, and baseline comparisons |

## Interpreting results

A Type I error experiment requires a true null hypothesis. For the built-in Welch
experiment, StatFuzz now verifies equal population means before the first draw.
Custom DGPs should expose a finite `population_mean`; otherwise the caller must
make the assumption explicit with `MeanEqualityNull(mean=..., note=...)`.

MCSE describes local Monte Carlo precision, while the reported Wilson interval
shows binomial uncertainty in the simulated rejection probability, including
boundary cases where MCSE is zero. Neither quantity automatically changes the
existing PASS/FAIL tolerance. The separate baseline-regression policy includes
its own uncertainty guard.

Searching many noisy estimates can select an unusually extreme result.
Independent validation re-estimates that candidate. The reported search ranks
and percentiles are descriptive; they do not provide corrected p-values or
general FWER/FDR control. A candidate applies to the specified experiment and
simulation budget, and shrinking follows the user-defined ordering without a
global-minimality guarantee.

For reproducible runs, retain the seeds, parameters, simulation budgets, and
software versions. Custom evaluators must use the supplied seeds and budgets.
Bootstrap coverage, OLS inference, power, and calibration are not built into
`stress_test` in this release.

## Documentation

| Topic | Reference |
| --- | --- |
| Supported imports and compatibility | [Public API](docs/PUBLIC_API.md) |
| Parameter spaces, search, and validation | [Search](docs/SEARCH.md) |
| Parameter and distribution-family simplification | [Shrinking](docs/SHRINKING.md) |
| Failure maps and exported reports | [Reporting](docs/REPORTING.md) |
| Assertions and regression comparisons | [StatCI](docs/STATCI.md) |
| Architecture | [Design](docs/DESIGN.md) |
| Runnable scripts | [Examples](examples) |
| Releases | [Changelog](CHANGELOG.md), [0.1.0 notes](docs/RELEASE_NOTES_0.1.0.md), [release checklist](docs/RELEASING.md) |

The [roadmap](ROADMAP.md) records completed development phases.
Its v0.1–v0.6 labels are independent of package versions.

## Development and contributions

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Statistical
contributions should state the target property and assumptions, use reproducible
seeds, and report Monte Carlo uncertainty.

Report bugs or propose features in [GitHub Issues](https://github.com/zhhqqwq/statfuzz/issues).
For simulation results, include a minimal example, dependency versions, seeds,
and simulation budgets. The project is maintained by
[zhhqqwq](https://github.com/zhhqqwq).

## License

[MIT](LICENSE).
