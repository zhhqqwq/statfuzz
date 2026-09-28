from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .model import StatCIResult
from .suite import StatCISuiteResult

STATCI_REGRESSION_SCHEMA_VERSION = "1.0"
UncertaintyMode = Literal["conservative", "independent"]


@dataclass(frozen=True, order=True)
class StatCIComparisonKey:
    """Stable experiment/assertion identity used to match baseline and current checks."""

    property: str
    target: float
    tolerance: float
    method: str
    metric: str
    dgp1: str
    dgp2: str
    n1: int
    n2: int

    @classmethod
    def from_result(cls, result: StatCIResult) -> StatCIComparisonKey:
        if not isinstance(result, StatCIResult):
            raise TypeError("result must be a StatCIResult")
        return cls(
            property=result.property,
            target=result.target,
            tolerance=result.tolerance,
            method=result.method,
            metric=result.metric,
            dgp1=result.dgp1,
            dgp2=result.dgp2,
            n1=result.n1,
            n2=result.n2,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "property": self.property,
            "target": self.target,
            "tolerance": self.tolerance,
            "method": self.method,
            "metric": self.metric,
            "dgp1": self.dgp1,
            "dgp2": self.dgp2,
            "n1": self.n1,
            "n2": self.n2,
        }

    def describe(self) -> str:
        return (
            f"{self.property} | {self.method} | "
            f"{self.dgp1} vs {self.dgp2} | n={self.n1}/{self.n2} | "
            f"target={self.target:g} ± {self.tolerance:g}"
        )


@dataclass(frozen=True)
class RegressionPolicy:
    """Engineering policy for deciding whether a matched check regressed."""

    minimum_worsening: float = 0.0
    uncertainty_multiplier: float = 2.0
    uncertainty_mode: UncertaintyMode = "conservative"
    strict_matching: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_worsening", self.minimum_worsening),
            ("uncertainty_multiplier", self.uncertainty_multiplier),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.uncertainty_mode not in {"conservative", "independent"}:
            raise ValueError(
                "uncertainty_mode must be 'conservative' or 'independent'"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "minimum_worsening": self.minimum_worsening,
            "uncertainty_multiplier": self.uncertainty_multiplier,
            "uncertainty_mode": self.uncertainty_mode,
            "strict_matching": self.strict_matching,
        }


def _comparison_mcse(
    baseline: StatCIResult,
    current: StatCIResult,
    policy: RegressionPolicy,
) -> float:
    if policy.uncertainty_mode == "conservative":
        return baseline.mcse + current.mcse

    if baseline.seed is None or current.seed is None:
        raise ValueError(
            "independent uncertainty mode requires known baseline and current seeds"
        )
    if baseline.seed == current.seed:
        raise ValueError(
            "independent uncertainty mode requires distinct baseline and current seeds"
        )
    return math.sqrt(baseline.mcse**2 + current.mcse**2)


@dataclass(frozen=True)
class StatCIRegressionResult:
    """Uncertainty-aware comparison for one matched StatCI check."""

    key: StatCIComparisonKey
    baseline: StatCIResult
    current: StatCIResult
    comparison_mcse: float
    uncertainty_allowance: float
    regression_threshold: float
    worsening: float
    regressed: bool
    direction: str
    pass_to_fail: bool
    fail_to_pass: bool

    @property
    def status(self) -> str:
        return "REGRESSION" if self.regressed else "PASS"

    @property
    def observed_change(self) -> float:
        return self.current.observed - self.baseline.observed

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key.as_dict(),
            "status": self.status,
            "regressed": self.regressed,
            "direction": self.direction,
            "pass_to_fail": self.pass_to_fail,
            "fail_to_pass": self.fail_to_pass,
            "observed_change": self.observed_change,
            "worsening": self.worsening,
            "comparison_mcse": self.comparison_mcse,
            "uncertainty_allowance": self.uncertainty_allowance,
            "regression_threshold": self.regression_threshold,
            "baseline": self.baseline.as_dict(),
            "current": self.current.as_dict(),
        }


def compare_results(
    baseline: StatCIResult,
    current: StatCIResult,
    *,
    policy: RegressionPolicy | None = None,
) -> StatCIRegressionResult:
    """Compare one exactly matched baseline/current StatCI result pair."""

    if not isinstance(baseline, StatCIResult) or not isinstance(current, StatCIResult):
        raise TypeError("baseline and current must be StatCIResult instances")

    resolved_policy = RegressionPolicy() if policy is None else policy
    if not isinstance(resolved_policy, RegressionPolicy):
        raise TypeError("policy must be a RegressionPolicy")

    baseline_key = StatCIComparisonKey.from_result(baseline)
    current_key = StatCIComparisonKey.from_result(current)
    if baseline_key != current_key:
        raise ValueError(
            "baseline and current checks do not have the same comparison key"
        )

    comparison_mcse = _comparison_mcse(baseline, current, resolved_policy)
    uncertainty_allowance = (
        resolved_policy.uncertainty_multiplier * comparison_mcse
    )
    regression_threshold = (
        resolved_policy.minimum_worsening + uncertainty_allowance
    )
    worsening = current.absolute_deviation - baseline.absolute_deviation
    regressed = worsening > regression_threshold

    if regressed:
        direction = "REGRESSION"
    elif worsening < 0:
        direction = "IMPROVED"
    elif worsening == 0:
        direction = "STABLE"
    else:
        direction = "WITHIN_GUARD"

    return StatCIRegressionResult(
        key=baseline_key,
        baseline=baseline,
        current=current,
        comparison_mcse=comparison_mcse,
        uncertainty_allowance=uncertainty_allowance,
        regression_threshold=regression_threshold,
        worsening=worsening,
        regressed=regressed,
        direction=direction,
        pass_to_fail=baseline.passed and not current.passed,
        fail_to_pass=not baseline.passed and current.passed,
    )


def _index_suite(
    suite: StatCISuiteResult,
    *,
    role: str,
) -> dict[StatCIComparisonKey, StatCIResult]:
    index: dict[StatCIComparisonKey, StatCIResult] = {}
    for result in suite.results:
        key = StatCIComparisonKey.from_result(result)
        if key in index:
            raise ValueError(
                f"{role} suite contains duplicate comparison key: {key.describe()}"
            )
        index[key] = result
    return index


@dataclass(frozen=True)
class StatCIRegressionSuiteResult:
    """Aggregate baseline-vs-current regression comparison."""

    name: str
    baseline_name: str
    current_name: str
    policy: RegressionPolicy
    comparisons: tuple[StatCIRegressionResult, ...]
    missing_current: tuple[StatCIComparisonKey, ...] = ()
    new_current: tuple[StatCIComparisonKey, ...] = ()
    schema_version: str = STATCI_REGRESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not self.comparisons:
            raise ValueError("regression suite requires at least one comparison")

    @property
    def passed(self) -> bool:
        return not any(comparison.regressed for comparison in self.comparisons)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def total(self) -> int:
        return len(self.comparisons)

    @property
    def regression_count(self) -> int:
        return sum(comparison.regressed for comparison in self.comparisons)

    @property
    def pass_to_fail_count(self) -> int:
        return sum(comparison.pass_to_fail for comparison in self.comparisons)

    @property
    def improvement_count(self) -> int:
        return sum(
            comparison.direction == "IMPROVED"
            for comparison in self.comparisons
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "baseline_name": self.baseline_name,
            "current_name": self.current_name,
            "status": self.status,
            "passed": self.passed,
            "total": self.total,
            "regression_count": self.regression_count,
            "pass_to_fail_count": self.pass_to_fail_count,
            "improvement_count": self.improvement_count,
            "policy": self.policy.as_dict(),
            "missing_current": [key.as_dict() for key in self.missing_current],
            "new_current": [key.as_dict() for key in self.new_current],
            "comparisons": [
                comparison.as_dict()
                for comparison in self.comparisons
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int = 2,
    ) -> Path:
        target = Path(path)
        target.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return target


def compare_suites(
    baseline: StatCISuiteResult,
    current: StatCISuiteResult,
    *,
    policy: RegressionPolicy | None = None,
    name: str = "StatCI regression",
) -> StatCIRegressionSuiteResult:
    """Match and compare two StatCI suites using an explicit regression policy."""

    if not isinstance(baseline, StatCISuiteResult):
        raise TypeError("baseline must be a StatCISuiteResult")
    if not isinstance(current, StatCISuiteResult):
        raise TypeError("current must be a StatCISuiteResult")

    resolved_policy = RegressionPolicy() if policy is None else policy
    if not isinstance(resolved_policy, RegressionPolicy):
        raise TypeError("policy must be a RegressionPolicy")

    baseline_index = _index_suite(baseline, role="baseline")
    current_index = _index_suite(current, role="current")

    baseline_keys = set(baseline_index)
    current_keys = set(current_index)
    missing_current = tuple(sorted(baseline_keys - current_keys))
    new_current = tuple(sorted(current_keys - baseline_keys))

    if resolved_policy.strict_matching and (missing_current or new_current):
        pieces = []
        if missing_current:
            pieces.append(
                "missing current checks: "
                + "; ".join(key.describe() for key in missing_current)
            )
        if new_current:
            pieces.append(
                "unexpected current checks: "
                + "; ".join(key.describe() for key in new_current)
            )
        raise ValueError("suite comparison key mismatch; " + " | ".join(pieces))

    matched_keys = tuple(sorted(baseline_keys & current_keys))
    if not matched_keys:
        raise ValueError("baseline and current suites have no matching checks")

    comparisons = tuple(
        compare_results(
            baseline_index[key],
            current_index[key],
            policy=resolved_policy,
        )
        for key in matched_keys
    )

    return StatCIRegressionSuiteResult(
        name=name,
        baseline_name=baseline.name,
        current_name=current.name,
        policy=resolved_policy,
        comparisons=comparisons,
        missing_current=missing_current,
        new_current=new_current,
    )


class StatCIRegressionError(AssertionError):
    """Assertion failure carrying a failed regression-comparison suite."""

    def __init__(self, result: StatCIRegressionSuiteResult) -> None:
        self.result = result
        super().__init__(
            f"StatCI regression suite {result.name!r} failed: "
            f"{result.regression_count}/{result.total} matched checks regressed"
        )


def assert_regression_suite(
    result: StatCIRegressionSuiteResult,
) -> StatCIRegressionSuiteResult:
    if not isinstance(result, StatCIRegressionSuiteResult):
        raise TypeError("result must be a StatCIRegressionSuiteResult")
    if not result.passed:
        raise StatCIRegressionError(result)
    return result


def _md(value: object) -> str:
    return str(value).replace("\", "\\").replace("|", "\|").replace("\n", "<br>")


def _number(value: float) -> str:
    return f"{value:.6g}"


def render_regression_summary(result: StatCIRegressionSuiteResult) -> str:
    """Render an uncertainty-aware baseline comparison for GitHub Actions."""

    if not isinstance(result, StatCIRegressionSuiteResult):
        raise TypeError("result must be a StatCIRegressionSuiteResult")

    lines = [
        f"## StatCI Regression — {_md(result.name)}",
        "",
        f"**Overall:** {result.status}  ",
        f"**Matched checks:** {result.total}  ",
        f"**Regressions:** {result.regression_count}  ",
        f"**PASS→FAIL transitions:** {result.pass_to_fail_count}  ",
        f"**Improvements:** {result.improvement_count}",
        "",
        (
            f"Policy: minimum worsening={_number(result.policy.minimum_worsening)}, "
            f"uncertainty={_number(result.policy.uncertainty_multiplier)} × "
            f"{result.policy.uncertainty_mode} comparison MCSE."
        ),
        "",
        "| Property | Direction | Baseline | Current | Worsening | Comparison MCSE | Guard threshold | PASS→FAIL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for comparison in result.comparisons:
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(comparison.key.property),
                    comparison.direction,
                    _number(comparison.baseline.absolute_deviation),
                    _number(comparison.current.absolute_deviation),
                    _number(comparison.worsening),
                    _number(comparison.comparison_mcse),
                    _number(comparison.regression_threshold),
                    "yes" if comparison.pass_to_fail else "no",
                )
            )
            + " |"
        )

    regressions = [
        comparison
        for comparison in result.comparisons
        if comparison.regressed
    ]
    if regressions:
        lines.extend(["", "### Regressions", ""])
        for comparison in regressions:
            lines.append(
                f"- **{_md(comparison.key.describe())}**: "
                f"worsening={_number(comparison.worsening)} > "
                f"guard={_number(comparison.regression_threshold)}"
            )

    if result.missing_current or result.new_current:
        lines.extend(["", "### Unmatched checks", ""])
        for key in result.missing_current:
            lines.append(f"- Missing current: {_md(key.describe())}")
        for key in result.new_current:
            lines.append(f"- New current: {_md(key.describe())}")

    return "\n".join(lines) + "\n"


def write_regression_summary(
    result: StatCIRegressionSuiteResult,
    path: str | Path | None = None,
) -> Path:
    if path is None:
        configured = os.environ.get("GITHUB_STEP_SUMMARY")
        if not configured:
            raise RuntimeError(
                "GITHUB_STEP_SUMMARY is not set; provide path explicitly "
                "when running outside GitHub Actions"
            )
        target = Path(configured)
    else:
        target = Path(path)

    with target.open("a", encoding="utf-8") as handle:
        handle.write(render_regression_summary(result))
    return target
