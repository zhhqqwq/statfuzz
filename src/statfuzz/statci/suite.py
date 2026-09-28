from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model import STATCI_SCHEMA_VERSION, StatCIResult


@dataclass(frozen=True)
class StatCISuiteResult:
    """Aggregate result for multiple StatCI checks."""

    name: str
    results: tuple[StatCIResult, ...]
    schema_version: str = STATCI_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

        results = tuple(self.results)
        if not results:
            raise ValueError("StatCISuiteResult requires at least one result")
        if not all(isinstance(result, StatCIResult) for result in results):
            raise TypeError("all suite results must be StatCIResult instances")

        object.__setattr__(self, "results", results)

    @classmethod
    def from_results(
        cls,
        results: list[StatCIResult] | tuple[StatCIResult, ...],
        *,
        name: str = "StatCI",
    ) -> StatCISuiteResult:
        return cls(name=name, results=tuple(results))

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed_count(self) -> int:
        return self.total - self.passed_count

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "total": self.total,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "results": [result.as_dict() for result in self.results],
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


class StatCISuiteError(AssertionError):
    """Assertion failure carrying an entire failed StatCI suite."""

    def __init__(self, suite: StatCISuiteResult) -> None:
        self.suite = suite
        super().__init__(
            f"StatCI suite {suite.name!r} failed: "
            f"{suite.failed_count}/{suite.total} checks failed"
        )


def assert_suite(suite: StatCISuiteResult) -> StatCISuiteResult:
    """Return a passing suite or raise AssertionError for an overall failure."""

    if not isinstance(suite, StatCISuiteResult):
        raise TypeError("suite must be a StatCISuiteResult")
    if not suite.passed:
        raise StatCISuiteError(suite)
    return suite
