from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model import STATCI_SCHEMA_VERSION, StatCIResult

_SUPPORTED_SUITE_SCHEMA_VERSIONS = {"1.0", STATCI_SCHEMA_VERSION}


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

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> StatCISuiteResult:
        """Reconstruct and validate a suite from its machine-readable payload."""

        if not isinstance(data, dict):
            raise TypeError("StatCISuiteResult payload must be a dictionary")
        schema_version = data.get("schema_version")
        if schema_version not in _SUPPORTED_SUITE_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported StatCI schema_version: {schema_version!r}"
            )

        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("suite name must be a non-empty string")

        raw_results = data.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            raise ValueError("suite results must be a non-empty list")
        results = tuple(
            StatCIResult.from_dict(item)
            for item in raw_results
            if isinstance(item, dict)
        )
        if len(results) != len(raw_results):
            raise ValueError("every suite result must be an object")

        suite = cls(name=name, results=results)

        expected = {
            "passed": suite.passed,
            "status": suite.status,
            "total": suite.total,
            "passed_count": suite.passed_count,
            "failed_count": suite.failed_count,
        }
        for field, value in expected.items():
            if data.get(field) != value:
                raise ValueError(f"suite field {field!r} is inconsistent with results")

        return suite

    @classmethod
    def from_json(cls, payload: str) -> StatCISuiteResult:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise TypeError("suite JSON must contain an object")
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> StatCISuiteResult:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


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
