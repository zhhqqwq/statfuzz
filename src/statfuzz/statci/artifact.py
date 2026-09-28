from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .suite import StatCISuiteResult

STATCI_STATUS_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class StatCIStatusArtifact:
    """Lightweight overall CI status derived from a StatCISuiteResult."""

    suite_name: str
    status: str
    passed: bool
    total: int
    passed_count: int
    failed_count: int
    badge_label: str = "StatCI"
    schema_version: str = STATCI_STATUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.suite_name, str) or not self.suite_name:
            raise ValueError("suite_name must be a non-empty string")
        if not isinstance(self.badge_label, str) or not self.badge_label:
            raise ValueError("badge_label must be a non-empty string")
        if self.status not in {"PASS", "FAIL"}:
            raise ValueError("status must be 'PASS' or 'FAIL'")
        if self.total <= 0:
            raise ValueError("total must be positive")
        if self.passed_count < 0 or self.failed_count < 0:
            raise ValueError("passed_count and failed_count must be non-negative")
        if self.passed_count + self.failed_count != self.total:
            raise ValueError("passed_count + failed_count must equal total")
        if self.passed != (self.status == "PASS"):
            raise ValueError("passed must agree with status")
        if self.passed != (self.failed_count == 0):
            raise ValueError("passed must agree with failed_count")

    @classmethod
    def from_suite(
        cls,
        suite: StatCISuiteResult,
        *,
        badge_label: str = "StatCI",
    ) -> StatCIStatusArtifact:
        if not isinstance(suite, StatCISuiteResult):
            raise TypeError("suite must be a StatCISuiteResult")

        return cls(
            suite_name=suite.name,
            status=suite.status,
            passed=suite.passed,
            total=suite.total,
            passed_count=suite.passed_count,
            failed_count=suite.failed_count,
            badge_label=badge_label,
        )

    @property
    def badge_message(self) -> str:
        return f"{self.status} · {self.passed_count}/{self.total}"

    @property
    def badge_color(self) -> str:
        return "brightgreen" if self.passed else "red"

    def badge_as_dict(self) -> dict[str, object]:
        """Return a Shields endpoint-compatible badge payload."""

        return {
            "schemaVersion": 1,
            "label": self.badge_label,
            "message": self.badge_message,
            "color": self.badge_color,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "suite_name": self.suite_name,
            "status": self.status,
            "passed": self.passed,
            "checks": {
                "total": self.total,
                "passed": self.passed_count,
                "failed": self.failed_count,
            },
            "badge": self.badge_as_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    def badge_to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.badge_as_dict(),
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

    def write_badge_json(
        self,
        path: str | Path,
        *,
        indent: int = 2,
    ) -> Path:
        target = Path(path)
        target.write_text(
            self.badge_to_json(indent=indent) + "\n",
            encoding="utf-8",
        )
        return target
