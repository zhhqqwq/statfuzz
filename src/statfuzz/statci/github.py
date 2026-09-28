from __future__ import annotations

import os
from pathlib import Path

from .suite import StatCISuiteResult


def _md(value: object) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _number(value: float) -> str:
    return f"{value:.6g}"


def render_github_summary(suite: StatCISuiteResult) -> str:
    """Render a GitHub Actions job-summary Markdown block."""

    if not isinstance(suite, StatCISuiteResult):
        raise TypeError("suite must be a StatCISuiteResult")

    lines = [
        f"## StatCI — {_md(suite.name)}",
        "",
        f"**Overall:** {suite.status}  ",
        f"**Checks:** {suite.passed_count} passed / {suite.failed_count} failed / {suite.total} total",
        "",
        "| Property | Status | Observed | Target | Tolerance | Deviation | MCSE | Simulations | Seed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for result in suite.results:
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(result.property),
                    result.status,
                    _number(result.observed),
                    _number(result.target),
                    _number(result.tolerance),
                    _number(result.deviation),
                    _number(result.mcse),
                    str(result.simulations),
                    "—" if result.seed is None else str(result.seed),
                )
            )
            + " |"
        )

    failed = [result for result in suite.results if not result.passed]
    if failed:
        lines.extend(
            [
                "",
                "### Failed checks",
                "",
            ]
        )
        for result in failed:
            lines.append(
                f"- **{_md(result.property)}**: "
                f"observed={_number(result.observed)}, "
                f"target={_number(result.target)}, "
                f"|deviation|={_number(result.absolute_deviation)} > "
                f"tolerance={_number(result.tolerance)}"
            )

    return "\n".join(lines) + "\n"


def write_github_summary(
    suite: StatCISuiteResult,
    path: str | Path | None = None,
) -> Path:
    """Append a rendered StatCI summary to GitHub's job-summary file.

    When path is omitted, GITHUB_STEP_SUMMARY is used.
    """

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
        handle.write(render_github_summary(suite))

    return target
