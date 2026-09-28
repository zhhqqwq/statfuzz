"""Machine-readable and standalone reporting for StatFuzz."""

from .html import render_html, write_html
from .map import FailureMap2D, FailureMapCell, failure_map_2d
from .model import (
    REPORT_SCHEMA_VERSION,
    SearchRecordSnapshot,
    SearchSnapshot,
    ShrinkSnapshot,
    StatFuzzReport,
    StressTestSnapshot,
    ValidationSnapshot,
    build_report,
)

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "FailureMap2D",
    "FailureMapCell",
    "SearchRecordSnapshot",
    "SearchSnapshot",
    "ShrinkSnapshot",
    "StatFuzzReport",
    "StressTestSnapshot",
    "ValidationSnapshot",
    "build_report",
    "failure_map_2d",
    "render_html",
    "write_html",
]
