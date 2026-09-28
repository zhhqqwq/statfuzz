"""Public reporting API for StatFuzz."""

from .html import render_html, write_html
from .map import FailureMap2D, failure_map_2d
from .model import REPORT_SCHEMA_VERSION, StatFuzzReport, build_report

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "FailureMap2D",
    "StatFuzzReport",
    "build_report",
    "failure_map_2d",
    "render_html",
    "write_html",
]
