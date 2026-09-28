from __future__ import annotations

from html import escape
from pathlib import Path

from .map import FailureMap2D
from .model import StatFuzzReport, StressTestSnapshot


def _text(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _e(value: object) -> str:
    return escape(_text(value), quote=True)


def _result_summary(result: StressTestSnapshot) -> str:
    status_class = (
        "status-pass" if result.status == "PASS" else "status-fail"
    )
    return (
        '<dl class="metrics">'
        f"<dt>Method</dt><dd>{_e(result.method)}</dd>"
        f"<dt>Metric</dt><dd>{_e(result.metric)}</dd>"
        f"<dt>Empirical</dt><dd>{_e(result.empirical)}</dd>"
        f"<dt>Nominal</dt><dd>{_e(result.nominal)}</dd>"
        f"<dt>Deviation</dt><dd>{_e(result.deviation)}</dd>"
        f"<dt>MCSE</dt><dd>{_e(result.mcse)}</dd>"
        f"<dt>Simulations</dt><dd>{_e(result.simulations)}</dd>"
        f'<dt>Status</dt><dd class="{status_class}">{_e(result.status)}</dd>'
        "</dl>"
    )


def _failure_map_html(failure_map: FailureMap2D) -> str:
    head = "".join(
        f"<th>{_e(value)}</th>"
        for value in failure_map.x_values
    )

    rows = []
    for y_value, cells in zip(failure_map.y_values, failure_map.cells):
        rendered_cells = []
        for cell in cells:
            if cell is None:
                rendered_cells.append(
                    '<td class="map-cell map-missing">Missing</td>'
                )
                continue

            status_class = (
                "map-pass"
                if cell.result.status == "PASS"
                else "map-fail"
            )
            rendered_cells.append(
                f'<td class="map-cell {status_class}">'
                f'<div class="map-estimate">{_e(cell.result.empirical)}</div>'
                f'<div>Δ {_e(cell.result.deviation)}</div>'
                f'<div class="map-uncertainty">MCSE {_e(cell.result.mcse)}</div>'
                f'<div>{_e(cell.result.status)}</div>'
                "</td>"
            )

        rows.append(
            f"<tr><th>{_e(y_value)}</th>{''.join(rendered_cells)}</tr>"
        )

    fixed = ", ".join(
        f"{_e(name)}={_e(value)}"
        for name, value in failure_map.fixed_parameters.items()
    )
    fixed_html = (
        f"<p><strong>Fixed slice:</strong> {fixed}</p>"
        if fixed
        else ""
    )

    return (
        '<section id="failure-map">'
        "<h2>Failure Map</h2>"
        f"<p><strong>X:</strong> {_e(failure_map.x_parameter)} · "
        f"<strong>Y:</strong> {_e(failure_map.y_parameter)} · "
        f"<strong>Objective:</strong> {_e(failure_map.objective)}</p>"
        f"{fixed_html}"
        f"<p>Coverage: {_e(failure_map.covered_cells)}/"
        f"{_e(failure_map.total_cells)} "
        f"({_e(failure_map.coverage_fraction)}) · "
        "Uncertainty overlay: MCSE</p>"
        '<div class="table-wrap"><table class="failure-map-table">'
        f"<thead><tr><th>{_e(failure_map.y_parameter)} \\ "
        f"{_e(failure_map.x_parameter)}</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
        "</section>"
    )


def _search_table(report: StatFuzzReport) -> str:
    parameter_names = report.search.parameter_names
    headers = "".join(
        f"<th>{_e(name)}</th>"
        for name in parameter_names
    )
    rows = []

    for record in report.search.records:
        params = "".join(
            f"<td>{_e(record.parameters.get(name))}</td>"
            for name in parameter_names
        )
        status_class = (
            "status-pass"
            if record.result.status == "PASS"
            else "status-fail"
        )
        rows.append(
            "<tr>"
            f"{params}"
            f"<td>{_e(record.objective_score)}</td>"
            f"<td>{_e(record.result.empirical)}</td>"
            f"<td>{_e(record.result.deviation)}</td>"
            f"<td>{_e(record.result.mcse)}</td>"
            f'<td class="{status_class}">{_e(record.result.status)}</td>'
            "</tr>"
        )

    return (
        '<section id="search-records">'
        "<h2>Search Records</h2>"
        f"<p>Strategy: {_e(report.search.strategy)} · "
        f"Objective: {_e(report.search.objective)} · "
        f"Root seed: {_e(report.search.root_seed)} · "
        f"Records: {_e(len(report.search.records))}</p>"
        '<div class="table-wrap"><table>'
        f"<thead><tr>{headers}<th>Objective score</th>"
        "<th>Empirical</th><th>Deviation</th><th>MCSE</th><th>Status</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody>"
        "</table></div>"
        "</section>"
    )


def _validation_html(report: StatFuzzReport) -> str:
    if report.validation is None:
        return ""

    validation = report.validation
    params = ", ".join(
        f"{_e(name)}={_e(value)}"
        for name, value in validation.parameters.items()
    )
    return (
        '<section id="validation">'
        "<h2>Independent Validation</h2>"
        f"<p><strong>Candidate:</strong> {params}</p>"
        '<div class="split">'
        '<div><h3>Search stage</h3>'
        f"{_result_summary(validation.search_result)}</div>"
        '<div><h3>Validation stage</h3>'
        f"{_result_summary(validation.validation_result)}</div>"
        "</div>"
        "</section>"
    )


def _shrink_html(report: StatFuzzReport) -> str:
    sections = []

    for label, shrink in (
        ("Scalar Shrinking", report.scalar_shrink),
        ("Family Shrinking", report.family_shrink),
    ):
        if shrink is None:
            continue

        rows = []
        for step in shrink.steps:
            rows.append(
                "<tr>"
                + "".join(
                    f"<td>{_e(step.get(key))}</td>"
                    for key in (
                        "attempt",
                        "parameter",
                        "from",
                        "to",
                        "from_family",
                        "to_family",
                        "accepted",
                        "empirical",
                        "deviation",
                        "mcse",
                    )
                )
                + "</tr>"
            )

        sections.append(
            "<section>"
            f"<h2>{_e(label)}</h2>"
            f"<p><strong>Criterion:</strong> {_e(shrink.criterion)} · "
            f"<strong>Complexity reduction:</strong> "
            f"{_e(shrink.complexity_reduction)}</p>"
            f"<p><strong>Start:</strong> {_e(shrink.start)}<br>"
            f"<strong>Final:</strong> {_e(shrink.final)}</p>"
            '<div class="table-wrap"><table>'
            "<thead><tr><th>Attempt</th><th>Parameter</th><th>From</th>"
            "<th>To</th><th>From family</th><th>To family</th>"
            "<th>Accepted</th><th>Empirical</th><th>Deviation</th>"
            f"<th>MCSE</th></tr></thead><tbody>{''.join(rows)}</tbody>"
            "</table></div>"
            "</section>"
        )

    return "".join(sections)


def render_html(report: StatFuzzReport) -> str:
    """Render a standalone HTML report from a frozen StatFuzzReport snapshot."""

    failure_map_html = ""
    if report.failure_map is not None:
        if not isinstance(report.failure_map, FailureMap2D):
            raise TypeError("HTML rendering requires failure_map to be FailureMap2D")
        failure_map_html = _failure_map_html(report.failure_map)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(report.title)}</title>
<style>
:root {{
  color-scheme: light dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
body {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  line-height: 1.5;
}}
h1, h2, h3 {{ line-height: 1.2; }}
section {{
  margin: 2rem 0;
  padding-top: 0.5rem;
  border-top: 1px solid #8886;
}}
.summary {{
  padding: 1rem;
  border: 1px solid #8886;
  border-radius: 0.75rem;
}}
.metrics {{
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 0.35rem 1rem;
}}
.metrics dt {{ font-weight: 700; }}
.metrics dd {{ margin: 0; }}
.split {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}}
.table-wrap {{ overflow-x: auto; }}
table {{
  border-collapse: collapse;
  width: 100%;
  font-variant-numeric: tabular-nums;
}}
th, td {{
  border: 1px solid #8886;
  padding: 0.5rem;
  text-align: left;
  vertical-align: top;
}}
.status-pass {{ font-weight: 700; }}
.status-fail {{ font-weight: 700; }}
.failure-map-table th,
.failure-map-table td {{ text-align: center; }}
.map-cell {{ min-width: 8rem; }}
.map-pass {{ background: color-mix(in srgb, Canvas, #2e8b57 16%); }}
.map-fail {{ background: color-mix(in srgb, Canvas, #c43d3d 18%); }}
.map-missing {{ opacity: 0.55; }}
.map-estimate {{ font-size: 1.1rem; font-weight: 700; }}
.map-uncertainty {{ font-size: 0.9rem; opacity: 0.8; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
</style>
</head>
<body>
<header>
<h1>{_e(report.title)}</h1>
<div class="summary">
<p><strong>StatFuzz report schema:</strong> {_e(report.schema_version)}</p>
<p><strong>Search:</strong> {_e(report.search.strategy)} ·
<strong>Objective:</strong> {_e(report.search.objective)} ·
<strong>Evaluated points:</strong> {_e(len(report.search.records))}</p>
</div>
</header>
{_validation_html(report)}
{failure_map_html}
{_shrink_html(report)}
{_search_table(report)}
</body>
</html>
"""


def write_html(report: StatFuzzReport, path: str | Path) -> Path:
    target = Path(path)
    target.write_text(render_html(report), encoding="utf-8")
    return target
