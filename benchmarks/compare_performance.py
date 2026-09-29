from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def _load(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    if not isinstance(data.get("rows"), list):
        raise TypeError(f"{path} must contain a rows array")
    return data


def _row_key(row: dict[str, object]) -> tuple[str, int, int]:
    dgp = row.get("dgp")
    n = row.get("n")
    simulations = row.get("simulations")
    if not isinstance(dgp, str):
        raise TypeError("benchmark row dgp must be a string")
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("benchmark row n must be an integer")
    if not isinstance(simulations, int) or isinstance(simulations, bool):
        raise TypeError("benchmark row simulations must be an integer")
    return dgp, n, simulations


def _end_to_end_us(row: dict[str, object]) -> float:
    value = row.get("end_to_end_us_per_replicate")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("end_to_end_us_per_replicate must be numeric")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError("end_to_end_us_per_replicate must be positive")
    return numeric


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def compare(
    reference: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    reference_rows = reference["rows"]
    candidate_rows = candidate["rows"]
    assert isinstance(reference_rows, list)
    assert isinstance(candidate_rows, list)

    reference_map = {_row_key(row): row for row in reference_rows}
    candidate_map = {_row_key(row): row for row in candidate_rows}

    if set(reference_map) != set(candidate_map):
        missing = sorted(set(reference_map) - set(candidate_map))
        extra = sorted(set(candidate_map) - set(reference_map))
        raise ValueError(
            "benchmark matrices do not match: "
            f"missing_candidate={missing}, extra_candidate={extra}"
        )

    paired: list[dict[str, object]] = []
    for key in sorted(reference_map):
        reference_row = reference_map[key]
        candidate_row = candidate_map[key]
        reference_us = _end_to_end_us(reference_row)
        candidate_us = _end_to_end_us(candidate_row)
        speedup = reference_us / candidate_us
        reduction = 1.0 - candidate_us / reference_us
        paired.append(
            {
                "dgp": key[0],
                "n": key[1],
                "simulations": key[2],
                "reference_us_per_replicate": reference_us,
                "candidate_us_per_replicate": candidate_us,
                "speedup": speedup,
                "runtime_reduction": reduction,
            }
        )

    speedups = [float(row["speedup"]) for row in paired]
    reductions = [float(row["runtime_reduction"]) for row in paired]
    return {
        "schema_version": "1.0",
        "reference_metadata": reference.get("metadata"),
        "candidate_metadata": candidate.get("metadata"),
        "summary": {
            "scenarios": len(paired),
            "median_speedup": median(speedups),
            "p05_speedup": _quantile(speedups, 0.05),
            "p95_speedup": _quantile(speedups, 0.95),
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "median_runtime_reduction": median(reductions),
        },
        "rows": paired,
    }


def _markdown(result: dict[str, object]) -> str:
    summary = result["summary"]
    rows = result["rows"]
    reference_metadata = result.get("reference_metadata")
    candidate_metadata = result.get("candidate_metadata")
    assert isinstance(summary, dict)
    assert isinstance(rows, list)

    lines = [
        "# StatFuzz paired performance comparison",
        "",
        "Reference and candidate were measured in the same workflow job on the same runner.",
        "",
        "## Execution modes",
        "",
        f"- Reference metadata: `{json.dumps(reference_metadata, sort_keys=True)}`",
        f"- Candidate metadata: `{json.dumps(candidate_metadata, sort_keys=True)}`",
        "",
        "## Summary",
        "",
        f"- Scenarios: {summary['scenarios']}",
        f"- Median speedup: {float(summary['median_speedup']):.3f}x",
        (
            f"- P05–P95 speedup: {float(summary['p05_speedup']):.3f}x–"
            f"{float(summary['p95_speedup']):.3f}x"
        ),
        (
            f"- Min–max speedup: {float(summary['min_speedup']):.3f}x–"
            f"{float(summary['max_speedup']):.3f}x"
        ),
        (
            "- Median runtime reduction: "
            f"{100.0 * float(summary['median_runtime_reduction']):.1f}%"
        ),
        "",
        "## Per-scenario results",
        "",
        "| DGP | n | sims | reference µs/rep | candidate µs/rep | speedup | reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        assert isinstance(row, dict)
        lines.append(
            "| {dgp} | {n} | {simulations} | "
            "{reference_us_per_replicate:.3f} | "
            "{candidate_us_per_replicate:.3f} | "
            "{speedup:.3f}x | {reduction:.1f}% |".format(
                **row,
                reduction=100.0 * float(row["runtime_reduction"]),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark-comparison.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmark-comparison.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare(_load(args.reference), _load(args.candidate))
    args.json_out.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
