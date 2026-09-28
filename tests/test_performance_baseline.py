import json
import subprocess
import sys
from pathlib import Path


def test_performance_baseline_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    json_path = tmp_path / "benchmark.json"
    markdown_path = tmp_path / "benchmark.md"

    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "performance_baseline.py"),
            "--dgp",
            "Normal",
            "--n",
            "10",
            "--simulations",
            "8",
            "--batch-size",
            "7",
            "--json-out",
            str(json_path),
            "--markdown-out",
            str(markdown_path),
        ],
        cwd=root,
        check=True,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.1"
    assert payload["matrix"] == {
        "dgps": ["Normal"],
        "n": [10],
        "simulations": [8],
    }
    assert payload["metadata"]["execution_batch_size"] == 7
    assert len(payload["rows"]) == 1

    row = payload["rows"][0]
    assert row["dgp"] == "Normal"
    assert row["n"] == 10
    assert row["simulations"] == 8
    for field in (
        "end_to_end_seconds",
        "sampling_seconds",
        "welch_seconds",
        "scipy_tail_seconds",
        "python_loop_seconds",
        "end_to_end_us_per_replicate",
        "sampling_us_per_replicate",
        "welch_us_per_replicate",
        "scipy_tail_us_per_replicate",
        "python_loop_us_per_replicate",
    ):
        assert row[field] >= 0

    rendered = markdown_path.read_text(encoding="utf-8")
    assert "Component timings are independent microbenchmarks" in rendered
    assert "Normal" in rendered
