import json
import subprocess
import sys
from pathlib import Path


def _payload(batch_size, runtime):
    return {
        "schema_version": "1.1",
        "metadata": {"execution_batch_size": batch_size},
        "matrix": {
            "dgps": ["Normal"],
            "n": [10],
            "simulations": [100],
        },
        "rows": [
            {
                "dgp": "Normal",
                "n": 10,
                "simulations": 100,
                "end_to_end_us_per_replicate": runtime,
            }
        ],
    }


def test_compare_performance_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    json_path = tmp_path / "comparison.json"
    markdown_path = tmp_path / "comparison.md"

    reference_path.write_text(
        json.dumps(_payload(1, 100.0)),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(_payload(64, 40.0)),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "compare_performance.py"),
            str(reference_path),
            str(candidate_path),
            "--json-out",
            str(json_path),
            "--markdown-out",
            str(markdown_path),
        ],
        cwd=root,
        check=True,
    )

    result = json.loads(json_path.read_text(encoding="utf-8"))

    assert result["schema_version"] == "1.0"
    assert result["summary"]["scenarios"] == 1
    assert result["summary"]["median_speedup"] == 2.5
    assert result["summary"]["median_runtime_reduction"] == 0.6
    assert result["rows"][0]["reference_us_per_replicate"] == 100.0
    assert result["rows"][0]["candidate_us_per_replicate"] == 40.0

    rendered = markdown_path.read_text(encoding="utf-8")
    assert "Median speedup: 2.500x" in rendered
    assert "60.0%" in rendered


def test_compare_performance_rejects_mismatched_matrix(tmp_path):
    root = Path(__file__).resolve().parents[1]
    reference = _payload(1, 100.0)
    candidate = _payload(64, 40.0)
    candidate["rows"][0]["n"] = 20

    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "compare_performance.py"),
            str(reference_path),
            str(candidate_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "benchmark matrices do not match" in completed.stderr
