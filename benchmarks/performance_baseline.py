from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t

from statfuzz import stress_test
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.methods import welch_ttest_pvalue

SCHEMA_VERSION = "1.0"
DEFAULT_NS = (10, 50, 200)
DEFAULT_SIMULATIONS = (1_000, 10_000, 100_000)
ROOT_SEED = 20260929
POOL_SIZE = 64


@dataclass(frozen=True)
class BenchmarkRow:
    dgp: str
    n: int
    simulations: int
    end_to_end_seconds: float
    end_to_end_us_per_replicate: float
    sampling_seconds: float
    sampling_us_per_replicate: float
    welch_seconds: float
    welch_us_per_replicate: float
    scipy_tail_seconds: float
    scipy_tail_us_per_replicate: float
    python_loop_seconds: float
    python_loop_us_per_replicate: float
    sampling_fraction_of_end_to_end: float
    welch_fraction_of_end_to_end: float
    scipy_tail_fraction_of_welch: float


def _dgp_factories() -> dict[str, Callable[[], object]]:
    return {
        "Normal": lambda: Normal(),
        "LogNormal": lambda: LogNormal(sigma=1.0),
        "StudentT": lambda: StudentT(df=5.0),
        "MixtureNormal": lambda: MixtureNormal(),
    }


def _elapsed(call: Callable[[], object]) -> float:
    started = time.perf_counter_ns()
    call()
    return (time.perf_counter_ns() - started) / 1_000_000_000


def _scenario_seed(dgp_index: int, n_index: int, simulation_index: int) -> int:
    sequence = np.random.SeedSequence(
        [ROOT_SEED, dgp_index, n_index, simulation_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _sampling_benchmark(dgp: object, *, n: int, simulations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    sink = 0.0

    def run() -> None:
        nonlocal sink
        for _ in range(simulations):
            x = dgp.sample(rng, n)
            y = dgp.sample(rng, n)
            sink += float(x[0]) + float(y[0])

    elapsed = _elapsed(run)
    if not np.isfinite(sink):
        raise RuntimeError("sampling benchmark produced a non-finite sink")
    return elapsed


def _sample_pool(
    dgp: object,
    *,
    n: int,
    simulations: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    size = min(POOL_SIZE, simulations)
    xs = [np.asarray(dgp.sample(rng, n), dtype=float) for _ in range(size)]
    ys = [np.asarray(dgp.sample(rng, n), dtype=float) for _ in range(size)]
    return xs, ys


def _welch_parameters(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n1 = x.size
    n2 = y.size
    mean1 = float(np.mean(x))
    mean2 = float(np.mean(y))
    v1 = float(np.var(x, ddof=1))
    v2 = float(np.var(y, ddof=1))
    denom2 = v1 / n1 + v2 / n2
    if not np.isfinite(denom2) or denom2 <= 0:
        raise RuntimeError("benchmark sample produced invalid Welch denominator")
    statistic = (mean1 - mean2) / np.sqrt(denom2)
    df_num = denom2**2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den
    if not np.isfinite(statistic) or not np.isfinite(df) or df <= 0:
        raise RuntimeError("benchmark sample produced invalid Welch parameters")
    return float(statistic), float(df)


def _welch_benchmark(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    *,
    simulations: int,
) -> float:
    sink = 0.0
    pool_size = len(xs)

    def run() -> None:
        nonlocal sink
        for index in range(simulations):
            slot = index % pool_size
            sink += welch_ttest_pvalue(xs[slot], ys[slot])

    elapsed = _elapsed(run)
    if not np.isfinite(sink):
        raise RuntimeError("Welch benchmark produced a non-finite sink")
    return elapsed


def _tail_benchmark(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    *,
    simulations: int,
) -> float:
    parameters = [_welch_parameters(x, y) for x, y in zip(xs, ys)]
    sink = 0.0
    pool_size = len(parameters)

    def run() -> None:
        nonlocal sink
        for index in range(simulations):
            statistic, df = parameters[index % pool_size]
            sink += float(2.0 * t.sf(abs(statistic), df=df))

    elapsed = _elapsed(run)
    if not np.isfinite(sink):
        raise RuntimeError("SciPy tail benchmark produced a non-finite sink")
    return elapsed


def _loop_benchmark(*, simulations: int) -> float:
    sink = 0

    def run() -> None:
        nonlocal sink
        for index in range(simulations):
            sink += index & 1

    elapsed = _elapsed(run)
    if sink < 0:
        raise RuntimeError("unreachable loop sink")
    return elapsed


def _end_to_end_benchmark(
    dgp: object,
    *,
    n: int,
    simulations: int,
    seed: int,
) -> float:
    def run() -> None:
        stress_test(
            method="welch_ttest",
            metric="type1_error",
            dgp=dgp,
            n1=n,
            n2=n,
            simulations=simulations,
            seed=seed,
        )

    return _elapsed(run)


def benchmark_scenario(
    *,
    dgp_name: str,
    dgp_index: int,
    n: int,
    n_index: int,
    simulations: int,
    simulation_index: int,
) -> BenchmarkRow:
    dgp = _dgp_factories()[dgp_name]()
    seed = _scenario_seed(dgp_index, n_index, simulation_index)

    # Small untimed warm-up avoids measuring one-time import/lazy-dispatch effects.
    warm_rng = np.random.default_rng(seed)
    warm_x = dgp.sample(warm_rng, n)
    warm_y = dgp.sample(warm_rng, n)
    welch_ttest_pvalue(warm_x, warm_y)
    t.sf(1.0, df=max(2, 2 * n - 2))

    end_to_end = _end_to_end_benchmark(
        dgp,
        n=n,
        simulations=simulations,
        seed=seed,
    )
    sampling = _sampling_benchmark(
        dgp,
        n=n,
        simulations=simulations,
        seed=seed + 1,
    )
    xs, ys = _sample_pool(
        dgp,
        n=n,
        simulations=simulations,
        seed=seed + 2,
    )
    welch = _welch_benchmark(xs, ys, simulations=simulations)
    scipy_tail = _tail_benchmark(xs, ys, simulations=simulations)
    python_loop = _loop_benchmark(simulations=simulations)

    scale = 1_000_000 / simulations
    end_to_end_us = end_to_end * scale
    sampling_us = sampling * scale
    welch_us = welch * scale
    scipy_tail_us = scipy_tail * scale
    loop_us = python_loop * scale

    return BenchmarkRow(
        dgp=dgp_name,
        n=n,
        simulations=simulations,
        end_to_end_seconds=end_to_end,
        end_to_end_us_per_replicate=end_to_end_us,
        sampling_seconds=sampling,
        sampling_us_per_replicate=sampling_us,
        welch_seconds=welch,
        welch_us_per_replicate=welch_us,
        scipy_tail_seconds=scipy_tail,
        scipy_tail_us_per_replicate=scipy_tail_us,
        python_loop_seconds=python_loop,
        python_loop_us_per_replicate=loop_us,
        sampling_fraction_of_end_to_end=sampling_us / end_to_end_us,
        welch_fraction_of_end_to_end=welch_us / end_to_end_us,
        scipy_tail_fraction_of_welch=scipy_tail_us / welch_us,
    )


def _metadata() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "root_seed": ROOT_SEED,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "github_actions": os.getenv("GITHUB_ACTIONS") == "true",
        "github_runner_os": os.getenv("RUNNER_OS"),
        "github_runner_arch": os.getenv("RUNNER_ARCH"),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_sha": os.getenv("GITHUB_SHA"),
    }


def _markdown(payload: dict[str, object]) -> str:
    metadata = payload["metadata"]
    rows = payload["rows"]
    lines = [
        "# StatFuzz Monte Carlo performance baseline",
        "",
        "Canonical environment: GitHub Actions ubuntu-latest + Python 3.12.",
        "",
        "Component timings are independent microbenchmarks and are **not additive**.",
        "They are diagnostic comparisons against end-to-end runtime, not a profiler",
        "decomposition.",
        "",
        "## Environment",
        "",
        f"- Python: {metadata['python']}",
        f"- NumPy: {metadata['numpy']}",
        f"- SciPy: {metadata['scipy']}",
        f"- Platform: {metadata['platform']}",
        f"- Runner arch: {metadata['github_runner_arch']}",
        f"- GitHub run id: {metadata['github_run_id']}",
        f"- Commit: {metadata['github_sha']}",
        "",
        "## Results",
        "",
        "| DGP | n | sims | end-to-end µs/rep | sampling µs/rep | Welch µs/rep | SciPy tail µs/rep | loop µs/rep | sampling/e2e | Welch/e2e | tail/Welch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dgp} | {n} | {simulations} | {end_to_end_us_per_replicate:.3f} | "
            "{sampling_us_per_replicate:.3f} | {welch_us_per_replicate:.3f} | "
            "{scipy_tail_us_per_replicate:.3f} | {python_loop_us_per_replicate:.3f} | "
            "{sampling_fraction_of_end_to_end:.3f} | {welch_fraction_of_end_to_end:.3f} | "
            "{scipy_tail_fraction_of_welch:.3f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dgp",
        action="append",
        choices=tuple(_dgp_factories()),
        help="Restrict to one or more DGPs.",
    )
    parser.add_argument(
        "--n",
        action="append",
        type=int,
        dest="ns",
        help="Restrict to one or more sample sizes.",
    )
    parser.add_argument(
        "--simulations",
        action="append",
        type=int,
        help="Restrict to one or more simulation budgets.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark-results.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmark-results.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dgp_names = tuple(args.dgp or _dgp_factories().keys())
    ns = tuple(args.ns or DEFAULT_NS)
    simulations_values = tuple(args.simulations or DEFAULT_SIMULATIONS)

    if any(n < 2 for n in ns):
        raise ValueError("all n values must be at least 2")
    if any(value <= 0 for value in simulations_values):
        raise ValueError("all simulation budgets must be positive")

    rows: list[BenchmarkRow] = []
    for dgp_index, dgp_name in enumerate(dgp_names):
        for n_index, n in enumerate(ns):
            for simulation_index, simulations in enumerate(simulations_values):
                print(
                    f"benchmarking {dgp_name}: n={n}, simulations={simulations}",
                    file=sys.stderr,
                    flush=True,
                )
                rows.append(
                    benchmark_scenario(
                        dgp_name=dgp_name,
                        dgp_index=dgp_index,
                        n=n,
                        n_index=n_index,
                        simulations=simulations,
                        simulation_index=simulation_index,
                    )
                )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": _metadata(),
        "matrix": {
            "dgps": list(dgp_names),
            "n": list(ns),
            "simulations": list(simulations_values),
        },
        "rows": [asdict(row) for row in rows],
    }

    args.json_out.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
