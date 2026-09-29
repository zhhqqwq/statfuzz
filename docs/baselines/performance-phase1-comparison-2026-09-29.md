# StatFuzz Phase 1 same-run performance comparison — 2026-09-29

This file records the canonical same-run A/B validation for Phase 1 batched execution.
Both execution modes were measured sequentially in GitHub Actions run
**36509821601** on the same runner, using the same commit,
Python / NumPy / SciPy versions, benchmark matrix, and root seed.

Reference mode uses `batch_size=1`. Candidate mode uses `batch_size=64`.

## Environment

- Python: 3.12.14
- NumPy: 2.5.3
- SciPy: 1.18.1
- Platform: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39
- Runner OS / arch: Linux / X64
- Root seed: 20260929
- Same-run workflow: 36509821601
- Merge-candidate SHA: `45b65275c56308a0e5835721f2940085fb0954d3`

## Summary

- Scenarios: **36**
- Median speedup: **2.145×**
- P05–P95 speedup: **1.872×–2.350×**
- Min–max speedup: **1.855×–2.396×**
- Median runtime reduction: **53.4%**

Because reference and candidate were executed on the same hosted runner, this comparison
does not attribute cross-runner CPU differences to Phase 1.

## By DGP

| DGP | Scalar median µs/rep | Batch64 median µs/rep | Median speedup | Median reduction |
| --- | ---: | ---: | ---: | ---: |
| LogNormal | 85.722 | 39.712 | 2.152× | 53.5% |
| MixtureNormal | 96.470 | 47.941 | 2.012× | 50.3% |
| Normal | 81.600 | 34.306 | 2.328× | 57.0% |
| StudentT | 83.243 | 38.876 | 2.139× | 53.3% |

## By sample size

| n | Scalar median µs/rep | Batch64 median µs/rep | Median speedup | Median reduction |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 82.798 | 37.157 | 2.242× | 55.4% |
| 50 | 84.693 | 39.333 | 2.146× | 53.4% |
| 200 | 94.060 | 47.565 | 1.967× | 49.1% |

## By simulation budget

| Simulations | Scalar median µs/rep | Batch64 median µs/rep | Median speedup | Median reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 86.864 | 40.879 | 2.104× | 52.5% |
| 10000 | 84.735 | 39.294 | 2.146× | 53.4% |
| 100000 | 84.637 | 38.935 | 2.165× | 53.8% |

## Interpretation

Phase 1 delivers a large, consistent improvement while preserving the exact fixed-seed
execution contract. The benefit is strongest for cheaper DGP/sample-size combinations,
where scalar tail-probability overhead was a larger fraction of total runtime. At
`n=200`, sampling and per-replicate moment calculations occupy a larger share, so the
speedup naturally narrows.

The slowest measured scenario still improves by 1.855×.
This is far above the percent-level hosted-runner noise observed in the pre-batching
baseline repeatability check.

The result supports keeping the Phase 1 default `batch_size=64`. It does **not** yet
justify vectorizing DGP sampling or changing random-stream semantics.

## Reproducibility evidence

Separate CI regression tests require `batch_size=1`, `2`, `7`, `64`, and a
value larger than the simulation budget to preserve:

- the complete `StressTestResult`;
- logical group-1/group-2 sample draw order;
- final NumPy bit-generator state;
- rejection count and uncertainty evidence;
- scalar-reference rejection decisions.

The optimization is therefore an execution change, not a statistical-experiment change.

## Full paired matrix

| DGP | n | sims | Scalar µs/rep | Batch64 µs/rep | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LogNormal | 10 | 1000 | 85.722 | 38.210 | 2.243× | 55.4% |
| LogNormal | 10 | 10000 | 83.746 | 38.570 | 2.171× | 53.9% |
| LogNormal | 10 | 100000 | 83.407 | 37.985 | 2.196× | 54.5% |
| LogNormal | 50 | 1000 | 86.863 | 41.361 | 2.100× | 52.4% |
| LogNormal | 50 | 10000 | 85.445 | 39.712 | 2.152× | 53.5% |
| LogNormal | 50 | 100000 | 84.892 | 38.955 | 2.179× | 54.1% |
| LogNormal | 200 | 1000 | 92.651 | 46.337 | 2.000× | 50.0% |
| LogNormal | 200 | 10000 | 93.841 | 45.145 | 2.079× | 51.9% |
| LogNormal | 200 | 100000 | 91.540 | 45.159 | 2.027× | 50.7% |
| MixtureNormal | 10 | 1000 | 95.281 | 46.644 | 2.043× | 51.0% |
| MixtureNormal | 10 | 10000 | 94.446 | 45.655 | 2.069× | 51.7% |
| MixtureNormal | 10 | 100000 | 93.643 | 45.803 | 2.044× | 51.1% |
| MixtureNormal | 50 | 1000 | 97.745 | 48.606 | 2.011× | 50.3% |
| MixtureNormal | 50 | 10000 | 96.470 | 47.941 | 2.012× | 50.3% |
| MixtureNormal | 50 | 100000 | 96.187 | 47.812 | 2.012× | 50.3% |
| MixtureNormal | 200 | 1000 | 103.990 | 55.988 | 1.857× | 46.2% |
| MixtureNormal | 200 | 10000 | 103.792 | 55.283 | 1.877× | 46.7% |
| MixtureNormal | 200 | 100000 | 104.510 | 56.325 | 1.855× | 46.1% |
| Normal | 10 | 1000 | 82.188 | 34.306 | 2.396× | 58.3% |
| Normal | 10 | 10000 | 80.770 | 33.936 | 2.380× | 58.0% |
| Normal | 10 | 100000 | 77.561 | 33.144 | 2.340× | 57.3% |
| Normal | 50 | 1000 | 81.600 | 35.357 | 2.308× | 56.7% |
| Normal | 50 | 10000 | 79.918 | 34.252 | 2.333× | 57.1% |
| Normal | 50 | 100000 | 79.267 | 34.046 | 2.328× | 57.0% |
| Normal | 200 | 1000 | 86.866 | 40.397 | 2.150× | 53.5% |
| Normal | 200 | 10000 | 84.024 | 38.455 | 2.185× | 54.2% |
| Normal | 200 | 100000 | 84.382 | 38.914 | 2.168× | 53.9% |
| StudentT | 10 | 1000 | 81.395 | 36.330 | 2.240× | 55.4% |
| StudentT | 10 | 10000 | 80.059 | 35.478 | 2.257× | 55.7% |
| StudentT | 10 | 100000 | 80.123 | 35.570 | 2.253× | 55.6% |
| StudentT | 50 | 1000 | 84.495 | 40.080 | 2.108× | 52.6% |
| StudentT | 50 | 10000 | 83.173 | 38.876 | 2.139× | 53.3% |
| StudentT | 50 | 100000 | 83.243 | 38.499 | 2.162× | 53.8% |
| StudentT | 200 | 1000 | 94.550 | 50.295 | 1.880× | 46.8% |
| StudentT | 200 | 10000 | 94.377 | 48.792 | 1.934× | 48.3% |
| StudentT | 200 | 100000 | 94.278 | 49.001 | 1.924× | 48.0% |

## Next step

Phase 1 is sufficient to move on to the next execution capability. The next recommended
node is a deterministic `SimulationProgress` callback emitted only after committed
logical batches. Checkpoint/resume should follow after progress semantics are stable.
