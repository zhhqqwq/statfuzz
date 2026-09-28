# StatFuzz Phase 1 batching benchmark — 2026-09-28

This file compares the frozen pre-batching baseline from GitHub Actions run
**36455152185** with the first completed Phase 1 batch-size-64 run
**36459592166**.

Both measurements use Python 3.12.14, NumPy 2.5.3,
SciPy 1.18.1, Linux/X64 GitHub-hosted runners, the same canonical
36-scenario matrix, and the same benchmark root seed.

The pre-batching raw data is stored in
[performance-2026-09-28.json](performance-2026-09-28.json). The Phase 1 raw data is
stored in [performance-phase1-batch64-2026-09-28.json](performance-phase1-batch64-2026-09-28.json).

## Correctness condition

Performance is accepted only together with the Phase 1 reproducibility tests:

- batch sizes 1, 2, 7, 64, and larger than the simulation budget;
- exact StressTestResult equality across built-in DGPs;
- exact logical group-1/group-2 sample call order and sample arrays;
- exact final NumPy bit-generator state;
- exact batched-vs-scalar Welch p-values on the tested sample pool;
- scalar zero-variance edge-case parity.

The optimized path does not batch random generation. It preserves the old draw order and
only vectorizes the SciPy t-tail evaluation.

## Overall result

| Metric | Pre-batching | Phase 1 batch=64 | Change |
| --- | ---: | ---: | ---: |
| Median end-to-end µs/rep | 148.958 | 63.766 | 56.9% lower |
| Median scenario speedup | — | 2.32x | — |
| Slowest scenario speedup | — | 1.95x | — |
| Fastest scenario speedup | — | 2.51x | — |

All **36/36** measured scenarios improved. The median speedup is far larger than the
approximately 1–2% hosted-runner variation observed when the pre-batching baseline was
repeated.

## By DGP

| DGP | Before median µs/rep | After median µs/rep | Median speedup | Median reduction |
| --- | ---: | ---: | ---: | ---: |
| Normal | 140.286 | 57.126 | 2.46x | 59.4% |
| LogNormal | 149.905 | 63.976 | 2.33x | 57.0% |
| StudentT | 146.948 | 62.034 | 2.35x | 57.5% |
| MixtureNormal | 170.857 | 79.208 | 2.15x | 53.5% |

## By sample size

| n | Before median µs/rep | After median µs/rep | Median speedup | Median reduction |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 145.328 | 60.712 | 2.39x | 58.2% |
| 50 | 148.377 | 63.766 | 2.33x | 57.1% |
| 200 | 157.389 | 73.857 | 2.15x | 53.4% |

## By simulation budget

| Simulations | Before median µs/rep | After median µs/rep | Median speedup | Median reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 150.433 | 64.551 | 2.31x | 56.6% |
| 10,000 | 147.934 | 62.962 | 2.35x | 57.4% |
| 100,000 | 147.699 | 62.949 | 2.33x | 57.0% |

## Full before/after matrix

| DGP | n | sims | Before µs/rep | After µs/rep | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 10 | 1000 | 141.244 | 57.126 | 2.47x | 59.6% |
| Normal | 10 | 10000 | 137.027 | 54.609 | 2.51x | 60.1% |
| Normal | 10 | 100000 | 136.617 | 54.672 | 2.50x | 60.0% |
| Normal | 50 | 1000 | 140.286 | 59.405 | 2.36x | 57.7% |
| Normal | 50 | 10000 | 137.906 | 55.961 | 2.46x | 59.4% |
| Normal | 50 | 100000 | 137.866 | 55.647 | 2.48x | 59.6% |
| Normal | 200 | 1000 | 145.018 | 62.928 | 2.30x | 56.6% |
| Normal | 200 | 10000 | 144.272 | 61.025 | 2.36x | 57.7% |
| Normal | 200 | 100000 | 142.372 | 61.397 | 2.32x | 56.9% |
| LogNormal | 10 | 1000 | 149.905 | 63.535 | 2.36x | 57.6% |
| LogNormal | 10 | 10000 | 146.970 | 61.756 | 2.38x | 58.0% |
| LogNormal | 10 | 100000 | 146.379 | 61.527 | 2.38x | 58.0% |
| LogNormal | 50 | 1000 | 150.961 | 65.435 | 2.31x | 56.7% |
| LogNormal | 50 | 10000 | 148.898 | 63.976 | 2.33x | 57.0% |
| LogNormal | 50 | 100000 | 149.019 | 63.864 | 2.33x | 57.1% |
| LogNormal | 200 | 1000 | 156.740 | 72.853 | 2.15x | 53.5% |
| LogNormal | 200 | 10000 | 155.869 | 71.575 | 2.18x | 54.1% |
| LogNormal | 200 | 100000 | 155.285 | 71.261 | 2.18x | 54.1% |
| StudentT | 10 | 1000 | 144.277 | 59.896 | 2.41x | 58.5% |
| StudentT | 10 | 10000 | 142.686 | 59.083 | 2.42x | 58.6% |
| StudentT | 10 | 100000 | 141.755 | 58.530 | 2.42x | 58.7% |
| StudentT | 50 | 1000 | 147.857 | 63.667 | 2.32x | 56.9% |
| StudentT | 50 | 10000 | 146.948 | 61.949 | 2.37x | 57.8% |
| StudentT | 50 | 100000 | 145.929 | 62.034 | 2.35x | 57.5% |
| StudentT | 200 | 1000 | 162.164 | 78.631 | 2.06x | 51.5% |
| StudentT | 200 | 10000 | 160.438 | 74.861 | 2.14x | 53.3% |
| StudentT | 200 | 100000 | 158.037 | 74.909 | 2.11x | 52.6% |
| MixtureNormal | 10 | 1000 | 170.281 | 77.896 | 2.19x | 54.3% |
| MixtureNormal | 10 | 10000 | 168.642 | 76.981 | 2.19x | 54.4% |
| MixtureNormal | 10 | 100000 | 168.549 | 76.331 | 2.21x | 54.7% |
| MixtureNormal | 50 | 1000 | 172.018 | 80.201 | 2.14x | 53.4% |
| MixtureNormal | 50 | 10000 | 170.259 | 79.122 | 2.15x | 53.5% |
| MixtureNormal | 50 | 100000 | 170.857 | 79.208 | 2.16x | 53.6% |
| MixtureNormal | 200 | 1000 | 181.216 | 93.086 | 1.95x | 48.6% |
| MixtureNormal | 200 | 10000 | 181.100 | 88.494 | 2.05x | 51.1% |
| MixtureNormal | 200 | 100000 | 180.431 | 88.072 | 2.05x | 51.2% |

## Interpretation

The Phase 1 result supports the original profiling hypothesis: scalar SciPy tail
evaluation was a major hot path. Batching only that part, while leaving random generation
and per-replicate mean/variance/statistic/df operations scalar, cuts median end-to-end
time by more than half.

The old scalar Welch and scalar tail columns in the benchmark remain independent
diagnostic microbenchmarks. After the optimized end-to-end path becomes faster than the
scalar Welch reference microbenchmark, ratios such as the historical
`welch_fraction_of_end_to_end` can exceed 1. They must not be interpreted as additive
runtime fractions.

The remaining cost is now relatively more concentrated in scalar sample generation and
scalar per-replicate moment/statistic construction, especially for larger n and
MixtureNormal. That does **not** automatically justify Stage B vectorization because
Stage B has a higher floating-point reproducibility risk.

## Next decision

Phase 1 meets both gates:

1. fixed-seed reproducibility is preserved by the strict test suite;
2. measured speedup is materially larger than hosted-runner noise.

The next Issue #40 work should therefore add progress events and checkpoint/resume around
the now-stable logical batch boundaries before considering more aggressive floating-point
vectorization.
