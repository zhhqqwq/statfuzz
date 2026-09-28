# StatFuzz performance baseline — 2026-09-28

This file freezes the first canonical pre-batching performance measurement produced by
GitHub Actions run **36455152185**.

The benchmark methodology is defined in [../BENCHMARKING.md](../BENCHMARKING.md).
Component timings are independent microbenchmarks and **must not be added together** as
an exact decomposition of end-to-end runtime.

## Environment

- GitHub Actions: true
- Runner OS / arch: Linux / X64
- Python: 3.12.14
- NumPy: 2.5.3
- SciPy: 1.18.1
- Platform: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39
- Benchmark schema: 1.0
- Root seed: 20260929
- Workflow run: 36455152185
- Measured PR merge-candidate SHA: `7696156fac372e4427c1a6df50c091b2883429a0`

## Executive measurements

Across all 36 matrix points:

| Metric | Min | Median | P95 | Max |
| --- | ---: | ---: | ---: | ---: |
| End-to-end µs/rep | 136.617 | 148.958 | 180.598 | 181.216 |
| Sampling µs/rep | 3.043 | 10.375 | 31.619 | 32.535 |
| Welch µs/rep | 121.353 | 122.726 | 124.117 | 128.210 |
| SciPy t-tail µs/rep | 56.323 | 57.355 | 58.213 | 58.311 |
| Bare loop µs/rep | 0.040 | 0.048 | 0.053 | 0.058 |

Median diagnostic ratios:

- Welch microbenchmark / end-to-end: **82.4%**
- SciPy t-tail / Welch microbenchmark: **46.7%**
- SciPy t-tail / end-to-end: **38.7%**
- DGP sampling / end-to-end: **7.0%**
- bare Python loop / end-to-end: **0.0%**

## Scaling with simulation budget

| Simulations | Median end-to-end µs/rep | Median sampling µs/rep | Median Welch µs/rep | Median tail µs/rep |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 150.433 | 10.461 | 123.181 | 57.854 |
| 10,000 | 147.934 | 10.402 | 122.384 | 57.404 |
| 100,000 | 147.699 | 10.309 | 122.186 | 57.039 |

The per-replicate cost is already close to stable by 10,000–100,000 simulations. The
1,000-replicate runs are only modestly higher, so fixed setup cost is not the main
performance problem.

## Scaling with sample size

| n | Median end-to-end µs/rep | Median sampling µs/rep | Median Welch µs/rep | Median tail µs/rep |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 145.328 | 8.070 | 122.698 | 57.882 |
| 50 | 148.377 | 10.375 | 121.915 | 57.244 |
| 200 | 157.389 | 19.368 | 123.108 | 57.165 |

Welch time is almost flat over n=10–200 in this regime. The increase in end-to-end time
with n is driven much more visibly by DGP sampling/allocation than by the scalar Welch
calculation.

## DGP comparison

| DGP | Median end-to-end µs/rep | Median sampling µs/rep | Median Welch/e2e | Median sampling/e2e |
| --- | ---: | ---: | ---: | ---: |
| Normal | 140.286 | 4.105 | 87.7% | 3.0% |
| LogNormal | 149.905 | 10.825 | 81.8% | 7.2% |
| StudentT | 146.948 | 9.978 | 83.4% | 6.8% |
| MixtureNormal | 170.857 | 24.332 | 71.1% | 14.3% |

MixtureNormal has the largest sampling cost, while Normal has the smallest. Even for
MixtureNormal, the scalar Welch path remains the largest single diagnostic component.

## What the baseline supports

1. **Bare Python loop dispatch is negligible by itself.** The median loop-only cost is
   0.048 µs/rep, roughly 0.0% of end-to-end time.
   Replacing the syntax of the loop alone has essentially no useful optimization ceiling.
2. **Welch evaluation is the dominant target.** Its median diagnostic ratio is
   82.4% of end-to-end runtime.
3. **The scalar SciPy t-tail call is a major part of Welch.** It accounts for about
   46.7% of the Welch microbenchmark
   and 38.7% of end-to-end runtime.
4. **Sampling matters for heavier generators and larger n.** It ranges from
   3.043 to 32.535 µs/rep and reaches about
   18.0% of end-to-end in
   the measured matrix.
5. **Large simulation budgets scale approximately linearly.** Per-replicate timings at
   10k and 100k are close, so the next design should reduce per-replicate statistical
   overhead rather than focus on one-time setup.

These findings support exploring batched/vectorized sample statistics and vectorized
tail-probability evaluation. They do **not** yet justify changing random-stream semantics.

## Full matrix

| DGP | n | sims | end-to-end µs/rep | sampling µs/rep | Welch µs/rep | SciPy tail µs/rep | loop µs/rep | sampling/e2e | Welch/e2e | tail/Welch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 10 | 1000 | 141.244 | 3.072 | 123.150 | 58.311 | 0.058 | 0.022 | 0.872 | 0.473 |
| Normal | 10 | 10000 | 137.027 | 3.043 | 122.936 | 58.156 | 0.047 | 0.022 | 0.897 | 0.473 |
| Normal | 10 | 100000 | 136.617 | 3.061 | 122.625 | 58.060 | 0.048 | 0.022 | 0.898 | 0.473 |
| Normal | 50 | 1000 | 140.286 | 4.160 | 123.077 | 57.910 | 0.041 | 0.030 | 0.877 | 0.471 |
| Normal | 50 | 10000 | 137.906 | 4.105 | 123.077 | 57.555 | 0.049 | 0.030 | 0.892 | 0.468 |
| Normal | 50 | 100000 | 137.866 | 4.081 | 122.003 | 57.094 | 0.048 | 0.030 | 0.885 | 0.468 |
| Normal | 200 | 1000 | 145.018 | 8.115 | 123.211 | 57.580 | 0.042 | 0.056 | 0.850 | 0.467 |
| Normal | 200 | 10000 | 144.272 | 8.092 | 124.351 | 57.292 | 0.050 | 0.056 | 0.862 | 0.461 |
| Normal | 200 | 100000 | 142.372 | 8.367 | 122.683 | 56.838 | 0.048 | 0.059 | 0.862 | 0.463 |
| LogNormal | 10 | 1000 | 149.905 | 9.295 | 128.210 | 57.857 | 0.041 | 0.062 | 0.855 | 0.451 |
| LogNormal | 10 | 10000 | 146.970 | 9.194 | 121.933 | 57.911 | 0.048 | 0.063 | 0.830 | 0.475 |
| LogNormal | 10 | 100000 | 146.379 | 9.188 | 122.883 | 57.762 | 0.053 | 0.063 | 0.839 | 0.470 |
| LogNormal | 50 | 1000 | 150.961 | 10.921 | 122.846 | 58.250 | 0.040 | 0.072 | 0.814 | 0.474 |
| LogNormal | 50 | 10000 | 148.898 | 10.825 | 122.381 | 57.726 | 0.052 | 0.073 | 0.822 | 0.472 |
| LogNormal | 50 | 100000 | 149.019 | 10.750 | 121.828 | 56.964 | 0.050 | 0.072 | 0.818 | 0.468 |
| LogNormal | 200 | 1000 | 156.740 | 17.050 | 123.097 | 57.161 | 0.041 | 0.109 | 0.785 | 0.464 |
| LogNormal | 200 | 10000 | 155.869 | 16.945 | 122.303 | 57.043 | 0.053 | 0.109 | 0.785 | 0.466 |
| LogNormal | 200 | 100000 | 155.285 | 16.965 | 123.120 | 57.169 | 0.048 | 0.109 | 0.793 | 0.464 |
| StudentT | 10 | 1000 | 144.277 | 6.952 | 122.770 | 57.183 | 0.042 | 0.048 | 0.851 | 0.466 |
| StudentT | 10 | 10000 | 142.686 | 6.918 | 122.077 | 57.908 | 0.047 | 0.048 | 0.856 | 0.474 |
| StudentT | 10 | 100000 | 141.755 | 6.890 | 121.799 | 57.395 | 0.047 | 0.049 | 0.859 | 0.471 |
| StudentT | 50 | 1000 | 147.857 | 10.001 | 123.946 | 58.201 | 0.042 | 0.068 | 0.838 | 0.470 |
| StudentT | 50 | 10000 | 146.948 | 9.978 | 121.465 | 57.031 | 0.050 | 0.068 | 0.827 | 0.470 |
| StudentT | 50 | 100000 | 145.929 | 9.867 | 121.704 | 57.393 | 0.053 | 0.068 | 0.834 | 0.472 |
| StudentT | 200 | 1000 | 162.164 | 21.740 | 123.668 | 57.318 | 0.043 | 0.134 | 0.763 | 0.463 |
| StudentT | 200 | 10000 | 160.438 | 21.788 | 122.387 | 57.204 | 0.050 | 0.136 | 0.763 | 0.467 |
| StudentT | 200 | 100000 | 158.037 | 21.686 | 122.370 | 56.983 | 0.053 | 0.137 | 0.774 | 0.466 |
| MixtureNormal | 10 | 1000 | 170.281 | 22.596 | 123.255 | 58.152 | 0.042 | 0.133 | 0.724 | 0.472 |
| MixtureNormal | 10 | 10000 | 168.642 | 22.394 | 122.575 | 57.516 | 0.052 | 0.133 | 0.727 | 0.469 |
| MixtureNormal | 10 | 100000 | 168.549 | 22.236 | 121.353 | 56.670 | 0.049 | 0.132 | 0.720 | 0.467 |
| MixtureNormal | 50 | 1000 | 172.018 | 24.828 | 121.738 | 56.323 | 0.040 | 0.144 | 0.708 | 0.463 |
| MixtureNormal | 50 | 10000 | 170.259 | 24.332 | 121.713 | 56.788 | 0.049 | 0.143 | 0.715 | 0.467 |
| MixtureNormal | 50 | 100000 | 170.857 | 24.213 | 121.492 | 56.497 | 0.049 | 0.142 | 0.711 | 0.465 |
| MixtureNormal | 200 | 1000 | 181.216 | 32.535 | 124.038 | 57.851 | 0.042 | 0.180 | 0.684 | 0.466 |
| MixtureNormal | 200 | 10000 | 181.100 | 31.922 | 123.276 | 56.623 | 0.050 | 0.176 | 0.681 | 0.459 |
| MixtureNormal | 200 | 100000 | 180.431 | 31.518 | 123.046 | 56.713 | 0.050 | 0.175 | 0.682 | 0.461 |

## Design consequence for the next node

The next execution design should prioritize batching **statistical work**, not merely
wrapping the existing scalar loop in chunks. Candidate work should separately benchmark:

- batched DGP sampling with a replicate axis;
- vectorized means/variances and Welch statistic/df calculation;
- vectorized `scipy.stats.t.sf`;
- validation cost for batched finite checks.

The required reproducibility invariant remains unchanged:

> For the same experiment configuration and root seed, changing only `batch_size`
> must not change the final result or the random stream assigned to any logical
> replicate.

This baseline establishes the before state; no performance threshold or optimization is
claimed here.
