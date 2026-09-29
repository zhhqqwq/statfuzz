# Performance benchmarking

StatFuzz measures Monte Carlo performance before changing the execution model. The
canonical baseline is produced by `benchmarks/performance_baseline.py` on GitHub
Actions `ubuntu-latest` with Python 3.12.

## Canonical matrix

The baseline covers all combinations of:

```text
DGP:
  Normal
  LogNormal(sigma=1)
  StudentT(df=5)
  MixtureNormal(default two-component mixture)

n:
  10
  50
  200

simulations:
  1,000
  10,000
  100,000
```

Both groups use the same DGP and the same sample size in this baseline. Every built-in
DGP has population mean zero, so the Type-I equal-means null is verified before timing
the simulation loop.

## What is measured

Each matrix point records five timings.

### End-to-end

A real `stress_test(..., method="welch_ttest", metric="type1_error")` call. This
includes the current null verification, per-replicate sample generation, sample
validation, Welch calculation, rejection counting, MCSE, and Wilson interval evidence.

### DGP sampling

Two raw DGP samples per replicate. This measures the relative cost of random generation
and each distribution family's transformation logic.

### Welch calculation

Repeated calls to the production `welch_ttest_pvalue()` on a small pre-generated pool
of valid sample pairs. It includes finite-input checks, means, variances, degrees of
freedom, and the SciPy t-tail call.

### SciPy tail probability

Repeated scalar calls to `scipy.stats.t.sf` on precomputed Welch statistic / degrees-of-
freedom pairs. This isolates the approximate cost of the probability-tail portion inside
the Welch calculation.

### Python replicate loop

A tiny integer operation in a Python `for` loop. This establishes the order of
magnitude of bare loop dispatch without statistical work.

## Interpretation

The component timings are **independent microbenchmarks**. They are not an additive
decomposition of end-to-end time.

For example, the Welch-only benchmark reuses pre-generated arrays, while the end-to-end
path allocates fresh arrays and performs checked sampling. Cache behavior and allocation
patterns differ. Ratios such as `Welch / end-to-end` or `SciPy tail / Welch` are
diagnostic signals, not accounting identities.

No performance PASS/FAIL thresholds are attached to the first baseline because hosted
GitHub runners are noisy. The first purpose is to identify which layer dominates by an
order of magnitude and to give later optimization work a reproducible reference.

## Outputs

The benchmark emits:

```text
benchmark-results.json
benchmark-results.md
```

JSON schema version `1.1` contains:

- runner and dependency metadata;
- the end-to-end execution batch size;
- the exact benchmark matrix;
- seconds per scenario;
- microseconds per replicate;
- diagnostic component/end-to-end ratios.

The GitHub Actions workflow now runs two full matrices on the **same runner**:

- scalar reference: `batch_size=1`;
- Phase 1 candidate: `batch_size=64`.

It then uses `benchmarks/compare_performance.py` to pair all 36 matching scenarios and
produce a same-run speedup comparison. This avoids attributing runner-to-runner CPU
differences to batching.

The workflow uploads the scalar, batched, and comparison JSON/Markdown files as
artifacts, and writes the paired comparison plus batched table to the job summary.

## Running a small local check

A local smoke run can use a reduced matrix:

```bash
python benchmarks/performance_baseline.py \
  --dgp Normal \
  --n 10 \
  --simulations 100 \
  --batch-size 64
```

Omitting those filters runs the full canonical matrix.

## Phase 1 batched execution

The first execution optimization keeps DGP sampling scalar and preserves the original
logical draw order exactly:

```text
replicate 1: group 1 -> group 2
replicate 2: group 1 -> group 2
...
```

Only the final SciPy t-tail evaluation is vectorized. Mean, variance, Welch statistic,
and degrees-of-freedom calculations still use the scalar reference operations for each
replicate. The default execution batch size is 64; `batch_size=1` uses the original
scalar Welch reference path.

The benchmark CLI records the end-to-end execution batch size and accepts
`--batch-size 1` for direct scalar-reference measurements. The canonical Actions
workflow runs `batch_size=1` and `batch_size=64` sequentially on the same hosted
runner before computing speedup, so the Phase 1 performance claim does not depend on
cross-runner comparisons.

## Batching reproducibility contract

Progress callbacks and checkpoint/resume are still future work, but batched execution
already obeys this required invariant:

> For the same experiment configuration and root seed, changing only execution
> `batch_size` must not change the final statistical result or the random stream
> assigned to any logical replicate.

The future execution design must therefore separate:

```text
experiment identity
        +
root seed
        +
logical replicate index
        ↓
replicate random stream

batch_size
        ↓
execution scheduling only
```

In particular, `batch_size` must never participate in seed derivation. Checkpoint/resume
must also resume at logical replicate boundaries without replaying or skipping a
replicate.

This invariant is stronger than merely obtaining a similar empirical rejection rate:
for a fixed seed and experiment, the logical sequence of replicate outcomes must remain
identical across supported batch sizes.

### Reproducibility regression matrix

The Phase 1 CI suite treats `batch_size=1` as the scalar reference and compares it
against `batch_size=2`, `7`, `64`, and a value larger than the simulation budget.

The regression suite covers all four built-in DGPs at representative sample sizes
`n=10` and `n=37`, and requires the complete `StressTestResult` to compare equal.
A recording custom DGP separately verifies the exact logical sample-draw sequence.
An internal executor test uses asymmetric `n1=13` / `n2=17` and requires the final
NumPy bit-generator state and rejection count to match the scalar reference exactly.

The batched Welch helper is also checked against the scalar oracle for rejection
decisions and zero-variance overrides. These tests intentionally verify execution
semantics rather than accepting approximate Monte Carlo agreement.

## Next design step

After Phase 1 is benchmarked against the frozen pre-batching baseline:

1. keep the scalar implementation as the reproducibility oracle;
2. decide whether vectorizing means/variances is justified by the measured remaining
   bottleneck;
3. add progress callbacks defined on completed logical batches;
4. add checkpoint state containing experiment identity, root seed, completed replicate
   count, rejection count, and exact NumPy bit-generator state;
5. require resumed execution to match an uninterrupted run exactly, even when resume
   uses a different batch size.
