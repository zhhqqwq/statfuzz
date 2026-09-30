# Bootstrap Mean Interval Coverage Design

This document defines the minimal design for adding bootstrap mean confidence
interval coverage to StatFuzz without weakening the reproducibility guarantees
established by Issue #40.

This is a design-only phase. It does not implement bootstrap resampling,
coverage simulation, a new public API, checkpoint integration, or report
changes.

## 1. Statistical goal

The first new statistical capability is deliberately narrow:

> Estimate the finite-sample coverage of a one-sample percentile bootstrap
> confidence interval for the population mean under a controlled DGP.

For each outer Monte Carlo replicate:

1. draw one sample of size n from a DGP;
2. construct a percentile bootstrap interval for the sample mean using a fixed
   number of bootstrap resamples;
3. compare the interval with the known population mean;
4. record one Bernoulli event: covered or not covered.

Across the outer simulations:

- coverage_count is the number of intervals containing the true mean;
- empirical is coverage_count / simulations;
- nominal is the bootstrap interval confidence level;
- deviation is empirical - nominal;
- MCSE and the Monte Carlo confidence interval are binomial-proportion evidence,
  exactly as for Type-I error.

The first method is percentile bootstrap only. BCa, basic bootstrap,
studentized intervals, other estimands, and two-sample bootstrap procedures are
out of scope.

## 2. Existing abstractions that are already reusable

### DGP sampling and identity

DataGenerator.sample(rng, n) and DGPIdentity are directly reusable. All built-in
DGPs already expose a finite population_mean, which supplies the truth required
by mean coverage.

### Parameter spaces and search seeds

ParameterPoint, ParameterSpace, deterministic point-seed derivation,
grid_search(), and random_search() are independent of Welch. Their core
contract remains:

~~~text
parameter point + search root seed
-> deterministic child experiment seed
~~~

### Discovery and independent validation

DiscoveryBudget, find_counterexample(), and validate_candidate() already
separate exploratory search from independent hold-out validation. Coverage
needs the same selection discipline.

### Search objectives

The current absolute, positive, and negative deviation objectives are directly
meaningful for coverage:

- absolute deviation: worst total miscoverage;
- positive deviation: over-coverage;
- negative deviation: under-coverage.

Their scoring formulas do not need to change.

### Shrinking

ShrinkPlan, FailureCriterion, ObjectiveThresholdCriterion,
OutsideToleranceCriterion, parameter shrinking, and family shrinking are also
conceptually reusable. Their algorithms only require a result exposing a small
common statistical-property interface.

## 3. Current Type-I-specific seams that need generalization

### stress_test()

The current entry point hard-codes method=welch_ttest and metric=type1_error. It
also assumes two groups, n1/n2, a mean-equality null, p-values, and rejection
decisions.

Do not grow this into a string dispatcher with bootstrap-specific branches.
The first bootstrap capability should use a dedicated bootstrap coverage API.

### StressTestResult

StressTestResult contains useful generic fields:

- method;
- metric;
- simulations;
- seed;
- nominal;
- empirical;
- mcse;
- tolerance;
- deviation;
- passed/status.

But it also embeds Type-I-specific structure:

- dgp1 / dgp2;
- n1 / n2;
- MeanNullCheck;
- rejection_count.

A one-sample coverage result should not fake a second group or call a successful
coverage event a rejection.

### Search result type checks

Search, validation, discovery, and shrinking currently require
StressTestResult by concrete type. Their algorithms are already generic; the
type boundary is not.

### Type1ErrorEvidence

Type-I error evidence is mathematically generic Bernoulli-rate evidence.
Coverage should reuse the same calculation rather than duplicate it.

### Progress and checkpoint count names

SimulationProgress.rejections and CheckpointState.rejections are Type-I-specific
names. They do not have to change before the first scalar bootstrap prototype,
but bootstrap should not expose those names as coverage semantics.

## 4. Minimal common result contract

Do not replace StressTestResult immediately.

Introduce a small structural result contract for algorithms that only need a
statistical property estimate:

~~~text
StatisticalPropertyResult
- method
- metric
- simulations
- seed
- nominal
- empirical
- mcse
- tolerance
- deviation
- passed
- status
~~~

StressTestResult already satisfies this shape.

Search, validation, discovery, objective scoring, and shrinking should depend on
this narrow contract instead of the concrete Type-I result class.

A future BootstrapCoverageResult can then expose the same common interface
without pretending to be a two-sample hypothesis-test result.

## 5. Minimal bootstrap result

Bootstrap coverage should use a dedicated immutable result with fields
conceptually equivalent to:

~~~text
BootstrapCoverageResult
- method = bootstrap_percentile_mean
- metric = coverage
- dgp / dgp_identity
- n
- simulations
- seed

- bootstrap_resamples
- bootstrap_interval_level
- bootstrap_quantile_method

- nominal
- empirical
- mcse
- tolerance
- coverage_count
- mean_target_check

- evidence_confidence_level
- evidence_interval_method
- evidence_interval_low
- evidence_interval_high
~~~

Two confidence levels must remain visibly distinct:

1. bootstrap_interval_level: nominal level of every inner bootstrap interval;
2. evidence_confidence_level: confidence level of the outer Monte Carlo Wilson
   interval around estimated coverage.

Do not reuse one ambiguous confidence_level name for both meanings.

## 6. Generalize Bernoulli-rate evidence

Introduce generic BinomialRateEvidence and binomial_rate_evidence(event_count,
trials, ...).

Then:

- Type-I error uses event_count = rejection_count;
- coverage uses event_count = coverage_count.

The current type1_error_evidence() should remain as a compatibility wrapper or
thin specialization so existing behavior does not change.

## 7. Mean truth contract

Coverage requires a known target parameter, not a null hypothesis.

MeanEqualityNull and MeanNullCheck should not be reused because that would
conflate an equal-means null with a known population parameter.

Add a separate truth contract conceptually equivalent to:

~~~text
MeanTarget(mean, note=None)

MeanTargetCheck
- source = population_mean | declaration
- mean
- declared_population_mean
- note
~~~

Resolution rules mirror the current conservative null logic:

- if the DGP exposes a finite population_mean, use it;
- if the caller also declares a mean target, require exact agreement;
- if the DGP does not expose a mean, require an explicit declaration;
- reject non-finite truth values before simulation.

The existing population_mean_of() helper can be reused.

## 8. Minimal bootstrap method model

Do not build a universal method framework yet.

Use one dedicated immutable method configuration:

~~~text
BootstrapMeanPercentile
- resamples
- interval_level
- quantile_method = linear
~~~

The scalar reference semantics are:

~~~text
sample x
for bootstrap replicate b = 0..R-1:
    draw n indices with replacement
    compute mean(x[indices])
take lower and upper empirical quantiles
return interval
~~~

The NumPy quantile method is statistical semantics and must be explicit. The
first implementation should support exactly one documented quantile method and
include it in method identity/fingerprinting.

Do not optimize bootstrap resampling before the scalar reference is locked by
fixed-seed tests.

## 9. RNG semantics

Bootstrap is the first StatFuzz method that consumes randomness internally.

### 9.1 Separate outer data randomness from bootstrap randomness

The outer DGP stream should preserve current StatFuzz semantics:

~~~text
root experiment seed
-> one outer data Generator
-> sample_0, sample_1, sample_2, ...
~~~

Bootstrap resampling must not consume from this same Generator.

With one shared Generator, changing bootstrap_resamples would change later DGP
samples and outer batch_size could change the interleaving of data and method
randomness.

### 9.2 Derive one bootstrap child RNG per logical outer replicate

For logical outer replicate i, derive a deterministic child seed from:

- root experiment seed;
- a bootstrap-specific domain separator;
- RNG derivation semantics version;
- logical outer replicate index.

Use a stable hash construction consistent with StatFuzz point/family seed
derivation.

Conceptually:

~~~text
bootstrap_child_seed_i =
    H(root_seed, bootstrap_mean_percentile, rng_semantics_version, i)
~~~

Instantiate an explicit concrete BitGenerator, for example PCG64, rather than
depending on the future default of default_rng().

### 9.3 Consequences

This ensures that changing only:

- outer batch_size;
- checkpoint boundary;
- resume batch_size;
- bootstrap evaluation scheduling;

cannot change the bootstrap stream assigned to a logical outer replicate.

Changing bootstrap_resamples also does not change the outer DGP sample sequence;
it only changes the method calculation for each fixed outer sample.

### 9.4 Canonical inner draw order

The scalar oracle defines:

~~~text
bootstrap replicate 0: n index draws
bootstrap replicate 1: n index draws
...
bootstrap replicate R-1: n index draws
~~~

Any future vectorized implementation must prove exact resampled-index
assignment, exact interval output, and unchanged coverage decisions.

## 10. Outer batching and commit semantics

Outer batch_size remains execution scheduling only.

For logical replicate i:

~~~text
draw outer sample from data_rng
-> construct interval with child_rng(i)
-> evaluate covered/not-covered event
~~~

A completed outer batch may be committed only after every interval in that
logical batch has been evaluated and coverage_count has been accumulated.

## 11. Checkpoint implications

The current checkpoint experiment schema is specific to Welch Type-I error. It
hard-codes two DGPs, n1/n2, a mean-equality null, and rejection state.

Do not force bootstrap fields into that schema.

A bootstrap experiment fingerprint should include at least:

~~~text
experiment_kind = bootstrap_mean_coverage

method
- name = bootstrap_percentile_mean
- semantics_version
- bootstrap_resamples
- interval_level
- quantile_method
- bootstrap_rng_scheme

metric
- name = coverage
- target = interval_level

dgp identity
mean target check
sample size n
outer simulations
tolerance
Monte Carlo evidence settings
root seed
~~~

batch_size remains excluded.

Because bootstrap method randomness is regenerated from root_seed +
logical_index, checkpoint state does not need a persistent bootstrap RNG state.
It only needs outer data RNG state plus committed aggregate state.

The committed aggregate count should eventually become generic event_count, or
an experiment-specific state payload, rather than pretending coverage events
are rejections.

## 12. Search / validation / shrinking integration

The algorithms should remain unchanged.

Only the result contract needs to widen from concrete StressTestResult to the
common StatisticalPropertyResult interface.

The following logic is reusable as-is:

- point seed derivation;
- grid/random search;
- objective scoring and ranking;
- independent validation seed derivation;
- search/validation budget split;
- outside-tolerance criteria;
- objective-threshold criteria;
- parameter shrinking;
- DGP-family shrinking.

Existing generic rows based on nominal, empirical, deviation, mcse, simulations,
seed, and status remain meaningful for coverage.

## 13. Recommended public API direction

For the first bootstrap capability, prefer a dedicated API rather than extending
stress_test(method=..., metric=...).

Conceptually:

~~~text
bootstrap_mean_coverage(
    dgp,
    n=20,
    simulations=1000,
    bootstrap_resamples=999,
    interval_level=0.95,
    tolerance=0.01,
    seed=0,
    mean_target=None,
    evidence_confidence_level=0.95,
    evidence_interval_method=wilson,
    batch_size=64,
    progress_callback=None,
)
-> BootstrapCoverageResult
~~~

Only after a second non-Type-I experiment exists should StatFuzz consider a
larger public Experiment / run_experiment() framework.

## 14. What should not be generalized yet

Do not generalize these parts merely for symmetry:

- optimized Welch batch executor;
- Welch p-value batching;
- two-sample Type-I null contract;
- report schema;
- StatCI serialization schema;
- generic arbitrary estimands;
- generic arbitrary interval methods.

The first bootstrap capability should prove the smallest useful seams before
expanding them.

## 15. Proposed implementation phases

### Bootstrap Phase A — result/search seam only

No bootstrap statistics yet.

- add StatisticalPropertyResult contract;
- make search/discovery/validation/shrinking depend on it;
- generalize binomial-rate evidence;
- keep all Welch behavior and exact tests unchanged.

### Bootstrap Phase B — truth + scalar bootstrap oracle

- add mean-target contract;
- add percentile bootstrap mean configuration;
- add scalar interval oracle;
- lock quantile semantics;
- lock per-outer-replicate child RNG derivation;
- add fixed-seed child-stream tests.

No outer coverage simulation yet.

### Bootstrap Phase C — coverage executor/result

- add BootstrapCoverageResult;
- run outer Monte Carlo coverage;
- prove outer batch_size cannot change DGP sample sequence, child bootstrap
  stream assignment, coverage decisions, or final outer RNG state.

### Bootstrap Phase D — checkpoint/progress integration

- generalize committed event-count naming/state;
- define bootstrap experiment fingerprint;
- add checkpoint/resume for coverage;
- prove fixed- and cross-batch resume equivalence.

### Bootstrap Phase E — search/shrink integration

- run coverage through existing grid/random search;
- independent validation;
- parameter shrinking;
- family shrinking;
- confirm objective ranking and deterministic search seeds remain unchanged.

## 16. Acceptance criteria

The first complete bootstrap capability requires:

- explicit validated population-mean truth;
- deterministic scalar bootstrap interval semantics for a fixed child seed;
- separation of outer DGP randomness from bootstrap method randomness;
- batch-size invariance of logical samples and child-stream assignment;
- internally consistent empirical coverage, MCSE, and Wilson evidence;
- search/validation/shrinking without Type-I-specific branches;
- exact checkpoint/resume once bootstrap checkpoint support is added.

The governing principle remains:

> improve the credibility of the statistical result before expanding the number
> of statistical methods.


## 17. Bootstrap Phase B1 implementation status

Phase B1 implements the mean-truth contract only.

- `MeanTarget` records an explicit population-mean declaration;
- `MeanTargetCheck` records the resolved truth evidence;
- built-in DGPs resolve automatically through their finite `population_mean`;
- custom DGPs without a population mean require an explicit declaration;
- a declaration that conflicts with a known DGP population mean fails before
  simulation;
- non-finite declared or DGP population means fail loudly;
- resolved checks have stable `as_dict()/from_dict()` serialization for later
  result/fingerprint integration.

Bootstrap resampling, percentile intervals, child RNG derivation, and coverage
execution remain out of scope after Phase B1.


## 18. Bootstrap Phase B2 implementation status

Phase B2 implements the immutable percentile-bootstrap method configuration only.

- `BootstrapMeanPercentile` fixes the method identity to
  `bootstrap_mean_percentile` with semantics version `1`;
- `resamples` must be a positive integer and is normalized to a Python `int`;
- `interval_level` must be finite and strictly between zero and one and is
  normalized to a Python `float`;
- the only supported quantile method is `linear`;
- `as_dict()`, `from_dict()`, and `canonical_json()` provide stable machine
  identity and exact round-trip semantics;
- unknown identity fields, unsupported method names, and unsupported semantics
  versions fail loudly.

Bootstrap sample generation, percentile interval calculation, and child RNG
derivation remain out of scope after Phase B2.


## 19. Bootstrap Phase B3 implementation status

Phase B3 implements the scalar percentile-bootstrap mean interval oracle only.

- the caller supplies a fixed one-dimensional finite sample and an existing
  NumPy `Generator`;
- each bootstrap replicate makes exactly one
  `rng.integers(0, n, size=n)` call;
- the replicate mean is computed immediately after that index draw;
- all replicate means are completed before the percentile quantiles are
  evaluated;
- the interval uses the method contract's only supported quantile method,
  `linear`;
- non-finite replicate means and invalid samples fail loudly;
- fixed-PCG64 reference tests lock the known resample indices, replicate means,
  interval result, and final RNG consumption.

Child-seed derivation, bootstrap RNG ownership, outer coverage simulation, and
checkpoint integration remain out of scope after Phase B3.


## 20. Bootstrap Phase B4 implementation status

Phase B4 locks deterministic bootstrap child-stream derivation only.

For each logical outer replicate, the child seed is derived from a canonical
JSON payload containing the bootstrap-specific domain separator, RNG semantics
version `1`, experiment root seed, and logical outer index. The payload is
hashed with BLAKE2b-128 using fixed personalization `statfuzz-bsprng1`, and
the digest is interpreted as an unsigned big-endian integer.

The child Generator is explicitly:

`numpy.random.Generator(numpy.random.PCG64(child_seed))`

and never depends on `default_rng()`.

Fixed-reference tests lock exact child seeds and initial PCG64 streams for
selected logical indices. Additional tests prove:

- the same logical index recreates the same stream;
- different logical indices receive distinct child seeds/streams;
- requesting child streams in a different order does not change any stream;
- consuming one child stream cannot perturb another;
- NumPy integral inputs normalize to the same derivation;
- derivation does not consume or depend on NumPy's legacy global RNG state.

Outer DGP sampling, coverage events, coverage aggregation, batching, progress,
and checkpoint/resume remain out of scope after Phase B4.


## 21. Bootstrap Phase C1 implementation status

Phase C1 introduces the aggregate result contract and one-logical-replicate
coverage event only.

`BootstrapCoverageResult` is a frozen statistical-property result backed by
the existing generic `BinomialRateEvidence`. Its nominal target is the
bootstrap method's `interval_level`, its event count is `coverage_count`,
and its PASS / OUTSIDE_TOLERANCE rule remains absolute deviation from nominal
coverage.

The internal scalar event helper composes the already-locked Phase B
contracts:

`MeanTargetCheck + BootstrapMeanPercentile + root seed + logical outer index`

then:

1. derives the deterministic Phase B4 child `PCG64` Generator;
2. calls the Phase B3 scalar percentile interval oracle on the supplied sample;
3. evaluates coverage using the inclusive rule
   `interval_low <= target_mean <= interval_high`;
4. returns an immutable event containing the logical index, target, interval,
   and covered flag.

Fixed-reference tests lock a known child-derived interval and prove that equality
with either interval boundary counts as covered. The event contract also rejects
a manually inconsistent covered flag.

Outer DGP sampling, simulations loops, batching, progress, checkpoint/resume,
reporting, and search integration remain out of scope after Phase C1.


## 22. Bootstrap Phase C2 implementation status

Phase C2 adds the scalar outer coverage executor only.

The public `bootstrap_mean_coverage(...)` function validates experiment
configuration and resolves the mean target before sampling. Its outer data RNG
is explicitly:

`numpy.random.Generator(numpy.random.PCG64(root_seed))`

The internal scalar reference loop evaluates logical outer replicates strictly
in ascending index order:

1. draw exactly one outer sample from the shared outer data Generator;
2. derive the already-locked per-index bootstrap child Generator;
3. call the Phase B3 scalar percentile interval oracle;
4. evaluate the Phase C1 inclusive coverage event;
5. increment the cumulative coverage count.

No batch buffering or reordering is present.

Fixed-reference tests lock the complete outer Normal sample sequence, complete
coverage-event sequence, final coverage count, and final outer RNG consumption
for a small experiment. A separate test changes bootstrap resample count while
holding the root seed and outer experiment fixed and proves that the complete
outer sample sequence and final outer RNG state remain exactly unchanged.

All four built-in DGPs are exercised through the public scalar executor.
Invalid configuration and unresolved mean truth fail before the first outer
draw, and malformed DGP samples report the exact logical outer index.

Batching, progress, checkpoint/resume, reporting, and search integration remain
out of scope after Phase C2.


## 23. Bootstrap Phase C3 implementation status

Phase C3 adds outer batching while deliberately keeping bootstrap evaluation
scalar.

`bootstrap_mean_coverage(...)` now accepts `batch_size` with default `64`.
The C2 scalar executor is retained as an independent reference oracle.

For each outer batch:

1. outer samples are generated one-by-one in ascending logical-index order from
   the shared outer `PCG64` Generator;
2. the completed samples in that batch are evaluated one-by-one in the same
   logical-index order;
3. each logical index still receives its deterministic Phase B4 child
   `PCG64` stream;
4. each interval still uses the Phase B3 scalar bootstrap oracle;
5. coverage counts are committed in logical order.

No bootstrap resampling is vectorized.

Strict equivalence tests cover all four built-in DGPs and
`batch_size=1/2/7/64/>simulations`. Every batched execution is compared
against the independent C2 scalar executor for:

- the complete outer DGP sample sequence;
- the complete `BootstrapCoverageEvent` sequence;
- final `coverage_count`;
- final outer BitGenerator state and following random stream.

The public `BootstrapCoverageResult` is also required to compare exactly equal
to the `batch_size=1` result for the same full matrix of DGPs and batch sizes.

Invalid batch sizes fail before the first outer draw.

Progress callbacks, checkpoint/resume, reporting, search integration, and
bootstrap vectorization remain out of scope after Phase C3.


## 24. Bootstrap Phase D1 implementation status

Phase D1 adds progress observation only; checkpoint/resume remains out of scope.

`BootstrapCoverageProgress(completed, total, covered, empirical)` is a frozen
cumulative snapshot. It validates that:

- `completed` and `total` are positive and `completed <= total`;
- `covered` is a non-negative integer not exceeding `completed`;
- `empirical` is finite and exactly equals `covered / completed`.

The batched coverage executor accepts an optional progress callback. The commit
order is:

1. generate the complete outer batch;
2. evaluate every coverage event in that batch;
3. add the batch's coverage events to cumulative `covered`;
4. emit one progress snapshot for the committed batch.

There is no progress event at zero, and a failed batch emits no progress
snapshot. Callback return values are ignored; callback exceptions propagate
immediately after the committed batch snapshot is delivered.

Exact invariance tests cover all four built-in DGPs and
`batch_size=1/2/7/64/>simulations`. With and without a callback, they require
exact equality of:

- public `BootstrapCoverageResult`;
- complete outer sample sequence;
- complete `BootstrapCoverageEvent` sequence;
- cumulative coverage count;
- final outer BitGenerator state and following RNG stream.

Checkpointing, resume, reporting, search integration, and bootstrap
vectorization remain out of scope after Phase D1.


## 25. Bootstrap Phase D2 implementation status

Phase D2 defines coverage-specific checkpoint identity and committed state only.

`BootstrapCoverageExperimentSpec` is the canonical persistent experiment
identity. Its fingerprint is computed through the existing
`ExecutionContract + ExperimentFingerprint(SHA-256)` infrastructure.

The fingerprint includes the canonical experiment payload containing:

- `BootstrapMeanPercentile` identity;
- coverage metric semantics version;
- stable `DGPIdentity`;
- resolved `MeanTargetCheck`;
- `n` and `simulations`;
- tolerance;
- evidence confidence level and interval method;
- root seed.

`batch_size` and progress callbacks are explicitly excluded from the
experiment payload and therefore cannot change its fingerprint. Unknown
execution-control fields are rejected by strict experiment deserialization.

Persistent bootstrap coverage experiments require an explicit stable
`DGPIdentity`; the transient repr-based custom-DGP fallback is not accepted.

`BootstrapCoverageCheckpointState` records only the committed execution state:

- completed logical outer replicates;
- cumulative covered intervals;
- lossless outer RNG snapshot.

The outer RNG snapshot is required to be the explicit PCG64 stream used by the
coverage executor. State round-trip tests restore the Generator and require the
following random stream to be exactly equal.

Phase D2 does not add checkpoint persistence, checkpoint cadence, resume
execution, or any `bootstrap_mean_coverage()` checkpoint argument.


## 26. Bootstrap Phase D3 implementation status

Phase D3 adds checkpoint persistence and an internal minimal resume path.

`BootstrapCoverageCheckpoint` is the persistent container for the Phase D2
experiment and committed state. Its schema is
`statfuzz.bootstrap_coverage.checkpoint/1`. The container requires a
self-consistent experiment/execution fingerprint, validates that the committed
state does not exceed the experiment budget, and requires the state outer RNG
type to match the execution contract.

Persistence semantics mirror the already-proven Welch checkpoint path:

- strict JSON object schemas;
- fingerprint validation during load;
- atomic same-directory temporary file;
- flush + `fsync`;
- `os.replace`;
- failed replacement preserves the old destination and cleans the temp file.

The coverage outer executor now has an internal logical-range kernel. An
uninterrupted run executes `[0, simulations)` from zero covered events. The
internal resume helper validates the runtime DGP and execution contract before
any draw, restores the exact outer PCG64 snapshot, then executes
`[completed, simulations)` with the checkpoint's cumulative covered count.

Fixed-batch resume tests run a real prefix, persist/read the checkpoint, resume,
and compare against uninterrupted execution. They require exact equality of:

- final `BootstrapCoverageResult`;
- complete outer DGP sample sequence;
- complete `BootstrapCoverageEvent` sequence;
- final cumulative coverage count;
- final outer BitGenerator state and following RNG stream.

The equivalence is tested across all four built-in DGPs for one fixed batch
size, with additional fixed-batch boundary tests for batch sizes 1, 2, and 7.
A mismatched runtime DGP must fail fingerprint validation before any new outer
draw, and a fully completed checkpoint resumes without drawing again.

Changing batch size across the checkpoint boundary is intentionally not tested
in D3 and remains Phase D4. No public `bootstrap_mean_coverage()` resume API
exists yet.


## 27. Bootstrap Phase D4 implementation status

Phase D4 validates cross-batch resume semantics and does not broaden the public
API.

The checkpoint boundary remains a fully committed logical boundary, but the
outer batch size used before and after that boundary may differ. The validated
transition set is `1->7`, `7->2`, `2->64`, and `64->1`.

For each transition and every built-in DGP, a real prefix is executed using the
prefix batch size, a checkpoint is captured at that complete batch boundary,
and the same checkpoint is resumed using the new batch size. The resulting
execution is compared against the independent scalar outer oracle.

Exact equality is required for:

- the complete outer DGP sample sequence;
- the complete coverage-event sequence;
- final `BootstrapCoverageResult`;
- final cumulative coverage count;
- final outer PCG64 state and following random stream.

The tests also confirm that batch scheduling remains absent from persistent
experiment/fingerprint identity, so the same checkpoint validates when the
resume batch size changes.

No production code change was required: the D3 range executor already starts
new scheduling at `checkpoint.state.completed`, outer sample assignment is
strictly logical-index ordered, and bootstrap child streams are derived solely
from root seed plus logical index.

Phase D is therefore complete at the internal reliability layer. A public
checkpoint/resume workflow remains intentionally deferred.


## 28. Bootstrap Phase E1 implementation status

Phase E1 validates search, discovery, and independent validation with real
`BootstrapCoverageResult` values.

The existing Phase A structural seam was sufficient: grid search, random search,
candidate validation, discovery budgeting, objective ranking, and multiplicity
diagnostics already depend on `StatisticalPropertyResult` rather than concrete
`StressTestResult`. No seed-derivation or ranking algorithm change was
required.

Integration tests now exercise `bootstrap_mean_coverage(...)` directly
through the existing search APIs and lock:

- exact deterministic per-point seeds for a canonical bootstrap parameter grid;
- deterministic random-search sampled indices and point seeds;
- reproducible objective ranking for repeated bootstrap searches;
- independent validation seeds derived from a distinct validation root seed;
- search and validation simulation budgets in `find_counterexample(...)`;
- real `BootstrapCoverageResult` objects at both search and validation stages.

Only stale Type-I-specific wording in seed-passthrough error messages was
generalized to refer to the statistical evaluation. Search algorithms,
objectives, seed derivation, discovery budgets, and validation selection logic
remain unchanged.

Shrinking, report, StatCI, and public checkpoint/resume integration remain out
of scope after Phase E1.


## 29. Bootstrap Phase E2 implementation status

Phase E2 validates parameter shrinking with real
`BootstrapCoverageResult` values.

The Phase A structural result seam was already sufficient:
`shrink_counterexample(...)` and its failure criteria operate on
`StatisticalPropertyResult`, so no shrinking algorithm, seed derivation,
criterion, or complexity rule needed to change.

The fixed Bootstrap reference trace uses:

- root seed `23`;
- 40 outer simulations per shrink-stage evaluation;
- percentile method with 19 bootstrap resamples and interval level 0.8;
- tolerance 0.10;
- start point `n=20, sigma=1.4`;
- shrink levels `n=(4,8,12,20)` and
  `sigma=(0.1,0.4,0.8,1.4)`.

Tests lock the exact deterministic seeds and coverage counts for the start and
every proposal. The trace contains both behaviors required by the greedy
shrinker:

- a simpler point that remains outside tolerance and is accepted;
- a simpler point whose coverage returns to the nominal target and is rejected.

The final point is `n=4, sigma=0.4`. The n dimension is already at its
simplest level, while the only simpler sigma proposal (`0.1`) no longer
satisfies the failure criterion, making the result a stable local minimum under
the explicit shrink plan.

Repeating the entire shrink run must produce exactly equal results and rows.
Repeating the same rejected proposal within the trace must reuse the same
point-derived seed and reproduce the same Bootstrap coverage result.

Only stale Type-I-specific seed-passthrough wording in the parameter shrinker
was generalized to refer to the statistical evaluation. Family shrinking,
report, and StatCI remain out of scope after Phase E2.


## 30. Bootstrap Phase E3 implementation status

Phase E3 validates DGP-family shrinking with real
`BootstrapCoverageResult` values.

The existing family shrinker already consumes `StatisticalPropertyResult`, so
no family-seed derivation, candidate ordering, criterion, or complexity rule
needed to change.

The fixed Bootstrap reference path uses:

- family shrink root seed `1`;
- 40 outer simulations per family evaluation;
- common sample size `n=8`;
- percentile method with 19 bootstrap resamples and interval level 0.8;
- tolerance 0.10;
- canonical family order:
  `Normal -> LogNormal -> StudentT -> MixtureNormal`.

The run starts from the most complex canonical MixtureNormal candidate. Tests
lock the exact deterministic family seeds and coverage evidence:

- MixtureNormal start: `24/40 = 0.6`, outside tolerance;
- Normal proposal: `31/40 = 0.775`, inside tolerance and rejected;
- LogNormal proposal: `16/40 = 0.4`, outside tolerance and accepted.

The shrinker therefore stops at LogNormal. Normal is the only simpler canonical
family and does not preserve the failure, so LogNormal is the simplest
still-failing family under the explicit plan.

Repeating the full family shrink must produce exactly equal
`FamilyShrinkResult`, trace rows, seeds, and Bootstrap coverage results.

Only stale Type-I-specific seed-passthrough wording in the family shrinker was
generalized to refer to the statistical evaluation. Report and StatCI remain
out of scope after Phase E3.
