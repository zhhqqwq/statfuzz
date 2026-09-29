# Checkpoint / Resume Design

This document defines the Phase 3A contract for resumable StatFuzz simulations.

Phase 3A is deliberately design-only. It does **not** implement checkpoint file
I/O, resume execution, or cross-batch resume tests. Those follow in later
phases of Issue #40.

## 1. Design goals

A checkpoint must let StatFuzz continue an interrupted Monte Carlo experiment
without changing the logical experiment or its random stream.

The eventual resumed run must be exactly equivalent to an uninterrupted run
with the same experiment configuration and RNG origin.

The design therefore separates three concepts:

1. **Experiment semantics**: what is being simulated and how the final result is
   interpreted.
2. **Execution compatibility**: the implementation/runtime assumptions required
   to reproduce the same stream and statistical decisions.
3. **Committed state**: the completed logical replicate count, accumulated
   rejection count, and exact RNG state at a committed boundary.

Execution scheduling is intentionally not experiment identity.

## 2. Current contracts this design builds on

### StressTestResult

The current result contains:

- method / metric;
- human-readable DGP names;
- structured DGP identities;
- verified null evidence;
- n1 / n2;
- total simulations;
- root seed;
- nominal alpha;
- empirical rejection rate;
- rejection count;
- MCSE;
- tolerance;
- confidence level / interval method / interval bounds.

A checkpoint fingerprint must not be built from the whole result. Empirical
values, MCSE, interval bounds, and status are outputs, not experiment
configuration.

Human-readable DGP names are display-only and must never be used as persistent
identity.

### DGPIdentity

Built-in DGPs expose a canonical structured identity containing:

- a stable family identifier; and
- full scalar parameter values.

Those identities are suitable for experiment fingerprints.

The fallback identity returned for custom DGPs without an explicit
`DGPIdentity` includes `repr(dgp)`. A default Python repr may contain a memory
address and is therefore not a reliable persistent identity.

**Checkpoint rule:** persistent checkpointing must require each custom DGP to
provide an explicit stable `DGPIdentity`. The conservative fallback remains
acceptable for transient results, but must not be accepted for checkpoint
fingerprints.

### Null contract

Type-I error experiments resolve to a `MeanNullCheck` before sampling starts.
The resolved check records:

- kind;
- verification source;
- common mean;
- group population means when known;
- optional declaration note.

The canonical checkpoint experiment payload should use the resolved null check,
not merely the caller's raw `null=` argument. This makes implicit verified
built-in nulls and explicit custom-DGP declarations machine-readable.

For the initial schema, the complete `MeanNullCheck.as_dict()` representation,
including `note`, is part of experiment semantics. This is intentionally
strict because the note is persisted in `StressTestResult`; changing it would
otherwise prevent exact result equality.

### Seed, batching, and progress

The current simulation creates one NumPy `Generator` from the root seed and
consumes draws in logical replicate order:

```text
x1 -> y1 -> x2 -> y2 -> ...
```

`batch_size` only schedules evaluation of already-generated samples and is
proven not to change the final result, logical sample sequence, or final RNG
state.

`SimulationProgress` is emitted only after a complete logical batch has been
committed.

Therefore:

- checkpoint boundaries are committed logical replicate boundaries;
- no partially generated/evaluated batch may be persisted;
- `batch_size` is excluded from the experiment fingerprint;
- progress callback identity is excluded from the experiment fingerprint;
- resume may choose a different `batch_size`.

## 3. Canonical experiment payload

Phase 3B should introduce a versioned canonical experiment payload equivalent
to the following structure:

```json
{
  "schema": "statfuzz.experiment/1",
  "method": {
    "name": "welch_ttest",
    "semantics_version": "1"
  },
  "metric": {
    "name": "type1_error",
    "semantics_version": "1"
  },
  "dgp1": {
    "family": "statfuzz.dgp.Normal",
    "parameters": {
      "mean": 0.0,
      "sd": 1.0
    }
  },
  "dgp2": {
    "family": "statfuzz.dgp.Normal",
    "parameters": {
      "mean": 0.0,
      "sd": 1.0
    }
  },
  "null": {
    "kind": "equal_means",
    "source": "population_means",
    "common_mean": 0.0,
    "group1_population_mean": 0.0,
    "group2_population_mean": 0.0,
    "note": null
  },
  "sample_sizes": {
    "n1": 20,
    "n2": 20
  },
  "budget": {
    "simulations": 10000
  },
  "decision": {
    "alpha": 0.05
  },
  "result_semantics": {
    "tolerance": 0.01,
    "confidence_level": 0.95,
    "interval_method": "wilson"
  },
  "rng_origin": {
    "root_seed": 42
  }
}
```

The payload must use the resolved group-2 DGP identity even when the public
caller passed `dgp2=None`.

### Included fields

The fingerprint must include fields that can change at least one of:

- generated samples;
- rejection decisions;
- total Monte Carlo budget;
- final `StressTestResult` semantics.

For the current Welch Type-I error experiment this means:

- experiment schema version;
- method name and method semantics version;
- metric name and metric semantics version;
- canonical DGP identities for both groups;
- resolved null contract;
- n1 / n2;
- total simulations;
- alpha;
- tolerance;
- confidence level;
- interval method;
- RNG origin.

### Explicitly excluded fields

The following are execution or presentation details and must not participate in
the experiment fingerprint:

- `batch_size`;
- progress callback identity;
- progress callback cadence;
- checkpoint path;
- checkpoint write frequency;
- temporary file path;
- wall-clock timestamps;
- elapsed time / ETA;
- human-readable DGP display names;
- current `completed` count;
- current rejection count;
- current RNG state.

The final four state values belong to a checkpoint instance, not to experiment
identity.

## 4. Execution compatibility contract

A semantic experiment payload alone is not sufficient for exact resume.

For the first implementation, StatFuzz should validate a deliberately strict
execution contract:

```json
{
  "statfuzz_execution_semantics": "1",
  "statfuzz_version": "0.1.1.dev0",
  "numpy_version": "<exact installed version>",
  "scipy_version": "<exact installed version>",
  "bit_generator": "<fully-qualified concrete BitGenerator class>"
}
```

Why this is separate from the semantic payload:

- NumPy's concrete BitGenerator determines the underlying random stream;
- NumPy distribution transforms can affect generated samples;
- SciPy tail evaluation can affect p-values close to the rejection boundary;
- StatFuzz method/DGP implementation changes can affect execution semantics.

The initial resume implementation should fail closed on an execution contract
mismatch. Compatibility can be relaxed later only after explicit evidence and a
versioned migration policy exist.

The fingerprint used for resume validation should cover both the semantic
experiment payload and this execution compatibility contract.

## 5. RNG origin and seed=None

A root integer seed is sufficient to identify the intended RNG origin only when
combined with the execution contract and concrete BitGenerator.

`seed=None` is different: NumPy obtains fresh unpredictable entropy, so two
runs with `seed=None` do not share an RNG origin even if every other experiment
field matches.

Therefore the Phase 3B implementation must choose one of these strict policies:

1. **Preferred:** support `seed=None` by capturing the exact initial
   BitGenerator state before the first draw and include a digest of that initial
   state in `rng_origin`.
2. **Simpler initial restriction:** reject persistent checkpointing when
   `seed=None`.

Do not treat `root_seed: null` by itself as a unique resume-compatible RNG
origin.

For integer seeds, Phase 3B should normalize accepted values to a plain Python
integer before canonical serialization and reject booleans.

## 6. Canonical fingerprint

The resume fingerprint should be a SHA-256 digest over canonical UTF-8 JSON of:

```text
{
  "experiment": <canonical experiment payload>,
  "execution": <execution compatibility contract>
}
```

Canonical JSON rules:

- UTF-8;
- object keys sorted;
- compact separators;
- no NaN / Infinity;
- DGP parameters use their full structured identity values;
- no human-readable display formatting;
- no batch/progress/checkpoint scheduling fields.

The checkpoint should store both:

- the canonical payloads, for diagnostics and validation errors; and
- the digest, for fast equality checking.

A digest mismatch must fail loudly before additional sampling begins.

## 7. Checkpoint schema

The initial machine-readable checkpoint should be equivalent to:

```json
{
  "schema": "statfuzz.checkpoint/1",
  "fingerprint": {
    "algorithm": "sha256",
    "value": "<hex digest>"
  },
  "experiment": {
    "...": "canonical experiment payload"
  },
  "execution": {
    "...": "execution compatibility contract"
  },
  "state": {
    "completed": 128,
    "rejections": 7,
    "rng": {
      "bit_generator": "<fully-qualified concrete BitGenerator class>",
      "state": {
        "...": "exact JSON-safe normalized BitGenerator state"
      }
    }
  }
}
```

### State invariants

A valid checkpoint must satisfy:

- `0 <= completed <= simulations`;
- `0 <= rejections <= completed`;
- if `completed == 0`, then `rejections == 0`;
- the state BitGenerator class matches the execution contract;
- the embedded BitGenerator state's own type marker, when present, is
  consistent with the declared class;
- no sample buffer or partial statistical batch is present.

Do not persist redundant derived values such as:

- empirical rejection rate;
- MCSE;
- Wilson interval;
- pass/fail status.

They can be recomputed from committed counts and final experiment configuration.
Avoiding redundant derived state prevents internal inconsistencies.

## 8. Exact BitGenerator state serialization

The implementation must restore the exact BitGenerator state rather than
re-seeding and replaying completed replicates.

Do not assume `bit_generator.state` is always plain JSON. Different NumPy
BitGenerators may contain NumPy scalar values or arrays.

Phase 3B should define one lossless recursive JSON normalization layer that:

- preserves dictionary keys;
- converts NumPy scalar integers/floats to Python scalars;
- preserves arbitrary-size Python integers exactly;
- encodes NumPy arrays with explicit dtype, shape, and data;
- rejects non-finite floating values;
- can reconstruct the original state object exactly.

The checkpoint writer must never silently coerce large RNG integers through a
floating-point representation.

## 9. Checkpoint commit boundary

The logical commit point is:

```text
generate complete logical batch
-> evaluate complete logical batch
-> add batch rejection count to cumulative state
-> completed becomes batch_stop
-> checkpoint snapshot is now valid
-> progress event may be emitted
```

No checkpoint may represent a partially generated or partially evaluated batch.

The checkpoint therefore does not need to store sample buffers.

A future implementation should prefer writing the checkpoint before invoking a
user progress callback at the same committed boundary, so a failing callback
cannot invalidate an already committed resumable state. This ordering is a
Phase 3B implementation detail but must not alter RNG semantics.

## 10. Atomic persistence requirements for Phase 3B

When file I/O is implemented, writes must be atomic at the checkpoint-file
level:

1. serialize and validate the complete next checkpoint;
2. write to a temporary file in the destination directory;
3. flush and close the temporary file;
4. atomically replace the destination.

Corrupt, truncated, unsupported-version, fingerprint-mismatched, or
execution-incompatible checkpoints must fail loudly before sampling resumes.

## 11. Resume validation requirements

Before consuming any new random draw, a future resume path must validate:

1. checkpoint schema version;
2. experiment payload shape and values;
3. execution compatibility contract;
4. recomputed fingerprint;
5. caller's current experiment fingerprint equals checkpoint fingerprint;
6. state invariants;
7. concrete BitGenerator type;
8. lossless RNG state restoration.

Only after all checks pass may the next logical replicate be sampled.

A resume call may choose a different `batch_size` and progress callback because
neither belongs to experiment identity.

## 12. Phase boundaries

### Phase 3A — this document

- define canonical experiment payload;
- define resume fingerprint inputs;
- define execution compatibility contract;
- define checkpoint schema and invariants;
- define RNG-state serialization requirements;
- define strict validation rules.

No runtime checkpoint behavior is added.

### Phase 3B — next implementation step

- implement canonical experiment/fingerprint model;
- implement checkpoint model and JSON-safe RNG-state normalization;
- implement schema validation;
- implement atomic checkpoint writing;
- still avoid full resume execution until model/serialization tests are stable.

### Phase 3C

- implement resume execution from a validated checkpoint;
- prove exact uninterrupted vs resumed equivalence for a fixed batch size.

### Phase 4

- prove exact resumed equivalence when `batch_size` changes across the
  checkpoint boundary;
- cover all built-in DGPs and representative sample sizes.


## 13. Phase 3B implementation status

Phase 3B now implements the model and persistence substrate defined above:

- `ExperimentSpec` provides the canonical semantic experiment payload;
- `ExecutionContract` records the strict runtime compatibility contract;
- `ExperimentFingerprint` computes SHA-256 over canonical experiment +
  execution JSON;
- `RNGSnapshot` uses a lossless tagged JSON encoding for NumPy RNG state;
- ndarray state is stored as dtype + shape + base64 raw bytes;
- `Checkpoint` validates schema, fingerprint, state invariants, and RNG type;
- `Checkpoint.write_atomic()` writes via same-directory temporary file,
  `fsync`, and `os.replace`;
- corrupt JSON, unsupported schemas, unknown fields, mismatched fingerprints,
  impossible counts, and RNG-type mismatches fail loudly.

The initial Phase 3B implementation chooses the conservative `seed=None`
policy: persistent checkpoint model construction rejects `seed=None`.
Supporting entropy-derived initial RNG identities is deferred to a separate
future change.

Phase 3B still does **not** make `stress_test()` resume from a checkpoint.
Resume execution and uninterrupted-vs-resumed equivalence remain Phase 3C.


## 14. Phase 3C implementation status

Phase 3C adds the minimal internal resume execution path while deliberately
keeping resume out of the top-level public API.

The resumable executor now accepts committed state:

- `start_index`, representing the number of completed logical replicates;
- `initial_rejections`, representing the cumulative committed rejection count;
- an exact restored NumPy `Generator`.

The internal resume path:

1. rebinds the checkpoint experiment to the concrete runtime DGPs;
2. rejects semantic/fingerprint mismatches before any new sample draw;
3. restores the exact supported BitGenerator state from the checkpoint;
4. validates the current execution contract against the checkpoint;
5. resumes logical simulation indices at `completed`;
6. carries forward the committed rejection count;
7. finalizes the result through the same evidence/result construction path as
   an uninterrupted `stress_test()`.

Fixed-`batch_size` tests prove exact equality between uninterrupted and
resumed runs for all built-in DGP families, including:

- complete `StressTestResult` equality;
- rejection-count equality;
- complete logical sample-sequence equality;
- final NumPy RNG-state equality;
- global progress counts after resume.

Phase 3C established the fixed-`batch_size` resume path. Phase 4 extends
that validation to changing `batch_size` across the checkpoint boundary.


## 15. Phase 4 cross-batch validation status

Phase 4 requires no production-code change. The existing resume semantics are
now validated across different execution batch sizes before and after a
checkpoint.

The validation matrix uses committed checkpoints created with
`batch_size=1/2/7/64` and resumes with a different
`batch_size=1/2/7/64/>remaining`. For every supported cross-batch pair, all
four built-in DGP families are checked against an uninterrupted scalar
reference.

The following invariants are exact:

- complete `StressTestResult` equality;
- rejection-count equality;
- complete logical sample order and values
  (`x1 -> y1 -> x2 -> y2 -> ...`);
- final NumPy RNG-state equality.

This confirms that `batch_size` remains execution scheduling only: it is not
part of experiment identity or seed derivation, and changing it after a
committed checkpoint does not change the statistical experiment.
