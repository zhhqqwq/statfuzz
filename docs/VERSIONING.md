# Versioning

StatFuzz uses a package release version that is independent from the historical roadmap
milestone labels.

## Package version

The single source of truth is:

```text
src/statfuzz/_version.py
```

`pyproject.toml` reads that value dynamically, and `statfuzz.__version__` imports the
same value. Do not maintain a second hard-coded package version.

The first public release candidate is **0.1.0**.

## Roadmap labels are not package versions

The repository roadmap used labels such as v0.1 through v0.6 to organize feature
development:

```text
statistical core -> search -> discovery -> shrinking -> reports -> StatCI
```

Those labels are development phases. Completing the roadmap through "v0.6" does not mean
the Python package must be released as version 0.6.0.

## Pre-1.0 policy

StatFuzz is alpha software and uses a SemVer-inspired `0.MINOR.PATCH` policy.

### Patch release: 0.x.Y

Use a patch release for compatible fixes such as:

- bug fixes;
- documentation corrections;
- packaging fixes;
- performance improvements that preserve documented behavior;
- additive metadata changes.

A patch release should not intentionally remove or rename documented public API.

### Minor release: 0.X.0

Use a minor release for:

- meaningful new capabilities;
- new public APIs;
- serialized-schema changes that accompany a larger feature;
- documented breaking API changes while the project remains pre-1.0.

Breaking changes should still use deprecation where practical.

## Schema versions

The following machine-readable formats have independent schema versions:

- StatFuzz report JSON;
- StatCI result JSON;
- StatCI status artifact JSON;
- StatCI regression JSON.

Schema versions are not derived from the package version.

## Version bump checklist

When preparing a release:

1. update `src/statfuzz/_version.py`;
2. update `CHANGELOG.md`;
3. update the matching release-notes file;
4. run the release quality gate;
5. merge the release PR;
6. tag the exact merge commit as `v<version>`;
7. create the GitHub Release from the same commit;
8. only then publish distribution artifacts, if desired.
