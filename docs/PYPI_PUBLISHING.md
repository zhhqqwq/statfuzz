# PyPI Trusted Publishing

StatFuzz publishes to PyPI with GitHub Actions OIDC Trusted Publishing. No long-lived
PyPI API token belongs in GitHub secrets.

## Publisher identity

For the first PyPI publication of `statfuzz`, configure a PyPI **Pending Publisher**
with these exact values:

```text
PyPI project name: statfuzz
GitHub owner:       zhhqqwq
Repository:         statfuzz
Workflow filename:  publish-pypi-v0.1.0.yml
Environment:        (leave blank)
```

The workflow file is:

```text
.github/workflows/publish-pypi-v0.1.0.yml
```

The environment field is intentionally blank for this one-time first publication. The
workflow itself is restricted to this repository and exact v0.1.0 release artifacts.

A pending publisher does not reserve the package name until the first successful publish.

## Exact-artifact policy

PyPI v0.1.0 must contain the exact files already attached to the GitHub Release:

```text
statfuzz-0.1.0-py3-none-any.whl
SHA-256:
55c0d706e829bb71074f66c9479d752bc6446b5e14d811fb66e2bd81369786cd

statfuzz-0.1.0.tar.gz
SHA-256:
46457ee685999496e04593ef15468b2fc6ef9dc60000eb41cb8cd4d00d3c6112
```

The PyPI workflow downloads those release assets directly instead of rebuilding them,
checks both hashes before upload, then checks PyPI's published hashes after upload.

## First publication

The workflow has two triggers:

- a push to `main` that introduces/updates this publishing workflow or this document;
- manual `workflow_dispatch` for safe retry after configuring a Pending Publisher.

The publishing job has:

```yaml
permissions:
  contents: read
  id-token: write
```

and uses `pypa/gh-action-pypi-publish@release/v1` without a username, password, or API
token.

## Verification

After successful upload, a second job:

1. waits for `https://pypi.org/pypi/statfuzz/0.1.0/json`;
2. verifies that PyPI exposes exactly the expected wheel and sdist hashes;
3. creates a fresh virtual environment;
4. installs `statfuzz==0.1.0` explicitly from `https://pypi.org/simple`;
5. runs `scripts/release_smoke.py`.

Only after this verification should README installation instructions switch to:

```bash
python -m pip install statfuzz
```

## Later releases

After the first publication, replace the one-time v0.1.0 workflow with a reusable release
publishing workflow. A protected GitHub environment named `pypi` is recommended for
future releases so publication can require explicit maintainer approval.
