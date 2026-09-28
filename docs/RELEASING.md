# Releasing StatFuzz

This checklist is for a pre-1.0 public release. It intentionally separates release
preparation from publication.

## 1. Prepare the release PR

- [ ] Start from an up-to-date `main`.
- [ ] Confirm the feature roadmap intended for the release is complete.
- [ ] Choose the package version according to `docs/VERSIONING.md`.
- [ ] Update `src/statfuzz/_version.py`.
- [ ] Update `CHANGELOG.md`.
- [ ] Update `docs/RELEASE_NOTES_<version>.md`.
- [ ] Review `docs/PUBLIC_API.md` and the namespace `__all__` tests.
- [ ] Confirm README examples use public import paths only.

## 2. Run the source quality gate

```bash
python -m pip install -e ".[dev,release]"
ruff check .
pytest
```

The normal CI matrix must pass on Python 3.10, 3.11, and 3.12.

## 3. Build distributions

From a clean checkout:

```bash
rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
```

Expected artifacts:

```text
dist/statfuzz-<version>.tar.gz
dist/statfuzz-<version>-py3-none-any.whl
```

Do not commit files from `dist/`.

## 4. Clean-install verification

Wheel:

```bash
python -m venv .venv-release-wheel
.venv-release-wheel/bin/python -m pip install --upgrade pip
.venv-release-wheel/bin/python -m pip install dist/statfuzz-<version>-py3-none-any.whl pytest
.venv-release-wheel/bin/python -m pytest tests
```

Then run an import/version smoke test and confirm `py.typed` is present in the installed
package.

Source distribution:

```bash
python -m venv .venv-release-sdist
.venv-release-sdist/bin/python -m pip install --upgrade pip
.venv-release-sdist/bin/python -m pip install dist/statfuzz-<version>.tar.gz
```

Run the same basic import/version smoke test.

The GitHub Actions package job mirrors these steps and uploads the built distributions as
a workflow artifact.

## 5. Merge, tag, and create release

Only after all required checks are green:

- [ ] Squash/merge the release PR.
- [ ] Confirm `main` still has green CI.
- [ ] Create an annotated or GitHub tag `v<version>` at the release commit.
- [ ] Create a GitHub Release using the prepared release notes.
- [ ] Attach or regenerate the exact wheel/sdist for the tagged commit.
- [ ] Verify the GitHub Release points to the same commit as the tag.

## 6. Optional package-index publication

PyPI publication is a separate action and should only happen after the tag/Release is
verified.

- [ ] Use a trusted publishing workflow or scoped token; never commit credentials.
- [ ] Upload only artifacts built from the tagged commit.
- [ ] Verify the package page metadata and README rendering.
- [ ] Install the published version in a fresh environment and run a smoke test.

## 7. Post-release

- [ ] Replace the release date placeholder in the changelog if it was not finalized before
      tagging.
- [ ] Move new work under `[Unreleased]`.
- [ ] Open the next milestone only after the release state is documented.
