# Contributing to StatFuzz

StatFuzz is an experimental statistics project. Contributions are welcome, but every new feature should make its statistical target explicit.

## Before opening a PR

1. State the statistical property being evaluated.
2. State the data-generating assumptions and the null/target quantity.
3. Include a deterministic random seed in tests and examples.
4. Report Monte Carlo uncertainty where a claim depends on simulation.
5. Add unit tests and, when appropriate, cross-check calculations against a trusted implementation.
6. Avoid presenting one Monte Carlo realization as a universal theorem about a method.

## Local setup

```bash
python -m pip install -e ".[dev,release]"
ruff check .
pytest
```

Public imports are defined in [docs/PUBLIC_API.md](docs/PUBLIC_API.md). New public names
should be added deliberately to the owning namespace `__all__` and covered by
`tests/test_public_api.py`.

Before release-oriented changes, also run:

```bash
python -m build
python -m twine check dist/*
```

The full clean-install process is documented in
[docs/RELEASING.md](docs/RELEASING.md).

## Design preference

Prefer small composable abstractions over a single large `stress_test` function with many special cases. DGPs, statistical methods, metrics, search strategies, and reports should remain separable.
