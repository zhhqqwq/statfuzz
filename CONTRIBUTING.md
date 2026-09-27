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
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Design preference

Prefer small composable abstractions over a single large `stress_test` function with many special cases. DGPs, statistical methods, metrics, search strategies, and reports should remain separable.
