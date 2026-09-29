import math

import numpy as np
import pytest

from statfuzz import MeanTarget, MeanTargetCheck
from statfuzz.dgp import LogNormal, MixtureNormal, Normal, StudentT
from statfuzz.targets import resolve_mean_target


@pytest.mark.parametrize(
    ("dgp", "expected"),
    [
        (Normal(mean=1.25), 1.25),
        (LogNormal(mean=-2.0), -2.0),
        (StudentT(mean=3.5), 3.5),
        (MixtureNormal(mean=0.75), 0.75),
    ],
)
def test_builtin_dgp_population_mean_is_resolved_automatically(dgp, expected):
    check = resolve_mean_target(dgp, None)

    assert check == MeanTargetCheck(
        source="population_mean",
        mean=expected,
        population_mean=expected,
        note=None,
    )
    assert check.as_dict()["kind"] == "mean"
    assert MeanTargetCheck.from_dict(check.as_dict()) == check


@pytest.mark.parametrize(
    "dgp",
    [
        Normal(mean=0.0),
        LogNormal(mean=0.0),
        StudentT(mean=0.0),
        MixtureNormal(mean=0.0),
    ],
)
def test_matching_explicit_target_preserves_verified_source_and_note(dgp):
    check = resolve_mean_target(
        dgp,
        MeanTarget(mean=0.0, note="known analytical mean"),
    )

    assert check.source == "population_mean"
    assert check.mean == 0.0
    assert check.population_mean == 0.0
    assert check.note == "known analytical mean"


class _CustomWithoutPopulationMean:
    name = "custom-without-population-mean"

    def sample(self, rng, n):
        return rng.normal(size=n)


class _CustomWithPopulationMean:
    name = "custom-with-population-mean"
    population_mean = 2.0

    def sample(self, rng, n):
        return rng.normal(loc=2.0, size=n)


class _CustomWithNonePopulationMean:
    name = "custom-with-none-population-mean"
    population_mean = None

    def sample(self, rng, n):
        return rng.normal(size=n)


def test_custom_dgp_without_population_mean_requires_explicit_target():
    with pytest.raises(ValueError, match="Provide MeanTarget"):
        resolve_mean_target(_CustomWithoutPopulationMean(), None)

    check = resolve_mean_target(
        _CustomWithoutPopulationMean(),
        MeanTarget(mean=4.0, note="external analytical result"),
    )

    assert check == MeanTargetCheck(
        source="declaration",
        mean=4.0,
        population_mean=None,
        note="external analytical result",
    )


def test_custom_dgp_with_none_population_mean_is_treated_as_unverifiable():
    check = resolve_mean_target(
        _CustomWithNonePopulationMean(),
        MeanTarget(mean=-1.0),
    )

    assert check.source == "declaration"
    assert check.mean == -1.0
    assert check.population_mean is None


def test_custom_dgp_population_mean_is_verified_when_available():
    check = resolve_mean_target(_CustomWithPopulationMean(), None)

    assert check.source == "population_mean"
    assert check.mean == 2.0
    assert check.population_mean == 2.0


def test_conflicting_target_fails_loudly():
    with pytest.raises(ValueError, match="does not match"):
        resolve_mean_target(
            Normal(mean=1.0),
            MeanTarget(mean=2.0),
        )

    with pytest.raises(ValueError, match="does not match"):
        resolve_mean_target(
            _CustomWithPopulationMean(),
            MeanTarget(mean=3.0),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_mean_target_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        MeanTarget(mean=value)


class _NonFinitePopulationMean:
    name = "non-finite-population-mean"
    population_mean = math.nan

    def sample(self, rng, n):
        return np.zeros(n)


def test_non_finite_dgp_population_mean_fails_loudly():
    with pytest.raises(ValueError, match="DGP population_mean must be finite"):
        resolve_mean_target(
            _NonFinitePopulationMean(),
            MeanTarget(mean=0.0),
        )


@pytest.mark.parametrize("note", ["", "   ", 123])
def test_mean_target_note_must_be_non_empty_string_or_none(note):
    with pytest.raises(ValueError, match="note"):
        MeanTarget(mean=0.0, note=note)


def test_mean_target_check_rejects_inconsistent_verified_state():
    with pytest.raises(ValueError, match="requires a verified"):
        MeanTargetCheck(
            source="population_mean",
            mean=0.0,
            population_mean=None,
        )

    with pytest.raises(ValueError, match="must equal"):
        MeanTargetCheck(
            source="declaration",
            mean=0.0,
            population_mean=1.0,
        )
