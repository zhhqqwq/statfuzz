import pytest

from statfuzz.metrics import (
    BinomialRateEvidence,
    Type1ErrorEvidence,
    binomial_rate_evidence,
    type1_error_evidence,
)


def test_binomial_rate_evidence_matches_type1_error_evidence_exactly():
    generic = binomial_rate_evidence(
        17,
        137,
        confidence_level=0.95,
        interval_method="wilson",
    )
    type1 = type1_error_evidence(
        17,
        137,
        0.05,
        confidence_level=0.95,
        interval_method="wilson",
    )

    assert isinstance(generic, BinomialRateEvidence)
    assert isinstance(type1, Type1ErrorEvidence)
    assert type1.rejection_count == generic.event_count
    assert type1.simulations == generic.trials
    assert type1.empirical == generic.empirical
    assert type1.mcse == generic.mcse
    assert type1.confidence_level == generic.confidence_level
    assert type1.interval_method == generic.interval_method
    assert type1.interval_low == generic.interval_low
    assert type1.interval_high == generic.interval_high


@pytest.mark.parametrize("event_count", [0, 17, 137])
def test_binomial_rate_evidence_handles_boundary_and_interior_counts(event_count):
    evidence = binomial_rate_evidence(event_count, 137)

    assert evidence.event_count == event_count
    assert evidence.trials == 137
    assert evidence.empirical == event_count / 137
    assert 0.0 <= evidence.interval_low <= evidence.empirical
    assert evidence.empirical <= evidence.interval_high <= 1.0


def test_binomial_rate_evidence_rejects_invalid_counts_and_method():
    with pytest.raises(TypeError, match="event_count"):
        binomial_rate_evidence(True, 10)
    with pytest.raises(ValueError, match="between 0 and trials"):
        binomial_rate_evidence(11, 10)
    with pytest.raises(ValueError, match="wilson"):
        binomial_rate_evidence(1, 10, interval_method="exact")
