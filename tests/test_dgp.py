import numpy as np
import pytest

from statfuzz.dgp import DGPIdentity, LogNormal, MixtureNormal, Normal, StudentT


def test_normal_sample_shape():
    x = Normal().sample(np.random.default_rng(0), 10)
    assert x.shape == (10,)


def test_lognormal_is_centered_in_expectation():
    rng = np.random.default_rng(123)
    x = LogNormal(sigma=0.8, mean=2.0).sample(rng, 200_000)
    assert np.mean(x) == pytest.approx(2.0, abs=0.03)


def test_student_t_requires_existing_mean():
    with pytest.raises(ValueError):
        StudentT(df=1.0)


def test_mixture_sample_shape():
    x = MixtureNormal().sample(np.random.default_rng(0), 25)
    assert x.shape == (25,)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Normal(mean=float("nan")),
        lambda: Normal(sd=float("nan")),
        lambda: Normal(sd=float("inf")),
        lambda: LogNormal(sigma=float("nan")),
        lambda: LogNormal(mean=float("inf")),
        lambda: StudentT(df=float("nan")),
        lambda: StudentT(mean=float("inf")),
        lambda: StudentT(scale=float("nan")),
        lambda: MixtureNormal(weight=float("nan")),
        lambda: MixtureNormal(mean1=float("inf")),
        lambda: MixtureNormal(sd1=float("nan")),
        lambda: MixtureNormal(mean2=float("-inf")),
        lambda: MixtureNormal(sd2=float("inf")),
        lambda: MixtureNormal(mean=float("nan")),
    ],
)
def test_builtin_dgps_reject_non_finite_parameters(factory):
    with pytest.raises(ValueError, match="finite"):
        factory()


def test_dgp_identity_preserves_full_precision_parameters():
    first = Normal(sd=1.0000001)
    second = Normal(sd=1.0000002)

    assert first.name == second.name == "Normal(mean=0, sd=1)"
    assert first.identity != second.identity
    assert first.identity.as_dict() == {
        "family": "statfuzz.dgp.Normal",
        "parameters": {"mean": 0.0, "sd": 1.0000001},
    }
    assert second.identity.as_dict()["parameters"]["sd"] == 1.0000002
    assert first.identity.canonical_json() != second.identity.canonical_json()


def test_dgp_identity_is_canonical_across_parameter_order():
    first = DGPIdentity.from_mapping(
        "custom",
        {"b": 2.0, "a": 1.0},
    )
    second = DGPIdentity.from_mapping(
        "custom",
        {"a": 1.0, "b": 2.0},
    )

    assert first == second
    assert first.parameters == (("a", 1.0), ("b", 2.0))
    assert first.canonical_json() == second.canonical_json()


def test_builtin_identity_normalizes_equivalent_numeric_inputs():
    assert Normal(mean=0, sd=1).identity == Normal(mean=0.0, sd=1.0).identity
    assert LogNormal(sigma=1, mean=0).identity == LogNormal(
        sigma=1.0,
        mean=0.0,
    ).identity
