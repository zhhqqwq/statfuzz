from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from statfuzz.methods import BootstrapMeanPercentile


def test_bootstrap_mean_percentile_defaults_are_stable():
    method = BootstrapMeanPercentile()

    assert method.resamples == 999
    assert method.interval_level == 0.95
    assert method.quantile_method == "linear"
    assert method.method == "bootstrap_mean_percentile"
    assert method.semantics_version == "1"


def test_bootstrap_mean_percentile_is_immutable():
    method = BootstrapMeanPercentile()

    with pytest.raises(FrozenInstanceError):
        method.resamples = 1000


@pytest.mark.parametrize(
    "value",
    [True, 1.5, "999", None],
)
def test_resamples_must_be_an_integer(value):
    with pytest.raises(TypeError, match="resamples"):
        BootstrapMeanPercentile(resamples=value)


@pytest.mark.parametrize("value", [0, -1, -100])
def test_resamples_must_be_positive(value):
    with pytest.raises(ValueError, match="positive"):
        BootstrapMeanPercentile(resamples=value)


def test_numeric_scalar_inputs_are_canonically_normalized():
    method = BootstrapMeanPercentile(
        resamples=np.int64(1999),
        interval_level=np.float64(0.9),
    )

    assert method.resamples == 1999
    assert type(method.resamples) is int
    assert method.interval_level == 0.9
    assert type(method.interval_level) is float


@pytest.mark.parametrize(
    "value",
    [True, "0.95", None, object()],
)
def test_interval_level_must_be_a_real_number(value):
    with pytest.raises(TypeError, match="interval_level"):
        BootstrapMeanPercentile(interval_level=value)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_interval_level_must_be_finite(value):
    with pytest.raises(ValueError, match="finite"):
        BootstrapMeanPercentile(interval_level=value)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.0, 1.1])
def test_interval_level_must_be_strictly_between_zero_and_one(value):
    with pytest.raises(ValueError, match="strictly between"):
        BootstrapMeanPercentile(interval_level=value)


@pytest.mark.parametrize("value", ["lower", "higher", "midpoint", None, 1])
def test_only_linear_quantile_method_is_supported(value):
    with pytest.raises(ValueError, match="linear"):
        BootstrapMeanPercentile(quantile_method=value)


def test_machine_identity_is_exact_and_canonical():
    method = BootstrapMeanPercentile(
        resamples=1999,
        interval_level=0.9,
    )

    assert method.as_dict() == {
        "method": "bootstrap_mean_percentile",
        "semantics_version": "1",
        "resamples": 1999,
        "interval_level": 0.9,
        "quantile_method": "linear",
    }
    assert method.canonical_json() == (
        '{"interval_level":0.9,"method":"bootstrap_mean_percentile",'
        '"quantile_method":"linear","resamples":1999,'
        '"semantics_version":"1"}'
    )


def test_machine_identity_round_trips_exactly():
    method = BootstrapMeanPercentile(
        resamples=499,
        interval_level=0.8,
    )

    restored = BootstrapMeanPercentile.from_dict(method.as_dict())

    assert restored == method
    assert restored.as_dict() == method.as_dict()
    assert restored.canonical_json() == method.canonical_json()


def test_machine_identity_changes_when_statistical_configuration_changes():
    baseline = BootstrapMeanPercentile()

    assert BootstrapMeanPercentile(resamples=1000) != baseline
    assert BootstrapMeanPercentile(interval_level=0.9) != baseline
    assert (
        BootstrapMeanPercentile(resamples=1000).canonical_json()
        != baseline.canonical_json()
    )
    assert (
        BootstrapMeanPercentile(interval_level=0.9).canonical_json()
        != baseline.canonical_json()
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.pop("resamples"),
        lambda data: data.__setitem__("unexpected", True),
    ],
)
def test_machine_identity_rejects_missing_or_unknown_fields(mutate):
    data = BootstrapMeanPercentile().as_dict()
    mutate(data)

    with pytest.raises(ValueError, match="unexpected keys"):
        BootstrapMeanPercentile.from_dict(data)


def test_machine_identity_rejects_unsupported_method_name():
    data = BootstrapMeanPercentile().as_dict()
    data["method"] = "bootstrap_basic_mean"

    with pytest.raises(ValueError, match="unsupported bootstrap method"):
        BootstrapMeanPercentile.from_dict(data)


def test_machine_identity_rejects_unsupported_semantics_version():
    data = BootstrapMeanPercentile().as_dict()
    data["semantics_version"] = "999"

    with pytest.raises(ValueError, match="semantics version"):
        BootstrapMeanPercentile.from_dict(data)


def test_machine_identity_requires_object():
    with pytest.raises(TypeError, match="object"):
        BootstrapMeanPercentile.from_dict([])
