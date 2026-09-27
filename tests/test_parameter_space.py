import json

import pytest

from statfuzz.search import ParameterSpace


def test_parameter_space_has_canonical_key_order():
    a = ParameterSpace({"sigma": [0.6, 1.4], "n": [8, 20]})
    b = ParameterSpace({"n": [8, 20], "sigma": [0.6, 1.4]})

    assert a.names == ("n", "sigma")
    assert [point.as_dict() for point in a] == [point.as_dict() for point in b]
    assert [point.as_dict() for point in a] == [
        {"n": 8, "sigma": 0.6},
        {"n": 8, "sigma": 1.4},
        {"n": 20, "sigma": 0.6},
        {"n": 20, "sigma": 1.4},
    ]


def test_parameter_points_are_json_serializable():
    point = next(iter(ParameterSpace({"n": [12], "sigma": [1.4]})))
    assert json.loads(point.to_json()) == {"n": 12, "sigma": 1.4}
    assert point["sigma"] == 1.4


def test_parameter_space_length():
    space = ParameterSpace({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(space) == 6


def test_empty_axis_is_rejected():
    with pytest.raises(ValueError, match="at least one candidate"):
        ParameterSpace({"sigma": []})


def test_duplicate_candidates_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ParameterSpace({"sigma": [1.0, 1.0]})


def test_non_finite_float_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        ParameterSpace({"sigma": [float("nan")]})
