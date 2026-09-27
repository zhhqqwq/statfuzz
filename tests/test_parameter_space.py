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


def test_point_at_matches_iteration_order():
    space = ParameterSpace({"a": [1, 2], "b": ["x", "y", "z"]})
    iterated = list(space)

    assert [space.point_at(i) for i in range(len(space))] == iterated


def test_point_at_rejects_invalid_index():
    space = ParameterSpace({"x": [1, 2]})

    with pytest.raises(IndexError, match="outside"):
        space.point_at(-1)
    with pytest.raises(IndexError, match="outside"):
        space.point_at(2)
    with pytest.raises(TypeError, match="integer"):
        space.point_at(1.5)
