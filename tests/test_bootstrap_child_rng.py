import numpy as np
import pytest

from statfuzz.methods import (
    bootstrap_mean_percentile_child_rng,
    bootstrap_mean_percentile_child_seed,
)

ROOT_SEED = 20260930

EXPECTED_CHILD_SEEDS = {
    0: 332941748150694023827286893033075392851,
    1: 292870143969549707061602695877815022983,
    3: 99120416707231458115994794293298153721,
    7: 3002290898806000946842741214723261422,
}

EXPECTED_STREAMS = {
    0: [
        556671999,
        719965892,
        1496689423,
        1833909323,
        1817150386,
        187021950,
        455370025,
        101810713,
    ],
    1: [
        1378144287,
        876288455,
        1504131067,
        1989882541,
        1718089779,
        1820184179,
        1790483072,
        110135247,
    ],
    3: [
        302341386,
        525832992,
        830959576,
        683780052,
        1530402797,
        1575938550,
        1750501527,
        518335123,
    ],
    7: [
        493734754,
        233916411,
        904845953,
        872820069,
        464912067,
        450158441,
        463467008,
        620581691,
    ],
}


@pytest.mark.parametrize("logical_index", [0, 1, 3, 7])
def test_child_seed_fixed_reference(logical_index):
    assert bootstrap_mean_percentile_child_seed(
        ROOT_SEED,
        logical_index,
    ) == EXPECTED_CHILD_SEEDS[logical_index]


@pytest.mark.parametrize("logical_index", [0, 1, 3, 7])
def test_child_rng_fixed_reference_stream(logical_index):
    rng = bootstrap_mean_percentile_child_rng(ROOT_SEED, logical_index)

    assert type(rng.bit_generator) is np.random.PCG64
    assert rng.integers(
        0,
        2**31,
        size=8,
        dtype=np.int64,
    ).tolist() == EXPECTED_STREAMS[logical_index]


def test_same_logical_index_recreates_identical_child_stream():
    first = bootstrap_mean_percentile_child_rng(ROOT_SEED, 7)
    second = bootstrap_mean_percentile_child_rng(ROOT_SEED, 7)

    np.testing.assert_array_equal(
        first.integers(0, 2**63, size=100, dtype=np.int64),
        second.integers(0, 2**63, size=100, dtype=np.int64),
    )


def test_different_logical_indices_get_distinct_child_seeds_and_streams():
    seeds = {
        bootstrap_mean_percentile_child_seed(ROOT_SEED, index)
        for index in range(16)
    }
    assert len(seeds) == 16

    streams = {
        tuple(
            bootstrap_mean_percentile_child_rng(
                ROOT_SEED,
                index,
            ).integers(0, 2**31, size=8, dtype=np.int64)
        )
        for index in range(16)
    }
    assert len(streams) == 16


def test_child_stream_assignment_is_independent_of_request_order():
    forward = {}
    for index in [0, 1, 3, 7]:
        forward[index] = bootstrap_mean_percentile_child_rng(
            ROOT_SEED,
            index,
        ).integers(0, 2**31, size=32, dtype=np.int64)

    reverse = {}
    for index in [7, 3, 1, 0]:
        reverse[index] = bootstrap_mean_percentile_child_rng(
            ROOT_SEED,
            index,
        ).integers(0, 2**31, size=32, dtype=np.int64)

    for index, stream in forward.items():
        np.testing.assert_array_equal(stream, reverse[index])


def test_consuming_one_child_stream_does_not_change_another_child_stream():
    baseline = bootstrap_mean_percentile_child_rng(ROOT_SEED, 3)
    expected = baseline.integers(0, 2**31, size=32, dtype=np.int64)

    unrelated = bootstrap_mean_percentile_child_rng(ROOT_SEED, 7)
    unrelated.integers(0, 2**31, size=1000, dtype=np.int64)

    actual = bootstrap_mean_percentile_child_rng(
        ROOT_SEED,
        3,
    ).integers(0, 2**31, size=32, dtype=np.int64)

    np.testing.assert_array_equal(actual, expected)


def test_root_seed_is_part_of_child_stream_identity():
    left = bootstrap_mean_percentile_child_seed(1, 0)
    right = bootstrap_mean_percentile_child_seed(2, 0)

    assert left != right


def test_numpy_integral_inputs_normalize_to_same_child_seed():
    assert bootstrap_mean_percentile_child_seed(
        np.int64(ROOT_SEED),
        np.int64(7),
    ) == bootstrap_mean_percentile_child_seed(ROOT_SEED, 7)


@pytest.mark.parametrize(
    ("root_seed", "logical_index", "error", "message"),
    [
        (True, 0, TypeError, "root_seed"),
        (1.5, 0, TypeError, "root_seed"),
        ("1", 0, TypeError, "root_seed"),
        (-1, 0, ValueError, "root_seed"),
        (0, True, TypeError, "logical_outer_index"),
        (0, 1.5, TypeError, "logical_outer_index"),
        (0, "1", TypeError, "logical_outer_index"),
        (0, -1, ValueError, "logical_outer_index"),
    ],
)
def test_child_rng_derivation_rejects_invalid_inputs(
    root_seed,
    logical_index,
    error,
    message,
):
    with pytest.raises(error, match=message):
        bootstrap_mean_percentile_child_seed(
            root_seed,
            logical_index,
        )


def test_derivation_does_not_consume_or_depend_on_global_numpy_rng_state():
    np.random.seed(123)
    before = np.random.get_state()

    bootstrap_mean_percentile_child_rng(ROOT_SEED, 7)

    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
