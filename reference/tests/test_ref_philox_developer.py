"""Developer checks for the Philox reference beyond the lead's acceptance suite.

These check stream/block mapping, selected fixed-point probes and the uniform grid. The lead's
known-answer and inverse-round tests establish the primitive's round count and reversible mapping.
"""

import numpy as np
import pytest

from tq_reference.philox import PhiloxError, block, uniform_f32, words

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
MASK32 = (1 << 32) - 1


def _split(value: int) -> tuple[int, int]:
    return value & MASK32, value >> 32


@pytest.mark.parametrize("seed", [0, 1, 0x0123456789ABCDEF, 1 << 63, (1 << 64) - 1])
def test_words_are_their_blocks_read_in_lane_order(seed):
    """A stream is only an assignment of counters, so a fill must equal block-by-block reads."""
    stream = 0xFEDCBA9876543210
    start = 4 * 7 + 2  # deliberately mid-block
    count = 23
    actual = words(seed, stream, start, count)
    expected = []
    for index in range(start, start + count):
        counter = np.array([*_split(index // 4), *_split(stream)], dtype=np.uint32)
        expected.append(block(counter, np.array(_split(seed), dtype=np.uint32))[index % 4])
    np.testing.assert_array_equal(actual, np.array(expected, dtype=np.uint32))


def test_distinct_streams_use_disjoint_counter_domains():
    """The upper counter words carry the stream, so no stream can reach another's blocks."""
    seed = 0x0123456789ABCDEF
    key = np.array(_split(seed), dtype=np.uint32)
    for stream in (0, 1, (1 << 64) - 1):
        counter = np.array([0, 0, *_split(stream)], dtype=np.uint32)
        np.testing.assert_array_equal(words(seed, stream, 0, 4), block(counter, key))


@pytest.mark.parametrize("seed", [0, (1 << 64) - 1])
def test_selected_inputs_are_not_fixed_points(seed):
    """These two inputs change under repeated application; this does not identify a round count."""
    counter = np.zeros(4, dtype=np.uint32)
    key = np.array(_split(seed), dtype=np.uint32)
    once = block(counter, key)
    twice = block(once.copy(), key)
    assert not np.array_equal(once, twice)
    assert not np.array_equal(once, counter)


def test_uniform_is_monotone_and_lands_on_the_exact_grid():
    """Every output is a multiple of 2**-24 in [0, 1), and the mapping never decreases."""
    probes = np.array([0, 1, 255, 256, 1 << 16, 1 << 31, MASK32 - 1, MASK32], dtype=np.uint32)
    values = uniform_f32(probes)
    assert np.all(np.diff(values) >= 0)
    assert values.min() == 0.0
    assert values.max() == np.float32(1.0) - np.float32(2.0**-24)
    scaled = values.astype(np.float64) * 2.0**24
    np.testing.assert_array_equal(scaled, np.floor(scaled))


def test_empty_requests_are_valid_at_the_last_addressable_word():
    for start in (0, (1 << 64) - 1):
        assert words(0, 0, start, 0).shape == (0,)
    with pytest.raises(PhiloxError, match="word range"):
        words(0, 0, (1 << 64) - 1, 2)
