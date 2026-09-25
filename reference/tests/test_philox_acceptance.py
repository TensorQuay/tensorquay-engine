"""Lead-owned RNG-001…005 acceptance; upstream vectors and independent integer maths."""

import hashlib
import importlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/philox-v1"
MASK = (1 << 32) - 1
MAX64 = (1 << 64) - 1
SEEDS = (0, 1, 0x0123456789ABCDEF, 1 << 63, MAX64)


def rows(name):
    return [line.split() for line in (FIXTURES / name).read_text().splitlines() if line[0] != "#"]


BLOCKS = [tuple(int(v, 16) for v in row) for row in rows("blocks.tsv")]
STREAMS = [
    (*[int(v, 16) for v in row[:3]], int(row[3]), tuple(int(v, 16) for v in row[4:]))
    for row in rows("streams.tsv")
]
UNIFORMS = [tuple(int(v, 16) for v in row) for row in rows("uniform.tsv")]
KATS = (
    (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
    (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
)
PHILOX_SOURCE_HASH = "6c2ef219a855885499a73b338d5f41dafe079618b2dae2f60ea86ee785d771e2"


@pytest.fixture
def philox():
    return importlib.import_module("tq_reference.philox")


def limb_product(a, b):
    """Four 16-bit products, independent of implementation's multiplication path."""
    a0, a1, b0, b1 = a & 65535, a >> 16, b & 65535, b >> 16
    full = a0 * b0 + ((a0 * b1 + a1 * b0) << 16) + (a1 * b1 << 32)
    return full >> 32, full & MASK


def oracle(counter, key, rounds=10):
    a, b, c, d = map(int, counter)
    for r in range(rounds):
        hi0, lo0 = limb_product(a, 0xD2511F53)
        hi1, lo1 = limb_product(c, 0xCD9E8D57)
        a, b, c, d = (
            hi1 ^ b ^ ((int(key[0]) + r * 0x9E3779B9) & MASK),
            lo1,
            hi0 ^ d ^ ((int(key[1]) + r * 0xBB67AE85) & MASK),
            lo0,
        )
    return a, b, c, d


def inverse(output, key):
    a, b, c, d = map(int, output)
    for r in reversed(range(10)):
        old_a = d * pow(0xD2511F53, -1, 1 << 32) & MASK
        old_c = b * pow(0xCD9E8D57, -1, 1 << 32) & MASK
        old_b = a ^ ((0xCD9E8D57 * old_c) >> 32) ^ ((key[0] + r * 0x9E3779B9) & MASK)
        old_d = c ^ ((0xD2511F53 * old_a) >> 32) ^ ((key[1] + r * 0xBB67AE85) & MASK)
        a, b, c, d = old_a, old_b, old_c, old_d
    return a, b, c, d


def uniform_bits(word):
    n = int(word) >> 8
    if n == 0:
        return 0
    exponent = n.bit_length() - 1
    return ((exponent + 103) << 23) | ((n - (1 << exponent)) << (23 - exponent))


def assert_array(actual, expected, dtype):
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == dtype
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=dtype))


def test_oracle_controls():
    for row, kat in zip(BLOCKS[:3], KATS, strict=True):
        assert row[6:] == kat == oracle(row[:4], row[4:6])
        assert inverse(kat, row[4:6]) == row[:4]
    assert oracle([0] * 4, [0] * 2, 7) == (0x5F6FB709, 0x0D893F64, 0x4F121F81, 0x4F730A48)
    assert oracle([0] * 4, [0] * 2, 9) != KATS[0]
    assert uniform_bits(MASK) == 0x3F7FFFFF


def test_manifest_and_canonical_digests():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["algorithm"] == "Philox4x32-10"
    assert manifest["mapping"] == "tq-philox-v1"
    assert manifest["uniform"] == "high24-times-2^-24"
    assert manifest["source"]["revision"] == "726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13"
    assert manifest["source"]["sha256"] == {
        "include/Random123/philox.h": PHILOX_SOURCE_HASH,
        "tests/kat_vectors": "aab5ebabf40003f63d6d87b24cbd2c8a02652e00cf8bad64226fd50586929183",
    }
    assert tuple(int(s, 16) for s in manifest["seeds"]) == SEEDS
    assigned = manifest["streams"]
    assert len({a["test_id"] for a in assigned}) == len(assigned) == 4
    assert len({a["stream"] for a in assigned}) == 4
    assert {int(a["stream"], 16) for a in assigned} == {r[1] for r in STREAMS}
    assert {r[0] for r in STREAMS} == set(SEEDS)
    canonical = {
        "blocks.tsv": (
            b"".join(struct.pack("<6I", *r[:6]) for r in BLOCKS),
            b"".join(struct.pack("<4I", *r[6:]) for r in BLOCKS),
            199,
        ),
        "streams.tsv": (
            b"".join(struct.pack("<4Q", *r[:4]) for r in STREAMS),
            b"".join(struct.pack("<" + "I" * r[3], *r[4]) for r in STREAMS),
            160,
        ),
        "uniform.tsv": (
            b"".join(struct.pack("<I", r[0]) for r in UNIFORMS),
            b"".join(struct.pack("<I", r[1]) for r in UNIFORMS),
            96,
        ),
    }
    assert set(manifest["files"]) == set(canonical)
    for name, (inputs, outputs, count) in canonical.items():
        assert manifest["files"][name] == {
            "sha256": hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(),
            "inputs_sha256": hashlib.sha256(inputs).hexdigest(),
            "outputs_sha256": hashlib.sha256(outputs).hexdigest(),
            "cases": count,
        }
        assert len(rows(name)) == count


@pytest.mark.parametrize("row", BLOCKS, ids=lambda row: "-".join(f"{v:x}" for v in row[:6]))
def test_blocks_against_upstream_and_inverse(philox, row):
    counter, key = np.array(row[:4], np.uint32), np.array(row[4:6], np.uint32)
    actual = philox.block(counter, key)
    assert_array(actual, row[6:], np.uint32)
    assert tuple(actual) == oracle(counter, key)
    assert inverse(actual, row[4:6]) == row[:4]
    assert not np.shares_memory(actual, counter)
    assert not np.shares_memory(actual, key)
    assert_array(counter, row[:4], np.uint32)
    assert_array(key, row[4:6], np.uint32)


@pytest.mark.parametrize("row", STREAMS, ids=lambda r: f"{r[0]:x}-{r[1]:x}-{r[2]:x}-{r[3]}")
def test_streams_against_upstream(philox, row):
    seed, stream, start, count, expected = row
    actual = philox.words(seed, stream, start, count)
    assert_array(actual, expected, np.uint32)
    scalar = []
    for i in range(start, start + count):
        counter = ((i // 4) & MASK, (i // 4) >> 32, stream & MASK, stream >> 32)
        scalar.append(oracle(counter, (seed & MASK, seed >> 32))[i % 4])
    assert_array(actual, scalar, np.uint32)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("start", [0, 1, 2, 3, (1 << 34) - 7, MAX64 - 63])
def test_partition_and_overlap(philox, seed, start):
    whole = philox.words(seed, MAX64, start, 64)
    for split in [0, 1, 2, 3, 4, 17, 31, 63, 64]:
        # Reverse request order to expose hidden mutable state.
        right = philox.words(seed, MAX64, start + min(split, 63), 64 - split)
        if split == 64:
            assert right.size == 0
        left = philox.words(seed, MAX64, start, split)
        assert_array(np.concatenate((left, right)), whole, np.uint32)
    assert_array(philox.words(seed, MAX64, start + 5, 17), whole[5:22], np.uint32)
    assert_array(philox.words(seed, MAX64, start, 64), whole, np.uint32)
    copy = philox.words(seed, MAX64, start, 64)
    copy[0] ^= np.uint32(MASK)
    assert_array(philox.words(seed, MAX64, start, 64), whole, np.uint32)


@pytest.mark.parametrize("word,expected", UNIFORMS)
def test_uniform_exact_scalar_bits(philox, word, expected):
    value = np.array(word, dtype=np.uint32)
    actual = philox.uniform_f32(value)
    assert actual.shape == ()
    assert_array(actual.view(np.uint32), expected, np.uint32)
    assert actual.dtype == np.float32
    assert expected == uniform_bits(word)
    assert not np.shares_memory(actual, value)


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3), (2, 3, 4)])
def test_uniform_array_layouts(philox, shape):
    values = (np.arange(np.prod(shape), dtype=np.uint32) * np.uint32(123456789)).reshape(shape)
    values = values.T[..., ::-1]
    before = values.copy()
    values.flags.writeable = False
    actual = philox.uniform_f32(values)
    expected = np.array([uniform_bits(v) for v in values.flat], np.uint32).reshape(values.shape)
    assert actual.dtype == np.float32
    assert_array(actual.view(np.uint32), expected, np.uint32)
    assert not np.shares_memory(actual, values)
    np.testing.assert_array_equal(values, before)


def test_uniform_discards_only_low_eight_bits(philox):
    values = np.array(
        [prefix | tail for prefix in (0, 0x80000000, 0xFFFFFF00) for tail in range(256)], np.uint32
    ).reshape(3, 256)
    result = philox.uniform_f32(values)
    assert_array(
        result.view(np.uint32), np.repeat([[0], [0x3F000000], [0x3F7FFFFF]], 256, 1), np.uint32
    )
    assert np.all(result >= 0) and np.all(result < 1)


def test_block_strided_readonly_inputs(philox):
    counter = np.arange(8, dtype=np.uint32)[::-2]
    key = np.arange(4, dtype=np.uint32)[::-2]
    counter.flags.writeable = key.flags.writeable = False
    assert_array(philox.block(counter, key), oracle(counter, key), np.uint32)


@pytest.mark.parametrize(
    "dtype", [int, np.int8, np.int32, np.int64, np.uint8, np.uint32, np.uint64]
)
def test_integer_scalar_types(philox, dtype):
    assert_array(philox.words(*map(dtype, [1, 1, 1, 1])), philox.words(1, 1, 1, 1), np.uint32)


@pytest.mark.parametrize("field", ["seed", "stream", "start", "count"])
@pytest.mark.parametrize(
    "bad",
    [
        -1,
        1 << 64,
        1 << 20000,
        True,
        np.bool_(False),
        1.0,
        np.float64(1),
        None,
        "1",
        [1],
        np.array(1),
    ],
    ids=[
        "negative",
        "u65",
        "huge",
        "bool",
        "np-bool",
        "float",
        "np-float",
        "none",
        "str",
        "list",
        "array",
    ],
)
def test_invalid_scalars(philox, field, bad):
    args = dict(seed=0, stream=0, start=0, count=0)
    args[field] = bad
    limit = sys.get_int_max_str_digits()
    assert issubclass(philox.PhiloxError, ValueError)
    with pytest.raises(philox.PhiloxError, match=field):
        philox.words(**args)
    assert sys.get_int_max_str_digits() == limit


@pytest.mark.parametrize("start,count", [(MAX64, 2), (MAX64 - 3, 5), (2, MAX64)])
def test_exhausted_word_range(philox, start, count):
    with pytest.raises(philox.PhiloxError, match="word range"):
        philox.words(0, 0, start, count)


@pytest.mark.parametrize("which", ["counter", "key", "uniform"])
@pytest.mark.parametrize(
    "bad,message",
    [
        ([1], "numpy.ndarray"),
        (None, "numpy.ndarray"),
        (np.array([0], dtype=np.uint64), "uint32"),
        (np.array([0], dtype=np.int32), "uint32"),
        (np.array([0], dtype=np.float32), "uint32"),
        (np.array([False]), "uint32"),
        (np.array([0], dtype=">u4"), "uint32"),
    ],
)
def test_array_validation(philox, which, bad, message):
    counter, key = np.zeros(4, np.uint32), np.zeros(2, np.uint32)
    with pytest.raises(philox.PhiloxError, match=message):
        if which == "uniform":
            philox.uniform_f32(bad)
        else:
            philox.block(bad if which == "counter" else counter, bad if which == "key" else key)


@pytest.mark.parametrize("which", ["counter", "key"])
@pytest.mark.parametrize("shape", [(), (0,), (1,), (3,), (5,), (2, 2), (1, 4)])
def test_block_shape_validation(philox, which, shape):
    counter, key = np.zeros(4, np.uint32), np.zeros(2, np.uint32)
    bad = np.zeros(shape, np.uint32)
    with pytest.raises(philox.PhiloxError, match="shape"):
        philox.block(bad if which == "counter" else counter, bad if which == "key" else key)
