"""RNG-001…005 reference for the shared Philox4x32-10 test stream.

Contract: ``reference/contracts/philox.md``. This is test infrastructure, not the production
token sampler. The generator is a counter-based bijection with no state, so the same counter and
key always give the same block and a stream is only an assignment of counters.

One round multiplies ``M0 * c0`` and ``M1 * c2`` as full 64-bit products, splits each into a high
and a low half, and emits ``(hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)``. Round ``r`` uses the key
``(k0 + r*W0, k1 + r*W1)`` modulo ``2**32``, so round zero uses the key exactly as supplied.

The word at logical offset ``i`` of a stream is lane ``i % 4`` of the block whose key is the seed
split low word first, and whose counter is ``[(i//4) low, (i//4) high, stream low, stream high]``.
Each stream owns ``2**64`` words and never carries into another stream's domain.
"""

import numpy as np

MASK32 = (1 << 32) - 1
MAX_U64 = (1 << 64) - 1
M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85
ROUNDS = 10
UNIFORM_SCALE = np.float32(1.0 / 16777216.0)  # 2**-24, exact in float32
UINT32 = np.dtype(np.uint32)


class PhiloxError(ValueError):
    """Invalid input to the RNG-001…005 reference."""


def _require_array(name: str, value, shape: tuple | None = None) -> np.ndarray:
    """Check one array argument. A native-order uint32 dtype is required, never converted."""
    if not isinstance(value, np.ndarray):
        raise PhiloxError(f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype != UINT32:
        raise PhiloxError(f"{name} must have dtype uint32, got {value.dtype}")
    if shape is not None and value.shape != shape:
        raise PhiloxError(f"{name} must have shape {shape}, got {value.shape}")
    return value


def _require_scalar(name: str, value) -> int:
    """Check one unsigned 64-bit scalar. The value is never formatted, so huge ints are safe."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PhiloxError(f"{name} must be an integer, got {type(value).__name__}")
    number = int(value)
    if not 0 <= number <= MAX_U64:
        raise PhiloxError(f"{name} must be in [0, 2**64)")
    return number


def _rounds(counter: list[np.ndarray], key0: int, key1: int) -> list[np.ndarray]:
    """Ten rounds over uint32 counter-word arrays; the keys stay Python ints."""
    c0, c1, c2, c3 = counter
    for round_index in range(ROUNDS):
        product0 = c0.astype(np.uint64) * np.uint64(M0)
        product1 = c2.astype(np.uint64) * np.uint64(M1)
        high0 = (product0 >> np.uint64(32)).astype(np.uint32)
        high1 = (product1 >> np.uint64(32)).astype(np.uint32)
        round0 = np.uint32((key0 + round_index * W0) & MASK32)
        round1 = np.uint32((key1 + round_index * W1) & MASK32)
        c0, c1, c2, c3 = (
            high1 ^ c1 ^ round0,
            product1.astype(np.uint32),
            high0 ^ c3 ^ round1,
            product0.astype(np.uint32),
        )
    return [c0, c1, c2, c3]


def block(counter, key) -> np.ndarray:
    """Return the four output words of one Philox4x32-10 block as a fresh uint32 ``[4]`` array."""
    counter = _require_array("counter", counter, (4,))
    key = _require_array("key", key, (2,))
    words = _rounds([counter[i : i + 1] for i in range(4)], int(key[0]), int(key[1]))
    return np.concatenate(words).astype(np.uint32, copy=False)


def words(seed, stream, start, count) -> np.ndarray:
    """Return ``count`` stream words from logical offset ``start`` as a fresh uint32 array."""
    seed = _require_scalar("seed", seed)
    stream = _require_scalar("stream", stream)
    start = _require_scalar("start", start)
    count = _require_scalar("count", count)
    if count and start + count - 1 > MAX_U64:
        raise PhiloxError("word range exceeds u64")
    if not count:
        return np.zeros(0, dtype=np.uint32)

    first = start // 4
    blocks = (start + count - 1) // 4 - first + 1
    group = np.uint64(first) + np.arange(blocks, dtype=np.uint64)
    counter = [
        (group & np.uint64(MASK32)).astype(np.uint32),
        (group >> np.uint64(32)).astype(np.uint32),
        np.full(blocks, stream & MASK32, dtype=np.uint32),
        np.full(blocks, stream >> 32, dtype=np.uint32),
    ]
    lanes = np.stack(_rounds(counter, seed & MASK32, seed >> 32), axis=-1).reshape(-1)
    offset = start - first * 4
    return np.ascontiguousarray(lanes[offset : offset + count])


def uniform_f32(words) -> np.ndarray:
    """Map words onto ``[0, 1)`` as ``(word >> 8) * 2**-24``, a grid of ``2**24`` float32 values.

    The top 24 bits are exactly representable in float32 and the scale is a power of two, so the
    product is exact. The largest result is ``1 - 2**-24``, never ``1``. The result keeps the
    input's shape, including a zero-dimensional array for scalar-shaped input.
    """
    values = _require_array("words", words)
    scaled = np.right_shift(values, np.uint32(8)).astype(np.float32) * UNIFORM_SCALE
    return np.asarray(scaled, dtype=np.float32, order="C")
