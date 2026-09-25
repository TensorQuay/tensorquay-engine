"""NVFP4 CPU reference codec: TEST-PLAN §2 and REF-003 / REF-004. Tests only, never shipped.

The ordinary path follows NVIDIA ModelOpt ``qtensor/nvfp4_tensor.py`` at commit b311c054,
in float32 throughout:

- global scale ``g = amax / (6 * 448)``;
- block scale ``s = E4M3(clamp(r, 2^-9, 448))`` with ``r = block_amax / (6 * g)``. A computed
  ``r`` of 0 is set to 1.0 before the clamp; that covers a zero block and a tiny block whose
  ratio underflows in float32;
- codes ``q = E2M1(w / (s * g))``. A normalised value that overflows to infinity saturates to
  6 (code 7 or 15);
- decode ``w = e2m1(q) * (s * g)``. Upstream's lookup decodes both zero codes (0, 8) to +0.0.

Intentional deviations from the upstream helper, and limits (TEST-PLAN §2):

- non-finite **source** weights are rejected;
- an all-zero tensor gets the canonical encoding (zero codes, unit block scales, ``g = 1.0``);
  the upstream helper would produce NaN block scales;
- **representability boundary:** the global scale, derived or supplied, must be finite and > 0
  after its float32 cast. Subnormal scales are accepted. A tensor is rejected only when an
  actual block step ``E4M3(s) * g`` underflows to zero in float32. That check runs before
  normalisation, so no 0 / 0 can occur. Upstream has no such check, so this is an intentional
  safety deviation.

``e2m1_decode`` and ``decode_exact`` keep the format's signed zero (code 8 is -0.0); only
``dequantize`` follows the upstream +0.0 convention.
"""

from dataclasses import dataclass

import numpy as np

E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MIN_SCALE = 2.0**-9
GROUPS = (16, 32)
UNIT_SCALE_CODE = 0x38  # E4M3 1.0

_E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
_E2M1_ODD_BOUNDS = np.array([0.75, 1.75, 3.5], dtype=np.float32)  # ties that round up
_E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_E2M1_VALUES = np.concatenate([_E2M1_MAGNITUDES, -_E2M1_MAGNITUDES])  # format: code 8 is -0.0
# Upstream's dequantization lookup decodes code 8 to +0.0.
_E2M1_DEQUANT_VALUES = np.where(_E2M1_VALUES == 0, np.float32(0.0), _E2M1_VALUES)


def _e4m3_values() -> np.ndarray:
    codes = np.arange(256)
    exponent = (codes >> 3) & 0xF
    mantissa = (codes & 0x7).astype(np.float64)
    normal = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7.0)
    magnitude = np.where(exponent == 0, mantissa * 2.0**-9, normal)
    values = np.where(codes & 0x80, -magnitude, magnitude)
    values[[0x7F, 0xFF]] = np.nan  # E4M3FN has no infinities; these two codes are NaN
    return values.astype(np.float32)


_E4M3_VALUES = _e4m3_values()
_E4M3_POSITIVE = _E4M3_VALUES[:0x7F].astype(np.float64)  # codes 0x00..0x7E, ascending


class Nvfp4Error(ValueError):
    """Invalid input for the NVFP4 reference codec."""


@dataclass(frozen=True)
class Nvfp4Tensor:
    """A quantized [rows, K] tensor: packed E2M1 codes, E4M3 block scales, float32 global scale."""

    packed: np.ndarray  # uint8 [rows, K // 2]; element 2i in the low nibble
    scales: np.ndarray  # uint8 E4M3 codes [rows, K // group]
    global_scale: np.float32
    group: int


def _require_dtype(a: object, dtype: type, name: str) -> None:
    if not isinstance(a, np.ndarray) or a.dtype != dtype:
        raise Nvfp4Error(f"{name} must be a numpy {np.dtype(dtype).name} array")


def _require_2d(a: np.ndarray, name: str) -> None:
    if a.ndim != 2:
        raise Nvfp4Error(f"{name} must be 2-D, got {a.ndim}-D")
    if a.size == 0:
        raise Nvfp4Error(f"{name} must not be empty")


def _require_finite(a: np.ndarray, name: str) -> None:
    if not np.isfinite(a).all():
        raise Nvfp4Error(f"{name} must be finite (NaN and Inf are rejected)")


def _is_integer(x: object) -> bool:
    return isinstance(x, int | np.integer) and not isinstance(x, bool | np.bool_)


def _require_codes(codes: object, limit: int, name: str) -> None:
    _require_dtype(codes, np.uint8, name)
    if (codes > limit).any():
        raise Nvfp4Error(f"{name} out of range (maximum {limit})")


def e2m1_decode(codes: np.ndarray) -> np.ndarray:
    """E2M1 codes (uint8, 0..15, any shape) to float32; bit 3 is the sign, code 8 is -0.0."""
    _require_codes(codes, 15, "E2M1 codes")
    return np.asarray(_E2M1_VALUES[codes], dtype=np.float32)


def e4m3_decode(codes: np.ndarray) -> np.ndarray:
    """E4M3FN codes (uint8, any shape) to float32 values; 0x7F and 0xFF decode to NaN."""
    _require_dtype(codes, np.uint8, "E4M3 codes")
    return np.asarray(_E4M3_VALUES[codes], dtype=np.float32)


def _e2m1_codes(x: np.ndarray) -> np.ndarray:
    magnitude = np.abs(x)
    codes = np.searchsorted(_E2M1_BOUNDS, magnitude, side="left").astype(np.uint8)
    codes += np.isin(magnitude, _E2M1_ODD_BOUNDS).astype(np.uint8)
    return np.asarray(codes | np.where(x < 0, 8, 0).astype(np.uint8), dtype=np.uint8)


def e2m1_encode(x: np.ndarray) -> np.ndarray:
    """Finite float32 values (any shape) to E2M1 codes, rounding as the pinned recipe does.

    Ties go to the even code, magnitudes above 5 saturate to 6 (code 7), and the sign bit is
    set only for x < 0.
    """
    _require_dtype(x, np.float32, "x")
    _require_finite(x, "x")
    return _e2m1_codes(x)


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """Finite float32 values in [0, 448] to E4M3FN codes: nearest, ties to the even code."""
    _require_dtype(x, np.float32, "x")
    _require_finite(x, "x")
    if ((x < 0) | (x > E4M3_MAX)).any():
        raise Nvfp4Error("x out of range [0, 448]")
    v = x.astype(np.float64)
    hi = np.searchsorted(_E4M3_POSITIVE, v, side="left")  # first code whose value is >= v
    lo = np.maximum(hi - 1, 0)
    d_hi = _E4M3_POSITIVE[hi] - v
    d_lo = v - _E4M3_POSITIVE[lo]
    take_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    return np.asarray(np.where(take_hi, hi, lo), dtype=np.uint8)


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """Pack 2-D uint8 codes (0..15, even K) two per byte; element 2i goes in the low nibble."""
    _require_codes(codes, 15, "codes")
    _require_2d(codes, "codes")
    if codes.shape[1] % 2:
        raise Nvfp4Error("codes: K must be a multiple of 2")
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_nibbles: 2-D uint8 [rows, cols] to codes [rows, 2 * cols]."""
    _require_dtype(packed, np.uint8, "packed")
    _require_2d(packed, "packed")
    codes = np.empty((packed.shape[0], 2 * packed.shape[1]), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return codes


def decode_exact(code: int, scale_code: int, global_scale: float) -> float:
    """The exact value e2m1(code) * e4m3(scale_code) * global_scale in float64 (REF-003).

    Codes must be integers (Python or NumPy; not bool): code 0..15, scale_code 0..255 but not a
    NaN code (0x7F, 0xFF).
    """
    valid = _is_integer(code) and _is_integer(scale_code)
    if not (valid and 0 <= code <= 15 and 0 <= scale_code <= 255) or scale_code in (0x7F, 0xFF):
        raise Nvfp4Error("code or scale code out of range (integers required)")
    if not np.isfinite(global_scale):
        raise Nvfp4Error("global scale must be finite")
    return float(_E2M1_VALUES[code]) * float(_E4M3_VALUES[scale_code]) * float(global_scale)


def _require_group(group: object) -> None:
    if not _is_integer(group) or group not in GROUPS:
        raise Nvfp4Error(f"group must be the integer 16 or 32, got {group!r}")


def _check_weights(w: object, group: int) -> None:
    _require_dtype(w, np.float32, "w")
    _require_2d(w, "w")
    _require_group(group)
    if w.shape[1] % group:
        raise Nvfp4Error(f"K = {w.shape[1]} must be a multiple of the group {group}")
    _require_finite(w, "w")


def _checked_global_scale(global_scale: float, origin: str) -> np.float32:
    with np.errstate(over="ignore"):  # too large for float32 becomes inf and is rejected below
        g = np.float32(global_scale)
    if not (np.isfinite(g) and g > 0):
        raise Nvfp4Error(
            f"{origin} global scale {global_scale!r} is not representable: after the float32 "
            "cast it must be finite and > 0"
        )
    return g


def _natural_global_scale(amax: np.float32) -> np.float32:
    return _checked_global_scale(amax / np.float32(E2M1_MAX * E4M3_MAX), "derived")


def _amax(w: np.ndarray) -> np.float32:
    return np.max(np.abs(w))


def _canonical_zero(shape: tuple[int, int], group: int) -> Nvfp4Tensor:
    rows, k = shape
    scales = np.full((rows, k // group), UNIT_SCALE_CODE, dtype=np.uint8)
    return Nvfp4Tensor(np.zeros((rows, k // 2), dtype=np.uint8), scales, np.float32(1.0), group)


def _encode(w: np.ndarray, group: int, g: np.float32) -> Nvfp4Tensor:
    rows, k = w.shape
    blocks = w.reshape(rows, k // group, group)
    with np.errstate(over="ignore", under="ignore"):  # inf clamps or saturates, as upstream
        scale = np.max(np.abs(blocks), axis=-1) / (np.float32(E2M1_MAX) * g)
        scale[scale == 0] = 1.0  # a computed zero ratio: a zero block or an underflow
        clamped = np.clip(scale, np.float32(E4M3_MIN_SCALE), np.float32(E4M3_MAX))
        scale_codes = e4m3_encode(clamped)
        step = _E4M3_VALUES[scale_codes] * g  # float32 product first, as upstream does
        if (step == 0).any():
            raise Nvfp4Error(
                f"global scale {g!r} is not representable for this tensor: a block step "
                "(E4M3 scale * global scale) underflows to zero in float32"
            )
        codes = _e2m1_codes((blocks / step[..., None]).reshape(rows, k))
    return Nvfp4Tensor(pack_nibbles(codes), scale_codes, g, group)


def quantize(w: np.ndarray, group: int, global_scale: float | None = None) -> Nvfp4Tensor:
    """Quantize 2-D float32 [rows, K] to NVFP4 with blocks of ``group`` (16 or 32) along K."""
    _check_weights(w, group)
    if global_scale is not None:
        return _encode(w, group, _checked_global_scale(global_scale, "supplied"))
    amax = _amax(w)
    if amax == 0:
        return _canonical_zero(w.shape, group)
    return _encode(w, group, _natural_global_scale(amax))


def quantize_w13(gate: np.ndarray, up: np.ndarray, group: int) -> tuple[Nvfp4Tensor, Nvfp4Tensor]:
    """Quantize an expert's gate and up projections with one shared global scale."""
    _check_weights(gate, group)
    _check_weights(up, group)
    if gate.shape != up.shape:
        raise Nvfp4Error(f"gate and up must have the same shape, got {gate.shape}, {up.shape}")
    amax = max(_amax(gate), _amax(up))
    if amax == 0:
        return _canonical_zero(gate.shape, group), _canonical_zero(up.shape, group)
    g = _natural_global_scale(amax)
    return _encode(gate, group, g), _encode(up, group, g)


def _check_encoded(t: Nvfp4Tensor) -> None:
    _require_group(t.group)
    _require_dtype(t.packed, np.uint8, "packed")
    _require_2d(t.packed, "packed")
    k = 2 * t.packed.shape[1]
    if k % t.group:
        raise Nvfp4Error(f"K = {k} must be a multiple of the group {t.group}")
    _require_codes(t.scales, 0x7E, "scales")  # non-negative, non-NaN E4M3 codes only
    if t.scales.shape != (t.packed.shape[0], k // t.group):
        raise Nvfp4Error(f"scales shape {t.scales.shape} does not match packed {t.packed.shape}")
    _checked_global_scale(t.global_scale, "encoded")


def dequantize(t: Nvfp4Tensor) -> np.ndarray:
    """Decode to float32 [rows, K] as e2m1(code) * (e4m3(scale) * g), the product first.

    This follows upstream's lookup, so both zero codes decode to +0.0. Malformed encoded tensors
    (dtype, shape, group, NaN or negative scale codes, global scale) are rejected.
    """
    _check_encoded(t)
    step = _E4M3_VALUES[t.scales] * np.float32(t.global_scale)
    values = _E2M1_DEQUANT_VALUES[unpack_nibbles(t.packed)]
    rows, k = values.shape
    return (values.reshape(rows, k // t.group, t.group) * step[..., None]).reshape(rows, k)
