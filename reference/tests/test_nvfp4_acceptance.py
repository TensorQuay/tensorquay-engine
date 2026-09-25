"""Independent acceptance of the documented NVFP4 codec, written before implementation review."""

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from tq_reference import nvfp4

FP4_VALUES = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)
PACKED_LEVELS = np.array([[0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE]], dtype=np.uint8)


def _scale_value(code):
    """Integer exponent/significand interpretation of the E4M3FN format."""
    exponent, fraction = (code >> 3) & 15, code & 7
    if exponent == 15 and fraction == 7:
        return math.nan
    magnitude = (
        math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(8 + fraction, exponent - 10)
    )
    return -magnitude if code & 128 else magnitude


def _f32_bytes(value):
    return struct.pack("<f", float(value))


def test_ref003_every_fp4_code_and_signed_zero():
    codes = np.arange(16, dtype=np.uint8).reshape(2, 8)
    decoded = nvfp4.e2m1_decode(codes)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, FP4_VALUES.reshape(2, 8))
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(FP4_VALUES.reshape(2, 8)))


def test_ref003_every_fp8_code_against_integer_format_definition():
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    expected = np.array([_scale_value(i) for i in range(256)], dtype=np.float32).reshape(16, 16)
    decoded = nvfp4.e4m3_decode(codes)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, expected)
    assert np.isnan(decoded).sum() == 2
    assert np.signbit(decoded.flat[128])
    assert not np.signbit(decoded.flat[0])


@pytest.mark.parametrize("global_scale", [0.0, 0.3, 1.0, 1.3, -2.75])
def test_ref003_float64_decode_does_not_round_global_scale_to_float32(global_scale):
    for code in range(16):
        for scale_code in (0, 1, 7, 8, 0x38, 0x55, 0x7E, 0xB8):
            expected = float(FP4_VALUES[code]) * _scale_value(scale_code) * global_scale
            actual = nvfp4.decode_exact(code, scale_code, global_scale)
            assert actual == expected
            assert math.copysign(1, actual) == math.copysign(1, expected)


def test_fp4_midpoints_and_both_adjacent_float32_values():
    boundaries = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
    below = np.nextafter(boundaries, np.float32(-math.inf))
    above = np.nextafter(boundaries, np.float32(math.inf))
    values = np.stack([below, boundaries, above])
    expected = np.array([list(range(7)), [0, 2, 2, 4, 4, 6, 6], list(range(1, 8))], dtype=np.uint8)
    np.testing.assert_array_equal(nvfp4.e2m1_encode(values), expected)
    np.testing.assert_array_equal(nvfp4.e2m1_encode(-values), expected | 8)


def test_fp4_saturation_and_zero_sign_encoding():
    values = np.array([-1e30, -6, -0.1, -0.0, 0.0, 0.1, 6, 1e30], dtype=np.float32)
    expected = np.array([15, 15, 8, 0, 0, 0, 7, 7], dtype=np.uint8)
    np.testing.assert_array_equal(nvfp4.e2m1_encode(values), expected)


def test_fp8_all_nonnegative_values_and_rounding_boundaries():
    exact = np.array([_scale_value(i) for i in range(127)], dtype=np.float32)
    np.testing.assert_array_equal(nvfp4.e4m3_encode(exact), np.arange(127, dtype=np.uint8))
    midpoints = (exact[:-1] + exact[1:]) / np.float32(2)
    lower = np.arange(126, dtype=np.uint8)
    tie = lower + (lower & 1)
    np.testing.assert_array_equal(nvfp4.e4m3_encode(midpoints), tie)
    np.testing.assert_array_equal(
        nvfp4.e4m3_encode(np.nextafter(midpoints, np.float32(-math.inf))), lower
    )
    np.testing.assert_array_equal(
        nvfp4.e4m3_encode(np.nextafter(midpoints, np.float32(math.inf))), lower + 1
    )


def test_scalar_arrays_are_supported_by_elementwise_codecs():
    assert nvfp4.e2m1_decode(np.array(8, dtype=np.uint8)).shape == ()
    assert nvfp4.e4m3_decode(np.array(0x38, dtype=np.uint8)).item() == 1
    assert nvfp4.e2m1_encode(np.array(1.5, dtype=np.float32)).item() == 3
    assert nvfp4.e4m3_encode(np.array(1.0, dtype=np.float32)).item() == 0x38
    for operation, dtype, value in (
        (nvfp4.e2m1_decode, np.uint8, 8),
        (nvfp4.e4m3_decode, np.uint8, 0x38),
        (nvfp4.e2m1_encode, np.float32, 1.5),
        (nvfp4.e4m3_encode, np.float32, 1.0),
    ):
        result = operation(np.array(value, dtype=dtype))
        assert isinstance(result, np.ndarray)
        assert result.shape == ()


def test_nibble_order_has_an_independent_byte_pattern():
    codes = np.arange(16, dtype=np.uint8).reshape(1, 16)
    expected = np.array([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=np.uint8)
    np.testing.assert_array_equal(nvfp4.pack_nibbles(codes), expected)
    np.testing.assert_array_equal(nvfp4.unpack_nibbles(expected), codes)


def test_unpack_covers_every_possible_packed_byte():
    packed = np.arange(256, dtype=np.uint8).reshape(16, 16)
    expected = np.array([(i % 16, i // 16) for i in range(256)], dtype=np.uint8).reshape(16, 32)
    np.testing.assert_array_equal(nvfp4.unpack_nibbles(packed), expected)


@pytest.mark.parametrize("group", [16, 32])
def test_ref004_known_levels_with_automatic_and_supplied_global_scale(group):
    values = np.tile(FP4_VALUES, (2, group // 16))
    expected = np.tile(PACKED_LEVELS, (2, group // 16))
    for supplied in (None, np.float32(1.0)):
        encoded = nvfp4.quantize(values, group, global_scale=supplied)
        np.testing.assert_array_equal(encoded.packed, expected)
        expected_scale = 0x7E if supplied is None else 0x38
        np.testing.assert_array_equal(encoded.scales, np.full((2, 1), expected_scale, np.uint8))
        expected_global = 6 / 2688 if supplied is None else 1
        assert _f32_bytes(encoded.global_scale) == struct.pack("<f", expected_global)
        restored = nvfp4.dequantize(encoded)
        assert restored.dtype == np.float32
        np.testing.assert_array_equal(restored, values)
        assert not np.signbit(restored[0, 8])


def test_ref004_independent_scalar_review_vectors():
    fixture = Path(__file__).parent / "fixtures" / "lead_nvfp4_vectors.json"
    data = json.loads(fixture.read_text())
    for case in data["cases"]:
        values = np.array(case["values"], dtype=np.float32).reshape(data["metadata"]["shape"])
        values.setflags(write=False)
        encoded = nvfp4.quantize(values, case["group"])
        np.testing.assert_array_equal(encoded.packed.ravel(), case["packed"])
        np.testing.assert_array_equal(encoded.scales.ravel(), case["scale_bytes"])
        assert _f32_bytes(encoded.global_scale) == struct.pack("<f", case["global_scale"])
        np.testing.assert_array_equal(values.ravel(), np.array(case["values"], dtype=np.float32))


def _upstream_cases():
    fixture = Path(__file__).parent / "fixtures" / "modelopt_nvfp4_b311c054.json"
    data = json.loads(fixture.read_text())
    assert data["provenance"]["commit"] == "b311c054de4052df9c7f3de9409b7598f44a0dba"
    return [pytest.param(case, id=case["name"]) for case in data["cases"]]


@pytest.mark.parametrize("case", _upstream_cases())
def test_ref004_pinned_upstream_results_including_dequantization_bits(case):
    def from_words(words):
        return np.frombuffer(bytes.fromhex("".join(words)), dtype=">f4").astype(np.float32)

    values = from_words(case["input_f32_hex"]).reshape(case["shape"])
    values.setflags(write=False)
    if case["kind"] == "w13":
        partner = from_words(case["partner_f32_hex"]).reshape(case["shape"])
        partner.setflags(write=False)
        halves = (values, partner) if case["role"] == "gate" else (partner, values)
        result = nvfp4.quantize_w13(*halves, case["group"])[case["role"] == "up"]
    else:
        word = case["supplied_global_scale_f32_hex"]
        supplied = None if word is None else from_words([word])[0]
        result = nvfp4.quantize(values, case["group"], global_scale=supplied)
    expected = case["expected"]
    assert result.packed.tobytes() == bytes.fromhex(expected["packed_hex"])
    assert result.scales.tobytes() == bytes.fromhex(expected["scales_hex"])
    assert struct.pack(">f", float(result.global_scale)).hex() == expected["global_scale_f32_hex"]
    restored = nvfp4.dequantize(result)
    assert restored.dtype == np.float32
    assert restored.shape == tuple(case["shape"])
    assert restored.astype(">f4").tobytes().hex() == "".join(expected["dequant_f32_hex"])


@pytest.mark.parametrize("group", [16, 32])
def test_ref004_strided_readonly_weights_preserve_rows_and_do_not_mutate_inputs(group):
    # Every block includes all FP4 levels; both row and column strides are nonstandard.
    backing = np.tile(np.repeat(FP4_VALUES, 2), (4, group // 16))
    values = backing[::2, ::-2]
    assert not values.flags.c_contiguous
    values.setflags(write=False)
    before = backing.copy()
    contiguous = nvfp4.quantize(values.copy(), group, global_scale=1.0)
    encoded = nvfp4.quantize(values, group, global_scale=1.0)
    np.testing.assert_array_equal(encoded.packed, contiguous.packed)
    np.testing.assert_array_equal(encoded.scales, contiguous.scales)
    np.testing.assert_array_equal(nvfp4.dequantize(encoded), values)
    np.testing.assert_array_equal(backing, before)


@pytest.mark.parametrize("group", [16, 32])
def test_ref004_canonical_zero_and_zero_block_inside_nonzero_tensor(group):
    zeros = np.zeros((2, group * 2), dtype=np.float32)
    canonical = nvfp4.quantize(zeros, group)
    assert canonical.global_scale == np.float32(1)
    np.testing.assert_array_equal(canonical.packed, np.zeros((2, group), dtype=np.uint8))
    np.testing.assert_array_equal(canonical.scales, np.full((2, 2), 0x38, dtype=np.uint8))
    np.testing.assert_array_equal(nvfp4.dequantize(canonical), zeros)
    mixed = zeros.copy()
    mixed[:, group:] = 6
    result = nvfp4.quantize(mixed, group)
    np.testing.assert_array_equal(result.scales, np.array([[0x38, 0x7E]] * 2, dtype=np.uint8))
    np.testing.assert_array_equal(result.packed[:, : group // 2], 0)
    np.testing.assert_array_equal(result.packed[:, group // 2 :], 0x77)


@pytest.mark.parametrize("group", [16, 32])
def test_ref004_w13_uses_one_global_scale_for_both_halves(group):
    gate = np.full((2, group * 2), 6, dtype=np.float32)
    up = np.full_like(gate, 3)
    gate_result, up_result = nvfp4.quantize_w13(gate, up, group)
    assert _f32_bytes(gate_result.global_scale) == struct.pack("<f", 6 / 2688)
    assert _f32_bytes(up_result.global_scale) == _f32_bytes(gate_result.global_scale)
    np.testing.assert_array_equal(gate_result.scales, 0x7E)
    np.testing.assert_array_equal(up_result.scales, 0x76)
    np.testing.assert_array_equal(gate_result.packed, 0x77)
    np.testing.assert_array_equal(up_result.packed, 0x77)
    np.testing.assert_array_equal(nvfp4.dequantize(gate_result), gate)
    np.testing.assert_array_equal(nvfp4.dequantize(up_result), up)


def test_ref004_w13_empty_half_preserves_shared_global_scale():
    zero = np.zeros((1, 16), dtype=np.float32)
    nonzero = np.full_like(zero, 3)
    for gate, up in ((zero, nonzero), (nonzero, zero)):
        outputs = nvfp4.quantize_w13(gate, up, 16)
        for source, result in zip((gate, up), outputs, strict=True):
            assert _f32_bytes(result.global_scale) == struct.pack("<f", 3 / 2688)
            np.testing.assert_array_equal(nvfp4.dequantize(result), source)
    for result in nvfp4.quantize_w13(zero, zero, 16):
        assert result.global_scale == np.float32(1)
        np.testing.assert_array_equal(result.scales, 0x38)
        np.testing.assert_array_equal(result.packed, 0)


def test_ref004_subnormal_scale_clamp_and_underflow_are_distinct():
    smallest = np.nextafter(np.float32(0), np.float32(1))
    values = np.concatenate(
        [np.full(16, smallest, np.float32), np.full(16, 0.0001, np.float32)]
    ).reshape(1, 32)
    result = nvfp4.quantize(values, 16, global_scale=np.float32(1))
    np.testing.assert_array_equal(result.scales, np.array([[0x38, 0x01]], np.uint8))
    np.testing.assert_array_equal(result.packed, 0)
    with pytest.raises(nvfp4.Nvfp4Error, match="global scale"):
        nvfp4.quantize(np.full((1, 16), smallest, np.float32), 16)


def test_ref004_supplied_scale_saturates_without_rejecting_finite_source():
    values = np.full((1, 16), np.finfo(np.float32).max, dtype=np.float32)
    values[0, 1::2] *= -1
    result = nvfp4.quantize(values, 16, global_scale=np.float32(1e-30))
    np.testing.assert_array_equal(result.scales, 0x7E)
    np.testing.assert_array_equal(result.packed, 0xF7)
    assert np.isfinite(nvfp4.dequantize(result)).all()


@pytest.mark.parametrize("exponent", [-127, -140, -149])
@pytest.mark.parametrize("group", [16, 32])
def test_ref004_representable_subnormal_global_scales_are_supported(exponent, group):
    global_scale = np.float32(math.ldexp(1, exponent))
    values = np.full((1, group), math.ldexp(2688, exponent), dtype=np.float32)
    for supplied in (None, global_scale):
        encoded = nvfp4.quantize(values, group, global_scale=supplied)
        assert _f32_bytes(encoded.global_scale) == _f32_bytes(global_scale)
        np.testing.assert_array_equal(encoded.scales, 0x7E)
        np.testing.assert_array_equal(encoded.packed, 0x77)
        np.testing.assert_array_equal(nvfp4.dequantize(encoded), values)
    gate, up = nvfp4.quantize_w13(values, values / np.float32(2), group)
    assert _f32_bytes(gate.global_scale) == _f32_bytes(up.global_scale)
    np.testing.assert_array_equal(nvfp4.dequantize(up), values / np.float32(2))


def test_ref004_reject_only_an_actual_block_step_underflow():
    smallest = np.nextafter(np.float32(0), np.float32(1))
    # With this block amax, the scale rounds to 0.171875, whose product with g is zero.
    with pytest.raises(nvfp4.Nvfp4Error, match="global scale"):
        nvfp4.quantize(np.full((1, 16), smallest, np.float32), 16, global_scale=smallest)
    # A zero block uses the unit block scale, so the same supplied global scale is usable.
    zeros = np.zeros((1, 16), np.float32)
    encoded = nvfp4.quantize(zeros, 16, global_scale=smallest)
    assert _f32_bytes(encoded.global_scale) == _f32_bytes(smallest)
    np.testing.assert_array_equal(encoded.scales, 0x38)
    np.testing.assert_array_equal(encoded.packed, 0)
    np.testing.assert_array_equal(nvfp4.dequantize(encoded), zeros)


def test_ref004_dequantization_matches_upstream_zero_convention():
    tensor = nvfp4.Nvfp4Tensor(
        packed=np.full((1, 8), 0x88, dtype=np.uint8),
        scales=np.array([[0x38]], dtype=np.uint8),
        global_scale=np.float32(0.3),
        group=16,
    )
    decoded = nvfp4.dequantize(tensor)
    np.testing.assert_array_equal(decoded, np.zeros((1, 16), dtype=np.float32))
    assert not np.signbit(decoded).any()


@pytest.mark.parametrize("group", [0, 8, 64, 16.0, True])
def test_quantize_rejects_invalid_group(group):
    with pytest.raises(nvfp4.Nvfp4Error, match="group"):
        nvfp4.quantize(np.ones((1, 32), np.float32), group)


@pytest.mark.parametrize(
    "shape, message",
    [
        ((16,), "2-D"),
        ((1, 1, 16), "2-D"),
        ((0, 16), "empty"),
        ((1, 0), "empty"),
        ((1, 17), "multiple"),
    ],
)
def test_quantize_rejects_invalid_shapes(shape, message):
    with pytest.raises(nvfp4.Nvfp4Error, match=message):
        nvfp4.quantize(np.zeros(shape, dtype=np.float32), 16)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_quantize_rejects_nonfinite_source_and_w13_halves(value):
    good = np.ones((1, 16), np.float32)
    bad = good.copy()
    bad[0, 3] = value
    with pytest.raises(nvfp4.Nvfp4Error, match="finite"):
        nvfp4.quantize(bad, 16)
    for gate, up in ((bad, good), (good, bad)):
        with pytest.raises(nvfp4.Nvfp4Error, match="finite"):
            nvfp4.quantize_w13(gate, up, 16)


@pytest.mark.parametrize("scale", [0, -1, math.nan, math.inf, -math.inf, 1e-50, 1e40])
def test_quantize_rejects_unusable_global_scales_after_cast(scale):
    with pytest.raises(nvfp4.Nvfp4Error, match="global scale"):
        nvfp4.quantize(np.ones((1, 16), np.float32), 16, global_scale=scale)


def test_w13_rejects_mismatched_shapes():
    with pytest.raises(nvfp4.Nvfp4Error, match="shape"):
        nvfp4.quantize_w13(np.ones((1, 16), np.float32), np.ones((2, 16), np.float32), 16)


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32, np.bool_])
def test_float_operations_reject_implicit_dtype_conversion(dtype):
    values = np.ones((1, 16), dtype=dtype)
    for operation in (nvfp4.e2m1_encode, nvfp4.e4m3_encode):
        with pytest.raises(nvfp4.Nvfp4Error, match="float32"):
            operation(values)
    with pytest.raises(nvfp4.Nvfp4Error, match="float32"):
        nvfp4.quantize(values, 16)


@pytest.mark.parametrize(
    "operation", ["e2m1_decode", "e4m3_decode", "pack_nibbles", "unpack_nibbles"]
)
def test_code_operations_reject_wrong_dtype(operation):
    with pytest.raises(nvfp4.Nvfp4Error, match="uint8"):
        getattr(nvfp4, operation)(np.ones((1, 16), dtype=np.int32))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_elementwise_encoders_reject_nonfinite_values(value):
    for operation in (nvfp4.e2m1_encode, nvfp4.e4m3_encode):
        with pytest.raises(nvfp4.Nvfp4Error, match="finite"):
            operation(np.array([value], np.float32))


@pytest.mark.parametrize("value", [-1, 449])
def test_fp8_scale_encoder_rejects_out_of_range(value):
    with pytest.raises(nvfp4.Nvfp4Error, match="range"):
        nvfp4.e4m3_encode(np.array([value], np.float32))


def test_packed_code_validation():
    for operation in (nvfp4.e2m1_decode, nvfp4.pack_nibbles):
        with pytest.raises(nvfp4.Nvfp4Error, match="range"):
            operation(np.array([[0, 16]], np.uint8))
    with pytest.raises(nvfp4.Nvfp4Error, match="multiple"):
        nvfp4.pack_nibbles(np.zeros((1, 3), np.uint8))
    for operation in (nvfp4.pack_nibbles, nvfp4.unpack_nibbles):
        with pytest.raises(nvfp4.Nvfp4Error, match="2-D"):
            operation(np.array([0, 1], np.uint8))
        with pytest.raises(nvfp4.Nvfp4Error, match="empty"):
            operation(np.zeros((1, 0), np.uint8))


@pytest.mark.parametrize("code,scale", [(-1, 56), (16, 56), (0, -1), (0, 256), (0, 127), (0, 255)])
def test_exact_decoder_rejects_invalid_codes(code, scale):
    with pytest.raises(nvfp4.Nvfp4Error):
        nvfp4.decode_exact(code, scale, 1.0)


@pytest.mark.parametrize("code,scale", [(1.0, 56), (1, 56.0), (True, 56), (1, False)])
def test_exact_decoder_rejects_noninteger_code_types(code, scale):
    with pytest.raises(nvfp4.Nvfp4Error, match="range"):
        nvfp4.decode_exact(code, scale, 1.0)


@pytest.mark.parametrize("scale", [math.nan, math.inf, -math.inf])
def test_exact_decoder_rejects_nonfinite_global_scale(scale):
    with pytest.raises(nvfp4.Nvfp4Error, match="finite"):
        nvfp4.decode_exact(1, 56, scale)


@pytest.mark.parametrize(
    "field, value",
    [
        ("packed", np.zeros((1, 8), np.int32)),
        ("packed", np.zeros(8, np.uint8)),
        ("packed", np.zeros((1, 9), np.uint8)),
        ("scales", np.ones((1, 1), np.float32)),
        ("scales", np.array([[0x7F]], np.uint8)),
        ("scales", np.ones((2, 1), np.uint8)),
        ("global_scale", np.float32(math.nan)),
        ("group", 8),
    ],
)
def test_dequantize_rejects_malformed_encoded_tensors(field, value):
    fields = {
        "packed": np.zeros((1, 8), np.uint8),
        "scales": np.full((1, 1), 0x38, np.uint8),
        "global_scale": np.float32(1),
        "group": 16,
    }
    fields[field] = value
    with pytest.raises(nvfp4.Nvfp4Error):
        nvfp4.dequantize(nvfp4.Nvfp4Tensor(**fields))
