"""REF-004: our NVFP4 quantizer against the TEST-PLAN §2 codec contract.

Expectations are derived by hand from the contract. Byte identity with the pinned upstream recipe
lives in test_ref004_modelopt_fixture.py.
"""

import numpy as np
import pytest

from tq_reference import nvfp4

G = 2.0**-4  # a power-of-two global scale, so scaled values are exact
TINY = float(np.finfo(np.float32).tiny)


def f32(values):
    return np.array(values, dtype=np.float32)


def block(values, group=16):
    row = np.zeros(group, dtype=np.float32)
    row[: len(values)] = values
    return row


def codes_of(t):
    return nvfp4.unpack_nibbles(t.packed)


# ---- E2M1 rounding: ties, saturation, signs ----------------------------------------------


def test_ref004_e2m1_ties_round_half_to_even_code():
    # Tie points between neighbouring magnitudes: the even code wins.
    ties = f32([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert nvfp4.e2m1_encode(ties).tolist() == [0, 2, 2, 4, 4, 6, 6]
    assert nvfp4.e2m1_encode(-ties).tolist() == [8, 10, 10, 12, 12, 14, 14]


def test_ref004_e2m1_nearest_off_ties():
    x = f32([0.2499, 0.2501, 0.7499, 0.7501, 1.2501, 1.7499, 2.4999, 2.5001, 3.4999, 4.9999])
    assert nvfp4.e2m1_encode(x).tolist() == [0, 1, 1, 2, 3, 3, 4, 5, 5, 6]
    assert nvfp4.e2m1_encode(f32([5.0001])).tolist() == [7]


def test_ref004_e2m1_saturates_above_6():
    x = f32([6.0, 6.5, 7.0, 1.0e30, -6.0, -1.0e30])
    assert nvfp4.e2m1_encode(x).tolist() == [7, 7, 7, 7, 15, 15]


def test_ref004_e2m1_signed_zero():
    # +0 and -0 are not negative, so they keep code 0; -0.1 keeps its sign bit (code 8).
    assert nvfp4.e2m1_encode(f32([0.0, -0.0, 0.1, -0.1])).tolist() == [0, 0, 0, 8]


# ---- E4M3 block-scale rounding: nearest, ties to even -------------------------------------


def test_ref004_e4m3_encode_round_to_nearest_even():
    x = f32([1.0, 1.0625, 1.1875, 448.0, 2.0**-9, 1.5 * 2.0**-9, 2.5 * 2.0**-9, 13.0, 13.5])
    # 1.0625: midpoint of 1.0 (0x38) and 1.125 (0x39), so the even 0x38.
    # 1.1875: midpoint of 1.125 (0x39) and 1.25 (0x3A), so 0x3A.
    # 1.5 * 2^-9: midpoint of 0x01 and 0x02, so 0x02. 2.5 * 2^-9: midpoint of 0x02, 0x03: 0x02.
    # 13.5: midpoint of 13 (0x55) and 14 (0x56), so 0x56.
    expected = [0x38, 0x38, 0x3A, 0x7E, 0x01, 0x02, 0x02, 0x55, 0x56]
    assert nvfp4.e4m3_encode(x).tolist() == expected


# ---- The quantizer: scales, zero blocks, saturation, clamps, nibble order ------------------


def test_ref004_global_scale_is_amax_over_2688_in_float32():
    w = np.stack([block([0.3, -1.7]), block([0.01])])
    t = nvfp4.quantize(w, group=16)
    assert t.global_scale == np.float32(1.7) / np.float32(2688.0)
    assert t.packed.shape == (2, 8) and t.scales.shape == (2, 1) and t.group == 16


@pytest.mark.parametrize("group", [16, 32])
def test_ref004_exact_values_with_power_of_two_scales(group):
    # Block amax 6 * G gives a block scale of exactly 1.0 (0x38), so every code is exact and
    # dequantization reproduces the input.
    magnitudes = [6.0, 0.5, 1.5, 3.0, 4.0, 2.0, 1.0, 0.0]
    w = block([m * G for m in magnitudes], group)[None, :]
    t = nvfp4.quantize(w, group=group, global_scale=G)
    assert t.scales.tolist() == [[0x38]]
    assert codes_of(t)[0, : len(magnitudes)].tolist() == [7, 1, 3, 5, 6, 4, 2, 0]
    np.testing.assert_array_equal(nvfp4.dequantize(t), w)


def test_ref004_first_value_goes_to_the_low_nibble():
    w = block([0.5 * G, 6.0 * G])[None, :]  # codes 1, then 7
    assert int(nvfp4.quantize(w, group=16, global_scale=G).packed[0, 0]) == 0x71


def test_ref004_ties_inside_a_block():
    ties = [6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, -5.0]
    t = nvfp4.quantize(block([v * G for v in ties])[None, :], group=16, global_scale=G)
    assert codes_of(t)[0, : len(ties)].tolist() == [7, 0, 2, 2, 4, 4, 6, 6, 10, 14]


def test_ref004_block_scale_rounding_down_saturates_codes():
    # amax / (6 G) = 1.06 rounds down to the E4M3 value 1.0, so 6.36 exceeds 6 and saturates.
    t = nvfp4.quantize(block([6.36 * G, -6.36 * G, 1.0 * G])[None, :], group=16, global_scale=G)
    assert t.scales.tolist() == [[0x38]]
    assert codes_of(t)[0, :3].tolist() == [7, 15, 2]


def test_ref004_block_scale_clamps_to_448_and_2_pow_minus_9():
    big = block([6.0 * G * 1000.0])  # scale 1000 clamps to 448; the value saturates (code 7)
    tiny = block([6.0 * G * 2.0**-12])  # scale 2^-12 clamps up to 2^-9; value 0.75 -> code 2
    t = nvfp4.quantize(np.stack([big, tiny]), group=16, global_scale=G)
    assert t.scales.tolist() == [[0x7E], [0x01]]
    assert int(codes_of(t)[0, 0]) == 7 and int(codes_of(t)[1, 0]) == 2


def test_ref004_zero_block_inside_nonzero_tensor():
    w = np.concatenate([block([0.5, -0.25]), np.zeros(16, dtype=np.float32)])[None, :]
    t = nvfp4.quantize(w, group=16)
    assert t.scales[0, 1] == 0x38  # a zero block gets scale 1.0
    assert (codes_of(t)[0, 16:] == 0).all()
    assert (nvfp4.dequantize(t)[0, 16:] == 0.0).all()


@pytest.mark.parametrize("group", [16, 32])
def test_ref004_all_zero_tensor_has_the_canonical_encoding(group):
    w = np.zeros((3, 2 * group), dtype=np.float32)
    w[1, 5] = -0.0
    t = nvfp4.quantize(w, group=group)
    assert t.global_scale == np.float32(1.0)
    assert (t.scales == 0x38).all() and (t.packed == 0).all()
    out = nvfp4.dequantize(t)
    assert (out == 0.0).all() and not np.isnan(out).any()


def test_ref004_roundtrip_error_bound():
    rng = np.random.default_rng(1234)
    w = (rng.standard_normal((8, 64)) * 0.02).astype(np.float32)
    w[3, 7] = 0.4  # an outlier sets the tensor amax
    for group in (16, 32):
        t = nvfp4.quantize(w, group=group)
        step = nvfp4.e4m3_decode(t.scales).astype(np.float64) * float(t.global_scale)
        step = np.repeat(step, group, axis=1)
        err = np.abs(nvfp4.dequantize(t).astype(np.float64) - w.astype(np.float64))
        # Half the largest E2M1 gap (2) is one step, plus anything beyond saturation at 6 steps.
        bound = step + np.maximum(0.0, np.abs(w) - 6.0 * step)
        assert (err <= bound * (1 + 1e-6)).all()


# ---- W13: gate and up share one global scale ---------------------------------------------


def test_ref004_w13_shares_one_global_scale():
    gate = block([0.2, -0.05])[None, :]
    up = block([-0.9, 0.3])[None, :]
    tg, tu = nvfp4.quantize_w13(gate, up, group=16)
    shared = np.float32(0.9) / np.float32(2688.0)
    assert tg.global_scale == shared and tu.global_scale == shared
    alone = nvfp4.quantize(gate, group=16, global_scale=shared)
    assert tg.packed.tolist() == alone.packed.tolist()
    assert tg.scales.tolist() == alone.scales.tolist()


def test_ref004_w13_all_zero_pair_is_canonical():
    zeros = np.zeros((1, 16), dtype=np.float32)
    tg, tu = nvfp4.quantize_w13(zeros, zeros, group=16)
    assert tg.global_scale == tu.global_scale == np.float32(1.0)


def test_ref004_w13_rejects_mismatched_shapes():
    with pytest.raises(nvfp4.Nvfp4Error, match="shape"):
        nvfp4.quantize_w13(np.ones((1, 16), np.float32), np.ones((2, 16), np.float32), group=16)


# ---- Boundary rules from the pinned source ------------------------------------------------


def test_ref004_tiny_block_ratio_underflow_gets_unit_scale_not_the_clamp():
    # g follows the 3e38 outlier, so the tiny block's ratio underflows to 0 in float32. Upstream
    # sets that computed 0 to 1.0 before the clamp: scale 1.0 (0x38), not the clamp 2^-9 (0x01).
    w = np.concatenate([block([3.0e38]), block([-1.0e-45])])[None, :]
    t = nvfp4.quantize(w, group=16)
    assert t.scales.tolist() == [[0x7E, 0x38]]
    # -1.4e-45 / g underflows to -0.0, which is not < 0: the sign bit stays clear (code 0).
    assert int(codes_of(t)[0, 16]) == 0


def test_ref004_dequantize_uses_positive_zero_for_both_zero_codes():
    w = block([6.0 * G, -0.1 * G])[None, :]  # -0.1 step rounds to zero with the sign bit: code 8
    t = nvfp4.quantize(w, group=16, global_scale=G)
    assert codes_of(t)[0, :2].tolist() == [7, 8]
    out = nvfp4.dequantize(t)
    assert out[0, 1] == 0.0 and not np.signbit(out[0, 1])  # upstream lookup: +0.0
    assert np.signbit(nvfp4.e2m1_decode(np.array([8], dtype=np.uint8))[0])  # format: -0.0


def test_ref004_rejects_a_derived_global_scale_that_underflows_to_zero():
    # amax / 2688 = 3.7e-46 rounds to 0 in float32: the scale is not representable.
    w = block([1.0e-42])[None, :]
    with pytest.raises(nvfp4.Nvfp4Error, match="global scale"):
        nvfp4.quantize(w, group=16)
    with pytest.raises(nvfp4.Nvfp4Error, match="global scale"):
        nvfp4.quantize_w13(w, w, group=16)


@pytest.mark.parametrize("amax", [1.0e-36, 1.0e-40])
def test_ref004_subnormal_derived_global_scale_is_accepted(amax):
    # g = amax / 2688 is subnormal but positive, and the only block's step 448 * g is non-zero.
    w = block([amax, -amax / 2])[None, :]
    t = nvfp4.quantize(w, group=16)
    assert 0 < t.global_scale < np.finfo(np.float32).tiny
    assert t.scales.tolist() == [[0x7E]]
    assert codes_of(t)[0, :2].tolist() == [7, 13]  # 6, and -3 (-amax/2 is half of 6 steps)


def test_ref004_rejects_only_an_actual_block_step_underflow():
    # g = 2^-149 and the block holds 2 * 2^-149: its ratio 1/3 rounds to the E4M3 value 0.34375,
    # and 0.34375 * 2^-149 is below half the smallest subnormal, so the step underflows to 0.
    smallest = float(np.nextafter(np.float32(0), np.float32(1)))
    with pytest.raises(nvfp4.Nvfp4Error, match="global scale"):
        nvfp4.quantize(block([2.0 * smallest])[None, :], group=16, global_scale=smallest)
    # A zero block uses the unit scale, so its step is 2^-149 > 0: the same g is usable.
    t = nvfp4.quantize(np.zeros((1, 16), np.float32), group=16, global_scale=smallest)
    assert t.scales.tolist() == [[0x38]] and (t.packed == 0).all()
    assert (nvfp4.dequantize(t) == 0.0).all()


def test_ref004_supplied_small_scale_overflow_saturates_codes():
    # With g = 2^-126 the block ratio and the normalised values overflow to inf: the scale
    # clamps to 448 and the codes saturate (7, 15), as upstream. The source weights are finite.
    t = nvfp4.quantize(block([3.0e38, -3.0e38])[None, :], group=16, global_scale=TINY)
    assert t.scales.tolist() == [[0x7E]]
    assert codes_of(t)[0, :3].tolist() == [7, 15, 0]
    assert np.isfinite(nvfp4.dequantize(t)).all()


@pytest.mark.parametrize("g", [1.0e-50, 1.0e40, 0.0, -1.0, float("nan"), float("inf")])
def test_ref004_supplied_scale_is_validated_after_the_float32_cast(g):
    # 1e-50 casts to 0 and 1e40 to inf: neither is a finite positive float32.
    with pytest.raises(nvfp4.Nvfp4Error, match="global scale"):
        nvfp4.quantize(np.ones((1, 16), dtype=np.float32), group=16, global_scale=g)


# ---- Input validation ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_ref004_rejects_non_finite_source_weights(bad):
    w = np.ones((1, 16), dtype=np.float32)
    w[0, 3] = bad
    with pytest.raises(nvfp4.Nvfp4Error, match="finite"):
        nvfp4.quantize(w, group=16)


@pytest.mark.parametrize(
    ("w", "group", "message"),
    [
        (np.ones((1, 16), dtype=np.float64), 16, "float32"),
        (np.ones(16, dtype=np.float32), 16, "2-D"),
        (np.ones((1, 1, 16), dtype=np.float32), 16, "2-D"),
        (np.ones((1, 24), dtype=np.float32), 16, "multiple"),
        (np.ones((1, 16), dtype=np.float32), 8, "group"),
        (np.ones((0, 16), dtype=np.float32), 16, "empty"),
    ],
)
def test_ref004_rejects_invalid_shapes_dtypes_and_groups(w, group, message):
    with pytest.raises(nvfp4.Nvfp4Error, match=message):
        nvfp4.quantize(w, group=group)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: nvfp4.e2m1_decode(np.array([1.0])), "uint8"),
        (lambda: nvfp4.e2m1_decode(np.array([16], dtype=np.uint8)), "range"),
        (lambda: nvfp4.e4m3_decode(np.array([1], dtype=np.int64)), "uint8"),
        (lambda: nvfp4.e2m1_encode(np.array([1.0])), "float32"),
        (lambda: nvfp4.e2m1_encode(f32([np.nan])), "finite"),
        (lambda: nvfp4.e4m3_encode(np.array([1.0])), "float32"),
        (lambda: nvfp4.e4m3_encode(f32([np.inf])), "finite"),
        (lambda: nvfp4.e4m3_encode(f32([-1.0])), "range"),
        (lambda: nvfp4.e4m3_encode(f32([449.0])), "range"),
        (lambda: nvfp4.pack_nibbles(np.array([[16, 0]], dtype=np.uint8)), "range"),
        (lambda: nvfp4.pack_nibbles(np.array([1, 2], dtype=np.uint8)), "2-D"),
        (lambda: nvfp4.pack_nibbles(np.zeros((0, 2), dtype=np.uint8)), "empty"),
        (lambda: nvfp4.pack_nibbles(np.zeros((1, 3), dtype=np.uint8)), "multiple"),
        (lambda: nvfp4.unpack_nibbles(np.zeros((1, 2), dtype=np.float32)), "uint8"),
        (lambda: nvfp4.unpack_nibbles(np.zeros(2, dtype=np.uint8)), "2-D"),
        (lambda: nvfp4.decode_exact(16, 0x38, 1.0), "range"),
        (lambda: nvfp4.decode_exact(-1, 0x38, 1.0), "range"),
        (lambda: nvfp4.decode_exact(1, 0x7F, 1.0), "range"),
        (lambda: nvfp4.decode_exact(1, 0xFF, 1.0), "range"),
        (lambda: nvfp4.decode_exact(1, 256, 1.0), "range"),
        (lambda: nvfp4.decode_exact(1, 0x38, float("nan")), "global scale"),
    ],
)
def test_ref004_helpers_reject_invalid_input(call, message):
    with pytest.raises(nvfp4.Nvfp4Error, match=message):
        call()


@pytest.mark.parametrize("group", [0, 8, 64, 16.0, True, "16"])
def test_ref004_group_must_be_the_integer_16_or_32(group):
    with pytest.raises(nvfp4.Nvfp4Error, match="group"):
        nvfp4.quantize(np.ones((1, 32), np.float32), group)


def test_ref004_numpy_integer_group_is_accepted():
    assert nvfp4.quantize(np.ones((1, 32), np.float32), np.int64(16)).group == 16


def encoded(**changes):
    fields = {
        "packed": np.zeros((1, 8), np.uint8),
        "scales": np.full((1, 1), 0x38, np.uint8),
        "global_scale": np.float32(1.0),
        "group": 16,
    }
    fields.update(changes)
    return nvfp4.Nvfp4Tensor(**fields)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"group": 8}, "group"),
        ({"packed": np.zeros((1, 8), np.int32)}, "uint8"),
        ({"packed": np.zeros(8, np.uint8)}, "2-D"),
        ({"packed": np.zeros((1, 9), np.uint8)}, "multiple"),
        ({"scales": np.ones((1, 1), np.float32)}, "uint8"),
        ({"scales": np.array([[0x7F]], np.uint8)}, "range"),
        ({"scales": np.array([[0xB8]], np.uint8)}, "range"),
        ({"scales": np.full((2, 1), 0x38, np.uint8)}, "shape"),
        ({"global_scale": np.float32("nan")}, "global scale"),
    ],
)
def test_ref004_dequantize_rejects_malformed_encoded_tensors(changes, message):
    with pytest.raises(nvfp4.Nvfp4Error, match=message):
        nvfp4.dequantize(encoded(**changes))


def test_ref004_supplied_subnormal_scale_is_accepted():
    g = 2.0**-140  # subnormal; the block step 448 * g is non-zero
    t = nvfp4.quantize(block([2688.0 * g])[None, :], group=16, global_scale=g)
    assert t.global_scale == np.float32(g)
    assert t.scales.tolist() == [[0x7E]] and int(codes_of(t)[0, 0]) == 7


@pytest.mark.parametrize(
    ("call", "dtype"),
    [
        (lambda: nvfp4.e2m1_decode(np.array(8, dtype=np.uint8)), np.float32),
        (lambda: nvfp4.e4m3_decode(np.array(0x38, dtype=np.uint8)), np.float32),
        (lambda: nvfp4.e2m1_encode(np.array(-0.0, dtype=np.float32)), np.uint8),
        (lambda: nvfp4.e4m3_encode(np.array(1.0, dtype=np.float32)), np.uint8),
    ],
)
def test_ref004_zero_dimensional_inputs_give_zero_dimensional_arrays(call, dtype):
    out = call()
    assert isinstance(out, np.ndarray) and out.shape == () and out.dtype == dtype


def test_ref004_zero_dimensional_decode_keeps_signed_zero():
    out = nvfp4.e2m1_decode(np.array(8, dtype=np.uint8))
    assert out == 0.0 and np.signbit(out)


@pytest.mark.parametrize(
    ("code", "scale_code"), [(1.0, 0x38), (1, 56.0), (True, 0x38), (1, False), (np.bool_(1), 0x38)]
)
def test_ref004_decode_exact_requires_integer_codes(code, scale_code):
    with pytest.raises(nvfp4.Nvfp4Error, match="range"):
        nvfp4.decode_exact(code, scale_code, 1.0)


def test_ref004_decode_exact_accepts_numpy_integers():
    assert nvfp4.decode_exact(np.uint8(7), np.int64(0x38), 1.0) == 6.0
