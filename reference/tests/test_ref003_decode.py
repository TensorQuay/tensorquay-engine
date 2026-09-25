"""REF-003: NVFP4 decode tables vs values written by hand from the OCP MX / NVIDIA formats.

Expected values are independent of the implementation: hand-listed E2M1 and E4M3 values and
exact rational arithmetic with `fractions.Fraction` (TEST-PLAN §2, §3).
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from tq_reference import nvfp4

# E2M1 (OCP MX v1.0, FP4 E2M1): code bit 3 is the sign, bits 2..0 index these magnitudes.
E2M1_MAGNITUDES = [
    Fraction(0),
    Fraction(1, 2),
    Fraction(1),
    Fraction(3, 2),
    Fraction(2),
    Fraction(3),
    Fraction(4),
    Fraction(6),
]

# E4M3 (FN): bias 7, subnormal step 2^-9, max finite 448, 0x7F / 0xFF are NaN. Hand samples.
E4M3_SAMPLES = {
    0x00: Fraction(0),
    0x01: Fraction(1, 512),  # smallest subnormal, 2^-9
    0x02: Fraction(2, 512),
    0x07: Fraction(7, 512),  # largest subnormal
    0x08: Fraction(1, 64),  # smallest normal, 2^-6
    0x38: Fraction(1),
    0x39: Fraction(9, 8),
    0x3F: Fraction(15, 8),
    0x40: Fraction(2),
    0x55: Fraction(13),  # 0b0_1010_101: 2^(10 - 7) * (1 + 5/8) = 13
    0x7E: Fraction(448),
}

# Global scales: unit, a power of two, and non-unit, non-power-of-two float32 values.
GLOBAL_SCALES = [
    1.0,
    2.0**-4,
    float(np.float32(1.0) / np.float32(2688.0)),
    float(np.float32(0.3)),
    3.7e-5,
]


def signed(code):
    return (-1 if code & 0x8 else 1) * E2M1_MAGNITUDES[code & 0x7]


def test_ref003_e2m1_all_16_codes():
    got = nvfp4.e2m1_decode(np.arange(16, dtype=np.uint8))
    assert got.dtype == np.float32
    for code in range(16):
        assert Fraction(float(got[code])) == signed(code), f"code {code:#x}"
        sign = -1.0 if code & 0x8 else 1.0
        assert math.copysign(1.0, float(got[code])) == sign, f"sign of code {code:#x}"


def test_ref003_e4m3_samples():
    codes = np.array(sorted(E4M3_SAMPLES), dtype=np.uint8)
    got = nvfp4.e4m3_decode(codes)
    for code, value in zip(codes, got, strict=True):
        assert Fraction(float(value)) == E4M3_SAMPLES[int(code)], f"code {int(code):#x}"
    negative = nvfp4.e4m3_decode(np.array([0x80, 0xB8, 0xFE], dtype=np.uint8))
    assert negative[0] == 0.0 and math.copysign(1.0, float(negative[0])) == -1.0
    assert Fraction(float(negative[1])) == -1
    assert Fraction(float(negative[2])) == -448
    assert np.isnan(nvfp4.e4m3_decode(np.array([0x7F, 0xFF], dtype=np.uint8))).all()


@pytest.mark.parametrize("global_scale", GLOBAL_SCALES)
def test_ref003_exact_decode_products(global_scale):
    g32 = float(np.float32(global_scale))  # the stored global scale is float32
    for code in range(16):
        for scale_code, scale in E4M3_SAMPLES.items():
            got = nvfp4.decode_exact(code, scale_code, g32)
            assert isinstance(got, float)
            expected = signed(code) * scale * Fraction(g32)
            assert Fraction(got) == expected, f"code {code:#x} scale {scale_code:#x} g {g32!r}"


def test_ref003_low_nibble_holds_the_first_value():
    codes = np.array([[0x1, 0x2, 0xF, 0x8]], dtype=np.uint8)
    packed = nvfp4.pack_nibbles(codes)
    assert packed.dtype == np.uint8
    assert packed.tolist() == [[0x21, 0x8F]]
    assert nvfp4.unpack_nibbles(packed).tolist() == codes.tolist()
