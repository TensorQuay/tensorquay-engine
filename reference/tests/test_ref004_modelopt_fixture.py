"""REF-004, ordinary finite non-zero path: byte identity with the pinned upstream ModelOpt recipe.

The fixture was produced by running the pinned upstream code itself (tools/modelopt_nvfp4_fixture.py
and the provenance block in the fixture). Our implementation never generated or touched it.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tq_reference import nvfp4

FIXTURE = Path(__file__).parent / "fixtures" / "modelopt_nvfp4_b311c054.json"
PINNED_COMMIT = "b311c054de4052df9c7f3de9409b7598f44a0dba"


def load():
    return json.loads(FIXTURE.read_text())


def f32_from_hex(words):
    return np.array([struct.unpack(">f", bytes.fromhex(w))[0] for w in words], dtype=np.float32)


def hex_of_f32(values):
    flat = np.asarray(values, dtype=np.float32).ravel()
    return [struct.pack(">f", float(v)).hex() for v in flat]


def test_ref004_fixture_provenance_is_the_pinned_recipe():
    prov = load()["provenance"]
    assert prov["commit"] == PINNED_COMMIT
    assert prov["upstream"] == "https://github.com/NVIDIA/Model-Optimizer"
    assert "modelopt/torch/quantization/qtensor/nvfp4_tensor.py" in prov["files_sha256"]
    assert all(len(h) == 64 for h in prov["files_sha256"].values())


def cases():
    return [pytest.param(c, id=c["name"]) for c in load()["cases"]]


def quantize_case(case, w):
    rows, cols = case["shape"]
    if case["kind"] == "w13":
        partner = f32_from_hex(case["partner_f32_hex"]).reshape(rows, cols)
        gate, up = (w, partner) if case["role"] == "gate" else (partner, w)
        tg, tu = nvfp4.quantize_w13(gate, up, group=case["group"])
        return tg if case["role"] == "gate" else tu
    g = case["supplied_global_scale_f32_hex"]
    supplied = None if g is None else f32_from_hex([g])[0]
    return nvfp4.quantize(w, group=case["group"], global_scale=supplied)


@pytest.mark.parametrize("case", cases())
def test_ref004_matches_pinned_modelopt_byte_for_byte(case):
    rows, cols = case["shape"]
    t = quantize_case(case, f32_from_hex(case["input_f32_hex"]).reshape(rows, cols))
    expected = case["expected"]
    assert t.packed.tobytes().hex() == expected["packed_hex"]
    assert t.scales.tobytes().hex() == expected["scales_hex"]
    assert hex_of_f32([t.global_scale]) == [expected["global_scale_f32_hex"]]
    assert hex_of_f32(nvfp4.dequantize(t)) == expected["dequant_f32_hex"]
