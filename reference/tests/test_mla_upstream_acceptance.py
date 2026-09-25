"""Independent checks of upstream fixture provenance, input semantics and MLA agreement."""

import base64
import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_mla_acceptance import _assert_result, _assert_vector, _expanded_oracle

from tq_reference import mla_attention

PIN = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
SOURCE_HASHES = {
    "modeling_glm5_next.py": "431eae74b857ec2a8e38a96f9f76cdc0b366bf7e35bd739b071c35761f113888",
    "configuration_glm5_next.py": (
        "f12b5876701d000f930cd8102124e07d05675a79a5591db29c18244f5f025dec"
    ),
    "sdpa_attention.py": "53c7229daca9ade4c5df874194448938c1edc925abbc71809f9750dd66381e6f",
}
CASE_NAMES = {
    "core": {"asym_multihead", "mask_edges", "dup_invalid_indices", "precision_full_mantissa"},
    "glm_width": {"glm_width"},
}


@pytest.fixture(params=["core", "glm_width"])
def fixture_data(request):
    path = Path(__file__).parent / "fixtures" / f"transformers_mla_770e4c40_{request.param}.json"
    assert path.stat().st_size < 2 * 1024 * 1024
    data = json.loads(path.read_text())
    assert {case["name"] for case in data["cases"]} == CASE_NAMES[request.param]
    assert len(data["cases"]) == len(CASE_NAMES[request.param])
    return data


def _array(record):
    dtype = np.dtype(record["dtype"])
    shape = tuple(record["shape"])
    assert all(isinstance(n, int) and n > 0 for n in shape)
    raw = base64.b64decode(record["b64"], validate=True)
    assert len(raw) == math.prod(shape) * dtype.itemsize
    value = np.frombuffer(raw, dtype=dtype).reshape(shape)
    assert np.isfinite(value).all()
    if dtype.kind == "f":
        assert dtype.itemsize == 8
        return value.astype(np.float64)
    assert dtype.kind in "biu"
    return value.astype(dtype.newbyteorder("="))


def _scale(case):
    encoded = _array(case["scale"]["value"])
    assert encoded.size == 1
    scale = float(encoded.item())
    dims = case["dims"]
    assert dims["qk_rope_head_dim"] == 0
    assert scale == dims["qk_nope_head_dim"] ** -0.5
    assert case["scale"]["qk_head_dim"] == dims["qk_nope_head_dim"]
    assert case["scale"]["kv_lora_rank"] == dims["kv_lora_rank"]
    return scale


def _inputs(case, batch):
    dims = case["dims"]
    inputs = case["inputs"]
    keys, values = mla_attention.split_kv_b(
        _array(inputs["kv_b_proj_weight"]),
        num_heads=dims["num_heads"],
        qk_nope_head_dim=dims["qk_nope_head_dim"],
        v_head_dim=dims["v_head_dim"],
    )
    return {
        "q": _array(inputs["q"])[batch].transpose(1, 0, 2).copy(),
        "latent": _array(inputs["latent"])[batch, 0],
        "w_uk": keys,
        "w_uv": values,
        "mask": _array(case["upstream"]["mask"])[batch, 0],
        "scale": _scale(case),
    }


def _compare_rows(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    _assert_vector(actual, expected)
    for observed, reference in zip(
        actual.reshape(-1, actual.shape[-1]), expected.reshape(-1, expected.shape[-1]), strict=True
    ):
        _assert_vector(observed, reference)


def test_upstream_fixture_names_source_pins_and_math_backend(fixture_data):
    provenance = fixture_data["provenance"]
    assert provenance["commit"] == PIN
    assert {Path(name).name: digest for name, digest in provenance["files_sha256"].items()} == (
        SOURCE_HASHES
    )
    assert provenance["torch"].split("+")[0] == "2.8.0"
    assert provenance["numpy"] == "2.3.3"
    assert provenance["dependencies"]["torch"] == provenance["torch"]
    assert provenance["dependencies"]["numpy"] == provenance["numpy"]
    assert provenance["attn_implementation"] == "sdpa"
    assert provenance["sdpa_backend"].startswith("MATH")
    assert provenance["dtype"] == "float64 throughout, on CPU"
    executed = " ".join(provenance["executed"])
    for name in (
        "q_b_proj",
        "expand_kv",
        "build_attention_mask_from_topk",
        "sdpa_attention_forward",
    ):
        assert name in executed
    assert any(
        "_scaled_dot_product_attention_math" in op for op in provenance["sdpa_dispatched_ops"]
    )
    assert provenance["fused_sdpa_ops_seen"] == []
    for case in fixture_data["cases"]:
        events = case["diagnostics"]["sdpa_events"]
        assert "aten::_scaled_dot_product_attention_math" in events
        assert not any(marker in op for op in events for marker in ("flash", "efficient", "cudnn"))
        assert case["diagnostics"]["output_has_nan"] is False


def test_upstream_fixture_input_projection_and_mask_provenance(fixture_data):
    for case in fixture_data["cases"]:
        dims, inputs, upstream = case["dims"], case["inputs"], case["upstream"]
        batch, tokens, count = dims["batch"], dims["q_len"], dims["kv_len"]
        heads, key_dim, value_dim = dims["num_heads"], dims["qk_nope_head_dim"], dims["v_head_dim"]
        rank, query_rank = dims["kv_lora_rank"], dims["q_lora_rank"]
        q_resid, q_weight = _array(inputs["q_resid"]), _array(inputs["q_b_proj_weight"])
        latent, weight = _array(inputs["latent"]), _array(inputs["kv_b_proj_weight"])
        assert q_resid.shape == (batch, tokens, query_rank)
        assert q_weight.shape == (heads * key_dim, query_rank)
        assert latent.shape == (batch, 1, count, rank)
        assert weight.shape == (heads * (key_dim + value_dim), rank)
        expected_q = (
            (q_resid @ q_weight.T).reshape(batch, tokens, heads, key_dim).transpose(0, 2, 1, 3)
        )
        _compare_rows(_array(inputs["q"]), expected_q)
        keys, values = _array(upstream["key_states"]), _array(upstream["value_states"])
        assert keys.shape == (batch, heads, count, key_dim)
        assert values.shape == (batch, heads, count, value_dim)
        for b in range(batch):
            for h in range(heads):
                start = h * (key_dim + value_dim)
                _compare_rows(keys[b, h], latent[b, 0] @ weight[start : start + key_dim].T)
                _compare_rows(
                    values[b, h],
                    latent[b, 0] @ weight[start + key_dim : start + key_dim + value_dim].T,
                )
        indices = _array(inputs["topk_indices"])
        assert indices.shape == (batch, tokens, dims["topk"])
        assert indices.dtype.kind == "i"
        expected_mask = np.zeros((batch, 1, tokens, count), dtype=bool)
        for b in range(batch):
            for t in range(tokens):
                selected = {int(n) for n in indices[b, t] if 0 <= n < count}
                for n in selected:
                    expected_mask[b, 0, t, n] = True
        actual_mask = _array(upstream["mask"])
        assert actual_mask.dtype == np.bool_
        np.testing.assert_array_equal(actual_mask, expected_mask)
        counts = expected_mask[:, 0].sum(axis=-1)
        assert case["rows"]["selected_counts"] == counts.tolist()
        assert case["rows"]["all_masked"] == np.argwhere(counts == 0).tolist()
        assert case["rows"]["single_key"] == np.argwhere(counts == 1).tolist()


def test_upstream_math_output_matches_core_and_small_decimal_oracle(fixture_data):
    for case in fixture_data["cases"]:
        expected = _array(case["upstream"]["output"])
        dims = case["dims"]
        assert expected.shape == (
            dims["batch"],
            dims["q_len"],
            dims["num_heads"],
            dims["v_head_dim"],
        )
        for b in range(dims["batch"]):
            inputs = _inputs(case, b)
            result = mla_attention.mla_attention_core(**inputs)
            _compare_rows(result.out, expected[b])
            if case["name"] != "glm_width":
                independent = _expanded_oracle(**inputs)
                _assert_result(result, independent)
                _compare_rows(expected[b], independent.out)
            for t in range(dims["q_len"]):
                if not inputs["mask"][t].any():
                    np.testing.assert_array_equal(expected[b, t], 0)
                    np.testing.assert_array_equal(result.out[t], 0)
                    np.testing.assert_array_equal(result.out_latent[t], 0)
                    assert np.isneginf(result.lse[t]).all()
                else:
                    assert np.isfinite(result.lse[t]).all()


def test_upstream_fixtures_cover_precision_scale_and_mask_edges(fixture_data):
    cases = {case["name"]: case for case in fixture_data["cases"]}
    if "glm_width" in cases:
        case = cases["glm_width"]
        assert case["dims"]["qk_nope_head_dim"] == 256
        assert case["dims"]["kv_lora_rank"] == 512
        assert _scale(case) == 1 / 16
        expected = _array(case["upstream"]["output"])[0]
        inputs = _inputs(case, 0) | {"scale": 512**-0.5}
        wrong = mla_attention.mla_attention_core(**inputs).out
        with pytest.raises(AssertionError):
            _compare_rows(wrong, expected)
    else:
        case = cases["precision_full_mantissa"]
        assert case["diagnostics"]["fp32_probe_max_rel_diff"] >= 1e-9
        expected = _array(case["upstream"]["output"])
        rounded = expected.astype(np.float32).astype(np.float64)
        assert math.hypot(*(rounded - expected).ravel()) / math.hypot(*expected.ravel()) >= 1e-9
        assert cases["mask_edges"]["rows"]["all_masked"]
        assert cases["mask_edges"]["rows"]["single_key"]
        case = cases["dup_invalid_indices"]
        indices = _array(case["inputs"]["topk_indices"])
        count = case["dims"]["kv_len"]
        assert (indices < 0).any()
        assert (indices >= count).any()
        rows = indices.reshape(-1, indices.shape[-1])
        assert any(
            len(set(row[(row >= 0) & (row < count)])) < ((row >= 0) & (row < count)).sum()
            for row in rows
        )
