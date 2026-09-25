"""Developer smoke checks that the upstream MLA fixtures are well formed and self-consistent.

These do not check the reference against upstream; that is the lead's independent acceptance test.
They only confirm the files decode, that their recorded shapes agree with their dimensions, and
that the stored provenance really describes the stored values. NumPy only: the generator's torch
and transformers live in its own isolated environment (`tools/transformers_mla_fixture.py`).
"""

import base64
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_FILES = sorted(FIXTURE_DIR.glob("transformers_mla_770e4c40_*.json"))
COMMIT = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
SIZE_LIMIT = 2_000_000
MATH_EVENT = "aten::_scaled_dot_product_attention_math"
FUSED = ("flash", "efficient", "cudnn")


def unpack(entry: dict) -> np.ndarray:
    """Decode one stored array; the fixture documents exactly this recipe."""
    buffer = base64.b64decode(entry["b64"])
    return np.frombuffer(buffer, dtype=np.dtype(entry["dtype"])).reshape(entry["shape"])


def load(path: Path) -> dict:
    return json.loads(path.read_text())


if not FIXTURE_FILES:
    raise AssertionError(
        f"no upstream fixtures in {FIXTURE_DIR}; run tools/transformers_mla_fixture.py"
    )

CASES = {case["name"]: case for path in FIXTURE_FILES for case in load(path)["cases"]}
case_param = pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: p.stem)
def test_fixture_is_pinned_and_small(path):
    assert path.stat().st_size < SIZE_LIMIT
    provenance = load(path)["provenance"]
    assert provenance["commit"] == COMMIT
    assert len(provenance["files_sha256"]) == 3
    assert all(len(digest) == 64 for digest in provenance["files_sha256"].values())
    assert provenance["sdpa_backend"].startswith("MATH")
    assert provenance["fused_sdpa_ops_seen"] == []
    assert provenance["dtype"].startswith("float64")
    assert provenance["device"].startswith("cpu")
    # Positive evidence: the profiler observed the MATH kernel and no fused kernel.
    assert MATH_EVENT in provenance["sdpa_dispatched_ops"]
    assert not [op for op in provenance["sdpa_dispatched_ops"] if any(m in op for m in FUSED)]
    assert provenance["dependencies"]["torch"] == "2.8.0"
    assert provenance["dependencies"]["numpy"] == "2.3.3"


@case_param
def test_shapes_match_recorded_dimensions(case):
    dims = case["dims"]
    batch, q_len, kv_len = dims["batch"], dims["q_len"], dims["kv_len"]
    heads, rank = dims["num_heads"], dims["kv_lora_rank"]
    key_dim, value_dim = dims["qk_nope_head_dim"], dims["v_head_dim"]
    assert dims["qk_rope_head_dim"] == 0
    assert dims["qk_head_dim"] == key_dim
    expected = {
        "inputs.q_resid": (batch, q_len, dims["q_lora_rank"]),
        "inputs.q_b_proj_weight": (heads * key_dim, dims["q_lora_rank"]),
        "inputs.q": (batch, heads, q_len, key_dim),
        "inputs.latent": (batch, 1, kv_len, rank),
        "inputs.kv_b_proj_weight": (heads * (key_dim + value_dim), rank),
        "inputs.topk_indices": (batch, q_len, dims["topk"]),
        "upstream.key_states": (batch, heads, kv_len, key_dim),
        "upstream.value_states": (batch, heads, kv_len, value_dim),
        "upstream.mask": (batch, 1, q_len, kv_len),
        "upstream.output": (batch, q_len, heads, value_dim),
    }
    for dotted, shape in expected.items():
        section, name = dotted.split(".")
        assert unpack(case[section][name]).shape == shape, dotted


@case_param
def test_mask_is_the_set_of_valid_indices(case):
    """Duplicates collapse and out-of-range or negative indices drop out."""
    indices = unpack(case["inputs"]["topk_indices"])
    mask = unpack(case["upstream"]["mask"])[:, 0]
    kv_len = case["dims"]["kv_len"]
    expected = np.zeros(mask.shape, dtype=bool)
    for b, q in np.ndindex(*indices.shape[:2]):
        row = indices[b, q]
        expected[b, q, row[(row >= 0) & (row < kv_len)]] = True
    np.testing.assert_array_equal(mask, expected)


@case_param
def test_query_comes_from_the_recorded_residual_and_weight(case):
    """Both comparison arms must see the same query: q is q_resid through q_b_proj."""
    q_resid = unpack(case["inputs"]["q_resid"])
    weight = unpack(case["inputs"]["q_b_proj_weight"])
    dims = case["dims"]
    shape = (dims["batch"], dims["q_len"], dims["num_heads"], dims["qk_head_dim"])
    expected = (q_resid @ weight.T).reshape(shape).transpose(0, 2, 1, 3)
    stored = unpack(case["inputs"]["q"])
    assert np.abs(stored - expected).max() <= 1e-13 * max(1.0, np.abs(expected).max())


@case_param
def test_row_classification_matches_the_mask(case):
    mask = unpack(case["upstream"]["mask"])[:, 0]
    counts = mask.sum(axis=-1)
    np.testing.assert_array_equal(np.array(case["rows"]["selected_counts"]), counts)
    for name, wanted in (("all_masked", 0), ("single_key", 1)):
        recorded = {tuple(pair) for pair in case["rows"][name]}
        assert recorded == {tuple(pair) for pair in np.argwhere(counts == wanted)}


@case_param
def test_fully_masked_rows_are_kept_as_exact_zeros(case):
    """Upstream's _safe_softmax returns zeros, not NaN, so these rows compare directly."""
    output = unpack(case["upstream"]["output"])
    assert case["diagnostics"]["output_has_nan"] is False
    assert np.isfinite(output).all()
    for b, q in case["rows"]["all_masked"]:
        assert np.all(output[b, q] == 0.0)


def test_the_designated_precision_case_exposes_a_float32_stage():
    designated = [name for name, c in CASES.items() if c["draw"]["fp32_probe_floor"] is not None]
    assert designated == ["precision_full_mantissa"]
    case = CASES["precision_full_mantissa"]
    assert case["draw"]["require_fp32_lossy"] is True
    assert case["diagnostics"]["fp32_probe_max_rel_diff"] >= case["draw"]["fp32_probe_floor"]


def test_scale_uses_the_query_width_not_the_latent_width():
    for name, case in CASES.items():
        scale = case["scale"]
        assert scale["formula"] == "qk_head_dim ** -0.5", name
        # The scale is stored as a one-element array, so extract the element before converting.
        assert float(unpack(scale["value"]).item()) == scale["qk_head_dim"] ** -0.5, name
    glm = CASES["glm_width"]["scale"]
    assert (glm["qk_head_dim"], glm["kv_lora_rank"]) == (256, 512)
    assert float(unpack(glm["value"]).item()) == 256.0**-0.5


@case_param
def test_each_case_observed_the_math_kernel(case):
    events = case["diagnostics"]["sdpa_events"]
    assert MATH_EVENT in events
    assert not [name for name in events if any(marker in name for marker in FUSED)]
