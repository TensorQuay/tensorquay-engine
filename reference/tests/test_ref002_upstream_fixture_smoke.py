"""Developer smoke checks that the upstream MoE fixture is well formed and self-consistent.

These do not check the reference against upstream; that is the lead's independent acceptance test.
They confirm the file decodes, that the recorded shapes and routes agree with the dimensions, and
that the stored evidence really describes the stored values. NumPy only: the generator's torch and
transformers live in its own isolated environment (`tools/transformers_moe_fixture.py`).
"""

import base64
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "transformers_moe_770e4c40_experts.json"
COMMIT = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
SIZE_LIMIT = 2_000_000

if not FIXTURE.exists():
    raise AssertionError(f"{FIXTURE} is missing; run tools/transformers_moe_fixture.py")

FIXTURE_DATA = json.loads(FIXTURE.read_text())
CASES = {case["name"]: case for case in FIXTURE_DATA["cases"]}
case_param = pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))


def unpack(entry: dict) -> np.ndarray:
    """Decode one stored array; the fixture documents exactly this recipe."""
    buffer = base64.b64decode(entry["b64"])
    return np.frombuffer(buffer, dtype=np.dtype(entry["dtype"])).reshape(entry["shape"])


def test_fixture_is_pinned_and_small():
    assert FIXTURE.stat().st_size < SIZE_LIMIT
    provenance = FIXTURE_DATA["provenance"]
    assert provenance["commit"] == COMMIT
    assert len(provenance["files_sha256"]) == 3
    assert all(len(digest) == 64 for digest in provenance["files_sha256"].values())
    assert provenance["experts_implementation"] == "eager"
    assert provenance["fused_experts_ops_seen"] == []
    assert provenance["dependencies"]["torch"] == "2.8.0"
    assert provenance["dependencies"]["numpy"] == "2.3.3"


@case_param
def test_shapes_match_recorded_dimensions(case):
    dims = case["dims"]
    tokens, width = dims["num_tokens"], dims["hidden_size"]
    intermediate, experts, slots = dims["intermediate_size"], dims["num_experts"], dims["top_k"]
    expected = {
        "inputs.hidden_states": (tokens, width),
        "inputs.gate_up_proj": (experts, 2 * intermediate, width),
        "inputs.down_proj": (experts, width, intermediate),
        "inputs.top_k_index": (tokens, slots),
        "inputs.top_k_weights": (tokens, slots),
        "upstream.output": (tokens, width),
        "upstream.output_f32": (tokens, width),
    }
    for dotted, shape in expected.items():
        section, name = dotted.split(".")
        assert unpack(case[section][name]).shape == shape, dotted
    assert unpack(case["upstream"]["output"]).dtype == np.dtype(">f8")
    assert unpack(case["upstream"]["output_f32"]).dtype == np.dtype(">f4")
    assert unpack(case["inputs"]["top_k_index"]).dtype == np.dtype(">i8")


@case_param
def test_routes_satisfy_the_reference_api(case):
    """IDs must be in range and distinct per token; weights finite and nonnegative."""
    ids = unpack(case["inputs"]["top_k_index"])
    weights = unpack(case["inputs"]["top_k_weights"])
    experts = case["dims"]["num_experts"]
    assert ids.min() >= 0 and ids.max() < experts
    for row in ids:
        assert len(set(row.tolist())) == len(row)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    assert sorted({int(e) for e in ids.ravel()}) == case["rows"]["selected_experts"]
    zero = {tuple(pair) for pair in case["rows"]["zero_weight_slots"]}
    assert zero == {tuple(pair) for pair in np.argwhere(weights == 0)}


@case_param
def test_eager_execution_counts_are_recorded(case):
    counts = case["diagnostics"]["call_counts"]
    assert counts["eager_forward"] >= 1
    assert counts["apply_gate"] == len(case["rows"]["selected_experts"])


@case_param
def test_only_unselected_banks_are_poisoned(case):
    gate_up = unpack(case["inputs"]["gate_up_proj"])
    down = unpack(case["inputs"]["down_proj"])
    selected = case["rows"]["selected_experts"]
    assert np.isfinite(gate_up[selected]).all() and np.isfinite(down[selected]).all()
    assert np.isfinite(unpack(case["inputs"]["hidden_states"])).all()
    assert np.isfinite(unpack(case["upstream"]["output"])).all()
    assert case["diagnostics"]["output_has_nan"] is False


def test_the_poisoned_case_really_carries_non_finite_unselected_banks():
    case = CASES["poisoned_unselected"]
    unselected = case["rows"]["unselected_experts"]
    assert unselected
    gate_up = unpack(case["inputs"]["gate_up_proj"])
    down = unpack(case["inputs"]["down_proj"])
    for expert in unselected:
        assert not np.isfinite(gate_up[expert]).any()
        assert not np.isfinite(down[expert]).any()


@case_param
def test_recorded_fp32_probe_matches_the_stored_outputs(case):
    """Recompute the global relative L2 from the two stored outputs, with no epsilon."""
    reference = unpack(case["upstream"]["output"]).astype(np.float64)
    got = unpack(case["upstream"]["output_f32"]).astype(np.float64)
    denominator = np.linalg.norm(reference)
    assert denominator > 0
    recomputed = np.linalg.norm(got - reference) / denominator
    assert recomputed == pytest.approx(case["diagnostics"]["fp32_probe_rel_l2"], rel=1e-12)


def test_the_designated_precision_case_clears_its_floor():
    designated = [name for name, c in CASES.items() if c["draw"]["fp32_probe_floor"] is not None]
    assert designated == ["precision_sensitive"]
    case = CASES["precision_sensitive"]
    assert case["diagnostics"]["fp32_probe_rel_l2"] >= case["draw"]["fp32_probe_floor"]


def test_router_case_is_float32_throughout_and_widened_exactly():
    case = CASES["router_authentic"]
    router = case["router"]
    for name in ("router_weight", "router_bias", "router_logits", "top_k_weights_f32"):
        assert unpack(router[name]).dtype == np.dtype(">f4"), name
    assert np.isfinite(unpack(router["router_bias"])).all()
    assert unpack(router["top_k_index"]).dtype == np.dtype(">i8")
    stored_index = unpack(case["inputs"]["top_k_index"])
    np.testing.assert_array_equal(unpack(router["top_k_index"]), stored_index)
    widened = unpack(router["top_k_weights_f32"]).astype(np.float64)
    np.testing.assert_array_equal(widened, unpack(case["inputs"]["top_k_weights"]))
    assert case["dims"]["num_experts"] == 12 and case["dims"]["top_k"] == 8


def test_clamp_boundaries_case_straddles_the_limit():
    case = CASES["clamp_boundaries"]
    limit = case["dims"]["swiglu_limit"]
    intermediate = case["dims"]["intermediate_size"]
    rows = unpack(case["inputs"]["gate_up_proj"])[0, :, 0]
    gate, up = rows[:intermediate], rows[intermediate:]
    for half in (gate, up):
        assert (half < -limit).any() and (half > limit).any()
        assert (np.abs(half) == limit).any()
