"""Independent REF-002 checks of upstream execution evidence, inputs and numerical agreement."""

import base64
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_moe_acceptance import _assert_close, _oracle

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
PIN = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
SOURCE_HASHES = {
    "modeling_glm5_next.py": "431eae74b857ec2a8e38a96f9f76cdc0b366bf7e35bd739b071c35761f113888",
    "configuration_glm5_next.py": (
        "f12b5876701d000f930cd8102124e07d05675a79a5591db29c18244f5f025dec"
    ),
    "moe.py": "706047c3850f41fd298c21df6fdd79b639c385351f1c12422c63b8ef3733f423",
}
CASE_NAMES = (
    "router_authentic",
    "nonuniform_routes",
    "zero_weights",
    "clamp_boundaries",
    "poisoned_unselected",
    "precision_sensitive",
)


@pytest.fixture(scope="module")
def fixture_data():
    path = Path(__file__).parent / "fixtures" / "transformers_moe_770e4c40_experts.json"
    assert path.stat().st_size < 2_000_000
    data = json.loads(path.read_text())
    assert len(data["cases"]) == len(CASE_NAMES)
    assert {case["name"] for case in data["cases"]} == set(CASE_NAMES)
    return data


@pytest.fixture(params=CASE_NAMES)
def case(request, fixture_data):
    return next(case for case in fixture_data["cases"] if case["name"] == request.param)


def _array(record, *, finite=True):
    dtype = np.dtype(record["dtype"])
    shape = tuple(record["shape"])
    assert all(isinstance(n, int) and n > 0 for n in shape)
    assert dtype.kind in "fi" and dtype.itemsize in (4, 8)
    raw = base64.b64decode(record["b64"], validate=True)
    assert len(raw) == math.prod(shape) * dtype.itemsize
    value = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(dtype.newbyteorder("="))
    if finite:
        assert np.isfinite(value).all()
    return value


def _inputs(case):
    inputs = case["inputs"]
    return {
        "x": _array(inputs["hidden_states"]),
        "w13": _array(inputs["gate_up_proj"], finite=False),
        "w2": _array(inputs["down_proj"], finite=False),
        "topk_ids": _array(inputs["top_k_index"]),
        "topk_weights": _array(inputs["top_k_weights"]),
        "swiglu_limit": case["dims"]["swiglu_limit"],
    }


def test_source_pins_eager_dispatch_and_versions(fixture_data):
    provenance = fixture_data["provenance"]
    assert provenance["commit"] == PIN
    assert {Path(name).name: digest for name, digest in provenance["files_sha256"].items()} == (
        SOURCE_HASHES
    )
    assert provenance["torch"].split("+")[0] == "2.8.0"
    assert provenance["numpy"] == "2.3.3"
    assert provenance["dependencies"]["torch"] == provenance["torch"]
    assert provenance["dependencies"]["numpy"] == provenance["numpy"]
    assert provenance["experts_implementation"] == "eager"
    assert provenance["apply_gate_owner"] == "Glm5NextTextExperts"
    assert set(provenance["experts_registered_keys"]) == {
        "deepgemm",
        "batched_mm",
        "grouped_mm",
        "sonicmoe",
    }
    assert provenance["fused_experts_ops_seen"] == []


def test_case_inputs_and_observed_call_counts(case):
    dims = case["dims"]
    tokens, width = dims["num_tokens"], dims["hidden_size"]
    intermediate, experts, topk = dims["intermediate_size"], dims["num_experts"], dims["top_k"]
    inputs = _inputs(case)
    assert inputs["x"].shape == (tokens, width)
    assert inputs["w13"].shape == (experts, 2 * intermediate, width)
    assert inputs["w2"].shape == (experts, width, intermediate)
    assert inputs["topk_ids"].shape == inputs["topk_weights"].shape == (tokens, topk)
    for name in ("x", "w13", "w2", "topk_weights"):
        assert inputs[name].dtype == np.float64
    ids = inputs["topk_ids"]
    assert ids.dtype == np.int64
    assert ((ids >= 0) & (ids < experts)).all()
    assert all(len(set(row)) == topk for row in ids)
    assert (inputs["topk_weights"] >= 0).all()
    selected = set(ids.ravel())
    for expert in selected:
        assert np.isfinite(inputs["w13"][expert]).all()
        assert np.isfinite(inputs["w2"][expert]).all()
    if case["name"] != "poisoned_unselected":
        assert np.isfinite(inputs["w13"]).all() and np.isfinite(inputs["w2"]).all()
    calls = case["diagnostics"]["call_counts"]
    assert calls["eager_forward"] == 1
    assert calls["apply_gate"] == len(selected)
    assert case["diagnostics"]["output_has_nan"] is False


def test_upstream_and_reference_match_independent_decimal(case):
    inputs = _inputs(case)
    upstream = _array(case["upstream"]["output"])
    assert upstream.dtype == np.float64
    expected = _oracle(**inputs)
    _assert_close(upstream, expected)
    module = importlib.import_module("tq_reference.moe_experts")
    actual = module.moe_experts(**inputs)
    _assert_close(actual, expected)
    _assert_close(actual, upstream)


def test_fp32_probe_recomputes_from_actual_stored_outputs(case):
    reference = _array(case["upstream"]["output"])
    probe = _array(case["upstream"]["output_f32"])
    assert probe.dtype == np.float32 and probe.shape == reference.shape
    denominator = math.hypot(*reference.ravel())
    assert denominator > 0
    error = math.hypot(*(probe.astype(np.float64) - reference).ravel()) / denominator
    assert case["diagnostics"]["fp32_probe_rel_l2"] == pytest.approx(error, rel=1e-12, abs=0)
    if case["name"] == "precision_sensitive":
        assert error >= 1e-9
        for key in ("hidden_states", "gate_up_proj", "down_proj"):
            original = _array(case["inputs"][key])
            assert (original.astype(np.float32).astype(np.float64) != original).all()


def test_actual_router_outputs_cross_the_precision_boundary_exactly(fixture_data):
    case = next(c for c in fixture_data["cases"] if c["name"] == "router_authentic")
    assert case["dims"]["num_experts"] == 12 and case["dims"]["top_k"] == 8
    router = case["router"]
    for name in ("router_weight", "router_bias", "router_logits", "top_k_weights_f32"):
        assert _array(router[name]).dtype == np.float32
    inputs = _inputs(case)
    raw_weights = _array(router["top_k_weights_f32"])
    np.testing.assert_array_equal(raw_weights.astype(np.float64), inputs["topk_weights"])
    np.testing.assert_array_equal(_array(router["top_k_index"]), inputs["topk_ids"])
    assert np.unique(raw_weights).size > 3
    assert np.all(np.abs(inputs["topk_weights"].sum(axis=1) - 2.5) <= 1e-6)
    for other in fixture_data["cases"]:
        if other["name"] != "router_authentic":
            assert other["router"] is None


def test_clamp_fixture_actually_crosses_both_boundaries(fixture_data):
    case = next(c for c in fixture_data["cases"] if c["name"] == "clamp_boundaries")
    inputs = _inputs(case)
    gates, ups = [], []
    intermediate = inputs["w2"].shape[-1]
    for token, ids in enumerate(inputs["topk_ids"]):
        for expert in ids:
            projected = inputs["w13"][expert] @ inputs["x"][token]
            gates.extend(projected[:intermediate])
            ups.extend(projected[intermediate:])
    limit = inputs["swiglu_limit"]
    for values in (np.array(gates), np.array(ups)):
        assert (values > limit).any() and (values < -limit).any()
        assert (values == limit).any() and (values == -limit).any()


def test_zero_weights_and_poisoned_unselected_banks_are_present(fixture_data):
    cases = {c["name"]: c for c in fixture_data["cases"]}
    zeros = _inputs(cases["zero_weights"])
    assert (zeros["topk_weights"] == 0).any()
    all_zero = (zeros["topk_weights"] == 0).all(axis=1)
    assert all_zero.any()
    output = _array(cases["zero_weights"]["upstream"]["output"])
    np.testing.assert_array_equal(output[all_zero], 0)
    poisoned = _inputs(cases["poisoned_unselected"])
    unused = set(range(poisoned["w13"].shape[0])) - set(poisoned["topk_ids"].ravel())
    assert unused
    assert any(not np.isfinite(poisoned["w13"][e]).all() for e in unused)
    assert any(not np.isfinite(poisoned["w2"][e]).all() for e in unused)
