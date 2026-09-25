"""Lead-owned REF-006 qualification against data from the actual pinned upstream router."""

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_router_acceptance import (
    U,
    assert_result,
    assert_weights,
    logit_case,
    scalar_oracle,
    scalar_sigmoid,
)

FIXTURE = Path(__file__).parent / "fixtures" / "transformers_router_770e4c40.json"
COMMIT = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
SOURCE_SHA256 = {
    "models/glm5_next/modeling_glm5_next.py": (
        "431eae74b857ec2a8e38a96f9f76cdc0b366bf7e35bd739b071c35761f113888"
    ),
    "models/glm5_next/configuration_glm5_next.py": (
        "f12b5876701d000f930cd8102124e07d05675a79a5591db29c18244f5f025dec"
    ),
}
CASES = (
    "dense_dyadic",
    "glm_288",
    "bias_selection",
    "cast_sensitive",
    "epsilon_dominated",
    "zero_tail",
    "saturated",
    "rounding_discrepancy",
    "negative_subnormal",
    "empty_tokens",
)


@pytest.fixture(scope="module")
def fixture():
    assert FIXTURE.is_file(), "Upstream fixture required; no skip on missing evidence"
    assert FIXTURE.stat().st_size < 2_000_000
    return json.loads(FIXTURE.read_text())


def tensor(encoded):
    assert set(encoded) == {"dtype", "shape", "data"}
    assert encoded["dtype"] in {"float32", "float64", "int64"}
    shape = encoded["shape"]
    assert all(type(dim) is int and dim >= 0 for dim in shape)
    data = encoded["data"]
    assert len(data) == math.prod(shape)
    assert all(type(v) in (float, int) for v in data)
    if encoded["dtype"] == "int64":
        assert all(type(v) is int for v in data)
    result = np.array(data, dtype=encoded["dtype"]).reshape(shape)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(
        result.ravel().astype(np.float64), np.array(data, dtype=np.float64)
    )
    return result


def case_by_name(fixture, name):
    found = [c for c in fixture["cases"] if c["name"] == name]
    assert len(found) == 1
    return found[0]


def inputs(case):
    assert set(case["inputs"]) == {"x", "weight", "correction_bias"}
    return {**{key: tensor(value) for key, value in case["inputs"].items()}, "top_k": case["top_k"]}


def upstream(case):
    result = case["upstream"]
    assert set(result) == {"logits", "weights", "ids"}
    logits, weights, ids = (tensor(result[name]) for name in ("logits", "weights", "ids"))
    assert logits.dtype == weights.dtype == np.float32
    assert ids.dtype == np.int64
    assert weights.shape == ids.shape == (logits.shape[0], case["top_k"])
    assert np.all((ids >= 0) & (ids < logits.shape[1]))
    assert all(len(set(row)) == len(row) for row in ids.tolist())
    order = np.argsort(ids, axis=-1, kind="stable")
    return logits, np.take_along_axis(weights, order, -1), np.take_along_axis(ids, order, -1)


def test_fixture_provenance_and_complete_case_inventory(fixture):
    assert fixture["schema_version"] == 1
    assert sorted(c["name"] for c in fixture["cases"]) == sorted(CASES)
    p = fixture["provenance"]
    assert p["commit"] == COMMIT
    assert p["source_sha256"] == SOURCE_SHA256
    assert p["torch"] == "2.8.0"
    assert p["numpy"] == "2.3.3"
    assert p["transformers"] == "5.18.0.dev0"
    assert p["python"].startswith("3.12.")
    assert p["device"] == "cpu"
    assert p["platform"] and p["machine"]
    assert p["generator"] == "reference/tools/transformers_router_fixture.py"
    assert p["callable"] == "Glm5NextTextTopkRouter.forward"
    assert p["input_recipe"] and p["claim"]
    for package in ("torch", "numpy", "transformers"):
        assert p["dependencies"][package] == p[package]


@pytest.mark.parametrize("name", CASES)
def test_actual_upstream_and_our_router_both_meet_scalar_oracle(fixture, name):
    case = case_by_name(fixture, name)
    supplied = inputs(case)
    tokens, width = supplied["x"].shape
    experts = supplied["weight"].shape[0]
    assert case["config"] == {
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "hidden_size": width,
        "n_routed_experts": experts,
        "num_experts_per_tok": case["top_k"],
    }
    assert case["evidence"]["forward_calls"] == 1
    expected = scalar_oracle(**supplied)
    actual_upstream = upstream(case)
    assert actual_upstream[0].shape == (tokens, experts)
    assert_result(actual_upstream, expected)
    router = importlib.import_module("tq_reference.router")
    actual = router.route(**supplied)
    assert_result(actual, expected)
    assert_result(actual, actual_upstream)


@pytest.mark.parametrize("name", CASES)
def test_separated_selection_boundary_is_real(fixture, name):
    case = case_by_name(fixture, name)
    logits, _, _ = upstream(case)
    bias = inputs(case)["correction_bias"]
    k = case["top_k"]
    for row in logits:
        choice = [float(np.float32(scalar_sigmoid(z) + b)) for z, b in zip(row, bias, strict=True)]
        if k < len(choice):
            ranked = sorted(choice, reverse=True)
            gap = ranked[k - 1] - ranked[k]
            assert gap > 64 * U * max(1, max(abs(v) for v in choice))


def test_288_expert_fixture_uses_top8_and_reaches_high_ids(fixture):
    case = case_by_name(fixture, "glm_288")
    logits, _, ids = upstream(case)
    assert logits.shape[1] == 288
    assert ids.shape[1] == 8
    assert np.max(ids) >= 256


def test_bias_fixture_actually_changes_the_selected_set(fixture):
    case = case_by_name(fixture, "bias_selection")
    supplied = inputs(case)
    _, _, biased_ids = upstream(case)
    supplied["correction_bias"] = np.zeros_like(supplied["correction_bias"])
    _, _, unbiased_ids = scalar_oracle(**supplied)
    assert not np.array_equal(biased_ids, unbiased_ids)


def test_cast_fixture_exposes_a_float64_projection_mistake(fixture):
    case = case_by_name(fixture, "cast_sensitive")
    supplied = inputs(case)
    assert supplied["x"].dtype == supplied["weight"].dtype == np.float64
    logits, _, ids = upstream(case)
    wrong_logits = supplied["x"] @ supplied["weight"].T
    assert not np.array_equal(logits, wrong_logits.astype(np.float32))
    assert logits.shape[0] == 1
    wrong = logit_case(wrong_logits[0], bias=supplied["correction_bias"], k=case["top_k"])
    assert not np.array_equal(ids, scalar_oracle(**wrong)[2])


def test_epsilon_fixture_exposes_missing_epsilon(fixture):
    case = case_by_name(fixture, "epsilon_dominated")
    logits, weights, ids = upstream(case)
    for row in weights:
        assert 0 < math.fsum(row) < 2.25
    wrong = np.zeros_like(weights)
    for t, row in enumerate(logits):
        selected = [scalar_sigmoid(row[e]) for e in ids[t]]
        total = math.fsum(selected)
        wrong[t] = [float(s) * 2.5 / total for s in selected]
    with pytest.raises(AssertionError):
        assert_weights(wrong, weights, case["top_k"])


def test_zero_and_saturated_fixtures_have_analytical_outputs(fixture):
    for name, expected in [("zero_tail", 0), ("saturated", 0.3125)]:
        _, weights, _ = upstream(case_by_name(fixture, name))
        np.testing.assert_array_equal(weights, np.full_like(weights, expected))


def test_negative_subnormal_fixture_really_exercises_subnormal_scores(fixture):
    logits, weights, ids = upstream(case_by_name(fixture, "negative_subnormal"))
    for t, row in enumerate(logits):
        selected = [scalar_sigmoid(row[e]) for e in ids[t]]
        assert all(0 < s < np.finfo(np.float32).tiny for s in selected)
    assert np.all(weights > 0)


def test_recorded_rounding_discrepancy_exposes_the_old_gate(fixture):
    case = case_by_name(fixture, "rounding_discrepancy")
    logits, weights, ids = upstream(case)
    np.testing.assert_array_equal(
        logits, [[-10.91289234161377, -9.825783729553223, -5.059233665466309, -4, -3, -2, -1, 0]]
    )
    np.testing.assert_array_equal(ids, [np.arange(8)])
    # Captured NumPy FP32 arm of the pre-implementation probe, on the recorded host.
    # Freeze that evidence so a different CPU's exp implementation cannot erase the example.
    other_fp32 = np.array(
        [
            [
                4.7454734158236533e-05,
                0.00014073083002585918,
                0.01643425039947033,
                0.0468420647084713,
                0.12351273000240326,
                0.3104439973831177,
                0.7004128694534302,
                1.3021661043167114,
            ]
        ],
        dtype=np.float32,
    )
    relative = math.hypot(*(other_fp32.astype(float) - weights.astype(float)).ravel()) / math.hypot(
        *weights.ravel()
    )
    assert relative > 1e-12
    assert_weights(other_fp32, weights, 8)


def test_empty_fixture_is_an_actual_empty_upstream_call(fixture):
    case = case_by_name(fixture, "empty_tokens")
    assert inputs(case)["x"].shape[0] == 0
    for value in upstream(case):
        assert value.shape[0] == 0
    assert case["evidence"]["forward_calls"] == 1
