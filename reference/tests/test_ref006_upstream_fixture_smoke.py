"""Developer smoke checks that the REF-006 upstream fixture is well formed and self-consistent.

These do not compare the reference against upstream; that is the lead's acceptance test. They
confirm the file decodes, that the recorded shapes, dtypes and configuration agree, and that the
stored evidence really describes the stored values. NumPy only: the generator's torch and
transformers live in its own isolated environment (`tools/transformers_router_fixture.py`).
"""

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "transformers_router_770e4c40.json"
COMMIT = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
SIZE_LIMIT = 2_000_000
REQUIRED = (
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

if not FIXTURE.exists():
    raise AssertionError(f"{FIXTURE} is missing; run tools/transformers_router_fixture.py")

DATA = json.loads(FIXTURE.read_text())
CASES = {case["name"]: case for case in DATA["cases"]}
case_param = pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))


def unpack(entry: dict) -> np.ndarray:
    return np.asarray(entry["data"], dtype=np.dtype(entry["dtype"])).reshape(entry["shape"])


def test_schema_and_provenance():
    assert FIXTURE.stat().st_size < SIZE_LIMIT
    assert DATA["schema_version"] == 1
    provenance = DATA["provenance"]
    assert provenance["commit"] == COMMIT
    assert set(provenance["source_sha256"]) == {
        "models/glm5_next/modeling_glm5_next.py",
        "models/glm5_next/configuration_glm5_next.py",
    }
    assert all(len(d) == 64 for d in provenance["source_sha256"].values())
    assert provenance["callable"] == "Glm5NextTextTopkRouter.forward"
    assert provenance["device"] == "cpu"
    assert provenance["generator"] == "reference/tools/transformers_router_fixture.py"
    assert provenance["dependencies"]["torch"] == "2.8.0"
    assert provenance["dependencies"]["numpy"] == "2.3.3"
    assert not any("/" in name for name in provenance["dependencies"])


def test_every_required_case_is_present():
    assert tuple(CASES) == REQUIRED


@case_param
def test_shapes_dtypes_and_config_agree(case):
    config, top_k = case["config"], case["top_k"]
    tokens = case["inputs"]["x"]["shape"][0]
    width, experts = config["hidden_size"], config["n_routed_experts"]
    assert (config["n_group"], config["topk_group"]) == (1, 1)
    assert config["norm_topk_prob"] is True and config["routed_scaling_factor"] == 2.5
    assert config["num_experts_per_tok"] == top_k
    expected = {
        "inputs.x": (tokens, width),
        "inputs.weight": (experts, width),
        "inputs.correction_bias": (experts,),
        "upstream.logits": (tokens, experts),
        "upstream.weights": (tokens, top_k),
        "upstream.ids": (tokens, top_k),
    }
    for dotted, shape in expected.items():
        section, name = dotted.split(".")
        assert tuple(case[section][name]["shape"]) == shape, dotted
    assert case["inputs"]["correction_bias"]["dtype"] == "float32"
    assert case["upstream"]["logits"]["dtype"] == "float32"
    assert case["upstream"]["weights"]["dtype"] == "float32"
    assert case["upstream"]["ids"]["dtype"] == "int64"


@case_param
def test_upstream_forward_ran_exactly_once(case):
    assert case["evidence"]["forward_calls"] == 1


@case_param
def test_routes_are_valid_and_values_finite(case):
    ids = unpack(case["upstream"]["ids"])
    experts = case["config"]["n_routed_experts"]
    assert ids.dtype == np.int64
    for row in ids:
        assert len(set(row.tolist())) == len(row)
        assert row.min() >= 0 and row.max() < experts
    for section, name in (
        ("inputs", "x"),
        ("inputs", "weight"),
        ("inputs", "correction_bias"),
        ("upstream", "logits"),
        ("upstream", "weights"),
    ):
        assert np.isfinite(unpack(case[section][name])).all(), name
    assert (unpack(case["upstream"]["weights"]) >= 0).all()


@case_param
def test_cutoff_gap_is_recorded_and_wide_enough(case):
    """The contract's 64u rule, recomputed here from the stored diagnostics."""
    diagnostics = case["diagnostics"]
    assert "recomputed" in diagnostics["note"]
    tokens = case["inputs"]["x"]["shape"][0]
    if not tokens or case["top_k"] >= case["config"]["n_routed_experts"]:
        assert diagnostics["cutoff_gap"] is None
        return
    corrected = unpack(diagnostics["corrected_scores"])
    ordered = np.sort(corrected, axis=1)[:, ::-1]
    gap = float((ordered[:, case["top_k"] - 1] - ordered[:, case["top_k"]]).min())
    limit = 64.0 * 2.0**-24 * max(1.0, float(np.abs(corrected).max()))
    assert gap == pytest.approx(diagnostics["cutoff_gap"], rel=1e-6)
    assert limit == pytest.approx(diagnostics["cutoff_gap_required_above"], rel=1e-6)
    assert gap > limit


def test_designed_boundaries_really_landed():
    assert (unpack(CASES["zero_tail"]["diagnostics"]["scores"]) == 0).all()
    assert (unpack(CASES["zero_tail"]["upstream"]["weights"]) == 0).all()
    assert (unpack(CASES["saturated"]["diagnostics"]["scores"]) == 1).all()
    subnormal = unpack(CASES["negative_subnormal"]["diagnostics"]["scores"])
    assert (subnormal > 0).all() and subnormal.max() < 1e-37
    cast = CASES["cast_sensitive"]
    assert cast["inputs"]["x"]["dtype"] == "float64"
    assert cast["inputs"]["weight"]["dtype"] == "float64"
    x, weight = unpack(cast["inputs"]["x"]), unpack(cast["inputs"]["weight"])
    in32 = x.astype(np.float32) @ weight.astype(np.float32).T
    in64 = (x @ weight.T).astype(np.float32)
    assert not np.array_equal(in32, in64)  # the case really is cast-sensitive
    np.testing.assert_array_equal(unpack(cast["upstream"]["logits"]), in32)
    assert CASES["glm_288"]["config"]["n_routed_experts"] == 288
