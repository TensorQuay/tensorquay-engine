"""Developer checks for REF-002 beyond the lead's acceptance suite (test_moe_acceptance.py)."""

import math

import numpy as np
import pytest

from tq_reference.moe_experts import moe_experts

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _dense(x, w13, w2, topk_ids, topk_weights, swiglu_limit):
    """Every expert applied to every token, then gathered: a different order of the same algebra."""
    intermediate = w2.shape[-1]
    gate = np.einsum("th,eih->tei", x, w13[:, :intermediate])
    up = np.einsum("th,eih->tei", x, w13[:, intermediate:])
    gate = np.minimum(gate, swiglu_limit)
    up = np.clip(up, -swiglu_limit, swiglu_limit)
    hidden = (gate / (1.0 + np.exp(-gate))) * up
    per_expert = np.einsum("ehi,tei->teh", w2, hidden)
    out = np.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(topk_ids.shape[1]):
            out[token] += topk_weights[token, slot] * per_expert[token, topk_ids[token, slot]]
    return out


@pytest.mark.parametrize("seed", range(3))
def test_wider_random_case_matches_dense_form(seed):
    rng = np.random.default_rng(seed)
    tokens, width, intermediate, experts, slots = 6, 12, 9, 10, 4
    x = rng.standard_normal((tokens, width))
    w13 = rng.standard_normal((experts, 2 * intermediate, width)) / 4
    w2 = rng.standard_normal((experts, width, intermediate)) / 4
    ids = np.stack([rng.permutation(experts)[:slots] for _ in range(tokens)]).astype(np.int64)
    weights = rng.random((tokens, slots))
    result = moe_experts(x, w13, w2, ids, weights, swiglu_limit=10.0)
    expected = _dense(x, w13, w2, ids, weights, 10.0)
    error = np.linalg.norm(result - expected, axis=-1) / np.linalg.norm(expected, axis=-1)
    assert error.max() <= 1e-12


@pytest.mark.parametrize("gate", [-800.0, -745.2, -720.0, -100.0, -1.0, 0.0, 1.0, 100.0, 800.0])
def test_silu_tail_is_stable_and_correctly_signed(gate):
    """A scalar SiLU with no exponential of a positive argument, compared against math.exp."""
    case = {
        "x": np.ones((1, 1)),
        "w13": np.array([[[gate], [1.0]]]),
        "w2": np.ones((1, 1, 1)),
        "topk_ids": np.zeros((1, 1), dtype=np.int64),
        "topk_weights": np.ones((1, 1)),
        "swiglu_limit": 1e9,
    }
    actual = moe_experts(**case).item()
    decay = math.exp(-abs(gate))
    expected = gate * decay / (1.0 + decay) if gate < 0 else gate / (1.0 + decay)
    assert math.isfinite(actual)
    # The deep negative tail may underflow to zero, which the contract allows; it never flips sign.
    assert actual <= 0 if gate < 0 else actual >= 0
    assert abs(actual - expected) <= 1e-12 * max(1.0, abs(expected)) + 801 * 5e-324
