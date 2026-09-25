"""Developer checks for REF-006 beyond the lead's acceptance suite (test_router_acceptance.py)."""

import numpy as np
import pytest

from tq_reference.router import route

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _inputs(tokens=5, width=4, experts=20, seed=0):
    """Dyadic values, so the float32 projection is exact and no tie can appear by accident."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-8, 9, size=(tokens, width)).astype(np.float64) / 8
    weight = rng.integers(-15, 16, size=(experts, width)).astype(np.float64) / 16
    bias = (rng.permutation(experts) / 64.0).astype(np.float32)
    return x, weight, bias


@pytest.mark.parametrize("seed", range(4))
def test_relabelling_experts_permutes_selection_but_not_the_rounding(seed):
    """Logits and the selected set permute exactly; the weights only agree to fp32 rounding.

    The contract fixes the denominator as a float32 left-to-right sum in increasing expert-ID
    order, so that sum is deliberately label-dependent: relabelling the experts reorders the
    additions and moves the last bits. Selection itself is unaffected.
    """
    top_k = 6
    x, weight, bias = _inputs(seed=seed)
    logits, weights, ids = route(x, weight, bias, top_k=top_k)
    order = np.random.default_rng(seed + 100).permutation(weight.shape[0])
    moved_logits, moved_weights, moved_ids = route(x, weight[order], bias[order], top_k=top_k)

    np.testing.assert_array_equal(moved_logits, logits[:, order])
    # Map the permuted selection back to original expert IDs, then compare pair by pair.
    recovered = order[moved_ids]
    back = np.argsort(recovered, axis=1)
    np.testing.assert_array_equal(np.take_along_axis(recovered, back, 1), ids)
    paired = np.take_along_axis(moved_weights, back, 1)
    counted = 2 * top_k + 16
    budget = counted * 2.0**-24 / (1 - counted * 2.0**-24)
    assert np.abs(paired - weights).max() <= budget * np.abs(weights).max()


@pytest.mark.parametrize("seed", range(4))
def test_selection_is_the_true_top_k_of_the_corrected_scores(seed):
    """Independent check of the selection rule via a full sort, not a partial one."""
    x, weight, bias = _inputs(seed=seed)
    top_k = 6
    logits, weights, ids = route(x, weight, bias, top_k=top_k)
    with np.errstate(over="ignore", under="ignore"):
        corrected = np.float32(1) / (np.float32(1) + np.exp(-logits)) + bias
    for token in range(x.shape[0]):
        ranked = sorted(range(weight.shape[0]), key=lambda e: (-corrected[token, e], e))
        np.testing.assert_array_equal(ids[token], sorted(ranked[:top_k]))
        assert corrected[token, ids[token]].min() > corrected[token, ranked[top_k]] - 1e-7
    # Well-scaled scores normalise to the scaling factor; the epsilon is negligible here.
    assert np.allclose(weights.sum(axis=1), 2.5, rtol=1e-6, atol=0)


def test_weights_ignore_the_bias_even_when_it_reorders_everything():
    """A bias large enough to invert the ranking must not appear in any returned weight."""
    x, weight, _ = _inputs(experts=10)
    rising = (np.arange(10) / 8.0).astype(np.float32)
    _, plain, ids_plain = route(x, weight, np.zeros(10, dtype=np.float32), top_k=10)
    _, biased, ids_biased = route(x, weight, rising, top_k=10)
    np.testing.assert_array_equal(ids_plain, ids_biased)  # top_k == E selects everything
    np.testing.assert_array_equal(plain, biased)
