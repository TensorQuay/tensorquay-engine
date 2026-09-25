"""Developer checks for REF-001 beyond the lead's acceptance suite (test_mla_acceptance.py)."""

import numpy as np
import pytest

from tq_reference.mla_attention import mla_attention_core, split_kv_b

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _expanded(q, latent, w_uk, w_uv, mask, scale):
    """Expanded float64 attention: per-head keys and values materialised (another order)."""
    keys = np.einsum("nr,hdr->hnd", latent, w_uk)  # [H, N, Dn]
    values = np.einsum("nr,hvr->hnv", latent, w_uv)  # [H, N, Dv]
    scores = np.einsum("thd,hnd->thn", q, keys) * scale
    scores = np.where(mask[:, None, :], scores, -np.inf)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    probabilities = weights / weights.sum(axis=-1, keepdims=True)
    return np.einsum("thn,hnv->thv", probabilities, values)


@pytest.mark.parametrize("seed", range(3))
def test_wider_random_case_matches_expanded_form(seed):
    rng = np.random.default_rng(seed)
    tokens, heads, key_dim, value_dim, rank, keys = 4, 4, 32, 24, 64, 300
    q = rng.standard_normal((tokens, heads, key_dim))
    latent = rng.standard_normal((keys, rank))
    w_uk = rng.standard_normal((heads, key_dim, rank)) / 8
    w_uv = rng.standard_normal((heads, value_dim, rank))
    mask = rng.random((tokens, keys)) < 0.5
    result = mla_attention_core(q, latent, w_uk, w_uv, mask, scale=key_dim**-0.5)
    expected = _expanded(q, latent, w_uk, w_uv, mask, key_dim**-0.5)
    error = np.linalg.norm(result.out - expected, axis=-1) / np.linalg.norm(expected, axis=-1)
    assert error.max() <= 1e-12


def test_split_single_head_returns_copies():
    # With one head the key block is contiguous in the input, so a view would alias it.
    weight = np.arange(12, dtype=np.float64).reshape(3, 4)
    w_uk, w_uv = split_kv_b(weight, num_heads=1, qk_nope_head_dim=2, v_head_dim=1)
    assert not np.shares_memory(w_uk, weight) and not np.shares_memory(w_uv, weight)
    np.testing.assert_array_equal(w_uk[0], weight[:2])
    np.testing.assert_array_equal(w_uv[0], weight[2:])
