"""REF-001 float64 reference for the NoPE multi-head latent attention (MLA) core.

Contract: ``reference/contracts/mla-attention.md``. One sequence per call:

- ``q``: ``[T, H, Dn]``, the query after its expansion projection;
- ``latent``: ``[N, R]``, the cached latent rows after normalization;
- ``w_uk``: ``[H, Dn, R]`` and ``w_uv``: ``[H, Dv, R]``, the per-head key and value expansions;
- ``mask``: bool ``[T, N]``, the keys each query may attend to, shared by all heads.

The key expansion is absorbed into the query, so attention runs over latent rows. For each query
``t`` and head ``h`` that select at least one row:

    q_abs = w_uk[h].T @ q[t, h]
    s[n] = (q_abs . latent[n]) * scale            for selected rows n only
    m = max(s);  w = exp(s - m);  p = w / sum(w)
    out_latent = sum_n p[n] * latent[n];  out = w_uv[h] @ out_latent;  lse = m + log(sum(w))

Probabilities come from the shifted weights, never from ``exp(s - lse)``: when a large common score
offset makes ``lse`` round to ``m``, that form loses normalization. A query with no selected row
returns zeros and ``lse = -inf`` without any arithmetic. ``scale`` is always supplied by the caller,
never derived from a dimension.
"""

import operator
from dataclasses import dataclass

import numpy as np


class MlaError(ValueError):
    """Invalid input to, or arithmetic overflow in, the REF-001 reference."""


@dataclass(frozen=True)
class MlaCoreResult:
    out: np.ndarray  # float64 [T, H, Dv]
    out_latent: np.ndarray  # float64 [T, H, R]
    lse: np.ndarray  # float64 [T, H]


def _require_array(name: str, value, dtype, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise MlaError(f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype != dtype:
        raise MlaError(f"{name} must have dtype {np.dtype(dtype).name}, got {value.dtype}")
    if value.ndim != ndim:
        raise MlaError(f"{name} must have {ndim} dimensions, got shape {value.shape}")
    if 0 in value.shape:
        raise MlaError(f"{name} has an empty dimension: shape {value.shape}")
    return value


def _require_finite_input(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise MlaError(f"{name} contains non-finite values")


def _require_finite_result(stage: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise MlaError(f"overflow in {stage}")


def _require_dimension(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MlaError(f"{name} must be an integer, got {type(value).__name__}")
    size = operator.index(value)
    if size <= 0:
        # The value itself is caller-supplied and unbounded; formatting it can raise on huge ints.
        raise MlaError(f"{name} must be > 0")
    return size


def _require_scale(scale) -> float:
    if isinstance(scale, bool) or not isinstance(scale, (int, float, np.integer, np.floating)):
        raise MlaError(f"scale must be a real scalar, got {type(scale).__name__}")
    try:
        with np.errstate(all="ignore"):
            value = float(scale)
    except OverflowError:
        value = np.inf
    if not (np.isfinite(value) and value > 0):
        # Report the converted float64, never the caller's object: an unbounded int cannot format.
        raise MlaError(f"scale must be finite and > 0 after float64 conversion, got {value!r}")
    return value


def split_kv_b(w_kv_b, *, num_heads, qk_nope_head_dim, v_head_dim) -> tuple[np.ndarray, np.ndarray]:
    """Split a packed ``[H * (Dn + Dv), R]`` key/value expansion into copies ``(w_uk, w_uv)``.

    Rows are interleaved per head: each head's ``Dn`` key rows, then its ``Dv`` value rows.
    """
    weight = _require_array("w_kv_b", w_kv_b, np.float64, 2)
    _require_finite_input("w_kv_b", weight)
    heads = _require_dimension("num_heads", num_heads)
    key_dim = _require_dimension("qk_nope_head_dim", qk_nope_head_dim)
    value_dim = _require_dimension("v_head_dim", v_head_dim)
    rows = heads * (key_dim + value_dim)
    if weight.shape[0] != rows:
        # ``rows`` derives from caller dimensions and is unbounded; only the real row count is safe.
        raise MlaError(
            "w_kv_b rows must equal num_heads * (qk_nope_head_dim + v_head_dim), "
            f"got {weight.shape[0]}"
        )
    per_head = weight.reshape(heads, key_dim + value_dim, weight.shape[1])
    w_uk = np.array(per_head[:, :key_dim, :], dtype=np.float64, order="C", copy=True)
    w_uv = np.array(per_head[:, key_dim:, :], dtype=np.float64, order="C", copy=True)
    return w_uk, w_uv


def _validate_core(q, latent, w_uk, w_uv, mask) -> None:
    arrays = {
        "q": _require_array("q", q, np.float64, 3),
        "latent": _require_array("latent", latent, np.float64, 2),
        "w_uk": _require_array("w_uk", w_uk, np.float64, 3),
        "w_uv": _require_array("w_uv", w_uv, np.float64, 3),
        "mask": _require_array("mask", mask, np.bool_, 2),
    }
    tokens, heads, key_dim = q.shape
    keys, rank = latent.shape
    expected = {
        "w_uk": (heads, key_dim, rank),
        "w_uv": (heads, w_uv.shape[1], rank),
        "mask": (tokens, keys),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise MlaError(
                f"shape mismatch: {name} has shape {arrays[name].shape}, expected {shape} from "
                f"q {q.shape} and latent {latent.shape}"
            )
    for name in ("q", "latent", "w_uk", "w_uv"):
        _require_finite_input(name, arrays[name])


def mla_attention_core(q, latent, w_uk, w_uv, mask, *, scale) -> MlaCoreResult:
    """Absorbed NoPE latent attention for one sequence; see the module docstring and contract."""
    _validate_core(q, latent, w_uk, w_uv, mask)
    scale = _require_scale(scale)
    tokens, heads, _ = q.shape
    value_dim, rank = w_uv.shape[1:]
    out = np.zeros((tokens, heads, value_dim), dtype=np.float64)
    out_latent = np.zeros((tokens, heads, rank), dtype=np.float64)
    lse = np.full((tokens, heads), -np.inf, dtype=np.float64)
    with np.errstate(all="ignore"):
        for t in range(tokens):
            selected = np.flatnonzero(mask[t])
            if selected.size == 0:
                continue  # all-masked query: zero outputs and lse = -inf, no arithmetic
            rows = latent[selected]  # [K, R], selected rows only
            q_abs = np.einsum("hd,hdr->hr", q[t], w_uk)  # [H, R]
            _require_finite_result("absorbed query", q_abs)
            scores = (q_abs @ rows.T) * scale  # [H, K]: dot product first, then the scale
            _require_finite_result("score", scores)
            maximum = scores.max(axis=1, keepdims=True)
            shifted = scores - maximum
            _require_finite_result("shifted score", shifted)
            weights = np.exp(shifted)  # underflow to zero is valid; the maximum contributes 1
            total = weights.sum(axis=1, keepdims=True)
            probabilities = weights / total
            latent_t = probabilities @ rows  # [H, R]
            _require_finite_result("latent output", latent_t)
            out_t = np.einsum("hvr,hr->hv", w_uv, latent_t)  # [H, Dv]
            _require_finite_result("output", out_t)
            lse_t = maximum[:, 0] + np.log(total[:, 0])
            _require_finite_result("lse", lse_t)
            out_latent[t] = latent_t
            out[t] = out_t
            lse[t] = lse_t
    return MlaCoreResult(out=out, out_latent=out_latent, lse=lse)
