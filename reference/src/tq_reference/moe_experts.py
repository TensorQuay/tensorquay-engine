"""REF-002 float64 reference for the routed MoE expert path.

Contract: ``reference/contracts/moe-experts.md``. One batch of tokens per call:

- ``x``: ``[T, H]``, token activations;
- ``w13``: ``[E, 2*I, H]``, per expert all gate rows followed by all up rows;
- ``w2``: ``[E, H, I]``, the output projection;
- ``topk_ids``: int32/int64 ``[T, K]``, distinct expert IDs per token, each in ``[0, E)``;
- ``topk_weights``: float64 ``[T, K]``, the finite nonnegative weight of each ID.

For each token ``t`` and selected expert ``e`` with weight ``a``:

    gate = w13[e, :I] @ x[t];  up = w13[e, I:] @ x[t]
    g = min(gate, swiglu_limit)                   # upper clamp only
    u = clip(up, -swiglu_limit, swiglu_limit)     # symmetric
    out[t] += a * (w2[e] @ (SiLU(g) * u))

Contributions are summed in increasing expert-ID order, so the result does not depend on the
order of the slots. Weights are used exactly as supplied: this core never normalizes them,
applies a second scaling factor, selects experts or reorders weights apart from their IDs.

``SiLU(z) = z / (1 + exp(-z))`` is evaluated as ``z * exp(-|z|) / (1 + exp(-|z|))`` for negative
``z``, so the exponential argument is never positive and the deep negative tail underflows to zero
instead of overflowing. Only selected experts are inspected, so an unselected expert may hold NaN
or infinity; a selected one must be finite even where its weight or input is zero.
"""

import numpy as np

FLOAT64 = (np.dtype(np.float64),)
INTEGERS = (np.dtype(np.int32), np.dtype(np.int64))


class MoeError(ValueError):
    """Invalid input to, or arithmetic overflow in, the REF-002 reference."""


def _require_array(name: str, value, ndim: int, dtypes: tuple, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise MoeError(f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype not in dtypes:
        raise MoeError(f"{name} must have dtype {label}, got {value.dtype}")
    if value.ndim != ndim:
        raise MoeError(f"{name} must have {ndim} dimensions, got shape {value.shape}")
    return value


def _require_finite_input(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise MoeError(f"{name} contains non-finite values")


def _require_finite_result(stage: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise MoeError(f"overflow in {stage}")


def _require_limit(swiglu_limit) -> float:
    if isinstance(swiglu_limit, bool) or not isinstance(
        swiglu_limit, (int, float, np.integer, np.floating)
    ):
        raise MoeError(f"swiglu_limit must be a real scalar, got {type(swiglu_limit).__name__}")
    try:
        value = float(swiglu_limit)
    except OverflowError:
        value = np.inf
    if not (np.isfinite(value) and value > 0):
        # Report the converted float64, never the caller's object: an unbounded int cannot format.
        raise MoeError(f"swiglu_limit must be finite and > 0, got {value!r}")
    return value


def _validate(x, w13, w2, topk_ids, topk_weights) -> tuple[int, int, int]:
    """Check every supplied input and return ``(tokens, width, intermediate)``."""
    _require_array("x", x, 2, FLOAT64, "float64")
    _require_array("w13", w13, 3, FLOAT64, "float64")
    _require_array("w2", w2, 3, FLOAT64, "float64")
    _require_array("topk_ids", topk_ids, 2, INTEGERS, "int32 or int64")
    _require_array("topk_weights", topk_weights, 2, FLOAT64, "float64")

    # A token dimension of zero is allowed; every other dimension must be non-empty.
    for name, value, sizes in (
        ("x", x, x.shape[1:]),
        ("w13", w13, w13.shape),
        ("w2", w2, w2.shape),
        ("topk_ids", topk_ids, topk_ids.shape[1:]),
        ("topk_weights", topk_weights, topk_weights.shape[1:]),
    ):
        if 0 in sizes:
            raise MoeError(f"{name} has an empty dimension: shape {value.shape}")

    tokens, width = x.shape
    experts, rows = w13.shape[:2]
    intermediate = rows // 2  # an odd row count cannot match 2 * intermediate below
    slots = topk_ids.shape[1]
    for name, got, want in (
        ("w13", w13.shape, (experts, 2 * intermediate, width)),
        ("w2", w2.shape, (experts, width, intermediate)),
        ("topk_ids", topk_ids.shape, (tokens, slots)),
        ("topk_weights", topk_weights.shape, (tokens, slots)),
    ):
        if got != want:
            raise MoeError(
                f"shape mismatch: {name} has shape {got}, expected {want} from x {x.shape} "
                f"and w13 {w13.shape}"
            )
    if slots > experts:
        raise MoeError(f"shape mismatch: {slots} slots per token exceeds {experts} experts")

    if ((topk_ids < 0) | (topk_ids >= experts)).any():
        raise MoeError(f"topk_ids holds an expert outside the range [0, {experts})")
    if (np.diff(np.sort(topk_ids, axis=1), axis=1) == 0).any():
        raise MoeError("topk_ids holds a duplicate expert within a token")

    _require_finite_input("x", x)
    _require_finite_input("topk_weights", topk_weights)
    if (topk_weights < 0).any():
        raise MoeError("topk_weights must be nonnegative")

    selected = np.unique(topk_ids)  # empty when there are no tokens, so nothing is inspected
    _require_finite_input("w13", w13[selected])
    _require_finite_input("w2", w2[selected])
    return tokens, width, intermediate


def _silu(z: np.ndarray) -> np.ndarray:
    """``z / (1 + exp(-z))`` with a never-positive exponent, so the negative tail underflows
    to zero instead of overflowing."""
    decay = np.exp(-np.abs(z))
    return z * np.where(z < 0, decay, 1.0) / (1.0 + decay)


def moe_experts(x, w13, w2, topk_ids, topk_weights, *, swiglu_limit) -> np.ndarray:
    """Weighted routed-expert combine for one batch; see the module docstring and contract."""
    tokens, width, intermediate = _validate(x, w13, w2, topk_ids, topk_weights)
    limit = _require_limit(swiglu_limit)
    out = np.zeros((tokens, width), dtype=np.float64)
    with np.errstate(all="ignore"):
        for token in range(tokens):
            for slot in np.argsort(topk_ids[token]):  # increasing expert ID, not slot order
                expert = topk_ids[token, slot]
                gate_up = w13[expert] @ x[token]
                _require_finite_result("gate/up projection", gate_up)
                gate = np.minimum(gate_up[:intermediate], limit)
                up = np.clip(gate_up[intermediate:], -limit, limit)
                hidden = _silu(gate) * up
                _require_finite_result("activation", hidden)
                projected = w2[expert] @ hidden
                _require_finite_result("down projection", projected)
                contribution = topk_weights[token, slot] * projected
                _require_finite_result("weighted output", contribution)
                out[token] += contribution
                _require_finite_result("combine", out[token])
    return out
