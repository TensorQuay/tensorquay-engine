"""REF-006 float32 reference for the single-group GLM-5.3-Flash router.

Contract: ``reference/contracts/router.md``. One batch of tokens per call:

- ``x``: float16/32/64 ``[T, H]``, token activations;
- ``weight``: float16/32/64 ``[E, H]``, the router projection;
- ``correction_bias``: float32 ``[E]``, added for selection only.

``x`` and ``weight`` are cast to float32 **before** the projection, as upstream does, and every
later operation stays float32 even when the caller supplied float64. The pinned configuration has
one group, so there is no group filtering and none is built here.

    logits = x32 @ weight32.T                       # float32 products and accumulation
    scores = 1 / (1 + exp(-logits))                 # float32; exp overflow gives score zero
    choice = scores + correction_bias               # selection only, never a weight
    ids    = the K largest choices, ties to the lower ID, returned in increasing ID order
    denom  = float32 zero, plus the gathered scores in increasing ID order, plus float32(1e-20)
    weights = (scores[ids] / denom) * float32(2.5)  # divide first, then scale

The sigmoid deliberately keeps the pinned CPU path's zero negative tail: once ``exp(-logit)``
overflows, the score is exactly zero. This fixes a deterministic reference reduction, not any
particular library's, so no bitwise claim is made across libraries or processors.
"""

import numpy as np

FLOATS = (np.float16, np.float32, np.float64)
ONE = np.float32(1)
EPSILON = np.float32(1e-20)
SCALE = np.float32(2.5)


class RouterError(ValueError):
    """Invalid input to, or arithmetic overflow in, the REF-006 reference."""


def _require_array(name: str, value, ndim: int) -> None:
    if not isinstance(value, np.ndarray):
        raise RouterError(f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    if name == "correction_bias":
        if value.dtype != np.float32:
            raise RouterError(f"correction_bias must have dtype float32, got {value.dtype}")
    elif value.dtype.type not in FLOATS:
        raise RouterError(f"{name} must have a floating dtype, got {value.dtype}")
    if value.ndim != ndim:
        raise RouterError(f"{name} must have {ndim} dimensions, got shape {value.shape}")


def _validate(x, weight, correction_bias) -> int:
    """Check every supplied input, including when there are no tokens, and return ``E``."""
    _require_array("x", x, 2)
    _require_array("weight", weight, 2)
    _require_array("correction_bias", correction_bias, 1)

    experts = weight.shape[0]
    if x.shape[1] == 0 or weight.shape[1] == 0:
        raise RouterError(f"x {x.shape} and weight {weight.shape} have an empty width")
    if experts < 2:
        raise RouterError(f"weight must hold at least 2 experts, got an empty or single {experts}")
    if x.shape[1] != weight.shape[1] or correction_bias.shape[0] != experts:
        raise RouterError(
            f"shape mismatch: x {x.shape}, weight {weight.shape}, "
            f"correction_bias {correction_bias.shape}"
        )
    for name, value in (("x", x), ("weight", weight), ("correction_bias", correction_bias)):
        if not np.isfinite(value).all():
            raise RouterError(f"{name} contains non-finite values")
    return experts


def _require_top_k(top_k, experts: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
        raise RouterError(f"top_k must be an integer, got {type(top_k).__name__}")
    if not 1 <= top_k <= experts:
        # The value itself is caller-supplied and unbounded, so it is never formatted.
        raise RouterError(f"top_k must be between 1 and {experts}")
    return int(top_k)


def _to_float32(name: str, value: np.ndarray) -> np.ndarray:
    """Cast before projecting, and reject a finite input that leaves the float32 range."""
    cast = value.astype(np.float32)
    if not np.isfinite(cast).all():
        raise RouterError(f"{name} holds a value outside the float32 range")
    return cast


def route(x, weight, correction_bias, *, top_k=8):
    """Project, score, select and normalize one batch; see the module docstring and contract."""
    experts = _validate(x, weight, correction_bias)
    selected = _require_top_k(top_k, experts)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        logits = _to_float32("x", x) @ _to_float32("weight", weight).T
        if not np.isfinite(logits).all():
            raise RouterError("overflow in projection")

        scores = ONE / (ONE + np.exp(-logits))  # exp overflow -> score exactly zero
        choice = scores + correction_bias  # selection only; never reaches a weight
        # Ascending sort of the negated scores is descending with ties on the lower ID.
        ids = np.sort(np.argsort(-choice, axis=1, kind="stable")[:, :selected], axis=1)
        ids = ids.astype(np.int64, copy=False)

        gathered = np.take_along_axis(scores, ids, axis=1)
        denominator = np.zeros((logits.shape[0], 1), dtype=np.float32)
        for slot in range(selected):  # left to right, in increasing expert-ID order
            denominator += gathered[:, slot : slot + 1]
        denominator += EPSILON
        weights = (gathered / denominator) * SCALE  # divide first, then scale
    return (
        np.ascontiguousarray(logits),
        np.ascontiguousarray(weights),
        np.ascontiguousarray(ids),
    )
