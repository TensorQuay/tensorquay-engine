# REF-002: routed MoE expert path

Status: contract and implementation accepted on 20 Sep 2026; see the
[independent evaluation](../../docs/evals/2026-09-20-moe-reference.md).
This is the float64 algebraic reference in TEST-PLAN REF-002. It includes selected experts and
their weighted combine. It excludes router selection, shared experts, quantization, production
dtype casts, tensor-parallel communication and GPU execution.

## Source and API

The mathematical boundary is `Glm5NextTextExperts` in
[Transformers 770e4c40](https://github.com/huggingface/transformers/blob/770e4c40d0436082a52dc380f07a9d3f389c99d4/src/transformers/models/glm5_next/modeling_glm5_next.py).
Write the implementation independently; the installed upstream implementation runs separately
to generate comparison data. Module: `tq_reference.moe_experts`.

```python
class MoeError(ValueError): ...

def moe_experts(x, w13, w2, topk_ids, topk_weights, *, swiglu_limit) -> np.ndarray: ...
```

| Input | Dtype and shape | Meaning |
|---|---|---|
| `x` | float64 `[T, H]` | Token activations |
| `w13` | float64 `[E, 2*I, H]` | Per expert, all gate rows followed by all up rows |
| `w2` | float64 `[E, H, I]` | Output projection |
| `topk_ids` | int32 or int64 `[T, K]` | Distinct expert IDs per token, each in `[0, E)` |
| `topk_weights` | float64 `[T, K]` | Finite, nonnegative weights corresponding to those IDs |

All arrays must be NumPy ndarrays; no implicit dtype conversion. Shapes must agree, `T >= 0`,
`E, H, I >= 1` and `1 <= K <= E`. Read-only, non-contiguous and negative-stride inputs are
supported and never modified. The result is a fresh C-contiguous float64 `[T, H]` array.

`swiglu_limit` is an explicit Python int/float or NumPy integer/floating scalar, excluding bool,
arrays, strings and complex values. It must stay finite and positive after conversion to
float64. GLM-5.3-Flash supplies `10.0`; do not silently infer a limit from model or shape.

Weights need not sum to one or 2.5. They already include any router normalization and scaling;
the core must not normalize, apply another factor, select experts or reorder weights separately
from their IDs. The upstream padding sentinel `E` is outside this API and must be rejected.

## Mathematical and numerical contract

For each token `t` and selected expert `e` with its supplied weight `a`:

1. `gate = w13[e, :I] @ x[t]`; `up = w13[e, I:] @ x[t]`.
2. `g = min(gate, swiglu_limit)` elementwise; **no lower clamp on the gate**.
3. `u = clamp(up, -swiglu_limit, swiglu_limit)` elementwise.
4. `hidden = SiLU(g) * u`, where `SiLU(z) = z / (1 + exp(-z))` mathematically.
5. `expert_out = w2[e] @ hidden`; `contribution = a * expert_out`.
6. Sum contributions in increasing expert-ID order for each token, starting from float64 zeros.

All arithmetic is float64. Evaluate SiLU without avoidable exponential overflow; exponential
underflow to zero is allowed. The reference need not reproduce an upstream implementation's
avoidable loss of tiny negative-tail values. The `1e-12` upstream gate applies to the curated
finite comparison cases; it is not a universal guarantee for ill-conditioned or subnormal data.

Only selected expert weights are inspected or evaluated. Unselected experts may contain NaN
or infinity without affecting the output. A selected expert must have finite weights even
when its routing weight or input happens to be zero; a zero routing weight does not deselect it.
All `x` and routing-weight values must be finite. With `T=0`, validate metadata and the limit,
then return the empty result without inspecting expert weight values.

Reject non-finite arithmetic at the first evaluated stage: `gate/up projection`, `activation`,
`down projection`, `weighted output` or `combine`. This includes a projection that overflows
before a later clamp could hide it. Raise `MoeError`; do not leak floating-point warnings or
repair non-finite values. Exact zero inputs and zero routing weights produce zero outputs when
the evaluated intermediates remain finite.

## Diagnostics

Invalid supplied inputs raise `MoeError` with these stable fragments. Missing required arguments
retain Python's normal `TypeError`. Ordering between independently invalid inputs is unspecified.

| Condition | Fragment |
|---|---|
| Not an ndarray | `numpy.ndarray` |
| Wrong data dtype / ID dtype | `float64` / `int32 or int64` |
| Wrong rank | `dimensions` |
| Empty non-token dimension | `empty` |
| Inconsistent shapes, odd fused row count or `K > E` | `shape mismatch` |
| Expert ID outside `[0, E)` | `range` |
| Duplicate expert ID within a token | `duplicate` |
| Non-finite required input or selected weight | `non-finite` |
| Negative routing weight | `nonnegative` |
| Invalid clamp limit | `swiglu_limit` |
| Evaluated arithmetic overflow | `overflow in <stage>` |

Diagnostics must handle very large integer scalar arguments without changing Python's global
integer formatting limits or raising a different exception while formatting an invalid value.

## Acceptance

- The lead owns `tests/test_moe_acceptance.py` and `tests/test_moe_upstream_acceptance.py`.
  A scalar Decimal oracle at 80 digits evaluates projections, clamped SiLU, output projection
  and weighted combine independently of NumPy matrix multiplication. Analytical tests check
  the oracle itself; deliberate broken results must fail the comparison gate.
- For curated nonzero outputs: relative L2 error `<= 1e-12`, per token and globally, with no
  denominator epsilon. Exactly representable analytical zero outputs require exact zeros.
- Independent acceptance alone must reach 100% of this module's lines and branches, without
  exclusions. Previously accepted reference artifacts remain unchanged and their suites pass.
- Before acceptance, freshly reproduce a small fixture by running the pinned upstream eager
  expert module in float64. Positively verify the actual eager callable, source hashes, dtype
  and device. The generator never imports `tq_reference` and stores no external source.
- At least one fixture obtains IDs and weights from the actual upstream FP32 router, then
  converts those weights exactly to float64 for both expert arms. This establishes the input
  boundary only; it does not accept REF-006 router semantics. Include separately supplied
  routes for zero weights, clamp edges and unselected poisoned experts.
- A designated fixture must expose a complete accidental FP32 expert path: global relative
  L2 output difference `>= 1e-9` from the float64 run. Both runs must be finite; no row exclusion
  or NaN repair. Record all dependency versions and a strict regeneration/check command.
- The evaluation report records independent results, fixture/source revisions, known numerical
  limits and remaining gates. CPU acceptance does not satisfy the later MOE GPU tests.
