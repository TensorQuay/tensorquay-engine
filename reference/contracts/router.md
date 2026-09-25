# REF-006: single-group FP32 router

Status: contract and implementation accepted on 20 Sep 2026; see the
[independent evaluation](../../docs/evals/2026-09-20-router-reference.md).
This is the production-semantics CPU reference for the pinned GLM configuration. It includes
the projection, sigmoid, correction-bias selection and normalized routing weights. It excludes
experts, grouped routing, checkpoint loading and GPU execution. Keep this a plain NumPy module;
the isolated upstream runner is a test dependency, not an engine dependency.

## Source and API

The boundary is `Glm5NextTextTopkRouter` in
[Transformers 770e4c40](https://github.com/huggingface/transformers/blob/770e4c40d0436082a52dc380f07a9d3f389c99d4/src/transformers/models/glm5_next/modeling_glm5_next.py).
The pinned configuration has one group, 288 experts, top-8, normalization enabled and scaling
factor 2.5. With one selected group there is no group filtering; do not build unused grouped
routing machinery. Write the implementation independently from these mathematical requirements.

```python
# tq_reference.router
class RouterError(ValueError): ...

def route(x, weight, correction_bias, *, top_k=8) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
```

| Input | Dtype and shape |
|---|---|
| `x` | float16, float32 or float64 `[T, H]` |
| `weight` | float16, float32 or float64 `[E, H]` |
| `correction_bias` | float32 `[E]` |

All inputs must be NumPy ndarrays with finite values. `T >= 0`, `H >= 1`, `E >= 2` and
`1 <= top_k <= E`. `top_k` accepts Python and NumPy integer scalars, excluding booleans and
arrays. Read-only, non-contiguous and negative-stride arrays are supported without mutation.
Convert `x` and `weight` to float32 **before** projection, as upstream does. Reject finite values
that become non-finite in that conversion. The bias must already be float32: silently accepting
a float64 bias would change selection precision. Validate all inputs even when `T=0`.

Return `(logits, topk_weights, topk_ids)` as fresh C-contiguous arrays with shapes `[T,E]`,
`[T,K]`, `[T,K]` and dtypes float32, float32, int64. There is no input/output aliasing.
`T=0` returns empty arrays after validation.

## Arithmetic and ordering

1. Project the float32 inputs: `logits = x32 @ weight32.T`, with float32 products/accumulation
   and output. Reject a non-finite result with `RouterError("overflow in projection")`.
2. Compute sigmoid in float32 as `1 / (1 + exp(-logits))`. Exponential overflow to infinity
   gives score zero; exponential underflow is allowed. This intentionally matches the pinned
   CPU production path, including its zero negative tail; do not replace it with a float64 or
   stable negative-tail sigmoid that changes those values. Suppress expected overflow/underflow
   warnings locally, without changing NumPy's global error policy.
3. Add the correction bias in float32 for selection only. Select the largest `K` corrected
   scores; equal scores prefer the lower expert ID. Return the selected IDs in increasing ID
   order, with the weights paired to those IDs. This is our canonical slot ordering; upstream
   uses `topk(sorted=False)` and makes no promise about slot order or tie selection.
4. Gather the **unbiased** sigmoid scores. Starting with float32 zero, add these scores in
   increasing expert-ID order, rounding each addition to float32. Add `float32(1e-20)` to that
   sum. Divide each gathered score by that denominator in float32, then multiply by
   `float32(2.5)` in float32. Never include the bias in the weights, normalize all experts or
   omit the epsilon. A row of zero scores produces exact zero weights, not a uniform row.

This fixes a deterministic reference reduction, not PyTorch's private reduction implementation.
No bitwise claim is made across numerical libraries, processor types or batch partitions.
Arithmetic stays FP32 even when the caller supplied float64 arrays; this is not an O1-alg path.

## Diagnostics

Invalid supplied values raise `RouterError` with these fragments. Missing/unknown arguments
retain normal Python `TypeError`. Ordering between independently invalid inputs is unspecified.

| Condition | Fragment |
|---|---|
| Not an ndarray | `numpy.ndarray` |
| Wrong `x` or `weight` dtype | `floating dtype` |
| Bias not float32 | `float32` |
| Wrong rank | `dimensions` |
| `H=0` or `E<2` | `empty` |
| Inconsistent dimensions | `shape mismatch` |
| NaN or infinity in an input | `non-finite` |
| Finite input overflows when cast to float32 | `float32 range` |
| Invalid `top_k` | `top_k` |
| Non-finite projection | `overflow in projection` |

Diagnostics must handle arbitrarily large integer parameters without converting them to text
or changing Python's integer-formatting limit. Valid and rejected calls emit no numeric warnings.

## Numerical gate correction, decided before implementation

The earlier REF-006 weight gate `1e-12` was inconsistent with its FP32 operation boundary.
An independent probe with `H=1`, `E=K=8`, input one, zero bias and identical projected logits
produced weight relative L2 error `1.7748198117999202e-7` between the pinned upstream router and
NumPy FP32. Seven of eight weights differed in bits while the selected set was identical.
The logits were `[-10.91289234161377, -9.825783729553223, -5.059233665466309, -4, -3, -2, -1, 0]`.
Sigmoid implementations and reduction order both contribute; retaining the old rule would
require copying a backend's rounding details or choosing only accidentally identical fixtures.
[PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)
explicitly does not promise bitwise equality for equivalent floating-point calculations.

The lead therefore replaces that rule for REF-006 only with a **fixed FP32 qualification budget**:
`u = 2^-24`, `n = 2*K + 16`, `B(K) = n*u / (1 - n*u)` (require `n*u < 1` in the gate).
At top-8 this is about `1.91e-6`. It budgets two K-term reductions and sixteen further rounding
steps for sigmoid and normalization in the two implementations. This is a conservative test
budget for the curated cases, not a proof of an arbitrary library's transcendental accuracy or
of well-conditioned projection for all inputs. It is fixed before implementation, never fitted
to its output. Float64 reference gates and all GPU/model gates remain unchanged.

- Require exact expert **sets**, pairing weights by expert ID before comparison. For cross-library
  selection cases with `K<E`, require a cutoff gap greater than
  `64*u*max(1, max(abs(corrected_scores)))`. Mere absence of exact ties is insufficient.
- Upstream qualification inputs make FP32 projection exactly reproducible; assert equality of
  the projected logits independently, rather than hiding projection errors in the weight budget.
  Include small dense dyadic projections, a float64-input cast-sensitive case and all 288 experts.
- Compare nonzero weights with relative L2 `<= B(K)`, both per token and globally, without a
  denominator epsilon. Check each nonzero component with that same relative budget too, so
  small weights cannot hide behind large ones. Require exact zeros for expected zero weights.
- Independently test projection casts, selection arithmetic, ties, canonical pairing, FP32
  normalization, the epsilon and tail behavior. A high-precision scalar oracle rounds the
  individual sigmoid and normalization operations to float32; it does not call NumPy exp,
  matrix multiplication or reductions. Analytical cases check that oracle.
- Cross-library ties and near-ties are not a portable equivalence claim. They still receive
  deterministic local acceptance tests; do not discard a failed well-separated fixture.

## Acceptance and fixture

The lead owns `tests/test_router_acceptance.py` and `tests/test_router_upstream_acceptance.py`.
Independent acceptance alone must cover 100% of module lines and branches, without exclusions.
Deliberately broken selection, cast, bias, epsilon, normalization and pairing results must fail.
Previously accepted artifacts and evaluation reports stay unchanged.

Before acceptance, freshly run the pinned upstream router in its isolated Torch 2.8.0 environment.
The generator must never import `tq_reference`, copy upstream source into the repository or
substitute a local router for the real callable. Record source hashes, all dependency versions,
input recipes, raw logits/weights/IDs, configuration, dtypes and positive evidence of entering the
original upstream router forward. A strict `--check` reruns upstream and compares metadata and
cases. Include bias-driven selection, GLM's 288/top-8 configuration, cast-sensitive inputs,
epsilon-dominated negative logits, zero tails, saturation and the recorded rounding discrepancy.
Keep fixture files below 2 MB and use the existing isolated environment without new dependencies.

The evaluation report must name the numerical gate revision, measured errors, exact-selection
results, coverage and limitations. CPU acceptance does not qualify grouped routing, CUDA,
end-to-end quality, throughput, latency or any later model gate.
