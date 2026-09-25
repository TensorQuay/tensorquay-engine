# REF-001: NoPE MLA attention core

Status: contract and implementation accepted on 20 Sep 2026; see the
[independent evaluation](../../docs/evals/2026-09-20-mla-reference.md).
This is the float64 algebraic reference in TEST-PLAN REF-001, for one sequence at a time.
It qualifies the absorbed attention identity, projection layout, supplied scale and boolean mask.
It does not qualify a model forward pass, production casts, paging, index generation or GPU execution.

## API

Module: `tq_reference.mla_attention`.

```python
class MlaError(ValueError): ...

@dataclass(frozen=True)
class MlaCoreResult:
    out: np.ndarray         # float64 [T, H, Dv]
    out_latent: np.ndarray  # float64 [T, H, R]
    lse: np.ndarray         # float64 [T, H]

def split_kv_b(w_kv_b, *, num_heads, qk_nope_head_dim, v_head_dim): ...
def mla_attention_core(q, latent, w_uk, w_uv, mask, *, scale): ...
```

Inputs are NumPy arrays, with no implicit conversion. Mathematical data has dtype exactly float64:

| Input | Shape | Meaning |
|---|---|---|
| `q` | `[T, H, Dn]` | Query after the query expansion projection |
| `latent` | `[N, R]` | Cached latent after normalization |
| `w_uk` | `[H, Dn, R]` | Per-head key expansion |
| `w_uv` | `[H, Dv, R]` | Per-head value expansion |
| `mask` | `[T, N]`, dtype bool | Allowed key set per query, shared across heads |

All dimensions are positive and consistent. Every input data value must be finite, including masked values.
Non-contiguous, negative-stride and read-only arrays are accepted and never modified. Every returned array is a
fresh C-contiguous float64 array that does not alias an input or another returned array.

`split_kv_b` accepts float64 `[H * (Dn + Dv), R]`. Rows are interleaved **per head**: each head's `Dn` key rows
followed by its `Dv` value rows. It returns independent copies `(w_uk, w_uv)`. Dimension arguments are positive
Python or NumPy integers; booleans are rejected.

`scale` is a required keyword: a Python int/float or NumPy integer/floating scalar, excluding booleans, arrays,
complex values and strings. It must remain finite and positive after float64 conversion. The core never infers
it from a dimension. GLM-5.3-Flash uses `256**-0.5`, although its latent width is 512.

## Mathematical and numerical contract

For each active query/head, absorb the key expansion: `q_abs = w_uk[h].T @ q[t, h]`. For **selected** latent rows,
compute the dot product in float64, then multiply by the supplied scale. Let `m` be their maximum score.
Compute `weights = exp(scores - m)` and `p = weights / sum(weights)`. Then:

- `out_latent = sum_n p[n] * latent[n]`;
- `out = w_uv[h] @ out_latent`;
- `lse = m + log(sum(weights))`.

Compute probabilities from the shifted weights directly. Reconstructing them as `exp(scores - lse)` loses their
normalization when a large common score offset causes `lse` to round to `m`.

An all-masked query has exact zero outputs and `lse = -inf`. Validate input types, shapes and finite values, then
skip its arithmetic entirely. Unselected scores are not evaluated. Exponential underflow to zero is valid.
A non-finite computed absorbed query, selected score, shifted score, latent output, output or active `lse` raises
`MlaError` without leaking floating-point warnings. This contract does not guarantee exact algebra across every
ill-conditioned or overflowing combination of finite float64 inputs.

The active `lse` finite check is defensive. Earlier checks ensure finite scores and `1 <= sum(weights) <= N`
(up to rounding), so `0 <= log(sum(weights)) <= log(N)` for an in-memory array. This increment cannot overflow
a finite float64 maximum. Tests check finite `lse` at both float64 limits; an input-triggered `lse` overflow
case is not required. At the latent-output limit, different float64 reduction orders can either stay finite
or overflow: acceptance requires an accurate finite result or the correct first-overflow error.

## Diagnostics

Invalid supplied inputs raise `MlaError` with these stable fragments. Missing required arguments retain Python's
normal `TypeError`. Validation order between independently invalid arguments is unspecified.

| Condition | Fragment |
|---|---|
| Not an ndarray | `numpy.ndarray` |
| Wrong data dtype / mask dtype | `float64` / `bool` |
| Wrong rank / empty dimension | `dimensions` / `empty` |
| Inconsistent dimensions | `shape mismatch` |
| Non-finite data | `non-finite` |
| Invalid scale | `scale` |
| Invalid split dimension type / value | `integer` / `> 0` |
| Wrong packed projection row count | `rows must equal` |
| Evaluated arithmetic overflow | `overflow in <stage>` |

Stage names are `absorbed query`, `score`, `shifted score`, `latent output`, `output` and `lse`.

## Acceptance

The lead's independent oracle expands keys and values with scalar Decimal arithmetic at 80 digits, then applies
attention. It never computes the absorbed query. Analytical cases check the oracle itself, including a one-key
selection, uniform attention, an empty selection and equal scores at `1e16`.

- `out` and `out_latent`: `rel_l2 <= 1e-12` per query/head and globally for curated nonzero reference vectors.
  Exactly representable analytical zero vectors require exact zeros; there is no denominator epsilon.
- Finite `lse`: `abs(error) <= 1e-12 * max(1, abs(reference_lse))`; the empty sentinel is exactly `-inf`.
- The independent acceptance suite alone must cover all new module lines and branches. Existing NVFP4 acceptance
  and the combined package coverage must remain at 100%, without exclusions.
- Before REF-001 is accepted, also reproduce an independent upstream fixture using Transformers commit
  `770e4c40d0436082a52dc380f07a9d3f389c99d4`, actual query projection, KV expansion, mask construction and float64
  CPU SDPA with its MATH backend. The source stays outside the repository. Record SHA pins, dependency versions,
  input provenance, the backend and a regeneration command. The generator must not import `tq_reference`.
- A precision-sensitive fixture must expose an accidental FP32 path. Backend bit differences may be reported but
  are not a correctness requirement. An eval report records each gate separately; pending upstream validation is
  never reported as passed.

The fixture derives `q` from a common post-normalization query and upstream projection weights, so both comparison
arms see identical inputs. SDPA has no `out_latent` or `lse` output; those remain checked by the independent oracle.
Duplicate indices, causal validity, poisoned unread memory and `k_len` belong to later SMLA tests. Their harnesses
must independently validate selected indices before constructing this core's boolean mask.
