# Routed MoE CPU reference review

Gate date: 20 Sep 2026. **Decision: accepted for REF-002, the float64 routed-expert path.**
This reference may serve as an oracle for later expert-kernel tests. Phase 0 step 0.2 remains
incomplete: REF-006 is still open. No GPU or model-serving gate is covered.

## Scope

- Contract: [routed MoE expert path](../../reference/contracts/moe-experts.md), written before implementation.
- Code: `reference/src/tq_reference/moe_experts.py`; float64 token activations and expert weights,
  explicit clamp limit, caller-supplied distinct expert IDs and nonnegative routing weights.
- The gate/up projection is laid out per expert as `[gate; up]`. Gate clamping is upper-only;
  the up projection is clamped at both ends. Routing weights multiply the output projection;
  contributions combine in ascending expert-ID order. No implicit normalization or extra scaling.
- Independent tests: `reference/tests/test_moe_acceptance.py` and `test_moe_upstream_acceptance.py`.
  An 80-digit scalar Decimal oracle checks the full expert path without NumPy matrix multiplication.
- Upstream source: [Transformers 770e4c40](https://github.com/huggingface/transformers/blob/770e4c40d0436082a52dc380f07a9d3f389c99d4/src/transformers/models/glm5_next/modeling_glm5_next.py),
  with its eager expert implementation and class-owned clamped activation. The generator must
  positively observe those original callables; resembling an eager operation trace is insufficient.
- A fixture obtains authentic IDs and weights from the upstream router in FP32, including its
  correction bias, then widens returned weights exactly to float64 for both expert arms. This
  is input provenance, not REF-006 router acceptance. Shared experts and model forwarding are excluded.
- Ordinary environment: macOS ARM64, Python 3.12.4, NumPy 2.3.3, pytest 8.4.2, pytest-cov 7.0.0,
  Ruff 0.13.1. Upstream generation uses the existing isolated PyTorch 2.8.0 and pinned Transformers
  environment. No GPU, model weights, serving workload or throughput measurement is involved.
- Revision: uncommitted working tree based on `d7f25560d4767e72eb6542b1d9e22489f3a58a45`.

## Review decisions

1. The lead checked the oracle before implementation: a known sigmoid value, the SiLU difference
   identity, exact zeros and six deliberate errors. The implementation tests initially failed
   because the module did not exist. The first implementation passed all 165 initial cases.
2. The upstream padding sentinel is not part of this API. IDs must be in `[0, E)` and unique
   within a token. Unselected expert banks may contain non-finite values; selected banks and
   arithmetic must be finite even at zero routing weight. Tests isolate that zero-weight case.
3. SiLU must avoid exponential overflow for large negative inputs. The lead separately ran
   PyTorch 2.8.0 CPU float64 SiLU at `-720`: it returned negative zero. The algebraic reference
   preserves the representable negative-tail value. This is an intentional numerical difference;
   the fixture comparison does not claim equivalence over every finite float64 input.
   The subnormal test uses an absolute allowance derived from exponential rounding before the
   multiplication; it does not apply the ordinary relative-error gate in the subnormal range.
4. A proposed float64 router would also promote its selection bias and change the precision
   boundary. The fixture must retain FP32 router parameters, bias and outputs, with exact
   promotion only after routing. The expert-only FP32 probe must reuse fixed IDs and weights.
5. The upstream expert loop batches tokens, so its last bits can differ from per-token NumPy
   products. The gate remains per-token and global relative L2 `<= 1e-12`, with exact analytical
   zeros and no denominator epsilon. No tolerance is adjusted to fit observed output.
6. Generator review found an observer that removed an existing profile hook instead of restoring
   it. The lead reproduced that failure on both normal and exceptional exits, then verified the
   correction. Activation ownership is now checked by actual qualified name and source file;
   each recorded observation must enter the original eager forward exactly once.
7. Input/output provenance distinguishes generated router weights and bias from actual router
   outputs. It also distinguishes literal clamp inputs, random routing weights and poisoned unused
   banks. Expected expert order is labeled as derived, not an observation of the upstream loop.
8. A zero reference norm must not hide a nonzero FP32 result. After that correction, an additional
   lead probe found that `torch.linalg.vector_norm` underflows on two or more `1e-300` values:
   the helper reported an undefined ratio against FP32 zeros instead of the correct `1.0`.
   The corrected helper uses `math.hypot` and retains exact true-zero checks. The lead separately
   reran the boundary at 1, 2, 4 and 24 elements: all now report `1.0`, true-zero pairs report an
   undefined ratio, and a nonzero result against exact zero is rejected. The generator retains
   regression guards. All six raw fixture input/output cases stayed bit-identical through this fix.

## Independent verification

| Check | Status |
|---|---|
| Independent core suite, including 288 experts, top-8 and the last bank | 166 passed; 82 statements and 34 branches, 100% |
| Independent upstream fixture checks | 22 passed across six cases |
| All independent reference suites | 491 passed; 348 statements and 104 branches, 100% |
| Full suite with warnings treated as errors | 688 passed; 100% lines and branches |
| Fresh pinned upstream reproduction after all corrections | Six cases and 24 provenance fields match |
| Observer restoration on normal and exceptional exits | Failure reproduced before correction; both paths now pass |
| Zero and tiny-reference metric boundaries | Correct rejection/undefined ratio at true zero; relative error 1.0 for tiny references at 1, 2, 4 and 24 elements |
| Format, lint, source/function lengths, working-tree content scan, guard audit and diff check | Clean |
| Previously accepted NVFP4 and MLA artifacts | 16 artifacts unchanged byte for byte |

The lead measured the following errors against the stored upstream eager expert outputs:

| Upstream case | Global relative L2 error | Largest relative L2 error per token |
|---|---:|---:|
| Authentic FP32 router, top-8 | 1.389e-16 | 1.820e-16 |
| Nonuniform routes, asymmetric dimensions | 8.740e-17 | 2.116e-16 |
| Zero routing weights | 1.059e-16 | 1.218e-16 |
| Asymmetric clamp boundaries | 0 | 0 |
| Unselected banks containing NaN/Inf | 2.388e-16 | 2.993e-16 |
| Precision-sensitive float64 case | 1.019e-16 | 1.603e-16 |

All six cases also agree with the independent Decimal oracle. The all-zero-weight token is
checked as exact zero and never excluded. The designated FP32 expert rerun has global relative
L2 error `8.955e-8`, above the frozen `1e-9` sensitivity floor. Its stored raw FP32 output allows
the independent suite to recompute that metric. Routes are kept fixed during the FP32 rerun.

Coverage applies to `reference/src/tq_reference`, without added exclusions. The generator is
separately reviewed, reexecuted and probed; the coverage figure does not cover tooling or
upstream libraries. The fixtures use synthetic small weights, not a model checkpoint.

## Reproduction

From the repository root, after installing `uv`:

```sh
cd reference
uv sync --locked
uv run --locked --offline pytest tests/test_nvfp4_acceptance.py tests/test_mla_acceptance.py \
  tests/test_mla_upstream_acceptance.py tests/test_moe_acceptance.py \
  tests/test_moe_upstream_acceptance.py --cov --cov-branch -W error
uv run --locked --offline pytest --cov --cov-branch -W error
uv run --locked --offline ruff format --check .
uv run --locked --offline ruff check .
uv run tools/transformers_moe_fixture.py --check
```

The final command reexecutes upstream and its observer/metric guards. It may need network access
to install its isolated environment and requires the recorded dependency versions. Ordinary
acceptance uses the stored fixture and does not require PyTorch, Transformers, model weights or
a GPU. Input data, raw FP64/FP32 outputs, source hashes and actual call counts are in the
[upstream fixture](../../reference/tests/fixtures/transformers_moe_770e4c40_experts.json).

## Reviewed artifact snapshot

SHA-256, with paths relative to `reference/`:

```text
ac0d56480486871998238084c0c9741ff24f37b27ddb7d717c515373aaf86401  contracts/moe-experts.md
d40187376bf2780e33d9075d633ce3d0ff66a0435f1e20e4d8e9a5237735445e  src/tq_reference/moe_experts.py
e87bc474a220a1532771a5fb3adcc8b836a629a348d7aa9e1a1bbab2c69d9dfb  tests/test_moe_acceptance.py
b9f9010e8a9f615a0911ef7b5157a6eeadd4b0c19724d01180bf399a29c0e7c1  tests/test_moe_upstream_acceptance.py
4e9ad8756f9f7bf68c3b9d59eab4c0f2e9c820bd7f46cb01a8ad1bb07565d3fc  tests/test_ref002_moe_core.py
c706964cebbb4a2286dd9ea14e194943899bce0564eac7d757de10173a09592a  tests/test_ref002_upstream_fixture_smoke.py
bc0c5b67e88d7c2dc3e2e2a908455cb871671d7f113839b5de04f76341bf2c31  tools/transformers_moe_fixture.py
c228d61ac10c9669c7959be55401a14b457b5172948284dfa94d60cfe7783071  tests/fixtures/transformers_moe_770e4c40_experts.json
```

CPU acceptance does not establish GPU correctness, tensor-parallel execution, production dtype
behavior, model quality, speed, concurrency or an advantage over another engine. REF-006 and
the subsequent kernel and model gates remain open.
