# MLA CPU reference review

Gate date: 20 Sep 2026. **Decision: accepted for REF-001, the float64 algebraic attention core.** This reference
may now serve as an oracle for subsequent contract and kernel tests. Phase 0 step 0.2 remains incomplete:
REF-002 and REF-006 are still open. No GPU or model-serving gate is covered.

## Scope

- Contract: [NoPE MLA attention core](../../reference/contracts/mla-attention.md), frozen before implementation.
- Implementation: `reference/src/tq_reference/mla_attention.py`; one sequence, float64 data, explicit attention
  scale, boolean key masks and a per-head interleaved key/value projection layout.
- Independent acceptance: `reference/tests/test_mla_acceptance.py` and `test_mla_upstream_acceptance.py`. The oracle
  expands keys and values with scalar Decimal arithmetic at 80 digits; it does not use the implementation's
  absorbed-query computation.
- Upstream comparison: actual query projection, KV expansion, mask construction and CPU MATH SDPA from
  [Transformers 770e4c40](https://github.com/huggingface/transformers/blob/770e4c40d0436082a52dc380f07a9d3f389c99d4/src/transformers/models/glm5_next/modeling_glm5_next.py).
  Both arms consume the same post-projection query, latent, mask and scale. We generate the weights, residual and
  latent inputs; upstream produces the projected query, expanded keys/values, mask and attention output. This
  does not qualify a complete model projection path, normalization, indexer, production casts or output projection.
  The [PyTorch 2.8 SDPA documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)
  identifies MATH as a float64-capable implementation; the executed source and recorded backend are checked too.
- Environment for ordinary tests: macOS ARM64, Python 3.12.4, NumPy 2.3.3, pytest 8.4.2, pytest-cov 7.0.0,
  Ruff 0.13.1. No GPU, model weights, context-serving workload or throughput measurement is involved.
- Upstream generation additionally uses PyTorch 2.8.0 and Transformers 5.18.0.dev0 at the pinned commit, in an
  isolated environment. The fixtures record resolved dependency versions and source SHA-256 values. Exact
  regeneration checks those versions; ordinary fixture acceptance uses NumPy and needs no upstream installation.
- Revision: working tree based on `d7f25560d4767e72eb6542b1d9e22489f3a58a45`; REF-001 is uncommitted.
  Artifact hashes below identify the reviewed snapshot. No commit, push or GPU work is part of this acceptance.

## Review findings

1. Before implementation, the proposed `exp(score - lse)` normalization failed the lead's analytical example with
   two equal scores at `1e16`. The corrected implementation normalizes shifted exponential weights directly and
   computes `lse` separately. Independent tests cover that case.
2. The lead independently reproduced latent-output overflow using a dyadic logit gap of `7/16` and two maximum
   float64 latent values. Probabilities sum to one, yet intermediate product/sum rounding can overflow. The test
   requires the correct first-overflow error, or an accurate finite result on a backend whose reduction remains
   finite. A later output error or a non-finite result fails.
3. The active `lse` finite check is retained as a defensive postcondition. Once earlier checks pass, the log-sum
   increment is bounded by the array size and cannot overflow the float64 range. Independent tests cover both
   signs of the largest finite float64; no fabricated input-overflow case is required.
4. After the original suite passed at 100% coverage, adversarial review found that formatting very large Python
   integer arguments could raise plain `ValueError` instead of `MlaError`. Three new independent tests reproduce
   this for scale, a negative split dimension and a positive projection row-count mismatch. The required fix is
   bounded diagnostics, retaining the contract's message fragments and Python's existing global settings. The lead
   inspected that correction and reran the expanded independent suite successfully.
5. Fixture review caught misleading backend evidence: a nested MATH context overrode the optional FLASH probe,
   and a dispatch observer returned no SDPA events. The corrected generator requires a positive CPU profiler event
   for MATH in every case and removes the optional FLASH claim. Outputs must be finite; no NaN replacement or row
   exclusion is permitted. The lead inspected the generator and freshly reproduced all five cases and 21 stable
   provenance fields after the corrections. All-masked rows returned exact zeros in the actual upstream run.

## Independent verification

| Check | Status |
|---|---|
| New extreme-integer regressions | 3 failures reproduced before correction; all pass after correction |
| Independent MLA core acceptance | 193 passed; 100 statements and 32 branches, 100% |
| Independent upstream-fixture checks | 8 passed across two fixture files; all five cases compared |
| All independent reference suites | 303 passed; 266 statements and 70 branches, 100% |
| Full suite, including developer checks, with warnings treated as errors | 453 passed; 100% lines and branches |
| Fresh pinned Transformers + MATH-SDPA reproduction | 5 cases and 21 provenance fields match |
| Format, lint, source length, working-tree content scan, guard audit and diff check | Clean |
| Previously accepted NVFP4 implementation, tests, fixtures, generator and dependency files | Seven artifacts unchanged byte for byte from the accepted commit |

Coverage applies to `reference/src/tq_reference`, with no added exclusions. Fixture generators are separately
reviewed and executed; the 100% figure does not claim coverage of upstream libraries or generator tooling.

The lead measured output error against the stored MATH-SDPA outputs:

| Upstream case | Global relative L2 error | Largest relative L2 error per query/head |
|---|---:|---:|
| Asymmetric multi-head dimensions | 2.362e-16 | 9.567e-16 |
| Single-key and all-masked rows | 1.267e-16 | 2.512e-16 |
| Duplicate and invalid indices | 8.461e-17 | 1.868e-16 |
| Precision-sensitive float64 case | 4.277e-16 | 1.245e-15 |
| GLM query width 256, latent width 512 | 5.210e-16 | 1.084e-15 |

The two all-masked head outputs are checked as exact zeros, without a denominator adjustment. The small cases
also agree with the independent 80-digit Decimal oracle. SDPA returns neither latent outputs nor LSE; those
outputs remain covered by that oracle. The GLM-width case has deliberately small head/value counts to keep the
fixture below 2 MB; it checks the 256-versus-512 scale distinction, not a full production layer.

The precision case's complete FP32 upstream rerun has maximum relative entry error `7.996e-6`, above the frozen
`1e-9` sensitivity floor. This diagnostic detects a downcast of that path; it does not establish sensitivity to
every possible isolated cast. The independent suite also rejects use of the latent width to derive the scale.

Output and latent-output gates remain `rel_l2 <= 1e-12` per query/head and globally. Exactly representable zero
vectors require exact zeros. Finite `lse` uses `abs(error) <= 1e-12 * max(1, abs(reference_lse))`; empty rows require
exact zero outputs and `lse = -inf`. No threshold was adjusted to fit implementation results.

## Reproduction

From the repository root, after installing `uv`:

```sh
cd reference
uv sync --locked
uv run --locked --offline pytest tests/test_nvfp4_acceptance.py tests/test_mla_acceptance.py \
  tests/test_mla_upstream_acceptance.py --cov --cov-branch -W error
uv run --locked --offline pytest --cov --cov-branch -W error
uv run --locked --offline ruff format --check .
uv run --locked --offline ruff check .
uv run tools/transformers_mla_fixture.py --check
```

The last command executes upstream again. It may need network access to install its isolated environment and
targets the recorded dependency versions. A version/backend mismatch is reported, not silently accepted. The
ordinary acceptance commands consume the stored fixtures and do not require PyTorch, Transformers or a GPU.

Raw inputs, outputs, source pins and backend observations are in
[the core fixture](../../reference/tests/fixtures/transformers_mla_770e4c40_core.json) and
[the GLM-width fixture](../../reference/tests/fixtures/transformers_mla_770e4c40_glm_width.json).

## Reviewed artifact snapshot

SHA-256, with paths relative to `reference/`:

```text
6ba8eeb4733b386cb707ef2b8463e40f8a41d46c2d7d1a92528b7d1901728c64  contracts/mla-attention.md
dd5fd4e4c1bb040281aee280cc3bace092b5ff5b6492557ca7c4686f4b6e896a  src/tq_reference/mla_attention.py
541f4e721a50dbe40b49c2b53f1a6b316bc27c531ed82e96d94364265b119e5f  tests/test_mla_acceptance.py
c4573b9fa15404b34ee402144721018c646ba0d4ae83a03730b5c7165bc2cf8b  tests/test_mla_upstream_acceptance.py
25368a1deeade98011cd7c078c0f96ff044a2bd3be211d6dc3ad58cf11ea49bf  tests/test_ref001_mla_core.py
dcff61e35e968141379f96cfed662ea0461086f0bc475f6acfa98c901e63b2f8  tests/test_ref001_upstream_fixture_smoke.py
942a804560b6edf76b9236cefb202f509501405d89b2ffc1f9a9ca4f5a74e7b8  tools/transformers_mla_fixture.py
cfae81a7f9af93489f95cba862aae166789230c02379adfef17340cb397fef1c  tests/fixtures/transformers_mla_770e4c40_core.json
758031ee1a70bcb1c119a50531d97cb9db998666646e7d665f102e6aab9fc4ac  tests/fixtures/transformers_mla_770e4c40_glm_width.json
```

This report establishes no claim about Rust kernels, GPU performance, production model quality, serving capacity
or superiority over another engine. Later SMLA and full-model gates remain required.
