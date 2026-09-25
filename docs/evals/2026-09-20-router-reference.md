# FP32 router CPU reference review

Gate date: 20 Sep 2026. **Decision: accepted for REF-006, the single-group FP32 router.**
This completes the P0 references in Phase 0 step 0.2: REF-001, REF-002, REF-003, REF-004 and
REF-006. REF-005 remains P1 until indexer work. The next work is the shared Philox stream and
small Rust workspace in steps 0.3–0.4; no Rust or GPU gate is accepted by this result.

## Scope and numerical decision

- Contract: [single-group router](../../reference/contracts/router.md), approved before implementation.
  Code: `reference/src/tq_reference/router.py`, 113 lines and 60 executable statements.
- Projection inputs are cast before FP32 arithmetic. Correction bias affects selection only.
  The reference returns increasing expert IDs, uses that order for its FP32 normalization sum,
  adds the FP32 epsilon before division and applies the 2.5 factor after division.
- Independent tests: `test_router_acceptance.py` and `test_router_upstream_acceptance.py`.
  Their scalar Decimal oracle does not call NumPy exp, matrix multiplication or reductions.
  Curated projection inputs have exact dyadic sums; separate tests expose input-cast mistakes.
- External reference: the actual `Glm5NextTextTopkRouter.forward` from
  [Transformers 770e4c40](https://github.com/huggingface/transformers/blob/770e4c40d0436082a52dc380f07a9d3f389c99d4/src/transformers/models/glm5_next/modeling_glm5_next.py),
  with one group, normalization enabled and factor 2.5. No upstream source is stored here.
- Ordinary environment: macOS ARM64, Python 3.12.4, NumPy 2.3.3, pytest 8.4.2, pytest-cov 7.0.0,
  Ruff 0.13.1. Upstream uses the existing isolated Torch 2.8.0 and Transformers 5.18.0.dev0 at
  the pinned commit. All resolved dependency versions are recorded in the fixture.
- Revision: uncommitted working tree based on `d7f25560d4767e72eb6542b1d9e22489f3a58a45`.

**The earlier REF-006 `1e-12` weight gate was corrected before implementation.** Two independent
investigations found that native FP32 sigmoid and reductions differ across numerical libraries.
The lead's eight-expert probe had identical logits and selected sets but weight relative L2
error `1.7748198117999202e-7`. Its inputs and captured comparison vector are retained in the
contract and lead tests. Requiring `1e-12` would effectively require reproducing private backend
rounding details. [PyTorch documents this numerical limitation](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html).

The fixed replacement is `gamma(2*K+16)`, with FP32 unit roundoff `u=2^-24`: about `1.91e-6`
for top-8. It checks each nonzero component, every token and the global relative L2 error;
expected zeros must remain exactly zero. The contract explains the operation budget and its
limits. This is a qualification budget for curated inputs, not a universal error theorem.
Exact projection and exact selected sets are still required on these cases; cutoff gaps must
exceed the contract's FP32 margin. Other reference, GPU and model gates are unchanged.

## Independent verification

| Check | Result |
|---|---|
| Lead core acceptance | 162 passed; all 60 statements and 30 branches covered |
| Lead upstream acceptance | 29 passed over ten recorded cases |
| All independent reference suites | 682 passed; 408 statements and 134 branches, 100% |
| Full suite, warnings treated as errors | 931 passed; 100% package line and branch coverage |
| Fresh generator `--check` after corrections | Ten cases and all 14 provenance fields match; strict root comparison |
| Separate lead reproduction without importing the generator | All ten raw output triples reproduced bit for bit; original forward entered once per case |
| Deliberate semantic mutations of the actual implementation | All 14 caught by lead acceptance tests |
| Independent generator source, observer and failure-path probes | All 12 checks pass |
| Format, lint, file/function caps, content scan, guard audit and diff check | Clean |
| Earlier accepted artifacts and historical evaluation reports | All 27 files unchanged byte for byte |

The 14 mutations included omitted bias, bias leaking into weights, omitted epsilon/scaling,
normalizing all experts, FP64 sigmoid, casting after projection, the wrong tie rule, swapped
weight/ID pairing, fused normalization, pairwise or reversed reduction, FP64 selection and
repairing an overflowing projection. These were temporary review experiments, not edits to
the production reference. They are additional evidence beyond the 931 ordinary test cases.

The generator probes rejected an altered source hash, a lookalike callable from the wrong file,
missing evidence of the original forward, changed outputs/provenance/call counts, and extra or
missing root fields. The observer preserves an existing profile hook on normal and exceptional
exits. `--help` exits before running upstream. Source review also corrected inaccurate input
provenance, a return annotation and the initially incomplete strict comparison. All recorded
case inputs, configurations and raw outputs stayed unchanged through those corrections.

## Numerical results

All ten cases have identical projected logits and expert sets. We pair weights by expert ID;
the raw upstream slot order is retained in the fixture. The lead measured:

| Case | T / H / E / K | Largest token relative L2 | Largest component relative error |
|---|---|---:|---:|
| Dense dyadic projection | 4 / 4 / 12 / 8 | 8.885e-8 | 1.067e-7 |
| All GLM experts | 2 / 4 / 288 / 8 | 1.216e-7 | 1.893e-7 |
| Bias changes selection | 1 / 1 / 16 / 8 | 1.666e-7 | 2.119e-7 |
| Cast-sensitive float64 inputs | 1 / 2 / 12 / 8 | 0 | 0 |
| Epsilon-dominated scores | 1 / 1 / 16 / 8 | 0 | 0 |
| Zero negative tail | 1 / 1 / 16 / 8 | Exact zeros | Exact zeros |
| Positive saturation | 1 / 1 / 16 / 8 | 0 | 0 |
| Recorded rounding discrepancy | 1 / 1 / 8 / 8 | 1.775e-7 | 1.831e-7 |
| Subnormal sigmoid scores | 1 / 1 / 16 / 8 | 0 | 0 |
| Empty tokens | 0 / 4 / 12 / 8 | No tokens | No tokens |

The largest global relative L2 is `1.775e-7`; the largest weight difference is three FP32 ULPs.
Every component also passes the scalar oracle. Zero norms are handled as exact-zero comparisons,
not assigned an artificial relative error. An independent core case covers hidden width 4096
with eight experts; the upstream 288-expert case intentionally uses a narrow hidden dimension.
These are small synthetic tests, not a full checkpoint or a full-model tensor-shape benchmark.

## Reproduction and limits

From the repository root, after installing `uv`:

```sh
cd reference
uv sync --locked
uv run --locked --offline pytest tests/test_nvfp4_acceptance.py tests/test_mla_acceptance.py \
  tests/test_mla_upstream_acceptance.py tests/test_moe_acceptance.py \
  tests/test_moe_upstream_acceptance.py tests/test_router_acceptance.py \
  tests/test_router_upstream_acceptance.py --cov --cov-branch -W error
uv run --locked --offline pytest --cov --cov-branch -W error
uv run --locked --offline ruff format --check .
uv run --locked --offline ruff check .
uv run tools/transformers_router_fixture.py --check
```

The final command runs the pinned upstream implementation and verifies sources and observer
restoration. Its isolated environment may require network access on first use. Exact fixture
regeneration targets its recorded environment; ordinary acceptance uses NumPy and the stored
[fixture](../../reference/tests/fixtures/transformers_router_770e4c40.json), which is 94,776 bytes.
Coverage applies to `reference/src/tq_reference`, without exclusions. It does not measure the
generator, external libraries, GPU kernels or a runtime that has not yet been built.

The single-group scope matches this pinned model configuration. General grouped routing remains
unsupported. The canonical local tie rule is tested, while ties/near-ties carry no cross-library
selection promise. Negative-tail zeros match the pinned CPU production path; they are not a
claim about every CUDA sigmoid implementation. No GPU, model weights, inference performance,
quality, concurrency or advantage over vLLM/SGLang is established by this CPU gate.

## Reviewed artifact snapshot

SHA-256, with paths relative to `reference/`:

```text
dcd8064bbc2b2ff2f60a349e6539df5f1e40db7e6e8d7d4697742d23fee68dc1  contracts/router.md
b7c1bbc3abfd4562a9fd1f5b5e045afbdd719fe9ca261fa8744a820c81223e63  src/tq_reference/router.py
d32f389e51a46fdb0bcecd02a56b2b14bcefb241e23fb1c55b0b469c602afdf5  tests/test_router_acceptance.py
659042d430b7468feeccd454d1edc636788f1d99e2f82812fa0d4fa940ab4515  tests/test_router_upstream_acceptance.py
1dedbe186964879edcf5a0527b045db065a35c79f2fc349fdfb8c4c2971d3fbb  tests/test_ref006_router_core.py
082787a3f60562ffc751b51ff218865d4a0e5247734e217261330e710b0cb5b9  tests/test_ref006_upstream_fixture_smoke.py
759c0dbc110db13660096a4805e7dd4d9b48e802a679a24d546549ce2ab16c73  tools/transformers_router_fixture.py
75367abd2482301549d49b7e44cca928d14b39fba0a2f2e2d9cf2db2974c8944  tests/fixtures/transformers_router_770e4c40.json
```
