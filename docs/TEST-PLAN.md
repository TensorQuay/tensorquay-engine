# TensorQuay Engine: Test Plan

Status: **draft v0.3 for PM review** (19 Sep 2026). v0.3 applies the adversarial review
(`reviews/2026-09-19-adversarial-review.md`; response in `reviews/2026-09-19-adversarial-review-response.md`). The
technical lead's follow-up (`reviews/2026-09-19-v0.3-follow-up.md`) decided P1–P3 and P5–P7 and did **not** approve P4
(see the response). Items still marked **[Pending]** need a future contract or evidence.
Before v0.3: v0.2 fixes the review findings (`reviews/2026-09-19-design-review.md`
#1–#3, #6, #8–#24, #27–#28, #32–#33, #38–#39) and adds practice from FlashAttention, FlashMLA, FLA and FlashInfer. Tests
are written **before** kernels. Priorities: **P0** is a gate requirement, **P1** is needed before a public release,
**P2** is nice to have. All IDs are prefixed (for example `SMLA-E-009`), because the manifest (DEV-GUIDELINES G-08)
depends on them.

## 1. Principles
1. **Every test has an oracle,** and each oracle is itself checked (§3).
2. **Tolerances come from a baseline, not from guessing.** A kernel may be at most 2× less accurate than a plain
   implementation at the same precision (the FlashAttention rule). Index semantics are checked **exactly** with probes,
   because aggregate tolerances can't see a dropped or duplicated index.
3. **Edge cases are first-class:** boundaries, invalid input, NaN containment, 64-bit offsets, graph replay and
   determinism.
4. **Performance is tested like correctness:** a fixed protocol, cold L2, paired A/B runs, and **same-card baselines**
   from the best existing kernels.
5. **Reproducible:** a counter-based RNG (Philox4x32-10) produces identical input streams in Rust and Python. The
   environment is captured and results are stored as JSON.

## 1a. Where tests run
| Where | Tests |
|---|---|
| **Mac (free, every change)** | REF-001…006 (float64 CPU); `tq-core` contract validation of shapes, strides and alignment (SMLA-E-011/E-012, MOE-E-008/E-009 host part); `tq-sched` property tests; `tq-server` API tests with a fake engine; harness maths |
| **CI Linux, no GPU** | Build; **compile every kernel specialisation for sm_120** (G-11); resource gate (shared memory ≤ 99 KB per block, spills) |
| **Remote RTX 5090 (development)** | All SMLA/MOE correctness and edge tests, sanitizers, H-001…008, tuning. Device-side range and duplicate checks run here (debug builds set a device error flag) |
| **Remote RTX PRO 6000 (gate)** | The full kernel suite plus gate performance numbers. Max-Q (300 W) if rentable; otherwise log the power limit and apply a margin |
| **2 × RTX PRO 6000 (end to end; PM rule)** | Model golden tests, MTP, QUAL, SYS, API, soak, WL-001 agent sessions |

The current CPU-only checks are specified in [CPU-CHECKS](CPU-CHECKS.md). Their local slice was accepted
on 20 Sep 2026: 246 new independent cases, exact coverage of both new Python modules, and 12 deliberate
faults caught; see the [evaluation](evals/2026-09-20-cpu-tooling.md). `tests/manifest.toml` maps only the
24 accepted CPU obligations to runnable tests. This mapping does not certify the GPU portions of
the cases below. The Linux workflow is prepared; hosted execution and G-11/G-12 remain unverified.

## 2. Oracles, baselines and pass rules
**Oracles:**
- **O1-alg: float64 algebraic reference.** Float64 everywhere, with **no** production dtype casts.
  - It supplies `y*` for every **kernel accuracy test** (SMLA, MOE and later kernels).
  - It is also used for algebraic-equivalence checks such as REF-001.
- **O1-prod: production-semantics reference,** written from `transformers` `modeling_glm5_next.py` (commit
  770e4c40d0). **It copies every dtype cast the reference makes on purpose:**
  - FP32 RMSNorm (lines 77–81);
  - FP32 softmax (eager path, line 1094);
  - FP32 router (line 161);
  - FP32 indexer scores (lines 863, 867);
  - FP32 mHC and KDA.

  It supplies `y*` for model-level golden tests, and for selections that depend on those casts (router and indexer:
  REF-005, REF-006).
- **Each test names its oracle.** Tolerances stated for float64-vs-float64 comparisons never apply across an FP32
  cast.
- **O2: the `transformers` module itself,** used to validate O1-alg and O1-prod.
- **O3: an official runtime.** vLLM on sm_120 once its PRs merge; before that, the official FP8 checkpoint on a
  supported GPU. Used for end-to-end and MTP.

**Baseline** (the same maths in storage precision; this fixes where rounding happens):
- **SMLA:** scores accumulated in FP32 → softmax in FP32 → P rounded to BF16 → P·V accumulated in FP32 → output rounded
  to BF16.
- **MOE:** dequant = e2m1 × e4m3, which is exact in BF16. BF16 × BF16 with FP32 accumulation. The f32 global scale is
  applied to the FP32 accumulator. g and u are rounded to BF16. h = silu(min(g, 10)) · clamp(u, ±10) in FP32, rounded to
  BF16. y is accumulated over top-8 in FP32 and rounded once.

**NVFP4 codec.** Pinned to NVIDIA ModelOpt `qtensor/nvfp4_tensor.py` at commit `b311c054`; REF-003 and REF-004
cross-check against it.
- **Input:** a 2-D float32 `[rows, K]` tensor with every weight finite. A NaN or ±Inf **source** weight is
  **rejected** with an error, and nothing is exported. `G` must be the integer 16 or 32 (not a float or bool), and
  `K` a multiple of it.
- **Global decode scale `g`:** f32, one per expert tensor, and W13 gate and up share one.
  - Non-zero tensor: `g = tensor_amax / (6 × 448)`.
- **Block scale `s_b`:** E4M3, one per G values along K.
  - `r = block_amax / (6 × g)` in float32, then clamped to [2⁻⁹, 448] and cast to E4M3 (round to nearest, ties to
    even).
  - A **computed `r` of 0 is set to 1.0 before the clamp** (the pinned recipe's rule). That covers a zero block inside
    a non-zero tensor, and also a tiny non-zero block whose ratio underflows in float32. Such a block gets
    `s_b = 1.0`, not the 2⁻⁹ clamp.
- **Codes:** `q = E2M1(w / (s_b × g))`, using `s_b` **after** its E4M3 rounding, with the pinned recipe's rounding
  (ties to the even code) and saturation.
  - A normalised value that overflows to ±∞ (possible with a small supplied `g`) saturates to code 7 or 15, as
    upstream. It is not treated as a non-finite source weight.
  - The sign bit is set only when the normalised value is < 0, so a value that underflows to −0.0 gets code 0.
- **Decode** (`dequantize`): `w = e2m1(q) × (s_b × g)`, with the product rounded to float32 first, as upstream.
  - Upstream's lookup decodes **both** zero codes (0 and 8) to +0.0.
  - The format oracle (`e2m1_decode`, `decode_exact`, REF-003) keeps −0.0 for code 8.
  - `dequantize` rejects malformed encoded tensors: dtype, shape, group, NaN or negative scale codes, and an invalid
    global scale.
- **Representability boundary** (approved by the technical lead): `g`, derived or supplied, must be finite and > 0
  **after its float32 cast**. **Subnormal scales are accepted.**
  - A tensor is rejected only when an **actual block step** `E4M3(s_b) × g` underflows to zero in float32. The check
    runs on the float32 product, before normalisation, so 0 / 0 can't occur.
  - This check is an intentional safety deviation: upstream has none.
  - A zero block's step is `1.0 × g`, so it stays usable with any positive `g`.
- **API types:**
  - `decode_exact` takes integer codes (Python or NumPy, not bool); anything else is rejected.
  - The element-wise codecs return an `ndarray` for every input shape, including 0-d.
- **All-zero tensor (canonical encoding; an intentional deviation):** `tensor_amax = 0` would give `g = 0`, and in the
  pinned helper `0 / (6 × 0)` is NaN, which its "zero → 1.0" rule doesn't catch. So we define all codes = 0, every
  `s_b = 1.0` and `g = 1.0`, which decodes to exact zeros. We don't require byte identity with the upstream helper for
  this degenerate case. The ordinary finite, non-zero path must match it byte for byte (REF-004).
- **Layout:** the first value goes in the low nibble. Scales are `[rows, K / G]` row-major before any kernel-specific
  swizzle, and each kernel documents its swizzle.
- **Formats:**
  - **`NVFP4-G16`** is standard NVFP4 (one E4M3 scale per 16 values), the format public kernels prepare.
  - **`NVFP4-G32-E4M3`** is a distinct TensorQuay/ModelOpt W4A16 variant (one E4M3 scale per 32 values). It is not
    native to the tensor cores and is not a standard baseline format.

**Metrics:** computed **per row** (token × head for SMLA, token for MOE) **and** globally:
- `rel_l2`, `max_abs`, `cosine`;
- **anomaly positions** (NaN/Inf) must match the test's oracle exactly (the FlashMLA rule);
- for recurrent kernels (Phase 1): `RMSE / RMS(ref)` below a per-test ratio (the FLA rule).

**Pass rule** (per row and global):
- `max_abs(kernel) ≤ 2 × max_abs(baseline) + 1e-6 · max|y*|`;
- `rel_l2(kernel) ≤ 2 × rel_l2(baseline) + 1e-4`;
- no unexpected anomalies.

**lse:** relative error ≤ 1e-5, or ≤ 2 × the baseline's.

**Seeds and data:** 5 seeds per test.
- Inputs are **generated where they run**, from (test ID, seed), by the shared Philox stream.
- O1 runs on the pod: float64 on CPU, or on GPU for large cases.
- Only small cases (< 50 MB in total) are committed. Manifests store SHA-256 of inputs and expected outputs.

## 2a. Shared test generator (RNG)
The [Philox contract](../reference/contracts/philox.md) freezes Philox4x32-10, numeric stream IDs,
word offsets, exhaustion behavior and exact float32 conversion before implementation. These tests
qualify test-data generation, not production sampling. Existing reference fixtures stay unchanged.

| ID | P | Test | Pass |
|---|---|---|---|
| RNG-001 | P0 | Official known answers, high-bit inputs, scalar oracle and inverse rounds | Exact words |
| RNG-002 | P0 | Five seeds, stream domains, lane offsets, counter carry and final legal word | Exact words |
| RNG-003 | P0 | Partition/overlap invariance, empty requests, typed inputs and atomic range rejection | Contract holds |
| RNG-004 | P0 | Uniform conversion, endpoints, low-byte truncation and Python array layouts | Exact float32 bits |
| RNG-005 | P0 | Fixture/input/output hashes; execute Rust and Python; fresh pinned Random123 check | All agree; missing tools fail |

RNG-001…005 were accepted on 20 Sep 2026. The [evaluation](evals/2026-09-20-philox-foundation.md)
records the upstream revision, independent tests, exact coverage scopes and reproduction commands.

## 3. Reference validation (REF), runs on the Mac
| ID | P | Test | Pass |
|---|---|---|---|
| REF-001 | P0 | **Oracle O1-alg.** Attention **core only**: absorbed (q·W_UK → latent attention → ·W_UV) vs O2 expanded, both fed the same post-norm `q_resid`/`k_pass` and the same top-k mask. O2 runs `attn_implementation="sdpa"`, which keeps float64 (the eager path casts softmax to FP32 at line 1094 and is not used). If the pinned SDPA path turns out to cast, compare against a float64 re-implementation of the expanded form instead | rel_l2 ≤ 1e-12 |
| REF-002 | P0 | **Oracle O1-alg.** MoE **expert path**: O1 vs O2, given O2's `topk_indices`/`topk_weights` (the router is compared separately with FP32 semantics, REF-006) | rel_l2 ≤ 1e-12 |
| REF-003 | P0 | NVFP4 decode table: all 16 E2M1 codes × E4M3 scale samples × global scales including **non-unit, non-power-of-two** values, vs a table from the OCP MX / NVIDIA spec | Exact in float64 |
| REF-004 | P0 | Our quantizer implements the §2 NVFP4 codec for G = 16 and G = 32. **Finite, non-zero tensors with representable scales and block steps:** exported bytes (codes, block scales, global scale) and dequantized values **match the pinned ModelOpt recipe byte for byte**, including zero blocks inside non-zero tensors, saturation, non-unit global scales and W13's shared global scale. **Separately:** an all-zero tensor gets the canonical encoding (zero codes, unit scales) and decodes to exact zeros; non-finite input or an unrepresentable scale/step is rejected as specified in §2 | Identical bytes on the ordinary path; canonical bytes for all-zero; explicit boundary rejection |
| REF-005 | P1 | **Oracle O1-prod.** Indexer (k-pool 4, tail, top-k; **no rotation**, because `qk_rope_head_dim = 0`) vs O2; inputs have score gaps > 1e-5 relative | Identical index sets |
| REF-006 | P0 | **Oracle O1-prod.** Single-group router: sigmoid, correction bias for selection only, FP32, +1e-20 normalisation, × 2.5; separated selection scores as defined in the contract | Identical top-8 sets; exact projection on curated inputs; weights within the fixed FP32 budget in the [router contract](../reference/contracts/router.md) |
| REF-007 | P1 | MTP layer (layer 45) vs the pinned MTP oracle: vLLM `vllm/models/glm5next/nvidia/mtp.py` at commit `98ed0856` (`transformers` skips layer 45 on load) | Per Phase 1b |

The [REF-001 CPU core contract](../reference/contracts/mla-attention.md) fixes the API, mask and overflow semantics,
independent Decimal oracle and per-output tolerances before implementation. Its tests do not qualify the GPU SMLA
contract below. REF-001 was accepted on 20 Sep 2026 after independent upstream reproduction;
the [evaluation report](evals/2026-09-20-mla-reference.md) records the results and remaining gates.

The [REF-002 routed-expert contract](../reference/contracts/moe-experts.md) fixes the explicit clamp limit,
gate/up layout, routing-weight boundary, selected-bank validation and overflow semantics. It was accepted on
20 Sep 2026 after independent upstream reproduction; see the [evaluation report](evals/2026-09-20-moe-reference.md).
The fixture's upstream router supplies inputs only; REF-006 is evaluated separately below.

The [REF-006 router contract](../reference/contracts/router.md) fixes FP32 casts, tie handling,
slot ordering and the numerical gate before implementation. On 20 Sep 2026 the technical lead
corrected the earlier `1e-12` FP32 weight threshold: independently measured library/reduction
rounding already exceeds it on identical logits. The replacement is `gamma(2*K+16)` with
`u=2^-24` (about `1.91e-6` for top-8), checked per component, per token and globally; exact zeros
and exact expert sets on well-separated inputs remain required. The contract records the probe,
budget rationale and limits. This changes REF-006 only. The reference was accepted after independent reproduction
on 20 Sep 2026; the [evaluation report](evals/2026-09-20-router-reference.md) records the results and limits.

## 4. SMLA: sparse-MLA decode (the kernel missing on sm_120)
**Contract**, per GPU. Validated host-side by `SmlaParams::new` in `tq-core`.
The [Rust host contract](HOST-CONTRACTS.md) defines the concrete metadata API, canonical empty strides, byte spans,
alias policy and borrowed query plan for §§4–5. Its CPU slice was accepted on 20 Sep 2026;
the [evaluation](evals/2026-09-20-host-contracts.md) distinguishes host evidence from the pending GPU cases.
- **Inputs:**
  - `q`: bf16 `[T, H, 512]`, absorbed.
  - `kv`: a paged pool `[pages, 64, row]`. `row` is **BF16 × 512 (1024 B)** or **FP8 E4M3 × 512 plus 4 × f32 scales
    (528 B)**. The runtime stride is ≥ the row size and 16 B-aligned. The rows are the latent after `kv_a_layernorm`.
    **FP8 row layout:** bytes 0–511 hold 512 E4M3 values. Bytes 512–527 hold 4 f32 scales, one per 128-value tile
    (value = fp8 × scale[i / 128]). This is FlashInfer's GLM53_NOPE payload and the prefix of vLLM's `fp8_ds_mla` row
    (flashinfer #5075).
  - `page_table`: i32 `[S, max_pages]`.
  - `seq_id`: i32 `[T]`.
  - `q_pos`: i32 `[T]`, the position of each query token.
  - `kv_len`: i32 `[S]`.
  - `idx`: i32 `[T, K]` with K ≤ 2051. −1 means empty.
  - `k_len`: i32 `[T]`. Entries at or after `k_len` are ignored, even when they are stale positive values.
  - `scale` = **256^-0.5**. It is a validated parameter, fixed for GLM, so later families only change validation.
  - Shapes: H ∈ {32, 64}. **At most 4 query tokens per sequence** (1 + up to 3 MTP drafts), also a validated
    parameter.
  - **Decode only.** Prefill chunks use the separate SMLA-PF contract (§9, Phase 1a).
- **Validity:** an entry j is valid if and only if `j < k_len[t]` **and** `0 ≤ idx ≤ q_pos[t]` **and** `idx < kv_len`.
  Invalid entries are never dereferenced from the allocatable pool. A masked lane either uses a masked load (no
  memory access), or reads a **reserved zero row outside the allocatable pool** (zero data, scales 1.0). It never
  clamps to page 0 / row 0, which SMLA-E-009 poisons.
- **Outputs:**
  - `out` bf16 `[T, H, 512]` = Σ softmax · row, accumulated in FP32.
  - `lse` f32 `[T, H]`, the natural log-sum-exp.
  - With no valid entry: `out = 0` and `lse = −inf`.
- **Execution:**
  - **The launch grid does not depend on the data**, so the kernel can be captured in a graph and replayed with new
    `idx`, `kv_len` and `q_pos`.
  - Split-K uses a fixed split size.
  - Bitwise identical run to run (D0).
  - 64-bit byte offsets.

### 4.1 Correctness (SMLA-C)
| ID | P | Case |
|---|---|---|
| SMLA-C-001 | P0 | T=1, H=32, kv_len 4096, K=2048 random unique indices, BF16 rows |
| SMLA-C-002 | P0 | H=64 |
| SMLA-C-003 | P0 | 16 sequences, kv_len ∈ {1, 63, 64, 65, 2047, 4096, 32768, 131072} |
| SMLA-C-004 | P0 | MTP: 2, 3 and 4 query tokens per sequence, each with its own indices and `q_pos` |
| SMLA-C-005 | P0 | Paged pool with shuffled pages (the page table is not the identity) |
| SMLA-C-006 | P0 | `lse` vs O1 (relative rule, §2) |
| SMLA-C-007 | P0 | Split-K merged result vs a single-split run |
| SMLA-C-008 | P0 | **Scale guard:** an input where 512^-0.5 fails and 256^-0.5 passes |
| SMLA-C-009 | P1 | Real data: layer-3 weights from the BF16 release, a real prompt, O1 indexer indices |
| SMLA-C-010 | P0 | **Causality:** the indices for query token 0 include the position of token 1 (> q_pos[0]), which must be ignored |
| SMLA-C-011 | P0 | **Exact index-set probe:** q = 0 so every weight is 1/n; row i = the one-hot eᵢ (kv_len ≤ 512, shifted windows for larger positions). Pass if: the support of `out` equals the valid set **exactly**, each value equals 1/n to BF16 rounding, and `lse = ln n` (±1e-6) |
| SMLA-C-012 | P0 | **FP8 528 B rows:** C-001…C-005 and C-011 again, with stride ∈ {528, 544}; O1 uses the dequantized rows |
| SMLA-C-013 | P0 | **Kernel capability:** identical index rows across a sequence's query tokens must work, with causality still from each token's `q_pos`. This is **not** a model-semantics test; which producer fills each row is specified in §9 (MTP) |

### 4.2 Edge cases (SMLA-E); the index-semantics cases use the C-011 probe
| ID | P | Case | Pass |
|---|---|---|---|
| SMLA-E-001 | P0 | All −1, or k_len = 0 | out == 0, lse == −inf, no NaN |
| SMLA-E-002 | P0 | Exactly one valid entry | out == that row, bitwise (BF16), or the dequantized row rounded (FP8) |
| SMLA-E-003 | P0 | Valid-count sweep: 1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049, 2051 | Probe exact |
| SMLA-E-004 | P0 | −1 leading, trailing and interleaved | Probe exact |
| SMLA-E-005 | P0 | Out-of-range values: kv_len, q_pos + 1, i32::MAX, −2, i32::MIN | Ignored; memcheck clean |
| SMLA-E-006 | P0 | Unsorted vs sorted indices | §2 rule |
| SMLA-E-007 | P0 | **64-bit:** a pool over 8 GB with indices in the last pages; kv_len = 1,048,576 (each tensor dimension < 2^31) | §2; memcheck clean |
| SMLA-E-008 | P0 | kv_len = 1; the page edges 64 and 65; a **kv_len = 0 sequence inside a batch** | Probe exact; empty rows per E-001 |
| SMLA-E-009 | P0 | **NaN containment:** NaN in unselected rows, in **page 0 row 0**, in the unselected rows' FP8 scales, and at every position a clamp could land on | Output bitwise equal to the clean run |
| SMLA-E-010 | P0 | Numerics: logits ±1e5; all-equal logits; one dominant logit; subnormal inputs | §2; lse relative rule; no NaN |
| SMLA-E-011 | P0 | T = 0 | No launch, no error |
| SMLA-E-012 | P0 | Host rejection: H=48, D≠512, K>2051, stride < row size or not 16 B-aligned, > 4 query tokens per sequence | `ContractError` before launch |
| SMLA-E-013 | P0 | Determinism: 100 runs | Bitwise identical |
| SMLA-E-014 | P1 | Two streams running at once | Each correct |
| SMLA-E-015 | P0 | Capacity: T = 64 (16 sequences × 4); stress at T = 256 is P1 | §2 |
| SMLA-E-016 | P1 | Duplicate indices (debug build) | Device error flag is set |
| SMLA-E-017 | P0 | **Empty splits:** all valid entries in the first split only, then in the last split only | No NaN (no −inf − −inf); §2 |
| SMLA-E-018 | P0 | **Graph replay:** capture once, replay with new idx, kv_len, q_pos and k_len | §2 for every replay |
| SMLA-E-019 | P0 | **Stale padding:** positive, in-range values after `k_len` | Ignored; memcheck clean |
| SMLA-E-020 | P1 | Non-contiguous q, idx and out (padded strides) | §2 |

### 4.3 Performance (SMLA-P), protocol in §8
| ID | P | Case | Metric and gate |
|---|---|---|---|
| SMLA-P-001 | P0 | T sequences × 1 token, T ∈ {1, 2, 4, 8, 16, 32, 64}; H=32; K=2051; kv_len 32K and 128K; FP8 and BF16 rows; **timing includes the merge** | Time; read GB/s ÷ H-002 peak. **Gate: ≥ 0.9 × the same-card baseline (SMLA-P-004) at T ≥ 8, FP8 rows** |
| SMLA-P-002 | P1 | Same as P-001 with H=64 | Reported |
| SMLA-P-003 | P1 | kv_len < 2048 (short context), latency | Reported |
| SMLA-P-004 | P0 | Baselines on the same pod and session: **FlashInfer `sparse_mla_sm120` GLM53_NOPE (#5075)** (primary; H=32 runs on its runtime-H path with 16-head tiles, so record which path ran), the vLLM Triton SM12x fallback (#54929) if it builds, and a PyTorch gather reference | Numbers for the gate |
| SMLA-P-005 | P0 | **Gather microbenchmark**, run first, on the **GLM row formats** (528 B rows at stride 544, i.e. 32-byte aligned; 1 KiB rows).<br>**Fixed setup:** indices uniform random and unique within each query over a pool ≥ 4 × L2; T × H concurrency as in SMLA-P-001; cold per §8.<br>**Accounting (whole launch):**<br>• *Useful read bytes* = Σ over query tokens of (valid indices × row bytes), each (token, row) counted **once**, however many heads reuse it, plus the q and index bytes.<br>• *Written bytes* (out, lse) are reported separately.<br>• *Predicted footprint* = the 32-byte sectors the addresses touch, a prediction only (528 B at stride 544 → 17 sectors = 97.1 % useful; 1 KiB → 32 sectors = 100 %).<br>• *Measured traffic* = profiler counters for requested global loads, L2 sectors and DRAM bytes read (H-007). If counters aren't available, it is recorded as **unmeasured**, never inferred.<br>**Variants:** `load_ptr_tko`, `load_gather_scatter_view_tko`, and a **matched CUDA C++ gather** (benchmark tool only) with identical indices and accounting | Useful read bytes ÷ time ÷ H-002 read-only peak. **Target: ≥ 80 %** (decision P1).<br>If a cuTile variant misses it, the matched CUDA gather must have been run before the failure is attributed to the compiler.<br>If **both** miss it, the result is recorded as an **access-pattern or hardware-limit investigation**, not as evidence that changing the kernel language would fix it.<br>Future-family row widths (68 B, 288 B) are measured **report-only**, because of Phase 0 scope. That is not a claim they can't reach the target: a 288 B row is 9 sectors (100 %) when 32-byte aligned |
| SMLA-P-006 | P1 | Hot vs cold L2 | Both reported; the gate uses cold |
| SMLA-P-007 | P0 | Profile at T = 8 and 64 (Nsight Compute if H-007 allows it, otherwise Nsight Systems plus the roofline) | Report archived |
| SMLA-P-008 | P0 | **Tile and shared-memory sweep:** BN ∈ {16, 32, 64}; P·V with and without a D-split; shared memory read from the cubin | Every variant ≤ 99 KB per block; the best is chosen |
| SMLA-P-009 | — | **Moved to SMLA-PF-P-001 (§9).** Prefill needs its own contract; the decode contract allows ≤ 4 query tokens per sequence | — |

## 5. MOE: NVFP4 W4A16 grouped experts
**Contract**, per GPU:
- **Inputs:**
  - `x`: bf16 `[T, 4096]`.
  - `topk_ids`: i32 `[T, 8]`.
  - `topk_w`: f32 `[T, 8]` (normalised, × 2.5).
  - W13 `[E, 2I, 4096/2]` and W2 `[E, 4096, I/2]`: packed E2M1, E4M3 scales per G, and one f32 global scale per expert
    and tensor. **W13 gate and up share one global scale.**
  - E = 288. I = 1024 (TP=2) or 2048. Formats **`NVFP4-G16`** and **`NVFP4-G32-E4M3`** (§2), treated as distinct
    formats.
  - The gate/up row order follows the checkpoint and is documented.
  - **TP layout:** each shard's W13 is `[gate rows of this shard ; up rows of this shard]`. A naive row split of the
    fused tensor (all gate rows to shard 0, all up rows to shard 1) is wrong, and MOE-C-008 must catch it.
- **Output:** y bf16 `[T, 4096]`, with precision as in the §2 baseline. The shared expert is not included. Only
  selected experts are read.
- **Execution:** the grid is independent of the routing data (graph-replayable); bitwise D0; 64-bit offsets.
- **Checks:** the device checks expert-id range and duplicates in debug builds. The host checks shapes, G and
  alignment.

### 5.1 Correctness (MOE-C)
| ID | P | Case |
|---|---|---|
| MOE-C-001 | P0 | T=1, 8 distinct experts, G=16 and G=32, against O1 run on the **same dequantized weights** |
| MOE-C-002 | P0 | T ∈ {2, 4, 8, 16, 32, 64}, random routing |
| MOE-C-003 | P1 | Prefill T=4096 with all 288 experts active |
| MOE-C-004 | P0 | TP: y(shard 0) + y(shard 1) with I=1024 equals y with I=2048 |
| MOE-C-005 | P0 | Non-uniform routing weights, including 0 |
| MOE-C-006 | P0 | **Layout guard:** swapping gate and up breaks tolerance by far |
| MOE-C-007 | P0 | Dequant exactness: one-hot x through a test-only entry that shares the dequant helper. Power-of-two global scales: exact vs REF-003. Non-unit, non-power-of-two global scales: within FP32 rounding of the float64 value |
| MOE-C-008 | P0 | **Shard-layout guard:** shards built by a naive row split of the fused W13 must fail MOE-C-004; shards built per the TP-layout clause must pass |

### 5.2 Edge cases (MOE-E)
| ID | P | Case | Pass |
|---|---|---|---|
| MOE-E-001 | P0 | Experts with zero tokens, whose scales are NaN-poisoned | Output clean, which proves they were not read |
| MOE-E-002 | P0 | Hot expert: every token goes to the same 8 experts | §2 |
| MOE-E-003 | P0 | Maximum spread: T=36, each of the 288 experts used once | §2 |
| MOE-E-004 | P0 | **SwiGLU clamp:** g > 10; u beyond ±10; exactly ±10 | §2 |
| MOE-E-005 | P0 | Extremes: FP4 ±6, ±0, ±0.5; E4M3 scales at 448 and 2^-9 | §2; no overflow |
| MOE-E-006 | P0 | **NaN containment:** NaN E4M3 scales (0x7F/0xFF) and NaN global scale in unselected experts (E2M1 has no NaN code) | Bitwise equal to the clean run |
| MOE-E-007 | P0 | Zero rows in x | Exactly 0 |
| MOE-E-008 | P0 | T = 0 | No launch, no error |
| MOE-E-009 | P0 | Host: I or 4096 not a multiple of the tile, G ∉ {16, 32}. Device (debug): an id outside [0, 288), or a duplicate in a row | `ContractError` / device error flag |
| MOE-E-010 | P0 | Determinism: 100 runs | Bitwise identical |
| MOE-E-011 | P0 | **Graph replay** with new routing and x | §2 |
| MOE-E-012 | P0 | **Expert bank over 2 GiB** with the selected expert last (64-bit offsets; FlashInfer has had overflows here) | §2; memcheck clean |

### 5.3 Performance (MOE-P)
| ID | P | Case | Metric and gate |
|---|---|---|---|
| MOE-P-001 | P0 | Decode T ∈ {1, 2, 4, 8, 16, 32, 64}, TP=2 shard shapes, each format | Time, count of unique experts, bytes ÷ H-002 read peak (reported) |
| MOE-P-002 | P0 | Parity at **`NVFP4-G16`** (decision P2) against **vLLM Marlin NvFp4 W4A16** and **FlashInfer cuTile NVFP4 W4A16** (`prepare_cutile_nvfp4_weights`: E4M3 per 16, #5099), with commits and weight layouts pinned in the image manifest before the GPU session.<br>**A baseline is gate-eligible only if it:**<br>(1) passes the same MOE-C and MOE-E cases as our kernel, including the SwiGLU clamp edges (MOE-E-004), against the same oracle and pass rule (§2);<br>(2) uses the same values, routing, precision contract and GLM shapes (H = 4096, I = 1024 at TP=2, E = 288, top-8);<br>(3) is timed over the same operation boundary (both expert GEMMs, the clamped activation and the weighted combine), including any adapter or extra kernels it needs.<br>A runner with semantic differences (for example an unclamped activation) is reported as **context only** and cannot satisfy M5 | **Gate: ≥ 0.9 × the best gate-eligible baseline at T ≤ 64.** If no baseline is gate-eligible, P3 applies: provisional go only |
| MOE-P-005 | P0 | **`NVFP4-G32-E4M3` as a distinct format:** correctness (MOE-C and MOE-E at G32), bytes moved, time vs our own G16 kernel, bytes-normalised throughput. Quality is decided by ADR-0008 (KL, QUAL) | Reported only (decision P2). G32 has **no demonstrated parity and no full-model reference support**; any future G32 gate needs a compatible reference first (§9 EQV-001) |
| MOE-P-003 | P0 | Skewed (Zipf) routing vs uniform. Real router traces follow in Phase 1 | Reported |
| MOE-P-004 | P2 | Prefill T ∈ {512, 2048, 8192} | TFLOPS |

## 6. Harness self-tests (H), run before any measurement
| ID | P | Test | Pass |
|---|---|---|---|
| H-001 | P0 | Environment capture: GPU, SMs, L2, clocks, **power limit**, driver, CUDA, `tileiras` fingerprint, cutile-rs commit, NCCL version, Rust version | JSON written |
| H-002 | P0 | Peak bandwidth. A **read-only streaming kernel** (cuTile; bytes read ÷ time) gives the **read peak**, used as the roofline for read-dominated kernels (SMLA, MOE, gathers). A device-to-device copy is reported separately, with its bytes counted as read + written | Both stored |
| H-003 | P0 | L2 flush works: buffer = ½ L2; the flush writes 2 × L2 | Warm read ≥ 1.5× faster than cold |
| H-004 | P0 | Timer sanity: a ≥ 100 ms copy of known size timed with CUDA events vs the host clock | Within 5 % |
| H-005 | P0 | Deliberately broken kernels are caught: wrong scale, off-by-one index, masked lane reading slot 0 | Each fails its test |
| H-006 | P0 | **W4A16 dequant microkernel:** compiles, E2M1 → BF16 through `convert_tile`, otherwise bit operations; matches REF-003 | Exact; throughput reported |
| H-007 | P0 | Profiler availability (`ncu` counter permission in the container) | Chooses the SMLA-P-007 path |
| H-008 | P0 | **No JIT after warm-up:** compile counter unchanged across replays of every specialisation and graph bucket | Unchanged |
| H-009 | P0 | Measured BF16 tensor-core peak (FP32 accumulate). With H-002 it gives the **attainable roofline** = min(DRAM, compute) for each kernel's arithmetic intensity, which SMLA-P-001 reports | Stored |

## 7. Robustness tooling
- `compute-sanitizer` (memcheck, racecheck, initcheck) plus cuTile's `sanitize_memcheck` compile option, on every P0 C
  and E test. **P0: clean for our code.**
- **Triage for upstream faults.** A fault inside cutile-rs host code (for example #252) is logged with its upstream
  issue and suppressed only by a named, reviewed suppression. A fault in our code blocks.
- Compile-only resource gate (G-11): shared memory ≤ 99 KB per block; no local-memory spills in hot kernels.

## 8. Performance protocol
1. Run H-001…H-004 and H-007 first, in the same session.
2. **Timing:**
   - kernels in the µs range: CUDA-graph replay or CUPTI;
   - ≥ 100 µs: CUDA events.
   - Reuse `cutile::bench`.
3. **Every measured invocation starts cold.** Before each one, either:
   - evict L2 by writing ≥ 2 × L2 (≥ 256 MB on the PRO 6000); or
   - use an **independent input buffer set**, rotating over ≥ ⌈5 × L2 ÷ working set⌉ sets.

   Inside a graph replay, each captured invocation uses its own buffer set. The **working set** is the bytes actually
   accessed (selected rows, q, out, indices), not the pool size. Eviction work is excluded from kernel time. Baselines
   follow identical rules.
4. **Steady state** (warm, with realistic reuse) is measured and reported separately. Gates use the cold numbers.
5. ≥ 10 warm-up runs and ≥ 50 timed runs. Report median, p10, p90 and IQR.
6. Baselines run on the same pod in the same session. **Regressions use paired A/B** (`do_bench_paired`):
   - for each pair i, rᵢ = t_newᵢ / t_oldᵢ;
   - fail if median(r) > 1 + max(0.05, 3 × IQR(r)), where the IQR is computed on the dimensionless ratios.
7. Record the power limit. Gate numbers name the card edition.
8. Results go to `bench/results/<date>-<gpu>/*.json` with the environment.

## 9. Phase 1+ tests (outline; detailed before each phase)
- **KDA:**
  - KDA-C: chunked and recurrent paths vs O2 (RMSE ratio); state carried across chunks and steps; conv state; gate
    lower bound −5.0; FP32 vs BF16 state (ADR-0003).
  - **KDA-R-001, rollback:** after accepting k of n drafts, the state equals the state from recomputing k tokens.
  - **KDA-R-002, checkpoint restore:** the restored state gives the same next-step output as continuing without the
    restore.
- **SPEC (Mac, P0 when `tq-core` lands):**
  - SPEC-001: `parse_hf_config` plus `ModelSpec::validate` golden tests on the real `config.json` of GLM-5.3-Flash,
    DeepSeek-V4-Flash, DeepSeek-V4.1-Flash and a Qwen3.5 hybrid. Families we don't support are rejected with a clear
    error.
  - SPEC-002: a consumer placed before its producer is rejected.
- **CACHE-R (property tests):** for every cache kind, including the indexer tail state, `commit(accepted_len)` after
  drafts leaves the cache equal to replaying only the accepted tokens.
- **IDX (indexer); REF-005 becomes P0 when indexer work starts:**
  - k-pool grouping from the first valid token; tail for kv_len mod 4 ∈ {0, 1, 2, 3}; a budget of 512 pools; padding;
    `index_kpool` read from the config (never the code default of 16).
  - **No rotation for this checkpoint** (`qk_rope_head_dim = 0`), with a regression case that fails if RoPE is
    applied. Rotation is kept only for a future config with a non-zero rotary span.
  - Per-query pool visibility for multi-token (MTP) queries.
  - The indexer cache (ADR-0012) holds **completed pooled keys plus a per-sequence tail**: the unfinished pool's raw K
    and gate scores, ≤ kpool − 1 tokens, like vLLM's tail cache. It gives index sets identical to the HF layout.
  - **IDX-R (tail state):** prefill → decode, request resume from a prefix checkpoint, and draft rejection, each at all
    four pool offsets (kv_len mod 4 ∈ {0, 1, 2, 3}), with several sequences in one batch.
  - A replicated indexer and a head-split indexer give identical index sets (ADR-0005).
  - **Tie rule: higher score first; on ties, lower pool index first.** Tie-free inputs are compared as sets with O2, and
    the tie rule is unit-tested.
- **MHC:** vs O2 (not "rows sum to 1").
- **MTP** (oracle: vLLM `glm5next/nvidia/mtp.py` at commit `98ed0856`, REF-007):
  - **Target verification:** every query position (the target token and each draft) uses the target layers' own
    indexer output, per query.
  - **Drafting:** the MTP layer has its own indexer and index buffer. Draft step 0 computes top-k, and steps ≥ 1 reuse
    it (`index_share_for_mtp_iteration`). A target layer's indices are never reused.
  - **MTP-C:** sequence compaction of index rows, unequal draft depths per sequence, and rejection, checking that no
    stale row is used for another request. Accept length reported.
- **SAMPLE (rejection sampler), separate from determinism:**
  - It must preserve the target distribution: a statistical test over ≥ 10⁵ draws per case, with total-variation
    distance within the sampling bound.
  - Cases: zero-probability tokens; every acceptance length 0…d; the bonus token; temperature, top-p and top-k;
    interleaved requests.
  - **Sampler correctness is tested with identical supplied target and draft distributions** (unit level), separately
    from any model-forward difference.
  - **Greedy MTP on vs off:** these runs use **different execution partitions**: multi-token verification vs
    one-token recurrent decode. So D1's fixed-partition guarantee (DEV-GUIDELINES §1.3) does **not** make them bitwise
    equal. They are compared by agreement and KL per API-006. A bitwise claim needs the canonical-schedule contract,
    which is **[Pending]** before Phase 1.
  - **Sampled output is not promised token-equal with MTP on vs off.** If a coupled random stream is adopted later, it
    is specified and tested explicitly.
- **SMLA-PF (sparse prefill contract, Phase 1a; specified before implementation):**
  - Per-query causality inside a chunk; pooled-index visibility per query; chunk boundaries at any pool offset; a
    workspace budget.
  - SMLA-PF-P-001 (was SMLA-P-009): one sequence, a 4096-token chunk, K = 2051 per token. Time (it affects TTFT and
    M2).
- **Golden tests:**
  - each layer type vs O2 with real BF16 weights;
  - the first N layers vs O2 on CPU;
  - the full model vs O3 (official runtime, **same checkpoint files**): greedy agreement and per-token KL on 200
    prompts.
  - **KL definition:** KL(P_ref ‖ P_test) per position, teacher-forced on the reference tokens, over the full
    vocabulary. Computed in FP64 from FP32 logits, on generated positions only (prompt excluded). Reported as mean and
    p99.
  - **Two separate gates**, with the engine-equivalence gate kept distinct from the quantization-quality gate:
    1. **EQV-001, engine equivalence (a named future gate; [Pending] a defined contract and reference evidence).** Our
       engine vs a reference runner on the same checkpoint. **Prerequisites, completed and frozen before any engine
       result is tested:**
       - (a) select the reference runner (runtime and pinned commit) and the checkpoint format it actually loads;
       - (b) verify its loader and model semantics against O1-prod;
       - (c) define the noise-floor experiment exactly: which runs or variants, why they represent legitimate
         variation, and how many;
       - (d) aggregation over prompts and positions;
       - (e) handling of a zero or near-zero floor (a deterministic runner can give KL = 0);
       - (f) the pass threshold.

       No floor is assumed in these documents. For `NVFP4-G32-E4M3`, no complete official runner is known to load the
       format, so EQV-001 for G32 builds needs a runner or adapter with verified semantics first.
    2. **Quantization quality:** our W4 build vs the BF16 reference (KL, QUAL). This is a measured quality decision, not
       an allowance derived from FP8 vs BF16. It needs a **selected** BF16 acquisition plan (below). Listing options
       doesn't execute a reference run.
  - **The BF16 reference needs an approved plan before it gates anything.** The routed experts alone are **580.5 GiB
    in BF16** (43 MoE layers including MTP × 288 × 3 × 4096 × 2048 × 2 B), so it cannot run on the 2-GPU box.
    Options: an official runtime on large-memory data-centre GPUs; CPU teacher forcing on a ≥ 1 TB RAM host; or a
    pinned published reference artefact. **[Pending]:** one option must be **selected** (decision owner: the technical
    lead) before the quantization-quality gate can be evaluated.
- **QUAL** (paired, ≥ 3 runs, 95 % CI non-inferiority at −1.5 pts vs O3):
  - QUAL-001 SWE-bench Verified;
  - QUAL-002 Terminal-Bench 2.0;
  - QUAL-003 GSM8K;
  - QUAL-004 RULER 32K/128K;
  - QUAL-005 coding-agent tasks with held-out tests and a versioned public task manifest.
  - **Pre-registration:** task versions, seed pairing, the CI method (paired bootstrap, 10,000 resamples) and the
    inconclusive-result policy are frozen in WL-001 before any result is seen. An inconclusive result is not a pass.
  - **QUAL-S, release-day smoke tests (track R).** Membership is accepted as the starting design (decision P7).
    **[Pending]:** the fixtures, task versions and justified thresholds are frozen before the first qualification run;
    **no M8 pass until then.**
    - QUAL-S1: teacher-forced KL on 50 fixed prompts vs the model's own reference implementation;
    - QUAL-S2: a fixed 200-item GSM8K subset;
    - QUAL-S3: a fixed set of 20 hidden-test coding tasks.

    The M8 clock stops only when QUAL-S passes.
  - **QUAL-006 compressed builds (G5, M9):** per-token KL vs the official model, plus QUAL-001/002/005 paired against
    both the official model and our 4-bit build, reported with the exact bit budget. Sensitivity scans must give the
    same allocation run to run with a fixed seed and calibration set.
- **WL-001, public workload manifest** (sanitized; frozen before SYS-P-003/004 run):
  - prompt and output length distributions, tool delays, arrival pattern, cache state (warm or cold);
  - reasoning effort fixed at `max`; speculative settings;
  - **Agent-step metrics** (three, never mixed):
    1. **Full agent step (gates M2):** from the request being sent through **tool completion**, when the agent is ready
       to send its next request. This is comparable with the historical harness (task wall time ÷ steps, which
       includes tool runs).
    2. **Model-turn latency:** from the request being sent to the last token of the assistant turn.
    3. **Tool time:** reported separately.

    Comparisons with historical results use the full-step metric only. **Never present full-step vs model-turn
    numbers as an engine speed-up.**
  - **Eight fixed, paired session traces** (s1–s8). Each is replayed identically on our engine and on baseline B, with
    the same checkpoint files.
  - **Committed-token accounting:** the numerator counts each **committed, emitted** token once. That includes reasoning
    tokens delivered to the client, and excludes rejected draft tokens, which never add to it. Prefill tokens are not
    counted.
  - **Decode time** for a step runs from its first committed token to its last. TTFT and tool time are reported
    separately. A session's rate = its committed tokens after the first of each step ÷ the sum of its decode times.
  - **M1 is enforced per session.** **Each** of the eight rates must be ≥ 30 tok/s **and** ≥ 0.95 × B's rate on the
    same trace. All eight rates are reported, plus the median, minimum and p10/p90 as context. A median can't pass M1:
    `[1, 1, 1, 30, 30, 30, 30, 30]` fails.
  - **[Pending]:** the actual fixture (trace files, generator, seeds and hashes) is stored in the repository before any
    SYS evaluation. Until then WL-001 is a specification, not an existing reproducible workload.
  - Private traces may supplement it, never replace it.
- **REL-001, release-day rehearsal (M8 baseline):** run track R end to end on an already-released model. Record the
  time per step, and the hash checks (FR-1).
- **SYS:**
  - SYS-P-001: NCCL all-reduce latency at T × 4096 × 2 B (T = 8…64), P2P on and off.
  - SYS-P-002: tok/s at 1–64 users.
  - **SYS-P-003: per-user tok/s at 8 agents vs baseline B (M1).**
  - **SYS-P-004: agent step time at 4/8/16 vs B (M2).**
  - SYS-P-005: prefix-hit rate in multi-turn agent replay (with preserved thinking).
  - SYS-P-006: MTP on vs off at 1, 2, 4 and 8 agents (feeds ADR-0013).
  - **SYS-P-008: host-RAM tier.** Restore time for 8K/64K/128K-token sessions vs recomputing the prefill; agent step
    time with and without the tier when 16 sessions exceed GPU memory.
  - **SYS-P-009: multi-GPU scaling** on the real 3–4 GPU box: tok/s per user and step time for each plan in
    ADR-0014.
  - SYS-P-007: **prefill interference.** While a 100K-token prompt is prefilled, the other agents' p99 inter-token
    latency stays ≤ 2× its idle value.
  - **SYS-M-001: measured memory budget and per-agent context.**
  - SYS-R-001: 24 h soak.
  - SYS-R-002: overload returns 429 or queues.
  - SYS-R-003: 128K prompts.
  - SYS-R-004: OOM protection.
  - **SYS-R-005: zero JIT after ready.**
  - SYS-R-006: every graph bucket replays.
  - SYS-R-007: P2P self-test falls back to NCCL.
  - SYS-R-008: NCCL version check.
  - **SYS-P-010, Rust-kernel share (M7; decision P6):** on the WL-001 8-agent workload, Nsight Systems records every
    decode-step kernel on **both GPUs**.
    - **Denominator:** the summed durations of all compute kernels, whatever their implementation origin (ours, cuBLAS,
      other libraries).
    - **Numerator:** the durations of kernels implemented in Rust (cuTile Rust or cuda-oxide). A Rust launcher does not
      make a foreign kernel Rust.
    - NCCL, memory copies and elapsed decode-step time are reported separately.
  - **SYS-R-011, two-GPU graph lifecycle gate (before full-model integration):**
    - capture and replay matching buckets on both ranks; switch buckets under load; cancellation and shutdown;
    - bounded failure when one worker or the P2P probe fails;
    - destruction order: graphs before communicators; a failed communicator is aborted and a new one created for the
      fallback;
    - the actual PCIe topology and P2P capability are measured and recorded, not assumed.
  - SYS-R-009: cold start to "ready" (weights load plus warm-up), compared with baseline B on the same box.
  - **SYS-R-010: host-tier consistency.** A parked-then-restored session gives the same next-step output as one that
    never left the GPU (API-006 tolerance); no leak after 10,000 park/restore cycles.
- **API:**
  - API-001 conformance; API-002 SSE; API-003 stop sequences; API-005 cancellation.
  - **API-004:** tool-call and reasoning parsing, plus **chat-template parity**: token-for-token rendering vs the pinned
    template (revision `eb9eb208`) for the defaults, `reasoning_effort` ∈ {low, high, max} (default max), and
    `clear_thinking` history handling. Unsupported values are rejected with a clear error.
  - API-010: the metrics endpoint exposes the FR-8 series.
  - **API-006 prefix hit vs miss:** per-token KL with the target tokens fed in (teacher forcing) within the noise floor
    (primary); and ≥ 95 % of 200 prompts give identical first 256 greedy tokens (secondary). **Bitwise only under D1 when
    hit and miss use the same execution partition** (the same chunk boundaries, verification grouping and kernel
    specialisations). A cache hit generally resumes at a different partition than a full prefill, so the tolerance
    applies. Chunk-invariance needs the canonical-schedule contract, **[Pending]** before Phase 1.
  - API-007: a LAN bind requires an API key.
  - API-008: limits.
  - API-009: image parts rejected.

## 10. Phase 0 gate (go / no-go)
**Go** only if **all** of the following hold:
- Every P0 test in §3–§8 passes on its designated hardware (§1a).
- SMLA-P-001 is ≥ 0.9 × the FlashInfer sm_120 kernel at T ≥ 8 (FP8 rows).
- MOE-P-002 is ≥ 0.9 × the best **gate-eligible** W4A16 baseline at `NVFP4-G16`, T ≤ 64.
- SMLA-P-005 is ≥ 80 % of the read peak on the GLM row formats.
- H-008 shows no JIT after warm-up.
- Operational spending complies with the authorization recorded outside this repository.

**Decision P3:** if no gate-eligible baseline exists (MOE-P-002 eligibility; SMLA-P-004), a result ≥ 70 % of the
attainable roofline (H-002 and H-009) is a **provisional go only**. It does **not** satisfy M5 (same-card parity), which
stays open until a compatible baseline runs. The technical lead decides whether to proceed.

**Otherwise:** a report with the measured gap, the cause (profiler evidence), and options: fix, a CUDA C++ fallback for
that kernel (ADR-0010), or stop.

## 11. Traceability
| PRD item | Tests |
|---|---|
| M1 and M2 | SYS-P-003, SYS-P-004 on WL-001 |
| M3 | QUAL-* (non-inferiority, unchanged), golden KL; EQV-001 and the quantization-quality gate are **[Pending]** their prerequisites (§9) |
| M4 | SYS-R-* |
| M5 | SMLA-P-001/004, MOE-P-002 (G16); MOE-P-005 reports G32 |
| M6 | Release checklist (manual review of every published number) |
| M7 | SYS-P-010 |
| M8 | REL-001, QUAL-S, track R |
| M9, M10 | QUAL-006, REL-001 timing |
| FR-1 | SPEC-001, REL-001 (hash checks) |
| FR-2 | QUAL-*, SAMPLE |
| FR-4 | SYS-P-002, SYS-M-001, SYS-R-002 |
| FR-8 | API-010 |
| FR-9 | IDX tie rule |
| FR-10 | SYS-P-008, SYS-R-010 |
| NFR-2 | H-001, release checklist |
| NFR-3, NFR-4 | DEV-GUIDELINES G-01…G-12 |
| FR-3 | API-* |
| FR-5 | KDA-R-002, SYS-P-005, API-006 |
| FR-6 | MTP-*, KDA-R-001 |
| FR-7 | SYS-M-001 |
| NFR-1 | API-007/008 |
| NFR-5 | H-008, SYS-R-005 |
