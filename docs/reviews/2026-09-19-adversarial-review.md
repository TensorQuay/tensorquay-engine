# Adversarial review of the engine specifications

Public technical extract: internal operational notes are omitted; locations refer to the reviewed revisions.

Reviewed 19 Sep 2026 against repository commit `0f684de` and upstream sources. Scope: PRD, DEV-PLAN,
TEST-PLAN, DEV-GUIDELINES, ARCHITECTURE, the handover, and the earlier reviews. This is a review proposal;
it does not approve or amend the contracts.

**Verdict: revise before implementation.** The overall direction and separation of responsibilities are
reasonable. Several model semantics, numerical contracts and measurement rules still need correction.
The earlier reviews' closure is not evidence that these contracts have been validated on hardware.

There are **8 high and 7 medium findings** below. Findings 1–5 affect the Phase 0 specification. Findings
6–15 must be resolved before their corresponding model, runtime or release work. They need not all be
implemented in the kernel spike.

## Findings affecting Phase 0

### 1. High: the NVFP4 quantizer formula omits the global scale

Location: TEST-PLAN §2 and REF-004; DEV-PLAN Phase 1c.

The contract reconstructs a weight as `code * block_scale * global_scale`, but REF-004 defines the
block scale as `amax / 6`. For unscaled input weights, these definitions disagree. NVIDIA ModelOpt's
basic dynamic recipe instead uses a global decode scale `g = tensor_amax / (6 * 448)` and a block
scale derived from `block_amax / (6 * g)`, with its documented rounding and exceptional-value handling.
See [ModelOpt's quantizer][modelopt].

**Consequence:** references and kernels can agree with each other while interpreting exported weights
incorrectly. For tensor and block maxima of 6, `g = 1/448`: the correct block scale is 448. The stated
formula gives 1, whose largest reconstructed weight is only `6/448 = 0.013392857`.

**Required change:** specify whether `amax` is measured before or after global normalization; specify
encode/decode scale direction, zero blocks, saturation, rounding, nibble order and scale layout. Pin a
quantizer recipe. Add non-unit, non-power-of-two global scales and cross-check exported bytes and
dequantized values against that recipe. Keep W13's shared-scale requirement explicit.

### 2. High: the gather gate can mistake memory-transaction overhead for a compiler failure

Location: TEST-PLAN SMLA-P-005, H-002 and §10; DEV-PLAN step 0.6.

All tested row widths, including 68 bytes, appear subject to the same 80% streaming-bandwidth stop
rule. NVIDIA documents global-memory transactions in 32-byte units. An isolated cold 68-byte row
needs at least three such sectors: useful-byte efficiency is at most `68/96 = 70.8%` before latency
and scheduling overhead. Adjacent-row reuse changes this calculation, so the access distribution
also matters. See [NVIDIA's memory-access explanation][coalescing].

This is a transaction-efficiency calculation, not a measured throughput ceiling relative to H-002.
It demonstrates why one uncalibrated percentage is unsuitable for every row width. H-002 must also
distinguish read-only traffic from a copy's read-plus-write traffic.

**Required change:** define useful bytes and transferred bytes separately; fix the index distribution,
working set, stride and concurrency. Compare against a matched CUDA gather on the same card. Gate
GLM's actual row formats; report the smaller future-model rows separately until a defensible threshold
exists. A poor result for a future format should not automatically reject the GLM experiment.

### 3. High: group-32 E4M3 is not a drop-in baseline configuration

Location: PRD §6–§7; TEST-PLAN MOE-P-001/002; DEV-PLAN steps 0.5 and 0.8.

The memory plan favors E2M1 weights with E4M3 scales every 32 values. FlashInfer's public NVFP4 weight
preparation requires E4M3 scales with a `K/16` dimension; its MXFP4 preparation uses group-32/E8M0.
Neither standard preparation describes the proposed group-32/E4M3 checkpoint. A lower-level adapter
may be feasible, but has not been specified or validated. See the [pinned preparation routines][flashinfer-prepare],
[kernel implementation][flashinfer-moe] and [upstream support PR][moe-pr].

**Consequence:** the purported like-for-like baseline can use different weight values, scale decoding
and storage costs. A faster result would not establish kernel parity for the intended checkpoint.

**Required change:** declare the group-32 E4M3 layout as a distinct format. Establish group-16 NVFP4
parity first, and define a separate correctness, quality, memory and timing comparison for group 32.
Any baseline adapter must disclose conversions and changed bytes. Pin the tested commits and layouts
before the GPU session, and verify the GLM shapes and clamped activation, not just upstream demo shapes.

### 4. Medium: REF-001 still has two incompatible precision instructions

Location: TEST-PLAN §2, lines 32–38, and REF-001.

The general O1 rule preserves FP32 softmax, while REF-001 compares an absorbed formulation against a
float64 SDPA formulation at `1e-12`. Feeding identical post-normalization inputs removes RMSNorm's
precision difference but does not remove this softmax difference. The pinned
[Transformers implementation][transformers] explicitly casts the eager softmax to FP32.

For a simple softmax over scores `[0, 1]`, rounding the larger probability to FP32 alone introduces
about `2.59e-8` relative error. That arithmetic illustration is not a run of REF-001.

**Required change:** distinguish the float64 algebraic-equivalence oracle from the oracle reproducing
production dtype boundaries. Explicitly exempt the former's attention core from the FP32-softmax
rule. Define which oracle supplies `y*` in each accuracy test. This completes the earlier review's
proposed fix without weakening precision-sensitive tests to make them pass.

### 5. Medium: the cold-cache timing protocol is underspecified for CUDA graphs

Location: TEST-PLAN §8, lines 241–248; DEV-GUIDELINES tier-2 performance gate.

The protocol combines N launches in one graph with an L2 flush, but does not state where the flush or
buffer rotation occurs. Flushing once before a graph that reuses inputs makes only its first launch
cold. For eight queries, `8 * 2051 * 528` is about 8.3 MiB of selected KV payload before head reuse;
the allocated pool's total size does not establish the accessed working set's cache behavior.

**Required change:** specify eviction or independent buffers before every measured invocation,
including invocations inside a replay. Exclude eviction work from kernel latency and apply identical
rules to the baseline. Keep a separate realistic steady-state measurement. Define regression as a
dimensionless paired ratio; `max(5%, 3 * IQR)` is ambiguous unless IQR is normalized too.

## Findings affecting model implementation

### 6. High: pooled indexer keys are insufficient to resume a partially completed pool

Location: ARCHITECTURE §4 `SlotPool`, §5 indexer and prefix cache; TEST-PLAN IDX and CACHE-R.

At a request end after 3 tokens of a 4-token pool, the next token must combine with the earlier raw
keys and their compression gate scores. Completed pooled keys and KDA/conv checkpoints do not hold
that information. The [pinned vLLM implementation][vllm-attention] explicitly stores an in-progress
tail pool; the [Transformers pooling code][transformers] also shows the required raw inputs.

**Required change:** make the unfinished indexer pool part of per-sequence state. Define its state
length, copy-on-write, checkpoint, commit and rollback semantics. Test prefill-to-decode, request
resume and draft rejection across all four pool offsets, including multiple sequences. The existing
three cache kinds can represent this; a new generic cache framework is unnecessary.

### 7. High: the GLM indexer is specified to rotate dimensions that this checkpoint does not have

Location: DEV-PLAN Phase 1a; TEST-PLAN REF-005 and IDX.

These documents require interleaved indexer RoPE. The [checkpoint config][glm-config] sets
`qk_rope_head_dim = 0`. In [vLLM's indexer][vllm-attention], `rope_dim` comes from that field and rotation
is skipped when it is zero. The `indexer_rope_interleave` flag does not create a nonzero rotary span.

**Required change:** specify no indexer rotation for the first checkpoint. Retain interleaved rotation
only as a future configuration-dependent behavior. Add a zero-rotary-dimension regression case and a
reference comparison that would catch accidentally applying RoPE.

### 8. High: MTP index sharing is assigned the wrong producer

Location: ARCHITECTURE §4 `IndexSrc`; TEST-PLAN SMLA-C-013 and REF-007; DEV-PLAN Phase 1b.

The architecture says MTP reuses target indices. The [official MTP implementation][vllm-mtp] allocates
its own index buffer: draft iteration zero computes its top-k and later draft iterations reuse that
result. This is not a license to reuse a target layer's indices, or one query's indices for every
query in target verification.

**Required change:** specify the index producer, lifetime and visibility separately for target
verification and drafting. Pin this oracle before designing the carry interface. Keep SMLA-C-013 as
a kernel capability test, but do not treat it as proof of model semantics. Add sequence compaction,
unequal draft depths and rejection tests so stale rows cannot be assigned to another request.

### 9. High: D1 numerical determinism does not guarantee identical speculative samples

Location: DEV-GUIDELINES §1.3; TEST-PLAN MTP, KDA-R and API-006.

Fixed reductions do not define how draft, acceptance, rejection and bonus-token draws share a random
stream. A correct speculative sampler can preserve the target distribution while producing a
different sequence for the same seed. [vLLM's tests distinguish distribution convergence from greedy
equality][speculation]. A concrete two-token counterexample is recorded in the checks below.

**Required change:** separate deterministic forward computation, greedy equality, and sampled
distribution preservation. Either specify and test a deliberate coupling of random draws or remove
the blanket sampled-token equality promise. Add rejection-sampler tests covering zero probabilities,
all acceptance lengths, bonus tokens, temperature/top-p/top-k and request interleaving. Also define
how D1 handles KDA chunked prefill versus recurrent decode; a fixed attention split alone does not
make different recurrent algorithms numerically identical. [Tile IR also limits MMA bit-identity
guarantees][tile-stability].

### 10. Medium: sparse prefill has a test but no compatible kernel contract

Location: TEST-PLAN SMLA contract, SMLA-E-012 and SMLA-P-009; DEV-PLAN Phase 1a.

The decode launcher rejects more than four query tokens per sequence, but P-009 asks it to handle a
single 4096-token prefill chunk. The later kernel list does not assign a separate sparse-prefill
contract. This leaves a necessary full-model path outside the documented implementation work.

**Required change:** explicitly schedule a prefill launcher/contract, or an approved generalization
of the existing one. Cover per-query causality, pooled-index visibility, chunk boundaries and its
workspace budget before implementation. Keep the Phase 0 decode contract small if that is the scope.

### 11. High: the required BF16 quality baseline has no executable acquisition plan

Location: PRD M3; TEST-PLAN golden tests and QUAL; DEV-PLAN Phase 1 reference runs.

Several gates depend on the full-model FP8-versus-BF16 KL floor, but the execution plan describes the
official FP8 run without specifying how BF16 logits will be obtained. From the documented GLM shapes,
the 43 routed-expert layers including MTP alone occupy **580.5 GiB at BF16**, before other weights,
cache and workspace. Two target GPUs cannot hold this reference.

**Required change:** define a feasible BF16 oracle run, or a pinned existing reference artifact with
matching tokens and full-distribution data. Specify KL direction, teacher-forced positions, masks,
aggregation and precision. Separate engine-equivalence error from deliberate weight/KV quantization
loss; an FP8/BF16 difference is not automatically an attainable error allowance for W4 weights. Treat
the latter as a measured quality decision, not as an already established property of the format.

## Findings affecting runtime and product acceptance

### 12. Medium: a thread per GPU does not establish NCCL graph safety

Location: ARCHITECTURE §3 and §6; DEV-PLAN Phase 1a; TEST-PLAN SYS-P-001 and SYS-R-006/007.

[NVIDIA's graph guidance][nccl-graphs] warns of multi-GPU, single-process deadlocks and requires
collective capture and replay to agree across ranks. Dedicated worker threads address one cause;
a latency microbenchmark does not prove the whole graph lifecycle is safe.

**Required change:** add a two-GPU integration gate before full-model integration: capture and replay
matching buckets, switch buckets under load, exercise cancellation and shutdown, and verify bounded
failure when one worker or a P2P probe fails. Define graph/communicator destruction order and how a
failed probe can fall back without reusing a broken communicator. Measure the actual PCIe topology
and P2P capability rather than treating host-setting recipes as guarantees.

### 13. Medium: the API promises a thinking-off mode the model does not expose

Location: PRD FR-3; TEST-PLAN API-004; DEV-PLAN Phase 2.

FR-3 describes GLM thinking on/off. The [official template][glm-template] and [vLLM recipe][glm-recipe]
instead specify always-on thinking with `low`, `high` and `max` effort, defaulting to `max`.
`clear_thinking` controls preservation of historical reasoning, not whether the current answer thinks.

**Required change:** match the pinned template and define unsupported API values explicitly. Test
token-for-token prompt rendering, defaults and history preservation. Fix the effort setting in all
quality and performance comparisons so a shorter reasoning budget cannot masquerade as an engine gain.

### 14. Medium: release gates do not enforce all of the stated success criteria

Location: PRD M5/M7 and §9; TEST-PLAN §10–§11; DEV-PLAN track R.

The release row requires M1–M6 and omits M7's Rust GPU-time requirement. M7 also lacks a test ID.
Separately, the test plan's roofline fallback can produce a Phase 0 go without demonstrating the
same-card parity required by M5. The release pipeline's smoke QUAL suite has no defined membership.

**Required change:** map every applicable metric to a test and gate. Define M7's measurement workload
and denominator, including how NCCL and other library kernels count. Mark missing-baseline roofline
evidence as provisional, or explicitly amend the product acceptance criterion. Name the release-day
smoke tests and their thresholds before using them to stop the time-to-support clock.

### 15. Medium: the headline performance comparison is not reproducible from the specification

Location: PRD M1/M2 and baseline B; TEST-PLAN SYS-P-003/004 and QUAL; DEV-PLAN Phase 2.

The private harness is referenced, but no public workload manifest defines prompt lengths, output
lengths, tool delays, cache state, arrival pattern, reasoning effort, speculative settings or what
starts and ends an agent step. "Same checkpoint class" also permits different quantization behavior.
Different choices can change whether the floor or relative target passes.

**Required change:** pin a sanitized workload manifest and benchmark configurations. Define per-user
aggregation, warm/cold cases, token accounting and whether queue/tool time counts. Freeze paired
quality evaluation's task versions, seed pairing, confidence-interval method and inconclusive-result
policy before results are observed. Private traces may supplement a reproducible public fixture.

## Assumptions checked and retained

- NVIDIA's [Blackwell tuning guide][blackwell] confirms the 99 KB shared-memory maximum per block for
  compute capability 12.0. Compile-time resource checks are appropriate; launch-time opt-in and actual
  device attributes must still be verified.
- FlashInfer [PR 5075][kv-pr] confirms the 528-byte NoPE payload and the need for safe masked gathers.
  These parts of the kernel contract are grounded in upstream work.
- FlashInfer's W4A16 implementation exists on sm_120. That establishes a candidate baseline, not a
  measured result for this engine's shapes or a guarantee of equivalent group-32 scales.
- At review time, vLLM [issue 53963][vllm-issue] and PRs 55277, 53969, 55778 and 54929 remained open.
  The upstream integration risk is current; it should be rechecked at the actual gate.
- The [DeepSeek reference configuration and model][deepseek] support the distinction between KV
  producers, index producers, shared attention and the encoder/decoder design. Keeping cache IDs and
  stage descriptions is reasonable. This review did not independently re-read all weight-shard
  headers or validate a compressed checkpoint's quality or fit.
- Pure host crates, fixed allocation pools, no compilation after readiness, exact index probes,
  poisoned invalid rows, and tests that fail when required GPU hardware is absent are useful rules.

## Checks performed and limits

- Read the five specifications, handover, prior reviews, and relevant research sections.
- Inspected the pinned Transformers model and official vLLM attention/MTP implementations; compared
  their behavior with the current upstream paths. Read NVIDIA CUDA/NCCL/Tile IR material, ModelOpt
  quantization code, FlashInfer code, and official GLM and DeepSeek configurations.
- Ran small arithmetic probes for finding 1, the 68/96 sector-efficiency calculation and FP32 rounding
  in finding 4. Recomputed expert memory as `43 * 288 * 3 * 4096 * 2048 * 2` bytes for BF16.
- Sampling counterexample: target `P(A)=0.75`, draft `Q(A)=0.5`, first uniform `0.6`. Direct inverse-CDF
  sampling returns A; drafting returns B. With acceptance uniform `0.1`, B is accepted because
  `P(B)/Q(B)=0.5`. The same random stream can therefore give different tokens without violating the
  target distribution.
- The existing repository guard audit reported clean before this review was added. This review
  contains no model weights or third-party PDFs. No engine code or GPU tests exist yet; no GPU
  benchmark, full-model quality evaluation or PyTorch oracle test was run for this review.

## Proposed implementation order

1. Resolve findings 1–5 in the Phase 0 contract and measurement documents; obtain PM approval of the
   resulting design and gates under AGENTS.md and DEV-GUIDELINES §3.
2. Implement the CPU references and contract tests, cross-language random stream, workspace skeleton
   and compile-only checks in the development plan's order.
3. Establish baseline adapters and measurement self-tests, then implement and measure the two kernels.
4. Before Phase 1, resolve the indexer, MTP, prefill and reference-execution findings and define the
   corresponding tests. Run the two-GPU graph integration experiment before full-model integration.
5. Define the API, workload and release acceptance details before serving and release evaluation.

## Primary sources

Links below identify the versions inspected where an immutable revision was available. Moving
documentation pages and issue states were checked on the review date.

[modelopt]: https://github.com/NVIDIA/Model-Optimizer/blob/b311c054de4052df9c7f3de9409b7598f44a0dba/modelopt/torch/quantization/qtensor/nvfp4_tensor.py#L170-L208
[coalescing]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
[flashinfer-moe]: https://github.com/flashinfer-ai/flashinfer/blob/dc04f50c9aa3eabcdaa5feb0934edb3d85e9529a/flashinfer/fused_moe/cutile/fp4.py
[flashinfer-prepare]: https://github.com/flashinfer-ai/flashinfer/blob/dc04f50c9aa3eabcdaa5feb0934edb3d85e9529a/flashinfer/fused_moe/prepare.py#L1456-L1513
[moe-pr]: https://github.com/flashinfer-ai/flashinfer/pull/5099
[transformers]: https://github.com/huggingface/transformers/blob/770e4c40d0/src/transformers/models/glm5_next/modeling_glm5_next.py
[vllm-attention]: https://github.com/vllm-project/vllm/blob/98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5/vllm/models/glm5next/nvidia/attention.py
[vllm-mtp]: https://github.com/vllm-project/vllm/blob/98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5/vllm/models/glm5next/nvidia/mtp.py
[glm-config]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json
[glm-template]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/chat_template.jinja
[glm-recipe]: https://recipes.vllm.ai/zai-org/GLM-5.3-Flash#reasoning-modes
[speculation]: https://docs.vllm.ai/en/latest/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding
[tile-stability]: https://docs.nvidia.com/cuda/tile-ir/13.3/sections/stability.html
[nccl-graphs]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
[blackwell]: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
[kv-pr]: https://github.com/flashinfer-ai/flashinfer/pull/5075
[vllm-issue]: https://github.com/vllm-project/vllm/issues/53963
[deepseek]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference
