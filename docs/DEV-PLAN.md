# TensorQuay Engine: Development Plan

Status: **v0.3 technical plan** (19 Sep 2026; publication extract 25 Sep 2026). v0.3 applies the adversarial review and the technical lead's follow-up (`reviews/2026-09-19-v0.3-follow-up.md`). The lead decided P1–P3 and P5–P7; P4 was not approved. **[Pending]** items await a future contract or evidence. **Phase 0 scope is unchanged.** Earlier:
the paper reports. This document plans **Phase 0** in detail and outlines later phases.
- Architecture: [ARCHITECTURE](ARCHITECTURE.md).
- Requirements: [PRD](PRD.md).
- Tests: [TEST-PLAN](TEST-PLAN.md).
- Rules: [DEV-GUIDELINES](DEV-GUIDELINES.md).
- Research: `research/` (GLM-5, DeepSeek-V4.1-Flash).
- Reviews: `reviews/`.

## 1. How we work
- **Design, then tests, then code.** No kernel code until the technical lead approves its contract and tests within
  the Product Owner's agreed direction.
- **Lean.** Build the smallest thing that passes the tests. Reuse what exists (for example `cutile::bench`, and
  FlashInfer as a baseline) instead of rewriting it.
- **Use CPU tests and compile-only checks first.** Reserve GPU runs for hardware correctness, profiling and
  performance measurements. Record the environment and results in the evaluation report.

**First implementation slice approved by the technical lead, 19 Sep 2026:** the CPU NVFP4 reference and REF-003/004
tests under the corrected TEST-PLAN contract. Build only the small Python test setup this needs. The larger workspace,
GPU image and kernels follow in separate increments after their reference and contract tests are ready. The storage
prerequisite for the full build below does not block this bounded CPU-only slice; keep its environment small and do not
download model weights. Future EQV-001, QUAL-S and workload evidence remains pending at the applicable phase.

**CPU slice accepted, 19 Sep 2026:** `reference/` implements REF-003/004. The lead's independently authored suite
passes 102 cases and covers all 166 statements and 38 branches in the reference package; the combined suite passes
214 cases. The lead also reproduced all 13 pinned ModelOpt fixture cases, inspected the implementation and ran the
format, lint and content checks. This completes the NVFP4 part of step 0.2, not the remaining references or GPU gates.
Run instructions are in [the reference guide](../reference/README.md); the
[independent evaluation report](evals/2026-09-19-nvfp4-reference.md) records the scope and limits of acceptance.

**MLA CPU slice accepted, 20 Sep 2026:** REF-001 passes 193 independent core tests and eight upstream-fixture
checks. The lead reproduced five cases from pinned Transformers with positive CPU MATH-SDPA backend evidence.
All independent reference suites pass 303 tests; the full suite passes 453, both at 100% line and branch coverage.
The [MLA evaluation report](evals/2026-09-20-mla-reference.md) records the numerical results and limits.

**MoE CPU slice accepted, 20 Sep 2026:** REF-002 passes 166 independent core cases and 22 upstream-fixture checks.
The lead reproduced six pinned upstream cases, including authentic FP32 router inputs widened exactly for float64
expert comparison, and verified the original eager expert and activation callables. All independent reference suites
pass 491 tests; the full suite passes 688, both at 100% line and branch coverage. The
[MoE evaluation report](evals/2026-09-20-moe-reference.md) records the numerical results and tooling corrections.
**Router CPU slice accepted, 20 Sep 2026:** REF-006 passes 162 independent core tests and 29 upstream-fixture checks.
The lead reproduced ten actual upstream calls independently, verified exact projection and expert sets, and caught
14 deliberate implementation mutations. The [router evaluation](evals/2026-09-20-router-reference.md) documents the
FP32 numerical gate correction made before implementation. All independent suites pass 682 tests; the full suite
passes 931, both at 100% of 408 statements and 134 branches.

**Shared generator accepted, 20 Sep 2026:** step 0.3 implements Philox4x32-10 in the dependency-free Rust
`tq-testkit` crate and the Python reference. The [evaluation](evals/2026-09-20-philox-foundation.md) records exact
cross-language/upstream agreement, independent coverage and mutation checks. The first Rust workspace pins the
toolchain and lints; it contains no placeholder runtime crates.

**Rust host contracts accepted, 20 Sep 2026:** `tq-core` validates SMLA and MoE metadata, including checked
64-bit spans, padded layouts, empty launches and the host query plan. Its 44 independent tests pass in debug and
release at 100% of 295 lines, 456 regions and 22 functions. Seven external-consumer probes and 19 deliberate fault
probes pass; see the [host-contract evaluation](evals/2026-09-20-host-contracts.md).

**CPU tooling accepted, 20 Sep 2026:** the remaining local step 0.4 checks pass the independent strict run.
The [CPU tooling evaluation](evals/2026-09-20-cpu-tooling.md) records 246 new acceptance cases, 100% coverage of
both new Python modules, 12 deliberate faults caught and all existing reference/Rust gates retained. The Linux
workflow is implemented and locally validated; a hosted run remains unverified until a remote exists.

**Current position: Phase 0, local step 0.4 complete.** Hosted Linux results are tracked in the
[CPU workflow](https://github.com/TensorQuay/tensorquay-engine/actions/workflows/cpu.yml). Step 0.2's P0 references
(REF-001, REF-002, REF-003, REF-004 and REF-006) remain accepted; REF-005 stays P1 until indexer work. The reference
package's independent suites pass 1,263 tests and its full suite passes 1,522, at 100% of 470 statements and 150 branches.
Next: step 0.5's pinned CUDA build-image and compile-only contract (G-11/G-12). ModelSpec parsing (SPEC-001/002)
remains a separate pending slice. No Rust kernel, GPU performance or full-model gate is claimed by these results.

**Step 0.5 started, 20 Sep 2026:** the [first compiler-image contract](CUDA-BUILD.md) separates the compiler
environment from the later inference-baseline image. It pins CUDA 13.3.1, Rust 1.95.0 and released cuTile 0.3.1.
The [recipe passes independent local checks](evals/2026-09-20-cuda-build-preparation.md) and is ready for
its first Linux x86_64 build, including the upstream IR/bytecode smoke check.
Actual image execution, sm_120 cubins, the resource gate and GPU-test builds remain unverified.

**Evaluate at each milestone.** Follow [the evaluation report requirements](evals/README.md) for CPU references,
kernels, one layer, the full model, eight agents, sixteen agents and release qualification. Advance only on evidence
for that gate; maintain the independent acceptance suite and compare eligible baselines under matched conditions.

## 2. Software stack
| Layer | What | Owner | Notes |
|---|---|---|---|
| Our code | Rust (stable, pinned), crates per ARCHITECTURE §2 | TensorQuay | |
| Kernel language | **cutile-rs** at a pinned commit (0.3.x). A planned upgrade PR each month | NVIDIA (NVlabs), Apache-2.0 | Its safe host API (`cuda-core`) owns the CUDA context. We may carry a minimal-patch TensorQuay fork if unavoidable, always rebased on upstream releases (ADR-0016) |
| Kernel compiler | `tileiras` (**CUDA Toolkit 13.3**), a runtime dependency | NVIDIA | Also used in CI for compile-only builds |
| Driver | Supports CUDA 13.3 | NVIDIA | |
| Two-GPU communication | **NCCL ≥ 2.31.2** via `cudarc` (feature `nccl`), inside `tq-gpu` only | NVIDIA / community | Handles borrowed from `cuda-core`. cudarc binds the NCCL 2.30 API; we load the ≥ 2.31.2 runtime (ABI-compatible; verified by SYS-R-008) |
| Model and text | `safetensors`, `tokenizers`, `minijinja` | HF / community | |
| HTTP | `axum` on `tokio` (tq-server only) | community | |
| Fallback kernels (ADR-0010) | Vendored CUDA C++ (for example FlashInfer, Apache-2.0) compiled to cubin, then cuda-oxide | NVIDIA / FlashInfer | cuda-oxide needs a nightly compiler and LLVM 21 |
| **Not in the product** | PyTorch, Python, vLLM, SGLang, Triton | — | Python/PyTorch only as test oracles; vLLM and FlashInfer only as benchmark baselines |

## 3. Phase 0: kernel spike
| Step | Work | Where | Done when |
|---|---|---|---|
| 0.1 | Agree on contracts, tests and scope. Pin Python and upstream reference versions; provision storage for the selected CPU and Linux build environments | CPU / Linux builder | Contracts approved; prerequisites in place |
| 0.2 | O1-alg and O1-prod references (TEST-PLAN §2); the NVFP4 codec pinned to ModelOpt `b311c054` and cross-checked byte for byte (REF-003/004); REF-001…006 | CPU | REF P0 tests pass |
| 0.3 | Shared Philox generator (Rust and Python, same stream); manifests; small committed cases | CPU | A cross-language stream test passes |
| 0.4 | Workspace skeleton: `tq-core` (contracts and validation), `tq-testkit`, and the DEV-GUIDELINES tooling (toolchain, lints, deny, length/deps/manifest scripts, tier-1 CI) | CPU + CI | CPU tests pass on the Mac |
| 0.5 | **Pinned compiler image first:** CUDA 13.3.1, Rust and cuTile per [CUDA-BUILD](CUDA-BUILD.md); build on Linux without a GPU. **Separate benchmark image:** official vLLM nightly, FlashInfer main **AOT-built for sm_120** (so no JIT during measurement), and CUDA 13.3 for `tileiras`; recheck baseline eligibility before building. **Pin baseline commits and weight layouts** (FlashInfer cuTile NVFP4 at G16, vLLM Marlin), and add a matched CUDA C++ gather microbenchmark (a benchmark tool, not product code). **Compile-only CI** of every specialisation to an sm_120 cubin plus the resource gate (G-11); kernels v1 written to the contract; H-006 compiles. Record image identities; registry publication requires Product Owner authorization | Linux builder / CI | G-11 green; image digest recorded |
| 0.6 | **GPU session 1: RTX 5090** . **First 15 minutes** (stop the session if anything fails): driver ≥ CUDA 13.3, the image runs, `ncu` permission (H-007), power limit. Then H-001…H-008, then **SMLA-P-005 (gather) and SMLA-P-008 (shared memory)**. Gathers on the GLM row formats must reach 80 % of the read peak (SMLA-P-005, decision P1; smaller rows are report-only). Run the matched CUDA gather **before** blaming the compiler for a miss. If both miss, open an access-pattern or hardware investigation; don't switch kernel language on that evidence | GPU host | Early risks measured |
| 0.7 | Correctness and tuning loop: all SMLA and MOE C and E tests plus sanitizers; sessions ≤ 3 h, each with a written goal | RTX 5090 | All P0 C and E tests green |
| 0.8 | **Gate run on 1 × RTX PRO 6000** : full suite, SMLA-P and MOE-P, **same-card baselines** (FlashInfer `sparse_mla_sm120` #5075; MOE like for like at `NVFP4-G16`: vLLM Marlin and FlashInfer cuTile NVFP4 #5099; `NVFP4-G32-E4M3` measured as a distinct format, MOE-P-005), profiles | GPU host | Gate report |
| 0.9 | Report and go/no-go (TEST-PLAN §10) | CPU | PM decision |

## 4. Hardware measurement discipline
- Verify the driver, GPU model, power limit, PCIe layout and profiler permissions before running a benchmark.
- Build and record the pinned image before the hardware session. Run scripted jobs that emit sanitized JSON.
- Keep the workload and timed operation identical across eligible baselines. Report setup and compile time separately.
- Phase 0 uses generated test inputs; model weights are unnecessary except for the optional SMLA-C-009 case.

## 5. Later phases (outline; detailed at each gate)
| Phase | Content | Key tests |
|---|---|---|
| 1a | **Two-GPU session first (about 1 h on 2 × PRO 6000):** the graph-lifecycle gate SYS-R-011 plus NCCL ≥ 2.31.2 latency with P2P on and off, and whether the host allows P2P. Then the remaining kernels: KDA (chunk, recurrent, verify, conv), a **fused indexer** (score, ReLU and head-weighted sum plus deterministic top-k with tail; no rotation for this checkpoint (`qk_rope_head_dim = 0`); FP8 K; pooled-key cache writer plus the per-sequence tail state), the **sparse prefill contract and kernel (SMLA-PF)**, a fused MLA prologue (q_a → norm → q_b → absorb; kv_a → norm → cache write), latent → 528 B KV writer, FP8 W8A16 and BF16 GEMMs, head-batched GEMMs (W_UK/W_UV), FP32 router, mHC, Philox sampler, device-side speculative verify | SYS-P-001, KDA-*, IDX-*, MHC-* |
| 1b | `tq-model` and `tq-engine`: one layer, then N layers, then the full model on 2 GPUs; CUDA graphs; warm-up. **Pin the MTP oracle** (vLLM `glm5next/nvidia/mtp.py` at commit `98ed0856`, REF-007) and specify the index producers (target per query; MTP step 0, then reuse) before designing the carry interface. Before the Phase 1 gate, complete the **EQV-001 prerequisites** (select the reference runner and format, verify its loader, freeze the noise-floor experiment, aggregation, zero-floor rule and threshold) and **select** the BF16 acquisition plan for the quantization-quality gate (P4 was not approved) | Golden tests vs O2 and O3; REF-007; H-008 |
| 1c | Our checkpoint from **the official BF16 release**: ModelOpt W4A16 NVFP4 (group per ADR-0008), with hashes, cross-checked against the public artifact of the same format | REF-004; QUAL; ADR-0008 |
| 2b | (v1.1) Host-RAM session tier (ADR-0015): park and restore over a copy stream | SYS-P-008, SYS-R-010 |
| 2c | (when a 3–4 GPU box is planned) PP and DP in `tq-engine`/`tq-gpu` per ADR-0014 | SYS-P-009 |
| 2 | `tq-sched`, `tq-server`: batching, **chunked prefill with decode priority**, session affinity, pools, hybrid prefix cache, adaptive MTP (ADR-0013) with the SAMPLE tests, async sampling copy, API with template parity (API-004: `reasoning_effort` low/high/max, `clear_thinking`), security; **the WL-001 fixture** (eight paired session traces, committed-token accounting, per-session M1 enforcement) **stored before SYS-P-003/004** | SYS, API, QUAL; M1–M4 vs baseline B |
| 3 | v1.0 release; then a DeepSeek or Qwen candidate **that fits** (PRD §6). The DeepSeek plan, tests and oracles are in `research/deepseek-v4.1-flash.md` §6(d); they are adopted at that gate | PRD §9 |

**Track R: release-day pipeline (G2, the primary goal; built during Phases 1–2 and exercised on every release).**

| Step | Work | Automated? |
|---|---|---|
| R0 | **Watch releases** from the model families we follow (Hugging Face organisations of GLM, DeepSeek, Qwen and others) and alert the team | Yes |
| R1 | **Pull the weights** to our own box or storage at full line speed (parallel downloads, hashes checked) | Yes |
| R2 | **Gap report:** `parse_hf_config` plus `ModelSpec::validate` list what the engine already supports and what is missing (new fields, layer kinds, formats), which gives the tier (A/B/C) | Yes |
| R3 | **Convert and quantize,** or compress through track C if the model doesn't fit | Yes |
| R4 | **Golden tests on real weights:** layer by layer against the model's own reference code (the `transformers` implementation at the release commit), then the whole model | Yes |
| R5 | **QUAL-S and benchmark:** the release-day smoke set (TEST-PLAN §9; membership accepted, P7; fixtures, task versions and thresholds frozen before the first qualification run, and no M8 pass until then) plus tok/s and step time on WL-001. Full QUAL follows | Yes |
| R6 | **Gaps (tier B/C):** the missing kernels or layers are written to contract, then R4–R5 re-run | Engineering |
| R7 | **Publish** (Hugging Face checkpoint, engine release, benchmark note), **PM approval only** | PM |

A rehearsal on an already-released model gives the baseline times for M8 before the first real release-day run.

**Track C: compression (G5; future scope agreed separately).** The aim is to make models
that don't fit the hardware at 4-bit fit, with minimal quality loss, and to do it quickly after each release.

| Step | Work |
|---|---|
| C0 | **Release-day pipeline**, built once and reused: download the official BF16 weights → sensitivity scan (per layer and per expert, from KL on calibration data plus routing frequency) → bit allocation under a memory budget → quantize → evaluate (per-token KL, QUAL-006) → report. Candidate allocations are reproducible and evaluated against fixed quality gates. **Nothing is published without PM approval** |
| C1 | **Methods to evaluate** (on smaller models first, where it is cheap): mixed precision per layer and per expert (rarely routed experts get fewer bits); error-compensating quantizers (GPTQ/AWQ-style); calibration on real coding and agent traces; low-bit codes for 2–3 bits (trellis or vector codes in the QTIP / EXL3 / AQLM line); careful expert pruning or merging, only if QUAL allows (earlier community data showed pruning hurt) |
| C2 | **Quality recovery** when needed: distillation from the official model or quantization-aware fine-tuning, restricted to the smallest effective set of parameters. A separate experiment with a written quality and resource contract |
| C3 | **Engine kernels** for the chosen low-bit formats (dequant GEMV / grouped GEMM in cuTile Rust), tested like the Phase 0 kernels. Lower bits mean fewer bytes per token, so they also make decode faster |
| C4 | **First target: DeepSeek-V4.1-Flash on 2 × RTX PRO 6000** (about 2.5–2.7-bit experts, Engram in ≥ 256 GB host RAM). Go only if M9 passes |

Order: C0 and C1 start on models that fit (for example GLM-5.3-Flash at 3 bits vs 4 bits).
C2 and C4 require separate scope and evidence before implementation.

**Baseline B for M1/M2:** the best eligible official runtime on the same 2 × PRO 6000 hardware, checkpoint and
WL-001 workload. If no official runtime works on that configuration, use the documented fallback in PRD §5 and
report that limitation. This is pending hardware evidence.

## 6. Reporting
- **After each hardware session:** goal, environment, result, tests passing out of total and next technical step.
- **At each gate:** an evidence report with test results, sanitized benchmark JSON, profiles and reproduction commands.
- Publish estimates, CPU acceptance and measured GPU results as distinct kinds of evidence.
