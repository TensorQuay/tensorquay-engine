# TensorQuay Engine: Product Requirements

Status: **v0.3 technical requirements** (19 Sep 2026; publication extract 25 Sep 2026). v0.3 applies the adversarial review and the technical lead's follow-up (`reviews/2026-09-19-v0.3-follow-up.md`; response in `reviews/2026-09-19-adversarial-review-response.md`). The lead **decided** P1–P3 and P5–P7 and did **not** approve P4. Items marked **[Pending]** await a future contract or evidence. Earlier:
(`reviews/2026-09-19-design-review.md`), the architecture research and the paper reports (`research/`). Roles: Product
is the Product Owner; engineering decisions and gate acceptance belong to the technical lead. Companion documents:
[ARCHITECTURE](ARCHITECTURE.md), [DEV-PLAN](DEV-PLAN.md), [TEST-PLAN](TEST-PLAN.md),
[DEV-GUIDELINES](DEV-GUIDELINES.md).

## 1. Mission
**TensorQuay Engine brings the latest, most advanced open-source models to consumer and workstation NVIDIA GPUs, fast
(primary goal), on a cutting-edge Rust + CUDA Rust stack.**

It is an open-source (Apache-2.0) LLM inference engine: Rust from the API down to the GPU kernels, which are written in
NVIDIA's CUDA Rust (cuTile Rust, with cuda-oxide as the SIMT track).
- **First target:** the RTX PRO 6000 / RTX 5090 generation (Blackwell, sm_120).
- **First model:** GLM-5.3-Flash on 2 × RTX PRO 6000, for a team's coding agents.

## 2. Goals (why this project exists)
**The primary goal is G2.** The other goals serve it:
- G1 is *how* we build (Rust + CUDA Rust).
- G5 is how models that are too big still fit.
- G3 and G4 follow from it.

IDs are stable and referenced elsewhere; the order below is the priority.

| Priority | # | Goal | What it means in practice |
|---|---|---|---|
| **1 (primary)** | G2 | **The latest frontier open models on consumer GPUs, blazing fast** | New open models (GLM, DeepSeek, Qwen families) run on sm_120 within **hours** when the architecture is already supported, and within days when it needs new kernels. This comes from an automated release-day pipeline, a spec-driven modular engine, contract-tested kernels and reproducible hardware validation. Sized for the memory these cards have (G5 when a model doesn't fit) |
| 2 | G5 | **Make too-big models fit, with minimal quality loss, fast** | When a new frontier model is too large for the hardware (for example DeepSeek-V4.1-Flash), we don't drop it. We compress it: sensitivity-aware mixed precision, calibration on real agent and coding workloads, low-bit codes, and, when needed, distillation or quantization-aware fine-tuning to recover quality. An **automated pipeline** runs this within days of a release, judged by the same quality gates. The engine gets Rust kernels for the resulting low-bit formats (G1) |
| 3 | G1 | **Lead in fast Rust solutions for CUDA** | A production-grade engine whose hot paths are Rust GPU kernels written on NVIDIA's CUDA Rust (cuTile Rust). Performance at or above the best C++/Python stacks on the same card. We build **on** NVIDIA's stack and may carry our own variation where we need it, but we **keep upgrading from NVIDIA's releases** (ADR-0016). Dependency updates require regression testing |
| 4 | G3 | **Open and trustworthy** | Every kernel has a written contract, reference tests and a benchmark. Every published number is reproducible |
| 5 | G4 | **Support local coding workloads** | A fast, private engine for teams' coding agents on one office workstation |

## 3. Design and evaluation approach
The engine targets sm_120 directly, with a Rust runtime, model-specific configuration and independently tested
kernels. vLLM, SGLang and FlashInfer provide external reference implementations and performance baselines.
No performance advantage is established until paired measurements pass the gates below.

The Phase 0 gate tests whether cuTile Rust can meet the required correctness, resource and performance targets on
RTX 5090 and RTX PRO 6000. Comparisons must match model weights, formats, operation boundaries and workloads.
Unsupported baseline configurations are reported explicitly; estimates and third-party results do not pass a gate.

## 4. Users
| User | Need |
|---|---|
| Community: owners of RTX PRO 6000 / RTX 5090 (and later other consumer NVIDIA GPUs) | The newest open models running fast on their own GPUs, soon after release |
| Rust and GPU developers | A reference Rust + CUDA Rust inference engine to learn from, use and contribute to |
| Local coding-agent users on a 2 × RTX PRO 6000 workstation | Fast, private coding agents on the office network |

## 5. Success metrics
**v1.0 (GLM-5.3-Flash on 2 × RTX PRO 6000).** All are measured in **the same hardware and a reproducible agent harness**, using the public workload manifest **WL-001** (TEST-PLAN §9), with reasoning effort fixed at `max`. Baseline
B = the best **official** runtime (vLLM or SGLang) on that box, running the **identical checkpoint files** and manifest. If no official
runtime runs on the box yet, B is the best public runtime we can run there, as a benchmark only, and the
report says so.

| ID | Metric | Target |
|---|---|---|
| M1 | Decode tok/s per user, 8 concurrent agent sessions | **Each** of the eight WL-001 session traces: **≥ 30 (floor) and ≥ 0.95 × B on the same trace.** Rates use committed-token accounting (rejected drafts never count). All eight are reported; a median can't pass (TEST-PLAN §9 WL-001) |
| M2 | **Full** agent step time p50 at 8 agents (request through tool completion, as in the historical harness; reasoning effort `max`). Model-turn latency and tool time are reported separately (TEST-PLAN WL-001) | **≤ 30 s (floor) and ≤ B** |
| M3 | Quality vs the official reference (O3) | **Non-inferior:** paired tests, ≥ 3 runs, 95 % CI above −1.5 pts on SWE-bench Verified, Terminal-Bench 2.0 and GSM8K. Per-token KL ≤ 2 × the measured noise floor (official FP8 vs BF16) |
| M3 (note) | **The P4 replacement was not approved.** M3 above stays in force | Before the Phase 1 gate: **EQV-001** (engine equivalence vs a selected reference runner and format, with a frozen noise-floor experiment, aggregation, zero-floor rule and threshold) and a **selected** BF16 acquisition plan for the quantization-quality gate (TEST-PLAN §9). Both are **[Pending]**; the existing quality obligations stay open |
| M4 | Reliability | 24 h soak at 8 agents: no crash, memory growth < 1 %. Overload queues or returns 429 |
| M5 | Kernel efficiency | Phase 0 gate: each spike kernel ≥ 0.9 × the best **gate-eligible** same-card kernel (TEST-PLAN §10). MOE parity is at `NVFP4-G16` (P2): a baseline qualifies only if it passes the same MOE correctness cases, including the clamp edges, over the same timed operation (MOE-P-002). `NVFP4-G32-E4M3` is reported only. A roofline-only result without an eligible baseline is a provisional go and doesn't satisfy M5 (P3) |
| M6 | Openness | Every published number has its configuration, raw data and a reproduction script |
| **M7** | **Rust-first (G1)** | ≥ 90 % of decode GPU time is spent in kernels written in Rust (cuTile Rust or cuda-oxide). Any C++ fallback kernel (ADR-0010) is tracked and has a plan to replace it. Measured by SYS-P-010 (decision P6): summed compute-kernel time across both GPUs by implementation origin. NCCL, copies and step time are reported separately; a Rust launcher doesn't make a foreign kernel Rust |

**Rollout order for the first deployment** (GLM-5.3-Flash on 2 × RTX PRO 6000):
- **8 agents first, then 16.** Each stage reports measured usable context per agent (SYS-M-001) and speed (SYS-P-003/004).
- Publish the harness, configuration and sanitized workload alongside each reported result.

**After v1.0.**

| ID | Metric | Target (to confirm at v1.0) |
|---|---|---|
| **M8** | **Time to a working build (G2, the primary metric):** from a model's open release to a build on sm_120 that passes golden tests and **QUAL-S** (TEST-PLAN §9; membership accepted, P7; fixtures and thresholds **[Pending]**, frozen before the first qualification run, so no M8 pass until then) | **Tier A** (new weights, known architecture): **≤ 6 h**, download time included. **Tier B** (a variant using existing kernels): **≤ 48 h**. **Tier C** (new kernels needed): **≤ 2 weeks**. Full QUAL follows within 48 h of a working build |
| **M9** | **Compression quality (G5):** a compressed build of a model that doesn't fit at 4-bit | Per-token KL and paired QUAL vs the official model, published with the bit budget. Target: within 3 pts on the coding/agent suites at ≤ 2.7 bits per expert weight (to confirm after the first study) |
| **M10** | **Compression speed (G5)** | First compressed candidate with QUAL results ≤ 7 days after the release of a model we choose to support |

## 6. Scope
**v1.0 (in):**
- GLM-5.3-Flash, text only. Image inputs are rejected with a clear error, and vision weights are not loaded.
- **Weights:**
  - Experts: NVFP4 W4A16, as standard **`NVFP4-G16`** or the **`NVFP4-G32-E4M3`** variant (non-standard, not a
    public-baseline format; see TEST-PLAN §2). Parity is established at G16 first. ADR-0008 chooses from measured
    KL/QUAL and memory.
  - Other weights keep the official formats: FP8 128×128 where the official checkpoint uses FP8, and BF16 for the ones
    it keeps in BF16 (KDA projections, `kv_b_proj`, indexer, routers, embeddings, `lm_head`, mHC). An optional FP8 for
    the KDA projections saves about 4.4 GiB, subject to QUAL.
  - We quantize **from the official BF16 release** (zai-org/GLM-5.3-Flash-BF16).
- **Latent KV:** FP8 528-byte rows (the same layout as FlashInfer) are the intended default, **subject to ADR-0002**
  (QUAL KL vs BF16); BF16 is the reference.
- **MTP** speculative decoding with **adaptive depth** (0–3 drafts). It is off at high load unless measurement shows a
  gain.
- TP=2 over PCIe with NCCL. P2P only after a self-test passes.
- Continuous batching, paged KV, **hybrid-aware prefix caching** (turn-end state checkpoints).
- OpenAI-compatible `/v1/chat/completions`: SSE, stop sequences, max_tokens, GLM tool calls and reasoning content.
- Linux x86_64, CUDA 13.3 toolkit and driver, sm_120 only.

**Non-goals for v1.0:** vision, training, LoRA, GGUF, other GPUs, multi-node, Windows, a web UI, an SSD KV tier.

**Future work: designed so it needs no redesign, but not built in Phase 0 or 1 (PM, 19 Sep):**
- **3–4 GPUs over PCIe** through a parallel plan: PP for 3 GPUs; two 2-GPU copies, or 2 × 2 pipelines, for 4
  (ARCHITECTURE §6, ADR-0014).
- **Agent conversation memory parked in host RAM** (ARCHITECTURE §5a, ADR-0015). Recommended for **v1.1**, or earlier
  if SYS tests show idle sessions being evicted. Restore versus recompute must be measured; the
  [host-memory study](research/host-memory-inference.md) separates copy estimates from hardware results.

**Later (each needs a PM decision; research in `research/`):**
- **DeepSeek V4-family.** New attention stack (compressed sparse attention, sliding window, FP4 KV).
  - **DeepSeek-V4-Flash** (about 284B) likely fits the 2-card box at 4-bit; its fit is to be verified.
  - **DeepSeek-V4.1-Flash** does **not** fit at 4-bit (277.8 GiB). It is the **first target of the compression track
    (G5)**: about 2.5–2.7-bit experts plus ≥ 256 GB of host RAM for Engram, if the quality gates pass.
- **Compression track (G5).** Methods, pipeline and gates are in DEV-PLAN §5 track C.
- **Qwen hybrid (Gated DeltaNet).** Qwen3.5-397B does not fit 2 × 96 GB at 4-bit, so choose a size that fits.
- **Single RTX 5090 profiles.**

## 7. Memory budget (planning estimate per GPU; measured by SYS-M-001)
| Item | GiB |
|---|---|
| Routed experts at group 32, including MTP experts | ≈ 77 |
| Non-expert weights (official formats) | ≈ 7–8 |
| CUDA/NCCL context, activations, graphs | ≈ 2–3 |
| **Pages and state slots** | **≈ 7–8** |

For example:
- 8 agents × 64K tokens in FP8 pages ≈ 3.0 GiB;
- 32 KDA slots in FP32 ≈ 2.1 GiB.
- At the FR-4 target of 16 sequences, 48 slots ≈ 3.2 GiB plus 512K tokens of pages ≈ 3.0 GiB. That fits, but only
  just, so admission control and ADR-0003 (BF16 state halves the slot cost) matter.

At group 16, about 4.6 GiB less per GPU is available. The context each agent can hold is **published per
configuration**; see FR-7.

## 8. Requirements
| ID | Functional requirement |
|---|---|
| FR-1 | Load GLM-5.3-Flash from a documented checkpoint layout; verify SHA-256 before serving |
| FR-2 | Output quality per M3; sampling with temperature, top-p, top-k and seed (a counter-based RNG) |
| FR-3 | OpenAI Chat Completions: SSE, stop sequences, max_tokens, tool calls (GLM), reasoning content, and a per-request `reasoning_effort` mapped per family. **GLM-5.3-Flash** (pinned chat template, revision `eb9eb208`): `low`, `high` or `max` (default `max`); there is **no thinking-off mode**; `clear_thinking` (default false) controls whether earlier turns' reasoning is kept. Unsupported values are rejected. Image parts are rejected |
| FR-4 | Continuous batching up to the memory-derived limit (target 16 sequences, at least 8); admission never over-commits |
| FR-5 | Prefix caching with KV pages plus KDA state checkpoints at request ends |
| FR-6 | MTP with adaptive depth (0–3) chosen per step by a measured cost model, switchable, and **exact KDA-state rollback** of rejected drafts |
| FR-10 | (v1.1) Host-RAM tier: park idle agent sessions' KV pages and state checkpoints in pinned host RAM and restore them over PCIe; the host RAM budget is configurable |
| FR-9 | Deterministic indexer top-k with a defined tie-break, part of D0 (the GLM-5 paper saw RL degrade with non-deterministic top-k) |
| FR-7 | Context up to 128K per request, **subject to the pool**. The published per-agent context budget for 8 agents is from SYS-M-001 |
| FR-8 | Prometheus metrics: tok/s, queue, pool use, MTP acceptance, prefix-hit rate, latency histograms |

| ID | Non-functional requirement |
|---|---|
| NFR-1 | **Security:** no telemetry and no outbound network. Loopback by default. A **mandatory API key when bound to the LAN**. Request and token limits. No prompt logging by default |
| NFR-2 | **Reproducibility:** pinned Rust, cutile-rs, CUDA 13.3, NCCL ≥ 2.31.2 and dependencies; every benchmark stores its environment and the `tileiras` fingerprint |
| NFR-3 | **Lean code** per DEV-GUIDELINES; each kernel has a contract, a reference, tests and a benchmark |
| NFR-4 | **Licensing:** Apache-2.0; upstream notices kept; model licences respected (GLM MIT). We check whether `tileiras` may be redistributed before bundling it |
| NFR-5 | **Operations:** one binary plus a config. The CUDA toolkit (`tileiras`) is a runtime dependency. The kernel cache is pre-warmed at install and warmed up before "ready"; there is **no JIT after ready** |

## 9. Milestones and exit criteria
| Milestone | Exit criteria |
|---|---|
| **Phase 0: kernel spike** | TEST-PLAN §10: all P0 tests pass; SMLA and MOE ≥ 0.9 × the best same-card kernel; report |
| Phase 1: kernels to full model | Layer and model golden tests; the two-GPU graph lifecycle gate (SYS-R-011) before full-model integration; QUAL non-inferior (M3) on 2 × RTX PRO 6000 |
| Phase 2: server | SYS/API tests; M1, M2 and M4 vs baseline B |
| v1.0 public | M1–M7 (M7 accepted as a release criterion, P5); PM sign-off |

## 10. Risks
| Risk | Mitigation |
|---|---|
| Baseline implementations change | Pin and recheck baseline eligibility before each comparison; publish paired results |
| cuTile churn: breaking releases every 2–3 weeks, an open use-after-free (#252), no upstream GPU CI | Pin a commit; our own regression suite; one planned upgrade a month |
| JIT compile stalls while serving | Rules in DEV-GUIDELINES §1.4; a zero-JIT-after-ready test |
| sm_120 limits: 99 KB shared memory per block, no TMA gather4 or multicast | FP8 rows; tile sweep (SMLA-P-008); fallback per ADR-0010 |
| Memory: replicated latent, recurrent slots × drafts × checkpoints | The budget in §7; ADR-0003/0004/0005; admission control |
| MTP gains little at 8 users (it reads more expert bytes) | Adaptive depth, decided by measurement |
| P2P unavailable on many hosts; past NCCL crashes | NCCL without P2P by default; self-test; NCCL ≥ 2.31.2 |
| Oracle mismatch (O3 on other GPUs; FP8 KV, group 32) | Statistical QUAL; upstream sm_120 as O3 once merged |
| Determinism costs speed (20–60 %) | D1 is opt-in; D0 is the default |
| Small team (bus factor) | Docs, tests, CI; open source |
| Sub-4-bit compression loses too much quality for frontier models | The M9 gates decide; publish honest results; recovery by distillation/QAT needs a separate technical scope |
| KDA state rollback for MTP is complex | ADR-0004; KDA-R-001; the MTP depth policy (ADR-0013) can fall back to 0 |
| Prefill interference: long agent prompts stall other users' decoding (to be measured on the selected workload) | Chunked prefill with decode priority (Phase 2); SYS-P-007 |
| No MTP reference in `transformers` (layer 45 is skipped on load) | Pin the official serving implementation as the MTP oracle (REF-007) |
| The quality reference (BF16) doesn't fit our hardware | Select the EQV-001 reference runner and protocol and the BF16 acquisition plan before the Phase 1 gate. Both remain pending; M3 stays in force |
| The spec could mis-state model semantics (the review found indexer RoPE, MTP index producer and thinking modes) | Every model-level contract cites its pinned oracle (TEST-PLAN §2 and §9); re-verify at each release (track R) |

## 11. Pending technical decisions
- The adversarial-review decisions P1–P3 and P5–P7 are accepted; P4 was not approved. The existing M3 obligations
  remain in force. See the [follow-up review](reviews/2026-09-19-v0.3-follow-up.md).
- Select the BF16 reference acquisition plan and the EQV-001 reference runner and protocol before the Phase 1 gate.
- Freeze QUAL-S fixtures and thresholds before claiming M8, and WL-001 before any SYS evaluation.
- Decide format, state precision and kernel choices through the measured ADR gates in ARCHITECTURE §9.
