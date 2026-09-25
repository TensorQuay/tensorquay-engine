# TensorQuay Engine: Architecture

Status: **draft v0.3 for PM review** (19 Sep 2026). v0.3 applies the adversarial review (`reviews/2026-09-19-adversarial-review-response.md`). Sources:
- the independent review (`reviews/2026-09-19-design-review.md`);
- architecture research (vLLM V1/MRv2, SGLang, TensorRT-LLM, mistral.rs, Grout, cutile-rs);
- the paper reports in `research/`.

Related: [PRD](PRD.md), [DEV-PLAN](DEV-PLAN.md), [TEST-PLAN](TEST-PLAN.md), [DEV-GUIDELINES](DEV-GUIDELINES.md).

## 1. Design goals
1. **Modular by responsibility.** Each crate has one job and a narrow interface. Pure host logic is separated from GPU
   code, so most of the engine is testable on a Mac.
2. **Lean.** One target (sm_120), one process, closed sets as `enum`s. Model families are separate modules compiled
   with the engine (§7). Dynamic plugin loading, backend registries and a graph compiler are deferred (§8).
   We add an abstraction only when a second real implementation exists.
3. **Built for agent workloads.** About 8 long-lived sessions with growing, append-only histories. Prefix reuse with
   hybrid (KV plus recurrent) state is a first-class feature, not an add-on.
4. **Fast by construction.**
   - All memory is allocated at start-up.
   - Decode runs as captured CUDA graphs: about 700 kernel launches per step at about 1.6 µs host cost each, so graphs
     are mandatory.
   - Per-step inputs live on the GPU, with one host sync per step.
   - No JIT compilation after "ready".

## 2. Crates and dependency direction
```
             tq (binary: wires server + engine)
            /                    \
     tq-server                  tq-engine ─────────────┐
 (HTTP, API, parsers)      (worker threads, step loop) │
            \                 /        |        \      │
             \           tq-model   tq-sched     \     │
              \        (families,  (scheduler,    \    │
               \        layers,     pools, radix   \   │
                \       loader)     cache, spec)    \  │
                 \        |            |         tq-kernels ── cutile-rs
                  \       |            |       (cuTile device code + launchers)
                   \      |            |            |
                    \     └──── tq-gpu ────────────┘   (CUDA host layer: context, streams, arena, graphs, NCCL)
                     \           |
                      └────── tq-core   (pure: ids, config, model spec, kernel contracts, budget maths, protocol)

  dev-only:  tq-testkit (vectors, metrics, GPU-test gating)    tq-bench (TEST-PLAN §8 protocol, HTTP load generator)
```
| Crate | Single responsibility | Builds and tests on the Mac? | `unsafe` |
|---|---|---|---|
| `tq-core` | Newtypes (`SeqId`, `PageId`, `SlotId`, `TokenPos`, `Bytes`); TOML config; `ModelSpec` parsed from HF `config.json`; **kernel contracts** (parameter types with host-side validation, the list of specialisations); memory-budget maths; the engine protocol (`EngineCmd` / `EngineEvent`); error enums | **Yes** | forbid |
| `tq-sched` | Pure host logic: admission, scheduling, page and slot allocators, the radix prefix cache with state checkpoints, speculative-decoding bookkeeping, choice of graph bucket. It emits a `StepPlan` | **Yes** | forbid |
| `tq-server` | HTTP edge (axum/tokio): OpenAI API, SSE, tokenizer, chat template (minijinja), incremental detokenizer, GLM tool-call and reasoning parsers, auth and limits, metrics. It talks to the engine **only** through `tq-core` protocol channels, so tests use a fake engine | **Yes** | forbid |
| `tq-gpu` | The only CUDA host layer: context and stream ownership (cuda-core); `DeviceArena` (allocate once); events and timers; `GraphCache` (capture and replay per bucket); `Comm` (NCCL via cudarc, later P2P); JIT warm-up plus a compile-counter guard; mmap loading of safetensors; environment capture | No (Linux + CUDA) | allowed, audited |
| `tq-kernels` | cuTile device modules (`*_kernel.rs`) plus launchers (`*_launch.rs`) that implement the `tq-core` contracts. Grouped by family: `attn/{sparse_mla, indexer}`, `linear/{kda, conv}`, `moe/{route, align, w4a16}`, `gemm/{fp8, bf16, head_batched}`, `norm`, `mhc`, `sample`, `topk`, `kvwrite`, `embed` | No | allowed, audited |
| `tq-model` | Model families (`glm5_next`, later others), shared layers (`mla_dsa`, `kda`, `moe`, `mhc`, `mtp`), the weight loader (TP `ShardSpec`), and the forward pass over a `StepCtx` | No | forbid |
| `tq-engine` | One worker thread per GPU; the step loop; building device metadata; graph replay; sampling and speculative verification; state-slot copies. It turns `tq-sched` plans into `tq-model` calls | No | forbid |
| `tq` | Binary: config → engine + server | No | forbid |
| `tq-testkit` (dev) | Vectors from a counter-based RNG (the same stream in Rust and Python); SHA-256 manifests; metrics (rel_l2, max_abs, cosine, RMSE ratio, anomaly masks); NaN poisoning and non-contiguous helpers; GPU tests that **fail, never skip**, when no GPU is present | CPU parts | forbid |
| `tq-bench` (dev) | Kernel benchmark protocol (it reuses `cutile::bench`), cold L2, paired A/B, an HTTP load generator for SYS tests | No | forbid |

Rules:
- Dependencies point downward only (CI gate G-07).
- Phase 0 builds only `tq-core` (contracts), a minimal `tq-gpu`, `tq-kernels` (2 kernels), `tq-testkit` and `tq-bench`.
- Bootstrap decision, 20 Sep 2026: step 0.3 starts with `tq-testkit` alone, a dependency-free `no_std`
  Philox test generator under the [approved contract](../reference/contracts/philox.md). Other crates are added
  when they have approved behavior and independent tests; the planned graph is not empty scaffolding to create now.

## 3. Runtime model (one process)
```
 tokio (tq-server) ──EngineCmd──▶ scheduler thread (tq-sched) ──StepPlan──▶ worker GPU0 ─┐ NCCL all-reduce
        ▲                                   ▲                                 worker GPU1 ─┘ (one thread per GPU)
        └──────────EngineEvent (tokens)─────┴──────────── StepOutcome ◀────────┘
```
- **One worker thread per GPU.** Each worker owns its context, stream and graphs. This follows NCCL's guidance: one GPU
  per thread when collectives are captured into CUDA graphs.
- **A step:**
  1. The scheduler emits `{seq → n_tokens}` plus slot copies and frees.
  2. The workers apply the diffs to device-resident state tables.
  3. The workers replay the graph for the bucket.
  4. Rank 0 samples on the device; there is one host sync per step.
  5. The outcome goes back to the scheduler.
- **Overlap** (plan step N+1 on the CPU while the GPU runs step N) is designed for, since inputs stay on the GPU, but
  it is built only if profiling shows host time above 5 % of a step.

## 4. Key interfaces (sketches; final types are decided in code review)
```rust
// tq-core — model structure is data; "registry" is a match. Caches are keyed by id, not by layer (ADR-0011),
// so cross-layer sharing (GLM IndexCache "shared", DeepSeek CSA2 Reindex/Reuse, CED decoder) needs no redesign.
pub enum Family { Glm5Next }                       // + DeepseekV4, Qwen3x when admitted (§7)
pub struct ModelSpec { family: Family, stages: Vec<StageSpec>, caches: Vec<CacheSpec>, drafter: Option<DrafterSpec> }
pub struct StageSpec { layers: Range<LayerIdx>, tokens: TokenSet /* All | LogitTokens | LastWindowPlusLogits(n) */,
                       epilogue: Option<StageOp> }  // GLM: one stage over all tokens
pub struct LayerSpec { mixer: MixerSpec, ffn: FfnSpec, resid: ResidSpec /* Plain | Mhc{streams, iters} */ }
pub enum MixerSpec {
    Kda { state: CacheId },                                                      // GLM linear layers
    SparseGlobal { global: GlobalSrc, index: IndexSrc, swa: Option<CacheId> },   // GLM DSA (swa = None)
}                                                                                // later: Gqa, Gdn, …
pub enum GlobalSrc { Own { cache: CacheId, compress: u8 }, Shared { cache: CacheId } }
pub enum IndexSrc { Compute { k_cache: CacheId, emits: IndexSlot }, Reuse { from: IndexSlot } }
// Reuse covers GLM "shared" indexer layers (unused in this checkpoint) and MTP draft steps ≥ 1 reusing MTP step 0's
// indices (the MTP layer has its own indexer and buffer). Target verification computes indices per query. MTP never
// reuses a target layer's indices (oracle: vLLM glm5next/nvidia/mtp.py @ 98ed0856).
pub struct CacheSpec { id: CacheId, kind: CacheKind, row: RowFmt, producer: Producer }
pub enum CacheKind { TokenPaged { tokens_per_entry: u8 }, Window { tokens: u32 }, PerSeqState }
pub fn parse_hf_config(json: &str) -> Result<ModelSpec, SpecError>;   // rejects unknown architectures and fields
pub fn gap_report(json: &str) -> GapReport;   // supported vs missing fields, layer kinds, formats → tier A/B/C (DEV-PLAN track R)
impl ModelSpec { pub fn validate(&self) -> Result<(), SpecError>; }   // producer before consumer; formats; memory fits

// tq-core contracts — validated params; kernels accept nothing else
pub enum KvRow { Bf16x512, Fp8x512F32Scales /* 528 B, 16 B-aligned, runtime stride */ }
pub struct SmlaParams { /* H, T, max_q_per_seq, K, row, stride, … */ }
impl SmlaParams { pub fn new(/* shapes, strides, alignment */) -> Result<Self, ContractError>; }

// tq-kernels — plain functions over validated params (no backend traits)
pub fn sparse_mla_decode(p: &SmlaParams, q: DevRef<bf16>, kv: &KvPoolView, idx: DevRef<i32>,
                         q_pos: DevRef<i32>, out: DevMut<bf16>, lse: DevMut<f32>, s: &Stream) -> Result<()>;

// tq-model — closed sets are enums; one context for all layers in a step
pub struct StepCtx<'a> { meta: &'a StepMeta, state: &'a StateViews, ws: &'a mut Workspace, comm: &'a dyn Comm, stream: &'a Stream }
pub struct Carry { idx: IdxArena /* IndexSlot → [T, K] i32 */, stage_hidden: DevBuf<bf16> } // fixed addresses → graph-safe
pub enum Mixer { Kda(KdaLayer), SparseGlobal(SparseGlobalLayer) }
pub enum WeightFormat { Bf16, Fp8Block128, Nvfp4G16W4A16, Nvfp4G32E4m3W4A16 }  // G32-E4M3 is a distinct, non-standard format
                                                                           // later: LowBit { bits, codec } (compression track, G5)

// Speculative decoding: an enum while only MTP is built (lean rule); it becomes a trait when a 2nd drafter lands.
pub enum Drafter { Mtp(MtpLayer) }                      // later: DSpark
impl Drafter { fn max_draft(&self) -> u8; fn draft(&self, cx: &mut StepCtx<'_>, out: &mut DraftBuf) -> Result<()>; }
pub enum VerifyPolicy { Off, Fixed(u8), AdaptiveEma }  // per-request verify length, chosen by load and acceptance
// Verification, rejection sampling and cache commit are drafter-agnostic (tq-engine). The rejection sampler preserves the
// target distribution; sampled tokens are not promised equal with MTP on vs off (DEV-GUIDELINES §1.3).

// tq-sched — pure, property-testable
pub struct PagePool { /* 64-token pages: DSA latent rows + indexer K share page ids */ }
pub struct SlotPool { /* one slot = all KDA layers' recurrent + conv state, plus the indexer tail, for one sequence position */ }
pub struct PrefixCache { /* token radix tree; node = { pages, checkpoint: Option<SlotId> } */ }
impl Scheduler { fn submit(&mut self, r: Request) -> Result<(), Backpressure>;
                 fn plan(&mut self) -> StepPlan; fn on_step(&mut self, o: &StepOutcome) -> Vec<Event>; }

// tq-gpu — the only trait today (two real implementations: single GPU and NCCL)
pub trait Comm: Send { fn all_reduce_sum(&self, buf: DevMut<bf16>, s: &Stream) -> Result<()>; fn rank(&self) -> u8; fn world(&self) -> u8; }
```

## 5. Memory design (hybrid KV + recurrent state)
**Three allocation shapes cover every known cache kind** (`CacheKind`):
- `TokenPaged`: grows with the sequence, with optional compression (1 token per entry for GLM; 2 or 4 for DeepSeek
  CSA).
- `Window`: a bounded ring for sliding-window attention.
- `PerSeqState`: one fixed slot per sequence, for KDA/GDN state, conv state and drafter state.

Row formats: `Bf16`, `Fp8InlineScale` (528 B), and later `Fp4` and `State{dtype}`. **Every kind implements
`commit(accepted_len)`**, so rejected draft tokens roll back uniformly.

**v1.0 builds two fixed pools** (GLM needs only `TokenPaged` and `PerSeqState`), sized at start-up from measured free
memory. There is one admission budget: pages, plus a state slot, plus draft capacity.

| Pool | Unit | Under TP=2 | Notes |
|---|---|---|---|
| Page pool | 64-token page: DSA latent rows (FP8 528 B, or BF16 1 KiB, per token per layer) plus indexer K | **Replicated** on both GPUs (MLA has one latent per token) | FP8 rows are the default; BF16 is the reference |
| Slot pool | One sequence's KDA state (34 layers × 32 heads × 128 × 128) plus conv state | Split by heads | FP32: 68 MiB per GPU per slot; BF16: 34 MiB. Choice by ADR-0003 |

**Planning budget per GPU** (estimate, to be measured by SYS-M-001):

| Item | GiB |
|---|---|
| Experts at group 32, including MTP | ≈ 77 |
| Non-expert weights | ≈ 7–8 |
| Context, workspace and graphs | ≈ 2–3 |
| **Left for pages and slots** | **≈ 7–8** |

For example:
- 512K tokens of FP8 pages ≈ 3.0 GiB.
- 32 FP32 slots ≈ 2.1 GiB (8 active, 8 for rollback, 16 checkpoints).

**Indexer cache (ADR-0012), in two parts.**
- **Completed pools:** one pooled key per 4 tokens (`TokenPaged { tokens_per_entry: 4 }`). A complete pool's key never
  changes. At the pinned FP8 layout, 128 key bytes plus one FP32 scale cost 132 B per pool per layer: **363 B/token**
  across 11 DSA layers. A BF16 pooled-key layout would cost 704 B/token; it is a different storage assumption.
- **Per-sequence tail** (`PerSeqState`, part of the slot): the unfinished pool's raw K and gate scores, up to 3 tokens.
  - The next token must be combined with them, so pooled keys alone can't resume a sequence (vLLM keeps an equivalent
    tail cache).
  - The tail is part of every checkpoint, is copied on write on prefix hits, and is rolled back by
    `commit(accepted_len)` like the KDA state.
- The indexer applies **no rotation** for this checkpoint (`qk_rope_head_dim = 0`).
- Under TP the indexer is replicated or head-split (ADR-0005).

**Prefix cache.** A cache hit must land on a node with both pages and a state checkpoint.
- Checkpoints are taken at the **end of each request** (where the next agent turn resumes) and at page-aligned chunk
  ends.
- Restore is copy-on-write into a private slot.
- This avoids the vLLM failure mode where checkpoints land inside request-unique tokens and hits silently drop to 0 %.

### 5a. Host-RAM tier for agent sessions (ADR-0015); future work, not built in Phase 0 or 1
Agents pause between turns, sometimes for minutes, while their conversation memory holds scarce GPU memory (about
7–8 GiB per GPU in total). A **second tier in pinned host RAM** holds idle sessions' pages and KDA state checkpoints.
- **Park** an idle session (or evict under memory pressure, least recently used first) with an async copy to host.
  **Restore** it on the next turn over PCIe instead of recomputing the prefill.
- **Scale, estimated with MTP off:** a 64K session uses about **456 MiB per GPU** for pages and state, or
  **526 MiB of unique host data** when parked. The host checkpoint includes both shards of the recurrent state;
  shared latent/indexer history is stored once. These estimates exclude extra checkpoints, staging and metadata.
  - Restoring both ranks sends about **957 MB** in total. At an assumed sustained 25–50 GB/s per GPU link, the
    per-link copy lower bound is about 9.6–19.1 ms; a shared 25–50 GB/s bottleneck needs about 19.1–38.3 ms before
    overhead. Measure the actual topology, simultaneous restores and NCCL contention.
  - Compare measured restore time with measured prefill recomputation; neither is established for this engine.
  - Bound the pinned pool after reserving OS, process and loading/staging memory. Installed RAM is not usable cache
    capacity. Full accounting and active sparse-cache research are in
    [the host-memory study](research/host-memory-inference.md).
- **Design:**
  - The radix prefix cache records each node's location (GPU or host).
  - Copies run on a dedicated copy stream, overlapped with compute.
  - Under TP the replicated latent is stored once on the host and uploaded to each GPU.
  - Placement is NUMA-aware on multi-socket hosts.
- **Tests:** a restored session gives the same next-step output as one that never left the GPU (tolerance per
  TEST-PLAN API-006); restore beats recompute (SYS-P-008).
- **Later:** an SSD tier behind host RAM, for long-lived sessions and shared prefixes.
- Weights are **not** offloaded to host RAM. Our 1-GPU + RAM tests were too slow for the speed targets.

**Speculative decoding on a recurrent model** (ADR-0004) is a choice between:
- (a) one slot per draft position, promoting the accepted one (SGLang); memory-heavy;
- (b) saving k, v, g, β for the draft tokens, then recomputing the accepted prefix into the committed slot.

Either way, a rejected draft must roll the state back exactly to the last accepted token.

## 6. Multi-GPU (1–4 GPUs over PCIe) and communication; v1.0 builds TP ≤ 2, and 3–4 GPU plans are future work
**Design for up to 4 GPUs; build what's needed.** v1.0 implements TP ∈ {1, 2}. The `ParallelPlan` type and weight
sharding are defined now, so that 3–4 GPUs need new code in `tq-engine` and `tq-gpu` only, with no redesign (ADR-0014).
```rust
// tq-core
pub struct ParallelPlan { dp: u8, pp: Vec<Range<LayerIdx>>, tp: u8, gpus: Vec<GpuId> } // validated: dp × pp × tp = GPUs
```
| Box | Recommended plan | Why |
|---|---|---|
| 1 GPU | TP=1 | Only for models that fit one card |
| **2 GPUs (v1.0)** | **TP=2** | The model needs both cards' memory; about 90 all-reduces per decode step over PCIe |
| 3 GPUs | **PP=3** (layer groups) | GLM's 64 heads don't split into 3; PP only sends one activation between groups per step |
| 4 GPUs, model fits in 2 | **DP=2 × TP=2** (two independent engines) | Twice the users with no extra communication; the simplest and fastest option |
| 4 GPUs, model needs 4 | **TP=2 × PP=2** | Keeps TP traffic inside each pair; pairs pass one activation per step |

Rules:
- **Pure TP=4 over PCIe is avoided.** Each of the roughly 90 all-reduces per step gets slower with more GPUs, and
  decode is latency-bound.
- PP needs micro-batching to keep all stages busy, which our 8–16 concurrent agents provide.
- **TP details (v1.0):** attention, KDA and dense MLPs are split by heads. Experts are split by intermediate size
  (I = 1024 per GPU); expert parallelism is an ADR-0005 option. The MLA latent is replicated; decode-context
  parallelism is a later option (ADR-0005).
- **Communication:**
  - `Comm` covers TP groups: an all-reduce captured inside the CUDA graph, with **NCCL ≥ 2.31.2** (2.26 crashes on
    dual RTX PRO 6000 for ≥ 512 KiB).
  - PP adds point-to-point send/receive between stages.
  - PCIe P2P needs host settings (IOMMU/ACS), which PCIe hosts often lack. So NCCL without P2P is the
    default, and a one-shot P2P all-reduce is enabled only after a start-up self-test passes (ADR-0009).
  - **Graph and communicator lifecycle.** One worker thread per GPU removes one known deadlock cause but does not
    prove the lifecycle safe (NVIDIA's NCCL graph guidance).
    - All ranks capture and replay matching buckets.
    - Graphs are destroyed before communicators.
    - A failed or hung communicator is aborted and never reused; the fallback path creates a new one.
    - The real topology and P2P capability are measured, not assumed.
    - Gate: SYS-R-011, before full-model integration.
- **Hardware note:**
  - Bandwidth depends on the platform. A desktop platform (for example AM5) runs 2 GPUs at PCIe 5.0 x8/x8, about half
    the per-GPU bandwidth of x16/x16.
  - 3–4 GPUs at x16 need a workstation platform (Threadripper PRO / Xeon W class).
  - This affects TP all-reduce time and host-RAM cache reloads (§5a), and belongs in the hardware plan.

## 7. Model families (what is shared vs added)
**Model extension boundary.** Each family has a module in `tq-model` that supplies its validated model description,
weight mapping and sharding, forward composition, cache/state requirements and optional drafter. A new checkpoint
using supported operations can reuse that module through configuration. A new family composes existing operations;
new maths or state semantics may require a new kernel or a reviewed extension to a core contract.

This provides a plugin-style contribution path through source modules: add the family, wire its explicit selection,
and supply pinned reference fixtures, independent acceptance tests and an eval report before claiming support.
Initially modules ship with the engine and require a rebuild. There is no stable external plugin API or binary ABI
yet; a separately installable plugin system can follow demonstrated demand and a tested interface.

Shared scheduling and memory management consume validated requirements and capabilities. Model-name conditionals
belong in family selection and model modules. Reusable kernels receive shapes, formats and explicit mathematical
parameters; they do not infer semantics from a model name. Measured execution choices depend on the model's
operations, GPU, context length and concurrency. They use validated configuration and offline tuning tables
(DEV-GUIDELINES §1.4); a model module cannot bypass shared resource accounting or the correctness gates.

**Shared engine services:**
- tq-core/sched/gpu/engine/server;
- cache kinds and pools; radix cache; graphs; sampler; TP.

**Reusable operations, selected where the model's maths matches:**
- the MoE pipeline (route → align → experts → combine);
- FP8/FP4 linear; mHC;
- the sparse gather-attention core; drafter verify and commit.

| | GLM-5.3-Flash (**v1.0**) | DeepSeek-V4-Flash (candidate) | DeepSeek-V4.1-Flash | Qwen hybrid (candidate) |
|---|---|---|---|---|
| Size | 321B (Hugging Face) | about 284–291B | 552B backbone plus 196B Engram | Pick a size that fits |
| Stages | 1 | 1 | Encoder 20 → project global KV → decoder 20 | 1 |
| Global attention | `SparseGlobal { Own(c=1), Compute }` | CSA (c=4) plus HCA (c=128) | CSA2 Full / Reindex / Reuse (c=2) | GQA paged |
| Local and state | KDA slot | SWA 128 | SWA 128, bounded replay | GDN slot |
| KV row | BF16 / FP8 528 B | FP8 656 B (with RoPE) | FP4 E2M1 + E4M3/16 | BF16/FP8 |
| New kernels | NoPE sparse MLA, k-pool indexer, KDA, mHC, W4A16 | compressor, HCA, SWA, hash router, grouped o-proj | hierarchical indexer, FP4-KV attention, Engram gather, single-pass mHC | GQA flash, GDN |
| Drafter | MTP (1 layer, iterated) | MTP | DSpark | MTP |
| Fits 2 × PRO 6000 at 4-bit? | **Yes** | **Likely** (to be verified) | **No** (277.8 GiB). **Compression-track target**: about 2.5–2.7-bit experts plus ≥ 256 GB host RAM, if M9 passes | Qwen3.5-397B: **no** (compression-track candidate) |

- v1.0 builds only the GLM variants. Other enum variants arrive with their model; exhaustive `match` lists every site to
  update.
- Research: `research/glm-5.md` and `research/deepseek-v4.1-flash.md`.
- **Risk:** DeepSeek changed its attention design three times in about 12 months (DSA → CSA/HCA → CSA2 + CED). Cache ids,
  stages and carry slots in the core interfaces absorb that churn. Always check that the next target **fits in memory**
  before committing to it.

## 8. What we deliberately do NOT build yet
- A device or backend trait (one target).
- Attention-backend or quant-method registries, dynamic model-plugin loading or a stable external plugin ABI
  (built-in family modules use `match`, §7).
- A graph IR or fusion compiler (capture the imperative forward pass).
- Multi-process or prefill/decode disaggregation.
- An SSD KV tier (the host-RAM tier in §5a is planned).
- Elastic pools.
- EP or DCP. PP and DP are designed for (§6) and built when a 3–4 GPU box is planned.
- An online autotuner.
- Grammar-constrained decoding.
- LoRA.
- Vision (explicitly rejected at the API).

## 9. Decision records to write (docs/adr/)
| ADR | Decision | Decided by |
|---|---|---|
| 0001 | Crate layout (this document) | PM approval |
| 0002 | KV row format: FP8 528 B default, BF16 reference | QUAL KL |
| 0003 | KDA state dtype: FP32 vs BF16 | Rollback and long-context QUAL |
| 0004 | Speculative state: slot per draft vs save-and-recompute | Memory plus SYS-P-003 |
| 0005 | Experts TP vs EP; latent replicated vs DCP vs DP-attention for the DSA layers; indexer replicated vs head-split | SYS-P-001, SYS-M-001, IDX set-equality test |
| 0006 | Determinism tiers D0/D1 (DEV-GUIDELINES §1.3) | — |
| 0007 | Where `unsafe` mmap lives (`tq-gpu`) | — |
| 0008 | Expert codec (NVFP4; INT4 if an official QAT checkpoint appears) and group size 16 vs 32 | Measured KL/QUAL and memory |
| 0009 | NCCL-only vs P2P one-shot all-reduce | P2P self-test, SYS-P-001 |
| 0010 | Kernel fallback language: vendored CUDA C++ (e.g. FlashInfer, Apache-2.0) before cuda-oxide (needs nightly plus LLVM 21) | Phase 0 gate |
| 0011 | Cache ids, stages and carry slots are v1.0 interfaces (only GLM's variants are built) | PM approval |
| 0012 | Indexer cache layout (pooled keys plus a per-sequence tail) and precision (BF16/FP8) | IDX and IDX-R tests, QUAL KL |
| 0013 | MTP draft-depth policy (a cost model from measured step time and acceptance) | SYS-P-006 |
| 0014 | Parallel plan for 3–4 GPUs (PP=3; DP=2 × TP=2; TP=2 × PP=2) | SYS-P-009 on the real box |
| 0015 | Host-RAM tier: park and restore policy, eviction, pinned-memory budget | SYS-P-008, SYS-R-010 |
| 0016 | Relationship to NVIDIA cutile-rs: pinned upstream, extensions in our crates, a minimal-patch fork only if unavoidable, monthly rebase and upgrade | PM approval |
