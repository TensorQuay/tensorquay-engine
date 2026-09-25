# TensorQuay Engine: architecture research (2026-09-19)

Sources checked this session. "Unverified" = I could not confirm it from a primary source. "Estimate" = my own arithmetic.

## A. Key facts that affect the plan

### A1. Upstream has nearly closed the gap the PRD is built on
1. **FlashInfer merged GLM-5.3 NoPE sparse MLA for SM120 on 2026-09-11** (#5075). It adds a canonical 528-byte FP8 row (512 × E4M3 plus 4 × FP32 inline scales), a runtime row stride and NaN-safe masked gathers. A refactor that unifies SM120 execution landed on 09-18 (#5197). https://github.com/flashinfer-ai/flashinfer/pull/5075 , https://github.com/flashinfer-ai/flashinfer/pull/5197
2. **The engine integrations are still open, but close.**
   - vLLM #53963 is still open. On 2026-09-18 a Z.ai maintainer wrote that GLM-5.3-Flash "runs on RTX PRO 6000 with FlashInfer's PR #5075 plus a temporary NoPE KV-cache writer workaround". vLLM PR #55277 is still open.
   - SGLang #39302 needs #5075 plus PR #38430, which is also open.
   - https://github.com/vllm-project/vllm/issues/53963 , https://github.com/vllm-project/vllm/pull/55277 , https://github.com/sgl-project/sglang/issues/39302 , https://github.com/sgl-project/sglang/pull/38430
   - **So the PRD §2 claim that official engines can't run the model will probably expire within weeks.**
3. **FlashInfer ships cuTile (Python) fused MoE kernels, including NVFP4 W4A16 on SM120.**
   - #4646 merged 09-01; #5099 (MXFP4 and W4A16) merged 09-15; tracker #4857.
   - Tile IR is the same compiler target as cutile-rs, so these kernels are a design reference and a same-card baseline.
   - https://github.com/flashinfer-ai/flashinfer/issues/4857
4. **mistral.rs already ships cuTile Rust kernels.**
   - NVFP4 W4A4/W4A16 dense, GEMV and MoE; FP8 block MoE; GDN prefill.
   - A load-time autotuner built on `cutile::experimental-tune`: per-bucket search, a correctness gate against the default config, a 3 % margin, cold-L2 timing by rotating model layers, and a cache keyed by kernel source, arch and tileiras build.
   - Its fused MoE is the Triton design (sorted_token_ids, `load_ptr_tko` gathers).
   - Stream sharing: it borrows candle's cudarc stream into cuTile with `Stream::borrow_with_owner`.
   - https://github.com/EricLBuehler/mistral.rs/pull/2413 , https://github.com/EricLBuehler/mistral.rs/pull/2418 , https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-quant/src/cutile/context.rs
5. **The bar on our own hardware.** ormandj's SGLang v0.4.3 on 2 × PRO 6000 Max-Q at 300 W, TP2, PCIe:
   - C1 197 tok/s; C4 425 tok/s aggregate (106 per request) at 35 forwards/s with MTP, about 3 tokens per forward.
   - Prefill 5.6–6.4K tok/s.
   - 524,288-token FP8 KV pool; 28 recurrent-state slots for 4 requests ("4–5 per live request").
   - Checkpoint: W4A16 NVFP4 **E4M3 scales over K=32** experts plus FP8 128×128 weight-only; MTP, indexer, embeddings, LM head and routers in BF16. 178 GB, GSM8K 96.9 %.
   - Their FlashInfer patch adds `sparse_mla_sm120` and CuTe-DSL `moe_w4a16` kernels. Their SGLang patch adds a k-pool indexer, radix top-k, FLA KDA plus `fused_kda_conv_recurrent_verify`, an mHC kernel, `mamba_radix_cache` and breakable CUDA graphs.
   - https://github.com/ormandj/sglang-glm53-flash-sm120 , https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
   - vLLM overlay TP2 without MTP: about 65 tok/s at bs 1; 184,755-token FP8 pool with a 177 GiB NVFP4 checkpoint (#53963, comment of 08-31).

### A2. cutile-rs (local clone e04245b; 0.3.1 released 2026-09-02; 0.4.0 in progress)
1. **The API is still moving.**
   - 0.3.0 (08-20) and 0.3.1 (09-02) broke APIs for soundness after a "September audit". 0.4.0 breaks `Global` again.
   - The README still says "expect bugs, incomplete features, and API breakage".
   - There is no GPU CI yet (#156 open).
   - https://github.com/NVlabs/cutile-rs/blob/main/CHANGELOG.md , https://github.com/NVlabs/cutile-rs/issues/156
2. **Open bugs that touch us:**
   - #252: possible use-after-free, a free queued on another stream before `sync_on` (seen on sm_120).
   - #215: bounds-check placement ignores control dependence; a hoisted check can trap valid programs. Fix in PR #293.
   - #274: the 0.3.1 JIT fails on some `&Tensor` generics.
   - #270 (open PR): lifetime-brand `CudaGraph` so captured buffers can't dangle.
   - Programmatic dependent launch (PDL) is not merged (#228, #298).
   - https://github.com/NVlabs/cutile-rs/issues/252 , /issues/215 , /issues/274 , /pull/270 , /pull/298
3. **Specialization.**
   - Dynamic dims (`-1`) and **integer scalar arguments** are bucketed by power-of-two divisibility, capped at 16. Moving from a value divisible by 8 to one divisible by 16 creates a new cache entry, and so a **JIT compile at first launch**.
   - `CompileOptions` and const generics are also part of the key.
   - The disk cache is opt-in. Its checksum gives "integrity, not authenticity".
   - https://github.com/NVlabs/cutile-rs/blob/main/cutile-book/guide/jit-compilation.md
4. **Compile-only testing without a GPU.**
   - `KernelCompiler` (default target sm_120) produces Tile IR and bytecode with no GPU. cuTile's own CPU test suite then lowers Tile IR to cubin with `tileiras`.
   - It reports `CheckPlacementCounts` (discharged/hoisted/in_place).
   - `deny_in_kernel_checks = true` turns any remaining in-kernel check into a compile error.
   - https://github.com/NVlabs/cutile-rs/blob/main/cutile-compiler/src/compile_api.rs
5. **Tensor metadata is i32 per dimension** (`from_raw_parts(.., Vec<i32>, Vec<i32>)`), so a flat view with more than 2^31 elements is impossible. The width of in-kernel offset arithmetic is **unverified**. (cutile/src/tensor.rs)
6. **Block-scaled MMA:** `mmaf_scaled` is native on sm_120 only for NVFP4 with E4M3 scales at group 16 and MXFP4/MXFP8 with E8M0 scales at group 32. Group-32 E4M3 is not native, which confirms the PRD's W4A16 dequant path. Mixed FP8 × FP4 is still an open PR (#230). https://github.com/NVlabs/cutile-rs/blob/main/cutile-book/tutorials/11-nvfp4-inference.md
7. **Gathers and hints:**
   - Gathers use `PointerTile` plus `unsafe load_ptr_tko` (with mask and fill value). There is no explicit shared-memory control.
   - TMA is a per-op hint (`tma::Enabled`).
   - Entry hints: `occupancy`, `num_cta_in_cga` and, for bytecode 13.3, `num_worker_warps_per_cta`.
   - Persistent kernels use mapped partitions.
   - https://github.com/NVlabs/cutile-rs/blob/main/cutile-book/reference/dsl-api.md
8. **Host side:**
   - A cache-hit launch costs about 1.6 µs of host time (5090), so about 700 launches per step is about 1.1 ms (estimate). **CUDA graphs are mandatory.**
   - cuda-core has CUDA VMM wrappers (#202).
   - `Tensor::from_foreign` and `borrow_with_owner` allow zero-copy interop with cudarc.
9. **Grout** (HF, cutile 0.2.0):
   - Batch-1 only. Runs cuBLAS through cudarc 0.19.2 for all GEMMs. Replays a captured CUDA graph for decode.
   - Tunes through environment variables. `model.rs` is 5,958 lines.
   - GPU tests **silently pass when no GPU is present**.
   - https://github.com/huggingface/grout

### A3. Other Rust CUDA options
1. **cuda-oxide is alpha** (v0.2.1, 2026-06-10). It needs a pinned nightly (nightly-2026-08-28) and LLVM 21, which **conflicts with our stable-toolchain pin**. It has shown inter-kernel interop with cuTile on the same stream. It recently fixed TMA global-to-shared copies on SM120 (#1249). https://github.com/NVlabs/cuda-oxide
2. **cudarc 0.19.9** (2026-08-11, repo moved to chelsea0x3b/cudarc) has the features `cuda-13030`, `nccl` (bindings up to 2.30), `cublas`, `cublaslt`, `cupti`, `nvtx` and `dynamic-loading`. cuTile's `cuda-core` has **no NCCL**. https://crates.io/crates/cudarc

### A4. Two-GPU communication on RTX PRO 6000
1. **P2P works only with IOMMU off (or passthrough) plus ACS off**, and `nvidia_uvm uvm_disable_hmm=1` according to one community guide. Otherwise NCCL hangs on the first collective. Community reports only; NVIDIA doesn't document it (unverified).
   - https://forum.level1techs.com/t/dual-rtx-pro-6000-blackwell-max-q-how-to-make-p2p-nccl-work/242403 , https://github.com/voipmonitor/rtx6kpro/blob/master/optimization/nccl-tuning.md
2. The working community vLLM TP2 configs set `NCCL_P2P_DISABLE=1 --disable-custom-all-reduce` (#53963).
3. **NCCL 2.26.2 hits an illegal memory access on dual RTX PRO 6000 for collectives ≥ 512 KiB; 2.31.2 is fine.** https://github.com/NVIDIA/nccl/issues/2418
4. Latency: NCCL all-reduce of 32–64 KB takes about 26 µs with a topology XML, about 48 µs without. A custom one-shot PCIe all-reduce is 1.4–6× faster below 512 KB and gives about +7 % decode on GLM-5, but **needs P2P** (rtx6kpro guide).
5. NCCL collectives can be captured in CUDA graphs (NCCL ≥ 2.9). **Multiple GPUs per process with graph launches can deadlock; NCCL recommends one GPU per process** (or one thread per GPU). https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
6. **Scale estimate:** 90 all-reduces per step × 25–50 µs is 2.3–4.5 ms per step.

### A5. sm_120 hardware
1. **RTX PRO 6000:** 188 SMs, 128 MB L2, 1,792 GB/s, 96 GB GDDR7, CC 12.0. Workstation Edition 600 W; Max-Q 300 W.
   - https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ , https://blog.us.fixstars.com/what-kind-of-gpu-is-the-nvidia-rtx-pro-6000-blackwell-max-q/
   - RTX 5090: 170 SMs; its L2 size is unverified, so measure it in H-001.
2. **Shared memory: 99 KB per block (101,376 B opt-in), 100 KB per SM**, verified with `cudaFuncSetAttribute`. Kernels written for Hopper's 228 KB fail: TileLang's sparse forward pass asks for 206,848 B. https://github.com/sgl-project/sglang/issues/39302
3. **Tensor cores and missing features.**
   - Only `mma.sync ... kind::mxf4nvf4.block_scale ... m16n8k64` exists, with UE4M3 scales at group 16 or UE8M0 at group 32.
   - **No tcgen05, TMEM, 2-CTA MMA or TMA multicast.** TMA itself exists. SM10x kernels do not run on SM12x.
   - Cluster shape is fixed at 1×1×1 because there is no multicast.
   - https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/ , https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
4. **TMA `tile::gather4` is rejected on sm_120a** according to a third-party ptxas reference (unverified with NVIDIA). So sparse gathers are per-row `cp.async` or `cp.async.bulk`; FlashInfer's SM120 sparse-MLA kernel requires 16 B-aligned rows for `cp.async.bulk` (#5075). https://gh.evko.io/crucible-notes/ptxas/sass-isa/async-copy.html
5. **FLA KDA:** FLA's Triton KDA kernels are what the SM120 SGLang build uses. MoonshotAI/FlashKDA (CUTLASS) says "SM90 and above", but whether it supports sm_120 is unverified. https://github.com/MoonshotAI/FlashKDA

### A6. Where the leading engines converged
1. **vLLM V1:**
   - The EngineCore process (scheduler plus executor) is separate from the API server, connected by ZMQ.
   - The scheduler output is a plain `{request_id: num_tokens}` dict with no prefill/decode split.
   - Persistent batch applies diffs each step; prefix caching costs < 1 % even at a 0 % hit rate; piecewise CUDA graphs.
   - https://vllm.ai/blog/2025-01-27-v1-alpha-release
2. **vLLM MRv2** (2026-03):
   - A persistent state table with `max_num_reqs` rows; block-table diffs staged on the CPU and applied by one kernel.
   - **GPU-native input preparation.** A Gumbel sampler with **stateless RNG seeded per request**. A `CUDAGraphManager` that captures all draft steps in one graph.
   - **"Async-first with zero CPU–GPU sync"**, and isolated model-specific logic.
   - https://vllm.ai/blog/2026-03-24-mrv2 , https://docs.vllm.ai/en/latest/design/model_runner_v2/
3. **vLLM MoE:** the modular kernel is Prepare/Finalize (activation quantization and dispatch/combine) + Experts (`apply`, `workspace_shapes`) + TopKWeightAndReduce. https://docs.vllm.ai/en/latest/design/fused_moe_modular_kernel/
4. **vLLM hybrid KV manager:**
   - It unifies page size (attention block size is raised until it matches the Mamba state, and state pages are padded).
   - The coordinator supports only two groups.
   - Mamba prefix caching is experimental ("align" mode). Hits silently drop to 0 % when the checkpoint lands in request-unique tokens (#45238).
   - It also needed stride tricks so that attention and state views don't corrupt each other.
   - https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/ , https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/ , https://github.com/vllm-project/vllm/issues/45238
5. **SGLang hybrid:**
   - Two separate fixed pools: a Mamba pool per request and a KV pool per token.
   - `MambaRadixCache`: a match copies the state; an insert forks a checkpoint; two LRUs.
   - **Speculative decoding gives each draft token its own state slot; the last accepted slot is promoted.**
   - Unified Radix Cache (2026-08) puts one token radix tree under FULL/SWA/MAMBA components, each with its own validator.
   - https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/ , https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/
6. **SGLang overlap scheduler:** the CPU schedules batch N+1 while the GPU runs batch N. https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/
7. **mini-sglang** (about 5K lines of Python): API server, tokenizer and a scheduler per GPU; radix cache, chunked prefill, overlap scheduling, TP. https://www.lmsys.org/blog/2025-12-17-minisgl/
8. **TensorRT-LLM (PyTorch backend):** PyExecutor = ModelEngine + two-stage Scheduler (Capacity, then MicroBatch) + Decoder + ResourceManagers with a `prepare/update/free_resources` lifecycle. https://nvidia.github.io/TensorRT-LLM/torch/arch_overview.html
9. **TGI:** maintenance mode, archived 2026-03-21. https://github.com/huggingface/text-generation-inference
10. **vLLM with DeepSeek-V3.2:** a separate FP8 indexer K cache (block 64); 656 B `fp8_ds_mla` rows. https://vllm.ai/blog/2025-09-29-deepseek-v3-2
11. **Decode context parallel (DCP):** under TP the MLA latent is **replicated on every rank**; DCP shards it along the sequence instead. https://vllm.ai/blog/2026-08-07-decode-context-parallelism

### A7. Determinism
1. Thinking Machines (2025-09-10): nondeterminism comes from **missing batch invariance**, not from atomics.
   - Fixes: a fixed reduction order per row; no split-K that changes with batch size (about 20 % slower than cuBLAS); a **fixed split size** for split-KV attention; the same KV layout for prefill and decode.
   - End to end: 26 s → 55 s → 42 s; 1000 of 1000 completions identical.
   - https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
2. SGLang's deterministic mode is about 34 % slower on average and works with chunked prefill, CUDA graphs and the radix cache. https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/

### A8. How kernel libraries test and benchmark
1. **FlashAttention** tests `max|out−ref| ≤ 2·max|out_pt−ref|`, the same rule as our TEST-PLAN §2. https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
2. **FLA** tests `RMSE/RMS(ref) < ratio` per test. https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils/_testing.py
3. **FlashMLA kernelkit:**
   - Inf/NaN positions must match exactly.
   - abs/rel tolerance plus cosine difference ≤ 1e-7, and bitwise-equality checks.
   - Non-contiguous inputs.
   - Cases with all indices invalid, zero seqlen, and per-row `topk_length`.
   - Timing with CUPTI after an 8 GB memset to flush L2.
   - https://github.com/deepseek-ai/FlashMLA/blob/main/tests/kernelkit/compare.py
4. **FlashInfer `bench_gpu_time`:**
   - Timing by CUPTI, CUDA-graph replay or CUDA events.
   - Cold L2 by flushing 2 × L2, or by **rotating buffers when the working set is below 5 × L2**.
   - https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/testing/utils.py
5. **FlashInfer #5075 found real bugs we should test for:**
   - Masked (−1) lanes were clamped to cache **slot 0** in 7 places, so a NaN there leaked through `0·NaN`.
   - Stale padding past `topk_len` caused an illegal memory access.
   - A calibration-cache schema needed a version bump.
   - It ships 782 tests on RTX PRO 6000.
6. **GitHub on self-hosted runners:** "should almost never be used for public repositories". Use just-in-time ephemeral runners; avoid `pull_request_target` with an untrusted checkout. https://docs.github.com/en/actions/reference/security/secure-use
7. **cargo-llvm-cov** 0.9.1: branch and MC/DC coverage are still nightly-only. https://github.com/taiki-e/cargo-llvm-cov

### A9. Model facts the drafts miss
1. **GLM-5.3-Flash** `config.json` (https://huggingface.co/zai-org/GLM-5.3-Flash):
   - **The indexer uses RoPE** (`indexer_rope_interleave: true`) even though the MLA is rope-free.
   - `indexer_types` can be "full" or "shared" (IndexCache cross-layer reuse); all layers are "full" in this model.
   - `index_share_for_mtp_iteration: true`: the MTP pass reuses the target's indices.
   - `index_kpool_compress: true`.
   - The router is sigmoid `noaux_tc` in FP32.
   - KDA has 64 heads × 128.
   - The config includes a vision tower.
2. **"DeepSeek next" now means V4-Flash** (284B total, 13B active, April 2026, arXiv 2606.19348):
   - CSA (compress 4) + HCA (compress 128) + sliding window 128, head_dim 512 with rope 64, one KV head, indexer top-512.
   - 256 experts top-6, 3 hash-routed layers, `o_groups` 8, mHC, MTP.
   - FP4 experts; FP8 blocks with UE8M0 scales.
   - **This is not "reuse sparse MLA"**, although FlashInfer already has SM120 DSV4 kernels.
   - https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
3. **Qwen3.5-397B-A17B** has GQA full attention every 4th layer (head dim 256, output gate), GDN, and 512 experts top-10. **At 4-bit it doesn't fit 2 × 96 GB.** https://huggingface.co/Qwen/Qwen3.5-397B-A17B

### A10. Performance envelope (estimate, TP=2, E4M3 group-32 experts, uniform routing)
- One expert shard per GPU is about 6.7 MB. The expected number of distinct experts per layer is `288·(1−(1−8/288)^T)`:
  - T=8: 58
  - T=16: 105
  - T=32: 171
  - T=64: 240
- Across 42 MoE layers, the **routed-expert bytes per step per GPU** are:

| T | Bytes | Time at 1.43 TB/s (80 % of peak) |
|---|---|---|
| 8 | 16 GB | 11 ms |
| 16 | 29 GB | 21 ms |
| 32 | 48 GB | 34 ms |

- **This matches ormandj's measurement:** our model gives about 39 forwards/s at C4 with 16 tokens; they measured 35.
- **At 8 agents without MTP**, about 45–50 tok/s per user is feasible, so M1 (≥ 30) has margin.
- **With 4-token MTP**, the extra expert bytes cut the gain to about 1.3× (from about 3× at C1). This matches our own llama.cpp finding that "MTP hurts at 4–32 users".
- **Conclusion:** MTP must be adaptive or off by default at C8; decide by measurement.
- **Memory:**
  - The MLA latent is replicated on both GPUs: 11 × 528 B plus the indexer is about 6.2 KB per token with FP8 rows, or about 11.5 KB with BF16. 512K tokens is 3.2 or 5.9 GB per GPU.
  - One KDA state slot is 34 layers × 32 heads × 128 × 128 = **68 MiB in FP32, 34 MiB in BF16, per GPU**.
  - One slot per draft token (the SGLang design) for 8 sequences × 5, plus 16 checkpoints, is 56 slots: 3.7 GiB in FP32. **That is a large share of the roughly 10 GiB left per GPU.**

## B. Proposed modular architecture

### B1. Crates (single responsibility; ↓ = may depend on)
| Crate | Responsibility | GPU? | Unsafe? |
|---|---|---|---|
| `tq-core` | Newtypes (SeqId, PageId, SlotId, TokenPos, Bytes); `ModelSpec` parsed from HF `config.json`; TOML config; memory-budget maths; checkpoint manifest and hashes; error enums | no | forbid |
| `tq-sched` | **Pure host logic**: scheduler and admission, page and slot allocators, radix prefix cache with state checkpoints, speculative-decode bookkeeping, graph-bucket choice. Emits a `StepPlan` | no | forbid |
| `tq-gpu` | The only CUDA host layer: device and stream ownership (cuda-core), `DeviceArena` (allocate once, VMM-backed), events and timers, `GraphCache` (capture and replay per bucket), `Comm` implementations (NCCL FFI; later P2P), JIT warm-up and compile-counter guard, environment capture | yes | allowed |
| `tq-kernels` | cuTile device modules (`*_kernel.rs`) plus checked launchers (`*_launch.rs`) with `Params::new() -> Result`. Grouped by family: `attn/{sparse_mla,indexer}`, `linear/{kda,conv}`, `moe/{route,align,w4a16}`, `gemm/{fp8_w8a16,bf16}`, `norm`, `mhc`, `sample`, `topk`, `kvwrite`, `embed` | yes | allowed |
| `tq-model` | Model families (`glm5_next.rs`, later `deepseek_v4.rs`, `qwen3_5.rs`), shared layers (`mla_dsa`, `kda`, `moe`, `mhc`, `mtp`), weight loader (mmap safetensors → arena, TP `ShardSpec`), forward pass over a `StepCtx` | yes | forbid (goes through tq-gpu/tq-kernels) |
| `tq-engine` | One worker thread per GPU; step loop; builds device metadata; graph replay; sampling and speculative verify; state-slot copies; bridges `tq-sched` plans to `tq-model` | yes | forbid |
| `tq-server` (bin `tq`) | axum/tokio edge; OpenAI API and SSE; tokenizer, chat template (minijinja), incremental detokenizer, GLM tool-call and reasoning parsers; `EngineHandle` channels; Prometheus | no | forbid |
| `tq-testkit` (dev-dependency only) | Vector and manifest loading with SHA-256; metrics (rel_l2, max_abs, cosine, RMSE ratio, anomaly masks); NaN-poison and non-contiguous helpers; GPU-test gating that **fails, never skips** | optional | forbid |
| `tq-bench` | TEST-PLAN §8 kernel protocol (events, CUPTI, graph replay, cold L2) plus an HTTP load generator | yes | forbid |

Dependency direction: `tq-core ← tq-sched`, `tq-core ← tq-gpu ← tq-kernels ← tq-model ← tq-engine ← tq-server`, and `tq-sched ← tq-engine`. `tq-bench` depends on tq-kernels and tq-gpu only; serving benchmarks talk HTTP. `tq-testkit` is a dev-dependency only.

Threads: tokio runs only in tq-server. There is one scheduler thread, and one worker thread per GPU, each owning its stream and graphs. This is the NCCL guidance and avoids GIL-style problems. Keep a single process.

### B2. Key interfaces (sketches)
```rust
// tq-core: model definition / "registry" is a match, not a plugin system
pub enum Family { Glm5Next /*, DeepseekV4, Qwen35 */ }
pub struct ModelSpec { family: Family, hidden: u32, vocab: u32, layers: Vec<LayerSpec>, mtp: Option<MtpSpec> }
pub struct LayerSpec { mixer: MixerSpec, ffn: FfnSpec, resid: ResidSpec /* Plain | Mhc{streams:4, iters:20} */ }
pub enum MixerSpec { Kda(KdaSpec), MlaDsa(MlaDsaSpec) /* v1.1: Csa, Hca, Swa; v1.2: Gqa, Gdn */ }
pub enum FfnSpec { Dense(DenseSpec), Moe(MoeSpec) }
pub fn parse_hf_config(json: &str) -> Result<ModelSpec, SpecError>; // rejects unknown architectures and fields

// tq-model: closed sets are enums; one per-step context shared by all layers
pub struct StepCtx<'a> { pub meta: &'a StepMeta /*device: ids, pos, cu_seqlens, page_table, slot_ids, spec_layout*/,
                         pub state: &'a StateViews, pub ws: &'a mut Workspace, pub comm: &'a dyn Comm, pub stream: &'a Stream }
pub enum Mixer { Kda(KdaLayer), MlaDsa(MlaDsaLayer) }
impl Layer { pub fn forward(&self, h: &mut Hidden, cx: &mut StepCtx<'_>) -> Result<(), ModelError>; }

// Attention: kernels are functions with validated params, not a backend trait
pub enum IndexSource { Compute, ReuseFrom(LayerIdx) }              // GLM: all Compute; MTP reuses target indices
pub enum KvRow { Bf16x512, Fp8x512InlineF32Scales /*528 B, 16 B aligned, runtime stride*/ }
pub fn sparse_mla_decode(p: &SmlaParams, q: DevRef<bf16>, kv: &KvPoolView, idx: DevRef<i32>, out: DevMut<bf16>, lse: DevMut<f32>, s: &Stream) -> Result<()>;
pub fn kda_chunk(p: &KdaParams, qkvgb: .., slots_in: DevRef<SlotId>, slots_out: DevRef<SlotId>, s: &Stream) -> Result<()>;
pub fn kda_recurrent(p: &KdaParams, .., slots: DevRef<SlotId>, draft: Option<DraftLayout>, s: &Stream) -> Result<()>; // verify writes per-position states or saves k,v,g,β for recompute (ADR)

// Quantized linear / MoE: an enum over the formats we ship; vLLM's split kept as plain functions
pub enum WeightFormat { Bf16, Fp8Block128 { scale: ScaleFmt /*F32 | Ue8m0*/ }, Nvfp4 { group: G /*16|32*/ } /* v1.1: Mxfp4 */ }
pub struct Linear { w: QWeight, fmt: WeightFormat, shard: Shard /*Col | Row | Replicated*/ }
pub struct MoeLayer { router: Router /*sigmoid noaux_tc f32, scale 2.5*/, bank: ExpertBank, shared: Option<Mlp> }
fn route(..) -> RouteOut; fn align(..) -> ExpertSchedule /*device-built*/; fn experts_w4a16(..); fn combine(..);

// tq-sched: pure; two pools, one admission budget (SGLang model, not vLLM's unified page)
pub struct PagePool { page_tokens: u32, free: Vec<PageId>, refcnt: Vec<u16> }   // DSA latent + indexer K share page ids
pub struct SlotPool { free: Vec<SlotId>, refcnt: Vec<u16> }                     // one slot = all KDA layers' state + conv
pub struct PrefixCache { /* token radix tree; node = { pages, ckpt: Option<SlotId> }; FULL + STATE validators */ }
pub struct Scheduler { .. }
impl Scheduler {
    pub fn submit(&mut self, r: Request) -> Result<(), Backpressure>;
    pub fn plan(&mut self) -> StepPlan;                      // {seq: n_tokens}, slot copies, frees, graph bucket
    pub fn on_step(&mut self, o: &StepOutcome) -> Vec<Event>; // accepted counts, finished, checkpoints to keep
}

// Sampler and speculative decoding: on device, counter-based RNG
pub struct SamplingParams { temperature: f32, top_p: f32, top_k: u32, seed: u64 }
// sample(logits, params_table, philox(seed, seq_id, position)); verify_greedy / verify_rejection on device; one host sync per step

// TP
pub trait Comm: Send { fn all_reduce_sum(&self, buf: DevMut<bf16>, s: &Stream) -> Result<()>; fn rank(&self) -> u8; fn world(&self) -> u8; }
// impls: SingleGpu (no-op), Nccl (≥ 2.31.2, graph-capturable); later P2pOneShot, only when the P2P self-test passes

// Server ↔ engine
pub enum EngineCmd { Submit { req: Request, tx: mpsc::Sender<EngineEvent> }, Cancel(ReqId), Stats(oneshot::Sender<Stats>) }
pub enum EngineEvent { Tokens { ids: Vec<u32> }, Finished { reason: Finish, usage: Usage }, Error(EngineError) }
```

### B3. Hybrid state design (the decision the drafts lack)
- **Two pools:**
  - KV pages: 64-token pages, replicated on both GPUs under TP.
  - Recurrent slots: split by heads under TP.
  - Both sizes are computed at start-up from measured free memory. Admission reserves pages plus 1 slot plus draft capacity.
- **Prefix cache:** a match must hit a node that has both KV pages and a state checkpoint. Checkpoint at the **end of each request**, which is where the next agent turn resumes, and at page-aligned chunk ends. Restore with copy-on-write into a private slot. This avoids vLLM's #45238 failure (checkpoint landing in unique tokens).
- **Speculative decoding with KDA** needs an ADR choosing between:
  - (a) one slot per draft position (SGLang; about 3.7 GiB per GPU at C8 in FP32);
  - (b) save k, v, g, β for the draft tokens and recompute the accepted ones into the committed slot at the start of the next step (about one scratch slot per sequence).
- **State dtype** (FP32 vs BF16 slots, where ormandj uses BF16) gets its own ADR, backed by a quality test.

### B4. Per model family
- **Shared across all three:** tq-core/sched/gpu/engine/server, the MoE pipeline, FP8 and FP4 linear, mHC (GLM and DSV4), MTP and the sampler, recurrent-slot and page pools, the radix cache and TP.
- **GLM-5.3 adds:**
  - sparse-MLA NoPE (BF16 and FP8 528 B rows), the k-pool FP8 indexer with RoPE plus radix top-k with tail;
  - KDA (chunk, recurrent, verify, short conv), the sigmoid `noaux_tc` router, mHC, W4A16 group 32, FP8 W8A16.
- **DeepSeek-V4-Flash adds:**
  - CSA and HCA compressed-KV writers and attention, sliding-window pages (a new cache component), dual-cache pages;
  - hash-routed layers, grouped `o_proj`, YaRN RoPE, the 656 B rope-carrying row, top-512 without k-pool;
  - FP4 expert and UE8M0 FP8 scale variants.
- **Qwen3.5 hybrids add:** GQA paged attention (prefill and decode, with output gate) and GDN (reusing the KDA chunk structure with a scalar gate). Choose a size that fits the box.

### B5. What not to abstract yet
- No device or backend trait (one target).
- No attention-backend or quant-method registry, and no model plugin system; use `match`.
- No graph IR or fusion compiler; capture the imperative forward pass into CUDA graphs.
- No multi-process design and no PD disaggregation.
- No HiCache or host offload, no elastic VMM pools, no EP/DCP/PP.
- No online autotuner in the server.
- No overlap scheduler until a profile shows host time above 5 % of a step. Keep inputs GPU-resident (MRv2) so it can be added later without a redesign.
- No grammar-constrained decoding and no LoRA.

## C. Recommended changes

### DEV-PLAN
1. **§2 crate table:** adopt B1, i.e. add `tq-core`, `tq-sched`, `tq-gpu`, `tq-testkit` and rename tq-runtime to tq-engine. Why:
   - a pure `tq-sched` gets property, mutation and Miri testing on the Mac;
   - unsafe FFI stays in two crates;
   - the lesson from Grout's 6K-line `model.rs` (A2.9).
2. **Phase 0, new step 0.4b "platform spike" (CPU CI plus 1 GPU hour):**
   - compile-only builds of every manifest specialization to cubin (A2.4);
   - a check that no JIT happens after warm-up (A2.3);
   - CUDA-graph capture of cuTile kernels together with NCCL through cudarc on one stream (A4.5);
   - a `from_foreign` interop smoke test (A2.8).
3. **Baselines at the gate:** build FlashInfer main (sparse_mla_sm120 with GLM53_NOPE from #5075, and the cuTile W4A16 MoE from #5099) for SMLA-P-004 and MOE-P-002. It is Apache-2.0 and can also be vendored as the CUDA C++ fallback. Why: A1.1 and A1.3.
4. **Fallback order:** vendored CUDA C++ (nvcc → cubin, loaded through cuda-core), then cuda-oxide. Why: cuda-oxide needs nightly plus LLVM 21 (A3.1).
5. **Add a two-GPU session early in Phase 1a**, about 1 h on 2 × PRO 6000:
   - NCCL ≥ 2.31.2 all-reduce latency with P2P on and off, and whether the rented host even allows P2P (IOMMU);
   - `NCCL_PROTO=LL`;
   - the result gates the P2P custom all-reduce.
   - Why: A4.
6. **Phase 1a kernel list additions:**
   - indexer RoPE (interleaved); FP8 indexer-K writer with k-pool compression;
   - radix top-k plus tail with deterministic tie-breaks;
   - latent→528 B FP8 KV writer; KDA conv-state update and `kda_verify`;
   - FP32 router; Philox sampler plus top-p/top-k; device-side speculative verify;
   - MTP reuse of target indices.
   - Why: A9.1 and A1.5.
7. **Phase 1c checkpoint:** target the ModelOpt "W4A16 NVFP4, E4M3 over K=32, FP8 128×128 weight-only" layout, produced by us with pinned official ModelOpt and hashes, and read the same format as the public artifact for cross-checks. Why: precedent and quality data (A1.5).
8. **KV format:** support the 528 B FP8 row from day one, with BF16 as the reference. Why: it halves gather bytes and shared-memory tiles within the 99 KB budget, and matches FlashInfer (A1.1, A5.2).
9. **MTP:** make it adaptive, off at high concurrency, and ship it only if SYS-P-003 improves. Why: the A10 estimate.
10. **Roadmap:** "DeepSeek" is V4-Flash (a new attention stack, B4). Pick a Qwen hybrid that fits 2 × 96 GB. Update PRD §5 "Later".
11. **PRD §2 and §10:** add "official engines reach SM120 within weeks" as a risk and a differentiation note (A1).

### DEV-GUIDELINES
1. **Determinism in two tiers** (§1.2 rule 7):
   - D0, required: bitwise identical run to run for the same batch composition.
   - D1, an opt-in mode: batch- and chunk-invariant, with fixed split sizes and no batch-dependent split-K.
   - "Cache hit ≡ miss" and "MTP on ≡ off" are bitwise only under D1 with aligned chunks; otherwise they are tolerance-based.
   - Why: A7.
2. **cuTile hot-path rules:**
   - no per-step-varying **integer scalar** kernel arguments; pass `kv_len` and similar through device metadata tensors;
   - bucket or pad T to a fixed divisibility, or set `max_divisibility`;
   - warm up every manifest specialization before graph capture;
   - `deny_in_kernel_checks=true` on hot kernels;
   - never build views wider than i32 per dimension.
   - Why: A2.3–A2.5.
3. **Tier-1 CI:** compile every manifest specialization for sm_120 to cubin on the CPU runner. Run a static resource gate with `cuobjdump -res-usage`: shared memory ≤ 99 KB per block, zero local-memory spills on hot kernels, and register deltas reported. Why: A2.4 and A5.2.
4. **Autotuning offline only:**
   - commit tuning tables keyed by (kernel, specialization, GPU name, tileiras fingerprint, cutile version, **schema version**);
   - at runtime, load the table or fall back to the default; no search.
   - Why: determinism, start-up time, and FlashInfer's calibration-schema bug (A8.5); mistral.rs's correctness gate is the template (A1.4).
5. **GPU tests fail if no GPU is present;** never skip-as-pass (the Grout pattern). Use a nextest `gpu` profile.
6. **No environment-variable tuning knobs;** everything goes in the validated TOML. Why: Grout and ormandj depend on dozens of them.
7. **One owner of the CUDA context** (cuda-core). cudarc is allowed only inside `tq-gpu` and borrows handles. Stream-ordered frees must use the owning stream (A2.2, #252).
8. **Unsafe gathers:** every `load_ptr_tko` / `int_to_ptr` SAFETY comment states the index-range invariant and the **zero-row** convention for masked lanes (A8.5).
9. **GPU CI:** no self-hosted runner on fork PRs. Use just-in-time ephemeral runners triggered by maintainers (label or `workflow_dispatch`), no `pull_request_target` with a checkout, a clean runner per job, no secrets, and a JIT disk cache writable only by the runner (A8.6, A2.3).
10. **Pins:** cutile-rs at a commit with a monthly upgrade PR that runs the whole suite; CUDA 13.3 (13.4 already ships but cuTile support is still PR #298); NCCL ≥ 2.31.2 checked at start-up with `ncclGetVersion`.

### TEST-PLAN
1. **§2 tolerances:**
   - cite FlashAttention for the 2× rule;
   - add FLA's RMSE ratio for recurrent/KDA over long sequences;
   - add FlashMLA's rule that anomaly positions must match exactly, plus cosine difference for FP8-KV tests (A8.1–A8.3).
2. **SMLA-E-009:** also NaN-poison page 0 row 0, the FP32 inline scales, and every clamp target. Add a new test for **stale positive padding past `topk_len`** (A8.5).
3. **SMLA-C and E:** add FP8 528 B-row variants with runtime stride ≥ 528 and 16 B alignment; `kv_len=0` rows inside a batch; per-row `topk_len`; non-contiguous q/idx/out (A8.3).
4. **MOE:** an expert bank over 2 GiB with offsets in the last expert (FlashInfer #4706 and #4687 overflowed there); unique-expert sweeps at T ∈ {8, 16, 32, 64}; realistic routing is P0, not P1 (A10).
5. **§8 performance protocol:**
   - time µs-scale kernels with CUPTI or CUDA-graph replay, not only events;
   - cold L2 by flushing ≥ 256 MB (2 × 128 MB L2), or by rotating buffers when the working set is below 5 × L2;
   - record the power limit, since Max-Q and WS differ 2× in TDP (A8.4, A5.1).
6. **New system tests:**
   - no JIT compile after "ready" (counter);
   - every graph bucket replays;
   - a restored state checkpoint matches recompute within tolerance (KDA chunk vs recurrent);
   - an agent multi-turn replay reports the prefix hit rate with turn-end checkpoints;
   - the P2P self-test falls back to NCCL.
7. **`tq-sched` property and state-machine tests** (proptest): no double free, reference counts, radix consistency, budget never exceeded, no admission deadlock. The mutation target stays at 80 %.
8. **Tier 2:** add cuTile's `sanitize_memcheck` compile option runs alongside compute-sanitizer (A2.4).
9. **SYS-P-001:** NCCL all-reduce latency at the real payload sizes (T × 4096 × 2 B, for T = 8…64) with P2P on and off.

## D. Top risks we haven't considered
1. **Upstream catches up within weeks** (A1.1–A1.2). The PRD problem statement weakens, and the ormandj numbers (106 tok/s per request at C4) become the public bar. Mitigation: differentiate on C8 agent throughput, determinism, reproducibility and a single binary. Use upstream as O3 on SM120.
2. **cuTile churn and soundness:** breaking releases every 2–3 weeks, an open use-after-free, bounds-check placement bugs, no upstream GPU CI (A2.1–A2.2). Mitigation: pin; our own regression suite; budget one upgrade a month.
3. **JIT specialization blow-up:** compile stalls of seconds while serving, and a long first boot (A2.3). Mitigation: the guideline rules plus a zero-JIT-after-ready test.
4. **P2P is unavailable on most PCIe hosts** without BIOS changes (IOMMU/ACS), and NCCL versions have had crashes (A4.1–A4.3). Mitigation: NCCL SHM default, P2P only after a self-test, and a documented host setup.
5. **Memory:**
   - the MLA KV is replicated per GPU;
   - recurrent slots multiply with draft tokens and checkpoints (68 MiB each in FP32);
   - only about 10 GiB is free per GPU.
   - Mitigation: an admission design and the state-dtype ADR (A10, B3).
6. **sm_120 limits for the gather kernel:**
   - 99 KB shared memory: at H=32, a BF16 Q tile is 32 KB and a 64×512 BF16 KV tile is 64 KB;
   - no TMA gather4 and no multicast;
   - no shared-memory control in cuTile.
   - Mitigation: FP8 rows, the Phase 0 gather benchmark (already planned), and the FlashInfer fallback.
7. **MTP gives little at 8 users** (A10). The plan assumes a benefit.
8. **The DeepSeek and Qwen scope assumptions are wrong** (A9.2–A9.3).
9. **Oracle mismatch:**
   - FP8 KV, group-32 E4M3 experts and BF16 vs FP32 KDA state change numerics;
   - O3 on Hopper runs different kernels.
   - Mitigation: QUAL on SM120 upstream as soon as it lands.
10. **Determinism vs speed:** batch invariance costs 20–60 % (A7). Keep D1 opt-in so it doesn't sink M1.
11. **Threading:** NCCL plus CUDA graphs with multiple GPUs in one thread can deadlock (A4.5). Use one worker thread per GPU.
12. **Supply chain:** the JIT disk cache has a checksum but no authenticity, and the `tileiras` binary is the compiler (A2.3). Keep the cache directories root-owned, and record the tileiras fingerprint in every benchmark JSON.
13. **The official model has vision** (A9.1). The v1.0 API must reject image parts explicitly, and the loader must skip the vision tensors.

---
## E. Revision 2 (after the DeepSeek-V4.1-Flash and GLM-5 paper facts). Supersedes B2–B4 and A10 where they differ.

Sources: team-lead summary; DeepSeek-V4.1 report §2.2 (CED), §2.3 (CSA2 modes and hierarchical indexer), §2.4.2 (Engram), §2.4.3 (DSpark), §2.4.4 (FP4 main KV) in local `dsv41.txt`, https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash ; GLM-5 arXiv 2602.15763.

### E1. What changes in the design
1. **KV is keyed by cache id, not layer index.** Several layers can read one global cache:
   - CSA2 Reindex and Reuse layers;
   - the CED decoder, whose global KV is projected from H_{L/2};
   - IndexCache "shared" layers in GLM.
   Each cache declares its kind, row format and producer. `ModelSpec::validate()` checks that every consumer runs after its producer.
2. **A model is a list of stages, not a flat layer list.** Each stage has a token set:
   - CED prefill runs the encoder on all prompt tokens;
   - the decoder runs only on the last n_win tokens (bounded SWA replay) plus the tokens that need logits;
   - a stage-boundary op projects the decoder's global KV.
   GLM is one stage over all tokens. This generalises vLLM's "logits indices" to per-stage token subsets carried in `StepMeta`.
3. **Carry across layers within a step:** index slots (top-k), candidate pools (from the hierarchical indexer) and the stage hidden state. They are allocated once in the workspace and resolved to fixed addresses when the model is built, so CUDA graphs stay valid.
4. **Cache kinds** use three allocation shapes plus row formats:
   - `TokenPaged { tokens_per_entry }`: 1 for GLM; 2 or 4 when CSA compresses;
   - `Window { tokens }`: SWA 128, a bounded ring of pages;
   - `PerSeqState`: KDA/GDN state, conv state, drafter state.
   - Row formats: `Bf16`, `Fp8InlineScale` (528 B), `Fp4E2m1E4m3g16` (DSV4.1 main KV, no global scale), `Mxfp4` (indexer), `State{dtype}`.
   - Prefix-cache components follow SGLang's Unified Radix Cache validators: FULL (token-paged), WINDOW (the trailing window must be present) and STATE (a checkpoint at the node).
   - Every kind implements `commit(accepted_len)` for speculative rollback. Compressed caches keep a pending partial block of fewer than r tokens.
5. **Speculative decoding becomes a trait,** because two real implementations exist now:
   - MTP: GLM-5.3 has 1 layer iterated with target indices reused; the GLM-5 paper has 3 shared-parameter layers with accept length 2.76 at 4 steps;
   - DSpark: 3 SWA-128 blocks, 5 positions in parallel, a Markov head, and a confidence head that drives a load-aware verify length.
   Verification, rejection sampling and cache commit stay shared and drafter-agnostic.

### E2. Interface sketches (replace B2's model/attention/speculative parts)
```rust
// tq-core
pub struct ModelSpec { family: Family, stages: Vec<StageSpec>, caches: Vec<CacheSpec>, drafter: Option<DrafterSpec>, /*..*/ }
pub struct StageSpec { layers: Range<LayerIdx>, tokens: TokenSet /*All | LogitTokens | LastWindowPlusLogits(u32)*/, epilogue: Option<StageOp> }
pub enum StageOp { ProjectGlobalKv { into: Vec<CacheId> } }            // CED: C_l = H_{L/2}·W_l^KV, Z_l = H_{L/2}·W_l^Z
pub struct LayerSpec { mixer: MixerSpec, ffn: FfnSpec, resid: ResidSpec /*Plain | Mhc{streams, iters} | MhcSinglePass*/, addons: SmallVec<[Addon; 1]> }
pub enum MixerSpec {
    Kda { state: CacheId },                                              // GLM
    SparseGlobal { global: GlobalSrc, index: IndexSrc, swa: Option<CacheId> }, // GLM DSA; DSV4/4.1 CSA/CSA2 (+SWA)
    /* v1.2: Gqa { kv: CacheId, window: Option<u32> }, Gdn { state: CacheId } */
}
pub enum GlobalSrc { Own { cache: CacheId, compress: u8 }, Shared { cache: CacheId } }  // CSA2 Reindex/Reuse; CED decoder
pub enum IndexSrc {
    Compute { k_cache: CacheId, candidates: Candidates /*All | Pool(PoolId)*/, emits: IndexSlot, emits_pool: Option<PoolId> },
    Reuse { from: IndexSlot },                                           // CSA2 Reuse; GLM "shared"; MTP reuse of target indices
}
pub enum Addon { Engram { table: HostTableId } }                         // DSV4.1 only; host tables, deterministic prefetch
pub struct CacheSpec { id: CacheId, kind: CacheKind, row: RowFmt, producer: Producer /*Layer(i) | StageOp*/ }
impl ModelSpec { pub fn validate(&self) -> Result<(), SpecError>; }      // producer-before-consumer, formats supported, memory fits

// tq-model: per-step carry, fixed addresses
pub struct Carry { idx: IdxArena /*IndexSlot → [T, K] i32*/, pools: PoolArena, stage_hidden: DevBuf<bf16> }

// Speculative decoding (tq-model: drafters; tq-sched: policies; tq-engine: verify and commit)
pub trait Drafter: Send {
    fn caches(&self) -> &[CacheSpec];                                    // MTP layer KV; DSpark SWA-128 KV
    fn max_draft(&self) -> u8;
    fn draft(&self, cx: &mut StepCtx<'_>, h: DevRef<bf16>, out: &mut DraftBuf /*ids, q-probs, confidence?*/) -> Result<()>;
    fn commit(&self, cx: &mut StepCtx<'_>, accepted: DevRef<u8>) -> Result<()>;
}
pub trait VerifyPolicy { fn verify_lens(&mut self, seqs: &[SpecView], load: &LoadEstimate, out: &mut [u8]); }
// Policies: Fixed(k) | AdaptiveEma (acceptance) | ConfidenceThroughput (DSpark survival probability × profiled throughput curve)
// StepPlan gains ragged per-sequence verify lengths; graph buckets are keyed on total tokens.
```

### E3. Per model family (replaces B4)
| | GLM-5.3-Flash (v1.0) | DeepSeek-V4-Flash | DeepSeek-V4.1-Flash | Qwen3.5 hybrid |
|---|---|---|---|---|
| Stages | 1 | 1 | Encoder 20 → ProjectGlobalKv → Decoder 20 | 1 |
| Global attention | `SparseGlobal{Own(c=1), Compute(All)}` | CSA `Own(c=4)` + HCA (c=128, dense over compressed) | CSA2 `Own(c=2)` Full / `Shared`+`Compute(Pool)` Reindex / `Shared`+`Reuse` | GQA paged |
| Local | none | SWA 128 on every layer | SWA 128 on every layer, bounded replay in decoder | none |
| Linear state | KDA slot | none | none | GDN slot |
| Global KV row | BF16 / FP8 528 B | FP8 656 B (rope) | FP4 E2M1 + E4M3/16 | BF16/FP8 |
| New kernels | NoPE sparse MLA, k-pool indexer, KDA, mHC, W4A16 g32 | compressor, HCA, SWA, hash router, grouped o-proj | CSA2 hierarchical indexer, FP4-KV dequant attention, Engram gather, single-pass mHC | GQA flash, GDN |
| Drafter | MTP (1 layer) | MTP | DSpark | MTP |
| Fits 2 × PRO 6000? | yes (4-bit experts) | likely (284B) | **no**: backbone about 276–290 GB at 4-bit, plus 196B Engram in host RAM | depends on size |

The design accommodates all four. v1.0 builds only the GLM variants. The other enum variants are added with their model, and exhaustive `match` makes the compiler list every site that needs updating.

### E4. Changes to earlier sections
- **A10:** using the GLM-5 paper's accept length of 2.76 at 4 steps (instead of about 3), MTP at 8 users gives about **1.1–1.2×** (estimate). Adaptive or off at C8 still stands.
- **C (DEV-PLAN):**
  - Put DeepSeek-V4.1-Flash on the roadmap as "spec parse, validate and CPU reference only" until a box fits it.
  - DeepSeek-V4-Flash is the realistic DeepSeek target for 2 × PRO 6000.
  - Add an ADR: "cache ids, stages and carry slots are v1.0 interfaces".
- **C (TEST-PLAN):**
  - `ModelSpec::validate` golden tests on the real `config.json` files (GLM-5.3, DSV4, DSV4.1, Qwen3.5), including rejection of consumer-before-producer.
  - Speculative commit/rollback property tests for every cache kind: after rollback, the state equals a replay of only the accepted tokens.
- **D (new risk): architecture churn.** DeepSeek changed its attention three times in about 12 months: DSA (V3.2) → CSA/HCA (V4) → CSA2 plus CED (V4.1). Mitigation: cache ids, stages and carry slots in the core interfaces, while kernels and model files stay per family. Also check that each next target fits in memory before committing to it.
