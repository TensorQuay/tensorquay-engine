# CPU and host memory for local inference

19 Sep 2026. Feasibility review for consumer and workstation NVIDIA GPUs. The calculations below are estimates,
not measured TensorQuay performance. They do not change release thresholds or approve an untested backend.

## Recommendation

Use GPU compute with an explicit, bounded host-memory cache. First measure preserving idle sessions and reusable
prefixes; then investigate active sparse-attention KV caching if active context limits concurrency. Keep CPU expert
execution as a separate capacity experiment. This research does not add a runtime dependency on another engine.

| Technique | Expected benefit | Main limitation | Priority |
|---|---|---|---|
| Park idle sessions and reuse prefixes from RAM | Avoid repeated prefill; retain more conversations | A normal active decoder still needs its working context on GPU | First experiment |
| Keep sparse-attention history in RAM and selected rows on GPU | Reduce active KV footprint; potentially admit more long-context decoders | PCIe misses, indexer storage and recurrent state remain | Second, bounded experiment |
| Run selected MoE experts on CPU | Fit weights beyond VRAM capacity | CPU memory bandwidth and routing-dependent work can dominate latency | Separate research |
| Stream dense weights or blindly oversubscribe GPU memory | Fit otherwise unsupported configurations | Repeated transfers can reduce generation speed substantially | No default path |

## What upstream evidence establishes

vLLM's [KV offloading guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) describes a host-backed
prefix cache with asynchronous transfers. [LMCache](https://docs.lmcache.ai/) similarly reuses cached history across
requests. Host caching is therefore an established baseline capability; evaluating against a GPU-only configuration
would not establish an advantage over these runtimes.

SGLang's [HiSparse report](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/) describes active sparse KV caching:
keep full history in host memory, retain frequently used rows on GPU and fetch misses. Its reported gains use
GLM-5.1 on data-centre hardware and large concurrency. It also reports overhead at low concurrency. Those results
are evidence for the mechanism, not a speed prediction for GLM-5.3-Flash on two workstation cards.

The [SGLang guide at 76f9213](https://github.com/sgl-project/sglang/blob/76f9213a411018547f4fd6a75f36feaa4d6bed58/docs/docs/advanced_features/hisparse_guide.mdx)
names an FP8 `flashinfer_sparse_mla` path on sm_120/121. However, it describes a disaggregated deployment, and its
[validation code](https://github.com/sgl-project/sglang/blob/76f9213a411018547f4fd6a75f36feaa4d6bed58/python/sglang/srt/arg_groups/hisparse_hook.py)
requires radix caching to be disabled. Losing prefix reuse can change an agent workload's result. Exact compatibility
with this model's KDA state, pooled indexer, MTP and TP=2 still needs a working run and correctness evidence.

vLLM's [HiSparse design at 41c4a3e](https://github.com/vllm-project/vllm/blob/41c4a3ed4e45792739bd8a5bbcd01269b06eae2c/docs/design/hisparse.md)
is experimental. It separates host identity from GPU residency, keeps replacement decisions on the GPU and retains
the indexer as a GPU cache group. These are useful design references; TensorQuay would implement its own contracts
in Rust. In particular, no per-layer CPU readback should be needed to choose cache victims.

[KTransformers' CPU backends](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md) demonstrate
CPU MoE execution with multiple instruction sets. They do not imply the same speed on a consumer dual-channel CPU
as on a many-channel server. CPU model, instruction set, memory channels and measured bandwidth must be recorded.

## GLM-5.3-Flash storage accounting

Model facts come from the [pinned configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)
and [pinned attention implementation](https://github.com/vllm-project/vllm/blob/98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5/vllm/models/glm5next/nvidia/attention.py).
Use TP=2, FP8 latent pages, FP32 KDA state, BF16 convolution state and MTP off for the initial estimate:

- DSA latent: 11 layers × 528 B = **5,808 B/token**, replicated on both GPUs; one unique host copy is sufficient.
- Pooled indexer: 128 FP8 bytes + one FP32 scale per four tokens, across 11 layers = **363 B/token**. This assumes
  replicated indexer storage. A BF16 pooled-key layout would instead cost 704 B/token; the formats must not be mixed.
- KDA state: 34 × 64 × 128 × 128 × 4 B = **136 MiB per sequence**, or 68 MiB on each GPU after head sharding.
- Convolution storage: three BF16 history positions for q/k/v, 34 layers and 64 × 128 channels: **4.78125 MiB** total,
  split across GPUs. The [pinned state layout](https://github.com/vllm-project/vllm/blob/98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5/vllm/model_executor/layers/mamba/mamba_utils.py)
  uses `kernel_size - 1 + num_spec`; this estimate sets `num_spec = 0`. Speculation scratch and padding add space.
- Incomplete indexer pools: provisioning four BF16 K/gate positions costs **22 KiB** total, replicated on each GPU.
  At most three unfinished positions are logically live; keeping four matches the reference's circular capacity.

The table includes one recurrent/conv checkpoint and one history per session, no prefix sharing, no weights,
allocator metadata, copy staging, graph workspace, MTP or extra rollback/checkpoint copies. K means 1,024 tokens.

| Context/session | Active MiB/session/GPU | Unique host MiB/parked session | Active GiB/GPU, 8 / 16 sessions | Host GiB, 8 / 16 parked sessions |
|---|---:|---:|---:|---:|
| 16K | 166.83 | 237.22 | 1.30 / 2.61 | 1.85 / 3.71 |
| 32K | 263.26 | 333.65 | 2.06 / 4.11 | 2.61 / 5.21 |
| 64K | 456.10 | 526.49 | 3.56 / 7.13 | 4.11 / 8.23 |
| 128K | 841.79 | 912.18 | 6.58 / 13.15 | 7.13 / 14.25 |

Unique host bytes differ from transfer bytes. A 64K restore sends about **478 MB to each GPU**, about **957 MB**
across both links, even when replicated history is stored only once on the host. At an assumed sustained 25–50 GB/s
per link, the per-link copy lower bound is about **9.6–19.1 ms**. A shared 25–50 GB/s bottleneck instead needs about
**19.1–38.3 ms** for the combined traffic, before software overhead or contention. Eight simultaneous restores multiply
the traffic; these are not eight free overlapping copies.

Do not allocate all installed RAM as pinned cache. Reserve measured process/OS headroom and loading/staging space.
As an arithmetic example, a usable 64 GiB cache holds about 124 independent 64K parked sessions under these assumptions,
roughly 8.1M tokens; that says nothing about how many can decode at once. A raw "128 GB = 20M tokens" calculation
omits recurrent checkpoints and available-memory limits.

The MTP layer has its own sparse-attention history and indexer: an extra 561 B/token plus 2 KiB tail capacity,
replicated on each GPU. At 64K that is about 35.06 MiB/session/GPU. A resumable MTP session needs that history too,
or an explicit rebuild/fallback. Rejected drafts also need target-state rollback and speculation scratch. Neither is
included in the table. Measure MTP off first, then account for every additional allocation when turning it on.

## Active sparse caching: a plausible, unmeasured opportunity

For this checkpoint, the indexer selects 512 four-token pools, expanding to **2,048 tokens**, plus at most three tail
tokens; it does not select 2,048 pools. Its configured indexers are all full, so do not assume GLM-5.2's cross-layer
shared-index prefetch optimization applies unchanged.

Ignoring the small tail, a step fetching every selected latent row from host would move
`batch × 2,048 × 528 × 11` bytes **per GPU**:

| Active decoders | Bytes/step/GPU | Copy-only ms at 25 / 50 GB/s per link | Combined host traffic at 30 steps/s |
|---|---:|---:|---:|
| 8 | 95.16 MB | 3.81 / 1.90 | 5.71 GB/s |
| 16 | 190.32 MB | 7.61 / 3.81 | 11.42 GB/s |

This is payload arithmetic, not an achieved PCIe rate or a decoder latency prediction. Cache hits can reduce traffic;
page amplification, scattered reads, layer serialization, MTP verification and competing NCCL traffic can increase
cost. The two GPUs share host bandwidth. The indexer still needs to search its pooled keys, and KDA state stays live.

For illustration, a 4,096-row latent buffer per DSA layer uses 22.69 MiB/session/GPU. With 64K pooled keys and the KDA
and convolution state above, the total is about **115.79 MiB/session/GPU**, versus 456.10 MiB with all latent history
resident, before management overhead. This could relieve capacity pressure. It must preserve every selected row and
the exact attention semantics; dropping cache misses would change the model.

NVIDIA's [zero-copy guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#zero-copy) warns that mapped
host memory on discrete GPUs needs appropriate access patterns. Compare explicit staged copies with a fused pinned-
host gather; do not assume managed-memory oversubscription is an equally fast substitute.

## Smallest useful verification sequence

1. Reuse the existing Qwen/5090 workload method for continuity, but store a sanitized, fixed WL-001 fixture before
   qualification. Keep model revision, quantization, prompt bytes, tool work and committed-token accounting identical.
2. On a working GLM decode path, compare GPU-only caching with request-boundary park/restore: 8 agents first, then 16;
   16K/32K/64K context; MTP off initially. Include tool-wait gaps and a separate all-active stress run.
3. Require byte-identical restored buffers at the same committed checkpoint, then use SYS-R-010/API-006 to check
   numerical continuation, including all tail lengths, slot reuse, cancellation and the 10,000-cycle leak check.
   Fail loudly on incomplete state. Use SYS-P-008 for restore versus recompute.
4. Record every session's committed tok/s, full-step p50/p95/p99, TTFT, queue time, copy bytes/time, cache hits,
   preemptions and host/VRAM peaks. Keep M1/M2 thresholds. Context retained in RAM is not active-decoder capacity.
5. Only if active context is the bottleneck, write a separate sparse-cache contract and independent tests. First
   measure gather bandwidth and miss-rate traces, with NCCL active; then add a bounded prototype behind an option.
   Benchmark against both GPU-only caching and an eligible upstream host-caching baseline, including prefix reuse.

Prefer an ordinary Rust scheduler and bounded pinned buffers. No new storage service, distributed cache, backend
registry or CPU kernel framework is needed for the first experiment. CPU tokenization, request preparation and
transfer orchestration can overlap GPU work; CPU tensor compute needs its own measured benefit.
