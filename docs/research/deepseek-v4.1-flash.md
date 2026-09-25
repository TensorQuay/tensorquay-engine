# Research note: DeepSeek-V4.1-Flash and what it means for TensorQuay Engine

Public technical extract: private deployment observations and operational plans are omitted.

| | |
|---|---|
| **Paper** | *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*, DeepSeek-AI, 51 pages |
| **Source** | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash (published 10 Sep 2026; PDF in the repo). Licence: MIT (README) |
| **Note date** | 19 Sep 2026. Research input only: no PRD, DEV-PLAN or TEST-PLAN text was changed |
| **What I read** | All 51 pages: body (pp. 1–37), references (pp. 38–45) and appendices A–C (pp. 46–51). Figures and tables were checked in the PDF, not only in the text extraction |
| **Also read** | `config.json`; `inference/config.json`, `model.py`, `kernel.py`, `engram.py`, `convert.py`, `README.md`; the safetensors headers of all 48 weight shards (dtypes and shapes, fetched by byte range, no weights downloaded) |

**Conventions**
- "p. N" is the printed page number, which equals the PDF page number.
- `cfg:key` is a key in `config.json` → `text_config`.
- `hdr:` means the safetensors headers.
- `ref:` means the official reference code in `inference/`.
- **(inferred)** marks my own calculation or reasoning, not a DeepSeek statement.
- GiB = 2³⁰ bytes, GB = 10⁹ bytes. "bpw" = bits per weight, including scale overhead.

**The five findings that matter most for us**
1. **It does not fit two 96 GiB GPUs at 4-bit.** The backbone as released (MXFP4 experts) is **277.8 GiB**, against
   about 191 GiB on 2 × RTX PRO 6000. The routed experts would need about **2.5–2.7 bpw** to fit. On top of that, the
   196B-parameter **Engram** tables (188.8 GiB) must live in **host RAM**. (§6a)
2. **KV cache is almost free.** The model needs **890 B/token** of global KV, so 8 agents × 256K tokens use only 1.7 GiB.
   On this model, memory pressure moves entirely from the KV cache to the weights. (§2, §6a)
3. **Our plan's assumption "DeepSeek reuses sparse-MLA and NVFP4 MoE" is only partly true.** V4.1 is not V3.2-style DSA
   with MLA. It uses **CSA2**: MQA over one shared 512-d latent, cross-layer shared KV, FP4 KV, a sliding-window (SWA)
   branch and attention sinks. The inner gather-and-softmax loop of our kernel carries over; the contract does not.
   (§6b)
4. **New runtime ideas are worth adopting for GLM too:**
   - typed KV classes (global, SWA ring, per-sequence state);
   - static layer "modes";
   - a tiered prefix cache in which recomputable state is not persisted but approximately **replayed**;
   - a generic speculative-decoding interface. (§6c)
5. **Quality:** it is the strongest open coding-agent model in its own table. DeepSWE v1.1 **74.2** vs GLM-5.3 **66.9**
   (Table 3, p. 33). That makes the fit question a product decision. (§7)

---

## 1. Key facts

| # | Fact | Value | Source |
|---|---|---|---|
| F1 | Backbone parameters | **552B** (headers: 551.6B; of which routed experts **543.6B** = 98.5 %) | p. 1, p. 7, p. 22; hdr (inferred sum) |
| F2 | Engram parameters | **196B** (headers: 196.6B = 2 tables × ~384M rows × 256) | p. 7, p. 13; `cfg:engram_num_embeddings` = [384006168, 384016682], `cfg:engram_head_dim` = 256 |
| F3 | Active parameters | **16B per token in decode, 8B in prefill** (CED) | p. 1, p. 7, p. 22 |
| F4 | Other parameters | Vision encoder and projector 0.49B (hdr). DSpark drafter 14.2B, of which 13.6B are experts (hdr) | hdr (inferred sums) |
| F5 | Checkpoint size | 510.3 GB = 475.2 GiB: backbone 298.3 GB (277.8 GiB), Engram 202.8 GB (188.8 GiB), DSpark 7.9 GB, vision 1.0 GB | hdr (inferred sums) |
| F6 | Layers | **40**: 20-layer causal encoder + 20-layer decoder; hidden size d = 5120 | p. 7, p. 21; `cfg:num_hidden_layers`, `cfg:hidden_size` |
| F7 | Layer layout | L0–1 SWA only. L2–19 CSA2 ratio 2 in 3 groups of 6 (Full at 2, 8, 14; the rest Reuse). L20–39 CSA2 ratio 1 in 5 groups of 4 (Full at 20; Reindex at 24, 28, 32, 36; the rest Reuse) | p. 7 (Fig. 3), p. 22; `cfg:compress_ratios`, `cfg:kv_source_layer_ids` = [2, 8, 14, 20], `cfg:index_source_layer_ids` = [2, 8, 14, 20, 24, 28, 32, 36] |
| F8 | Mode counts | Full 4, Reindex 4, Reuse 30, SWA-only 2 | (inferred from F7) |
| F9 | Attention dims | 64 query heads × head dim 512; **one KV head** (MQA; K = V = one 512-d latent); RoPE on the last 64 dims; q LoRA rank 1280; grouped output projection with 8 groups × 1024 | p. 22; `cfg:num_attention_heads`, `cfg:num_key_value_heads` = 1, `cfg:head_dim`, `cfg:qk_rope_head_dim`, `cfg:q_lora_rank`, `cfg:o_groups`, `cfg:o_lora_rank`; ref `Attention` |
| F10 | Softmax scale and sink | 512^-0.5, plus a learned per-head **attention sink** | ref `Attention.softmax_scale`, `attn_sink`; hdr `attn.attn_sink` [64] |
| F11 | Indexer | 32 heads × 128, top-k **512**; indexer K is projected from main KV (512 → 128); FP4 Q and K | p. 10, p. 22; `cfg:index_n_heads`, `cfg:index_head_dim`, `cfg:index_topk`; hdr `indexer.wk` [128, 512] |
| F12 | Hierarchical indexer | Decoder only: at most 2048 blocks × 8 positions = **16,384** candidates, built by L20 (Full) and searched by the Reindex layers | p. 11–12, p. 22; `cfg:candidate_source_layer_id` = 20, `cfg:candidate_topk_blocks`, `cfg:candidate_block_size` |
| F13 | SWA | Window n_win = **128** in every layer; SWA KV is FP8 | p. 14, p. 22; `cfg:sliding_window` |
| F14 | FP4 main KV | E2M1 + one **E4M3 scale per 16 channels**, no global scale (maximum magnitude 448 × 6 = 2688); quantised after RoPE; QAT in post-training | p. 14; ref `fp4_act_quant(latent, 16, …, e4m3)` |
| F15 | Indexer K format | MXFP4: E2M1 + UE8M0 scale per 32 | p. 14; ref `fp4_act_quant(k, 32, …)` |
| F16 | Global KV per token | **890 B** (V4-Flash 3,514; V3.2 48,068; V1 389,120). Always in HBM | p. 1 (Fig. 1b), p. 37 |
| F17 | 890 B breakdown | Main entry = 256 B (FP4) + 32 B (scales) = 288 B. Indexer-K entry = 64 + 4 = 68 B. Encoder: 3 sources × 356 B ÷ 2 = 534 B/token. Decoder: 1 × 356 B = 356 B/token. **Total 890 B** | (inferred from F7, F14, F15; matches p. 1 exactly) |
| F18 | SWA KV bytes | About 528 B per layer per window slot (512 FP8 + UE8M0/32 scales), i.e. about 2.7 MB per sequence for 40 layers, independent of length | (inferred from ref `act_quant(kv, 32, ue8m0)`; the paper only says FP8, p. 14) |
| F19 | Persistent KV | About **1/8 of V4-Flash**: SWA KV is no longer persisted (×½) and global KV is 1/4. Global KV lives on SSD/host with ≥ 72 h retention; SWA KV lives in a host pool (10 % of DRAM) with a TTL of minutes | p. 1, p. 19 |
| F20 | MoE | 1 shared + **384 routed** experts, **top-6**; expert intermediate size 2304; SwiGLU clamp 10; score function `sqrtsoftplus`; `noaux_tc`; routed scale 1.5; separate text and image bias | p. 22; `cfg:n_routed_experts`, `cfg:num_experts_per_tok`, `cfg:moe_intermediate_size`, `cfg:swiglu_limit`, `cfg:scoring_func`, `cfg:routed_scaling_factor`; hdr `gate.bias_vl` |
| F21 | Weight formats as released | Routed experts **MXFP4** (packed E2M1 + UE8M0 per 32). Other linear layers FP8 E4M3 with UE8M0 scales on **32 × 32** blocks. Embedding and head BF16; mHC weights FP32 | `cfg:quantization_config` (`weight_block_size` [32, 32], `scale_fmt` ue8m0, `expert_dtype` fp4); hdr |
| F22 | Context | **1M tokens** (YaRN factor 16 from 64K; `rope_theta` 1e4; compressed KV uses θ = 1.6e5) | p. 1, p. 22; `cfg:max_position_embeddings`, `cfg:rope_scaling`, `cfg:compress_rope_theta` |
| F23 | mHC | 4 streams, 20 Sinkhorn iterations, **Single-Pass** (input mixing uses the previous block's coefficients). The Mega-mHC kernel moves (2n+2)d = 10d of activations instead of (4n+4)d = 20d (n = 4) | p. 12–13, p. 22; `cfg:hc_mult`, `cfg:hc_sinkhorn_iters` |
| F24 | Engram | 2 modules at layers **1 and 14**; n-gram orders {2, 3, 4} × 8 hash heads × 256 dims (2048 per order); about 16M rows per head (distinct primes); FP8 tables. No short convolution. Prefetched **from host memory** via background RDMA | p. 13, p. 18; `cfg:engram_layer_ids`, `cfg:engram_n_heads`, `cfg:engram_max_ngram_size`, `cfg:engram_compressed_vocab_size` = 99092 |
| F25 | Engram placement | Inference: prefetched from host memory; tables sharded. RL rollouts: resident in GPU memory. SSD: **not stated** | p. 6, p. 13, p. 18 |
| F26 | DSpark | Speculative decoding with 3 blocks (SWA window 128), 5 draft positions per pass, a Markov head (rank 256), a confidence head, and a scheduler that picks the verify length from engine throughput curves. Drafter MoE: 128 experts, top-3. Fed by layers 37–39 | p. 13–14; `cfg:num_nextn_predict_layers` = 3, `cfg:dspark_block_size` = 5, `cfg:dspark_target_layer_ids`, `cfg:dspark_markov_rank`, `cfg:dspark_n_routed_experts`, `cfg:dspark_num_experts_per_tok` |
| F27 | No MTP | The MTP module is omitted from pre-training; DSpark is trained afterwards on a frozen backbone | p. 8, p. 14 |
| F28 | Inference kernels | Each **Reuse-mode layer runs 15 kernels in prefill and 11 in decode**. Named fused kernels: FlashMLA fused-RoPE-attention-RoPE-cast; DeepGEMM Mega-Gate, Mega-mHC and Mega-MoE; TileKernels; DeepSelect TopK | p. 6, p. 18–19 |
| F29 | Decode cost vs context | Growing the context 256× (4K → 1M) raises decode FLOPs by only about ¼ | p. 5 (Fig. 2) |
| F30 | Prefill saving | CED makes prefill O(NL/2 + n_win·L/2), "nearly halving" it | p. 9, p. 20 |
| F31 | Vision | DeepSeek-ViT: 32 layers × 1024, patch 14, 3 × 3 pixel-unshuffle, up to about 1344 × 1344 px | p. 8, p. 22 |
| F32 | Pre-training | **45T tokens** (text : multimodal = 7 : 1); 64K sparse attention from scratch; 1M from 34T | p. 6, p. 21–22 |

---

## 2. Architecture, component by component (how each one works at inference time)

### 2.1 Causal Encoder–Decoder (CED), pp. 8–9
- The 40 layers are split 20/20. **Decoder global KV is not computed from each decoder layer's own hidden state.**
  - It is projected from the encoder's last hidden state: `C_l = H_{L/2} W_l^KV` (Eq. 1, p. 9).
  - In V4.1 only one decoder layer (L20, Full mode) produces global KV. It reads the encoder output. All other decoder
    layers reuse it (F7).
- **Prefill consequence.** For prompt tokens, run the 20 encoder layers plus L20's KV and indexer-K projection. The decoder
  layers run only over the **last 128 prompt tokens**, to build their SWA state (Decoder SWA Bounded Replay, §2.4). Result:
  about 8B active parameters per prefill token (p. 7) and about half the prefill FLOPs (p. 9).
- **Decode consequence.** Every new token still goes through all 40 layers (16B active).
- **In the reference code** (`ref: Transformer.forward`), prefill runs all 40 layers over the whole prompt, which is exact.
  Production DeepSeek uses the bounded-replay shortcut instead. The two are *not* bit-identical (p. 20). See the open
  questions.

### 2.2 CSA2: Compressed Sparse Attention 2 with Full / Reindex / Reuse modes, pp. 9–11
Each attention layer computes its own query and its own SWA KV. It then attends over the **concatenation** of:
- the 128-slot SWA window, and
- the **top-512** selected entries of the shared global ("main") KV.

The modes differ only in where main KV, indexer K and the top-k indices come from:

| Mode | Main KV and indexer K | Top-k indices | Layers |
|---|---|---|---|
| **Full** | Computes its own. The compressor pools `ratio` tokens into one 512-d entry (softmax gate over the group, no overlap, no absolute position embedding), then RMSNorm, RoPE and FP4. Indexer K = RMSNorm(W·latent) + RoPE, then MXFP4 | Runs the indexer over all causally visible entries | 2, 8, 14, 20 |
| **Reindex** | Reuses the most recent Full layer's KV | Own indexer query; rescores the shared indexer K (in the decoder, only inside the candidate pool) | 24, 28, 32, 36 |
| **Reuse** | Reuses | Reuses the latest indices | the other 30 |

- **Indexer score.** `Σ_h w_h(x) · ReLU(q_h · k_j)` over 32 FP4 heads. The per-head weights w_h come from the hidden state.
  Then take the top-512, sorted into position order (`ref: Indexer.forward`).
- **Ratio 2 in decode.** An encoder source emits one compressed entry every 2 tokens. A pending odd token is kept in a
  small per-sequence "tail" state. Until its group completes, that token is visible only through the SWA window
  (`ref: Compressor`).
- **Why it is cheap.** Storage scales with the **4 KV-source layers**, not with 40 layers. The indexer runs in 8 layers,
  not 40.
- **V4 comparison.** V4 used a CSA + HCA hybrid; V4.1 is pure CSA2 (p. 4).

### 2.3 Hierarchical sparse indexer, decoder only, pp. 11–12
- **L20 (Full)** scores every position. It then takes the maximum score per block of 8 positions and keeps the top 2048
  blocks, i.e. a **candidate pool of at most 16,384 positions**. The newest, partly filled block is always kept
  (`ref: select_candidate_blocks`).
- The Reindex layers (24, 28, 32, 36) score **only the pool** and each pick their own top-512. Their per-query cost is
  therefore constant in context length.
- It was introduced in **post-training**, with the same restriction applied in training and inference (p. 12).
- The encoder Full layers (2, 8, 14) still scan everything (ratio 2, so N/2 entries each).

### 2.4 SWA and bounded replay, pp. 9, 19–20
- Every layer has a 128-token sliding window of **FP8** local KV, generated from that layer's own hidden state. It is a
  per-sequence ring buffer, so its size doesn't grow with context.
- **The problem.** Exact reconstruction of SWA state after a cache miss needs L × n_win tokens replayed, because the
  dependencies compound across layers.
- **Bounded replay.** Replay only the last n_win = 128 tokens, and truncate SWA to the replay segment: a query at position
  i sees keys in `[max(s, i−W+1), i]` (p. 20). There are two uses:
  - **Encoder SWA Bounded Replay.** On a prefix-cache hit whose SWA KV was evicted, replay the last 128 cached tokens
    through the encoder. This regenerates only SWA KV and reuses the cached global KV. The uncached suffix is then
    processed normally.
  - **Decoder SWA Bounded Replay.** At *every* prefill, run the decoder only over the last 128 prompt tokens. The
    resulting decoder SWA KV is used for decoding and never cached.
- **Quality impact.** DeepSeek reports only "negligible" impact, and simulates the decoder replay during post-training
  (p. 20). Outputs therefore depend on where the cache was hit (p. 20) and are not bit-identical to a full recompute.

### 2.5 Single-Pass mHC, pp. 12–13
- **Standard mHC.** There are 4 residual streams. Per block, predict the coefficients (A, B, C) from the streams:
  `X_{l+1} = B X_l + C F(A X_l)`. A depends on a full reduction over X_l, so a naive implementation needs 3 dependent
  kernels (20d of activation traffic for n = 4).
- **Single-Pass.** Use the *previous* block's A: `X_{l+1} = B_l X_l + C_l F(A_{l−1} X_l)`. Then one tiled pass can do
  the residual update, input mixing, coefficient prediction, pre-norm and FP8 cast together: the **Mega-mHC** kernel,
  with (2n+2)d traffic. That is the theoretical minimum and half of the original.
- **In the reference.** Each sub-layer's `hc_mixes` produces the `pre` mix used by the *next* sub-layer
  (`ref: Block.forward`).

### 2.6 Engram (conditional memory), pp. 13, 18; `ref: engram.py`
- **What it is.** A giant hashed n-gram lookup table added into the residual stream at layers 1 and 14. It stores
  "memorised" knowledge cheaply, so the MoE doesn't have to.
- **Per token, per module** (ref):
  1. Normalise token IDs into a compressed vocabulary of 99,092 IDs (lowercase, accents stripped, whitespace folded).
  2. Hash the 2-, 3- and 4-grams ending at this token. This is a multiplicative XOR hash, taken modulo a distinct prime
     for each of the 8 heads, giving **24 row IDs**.
  3. Gather 24 FP8 rows × 256 dims.
  4. A 6144 → 25,600 projection gives one key per mHC stream plus one shared value.
  5. A per-stream sigmoid gate (normalised dot product of stream and key) decides how much of the value is added.
- **Addresses depend only on the token IDs,** so rows can be fetched *before* the layer runs. DeepSeek prefetches from
  host memory over RDMA, overlapping layer 0 (p. 13).
- **Host traffic.** 2 modules × 24 rows × (256 + 8) B ≈ **12.4 KiB per token** (inferred). That is tiny even at 10K
  prefill tokens/s (≈ 127 MB/s).
- **Porting detail.** The hash multipliers come from NumPy's `default_rng(10007 × layer_id)`, the primes from a sympy
  search starting at 16M, and the token map from Hugging Face normalisers (`ref: engram.py`). We should precompute all
  three at checkpoint-conversion time and not re-implement them in Rust (inferred).

### 2.7 DSpark speculative decoding, pp. 13–14; `ref: DSparkBlock`
- **The drafter.** Three extra blocks (SWA-only attention with window 128; MoE with 128 experts, top-3). They read a
  projection of the backbone's hidden states at layers 37, 38 and 39.
- **Drafting.** One pass drafts **5 positions in parallel**; the placeholders are filled with a "noise" token. A rank-256
  **Markov head** then adds a per-position logit bias from the previous draft token (sequential, but tiny). A
  **confidence head** predicts per-position acceptance probability.
- **Scheduler.** It combines prefix-survival probabilities with **profiled engine throughput curves** to choose the
  verify length per request, under the current load (p. 14).
- **Scope of the reference.** The reference implements only the draft forward pass. The draft–verify loop and the
  scheduler are "out of scope" (`ref: ModelArgs` comment).

### 2.8 FP4 main KV, p. 14
- **Format.** E2M1 values with an E4M3 scale per 16 channels, applied after RoPE to both the no-RoPE and RoPE parts. This
  is NVFP4 without its second-level scale. That is safe because the latent's L2 norm is at most √512 (so no channel
  exceeds ≈ 22.6), and the observed maximum is about 10.
- **Why FP4 KV works without FP4 maths.** FP4 here saves **storage**, not maths: entries are dequantised before attention,
  so no FP4 tensor-core support is needed.
- **Indexer Q/K** use OCP MXFP4 (group 32, UE8M0 scale).
- **SWA KV stays FP8** because it is quantisation-sensitive.

---

## 3. Inference-system insights (§3.2 and elsewhere)
- **Kernel economy.** 30 of 40 layers are Reuse layers, and each is fused into 15 kernels (prefill) or 11 (decode). The
  big fused kernels are Mega-mHC, Mega-Gate and Mega-MoE (DeepGEMM), fused RoPE-attention-RoPE-cast (FlashMLA) and TopK
  (DeepSelect) (p. 18–19).
  - For us (inferred): 40 × 11 ≈ 440 launches per decode step, so CUDA-graph capture or persistent kernels are needed.
    Otherwise launch overhead alone is about 2 ms per step.
- **Deployment.** Encoder–Prefill–Decode (EPD) disaggregation: vision encoding, prefill and decode scale separately
  (p. 19). Also communication–computation overlap and sharded Engram tables (p. 6).
- **Persistent KV redesign** (p. 19):
  - *V4 deployment:* global KV and SWA KV both persisted to SSD with LRU eviction and more than 72 h residency. SWA KV
    was snapshotted at the end of the prompt and the end of the output, and made up about half the capacity.
  - *V4.1:* SWA KV leaves the persistent cache. It goes to a **distributed memory pool of 10 % of host DRAM, with a TTL
    of minutes**, because SWA KV is only reused within a minute-scale window of an active session. Global KV keeps a
    retention of at least 72 h.
  - A miss on SWA with a hit on global KV costs one 128-token encoder replay: "a catastrophic miss becomes a graceful
    degradation".
- **Lifetime classes.** Long-lived global KV is stored separately from short-lived encoder SWA KV in host memory (p. 6).
- **Decode FLOPs are nearly flat in context length** (Fig. 2, p. 5). A long agent history therefore costs little extra
  per generated token. The only linear part is the encoder and L20 index scans, done in FP4.
- **Rollout infrastructure worth copying (pp. 30–31).**
  - *Token-level interruption:* generation can stop at any token boundary.
  - *Token-granular state:* KV and expert routing are persisted per token, so a sample resumes without re-prefill.
  - *Sample-level garbage collection.*

---

## 4. Training and post-training (short)

### Pre-training
- **Data.** 45T multimodal tokens, text : multimodal = 7 : 1. Model-generated, low-information text is filtered out as
  "implicit duplication". More recent code is included (pp. 20–21).
- **Setup.**
  - batch of 100.6M tokens;
  - learning rate 2.6e-4 until 28T, cosine decay to 2.6e-5 by 40T, flat to 45T;
  - 64K sparse attention from scratch, extended to 1M at 34T;
  - "no instability" (p. 22).
- **Optimisers.**
  - Head-wise Muon for Q and K;
  - Muon for the other matrices;
  - AdamW for norms and biases;
  - a new **momentum + Sinkhorn-balanced update** for Engram, the embeddings and the head. It needs no Adam state
    (K = 11 steps, γ = 0.18). Engram's learning rate is ×5 (pp. 14–16, p. 22).
- **Load balancing.** Separate text and image bias terms (pp. 8, 22).
- **Vision.** SigLIP contrastive training on about 47B pairs at 224², then autoregressive training with a 4B MoE on 236B
  tokens at 544–1344 px (pp. 22–23).
- **Base-model results.** Matches V4-Pro-Base with 1/3 of the parameters and 1/4 of the active parameters: MMLU-Pro 74.1,
  HumanEval 79.4. LongBench-V2 is 45.2 vs 51.5 for V4-Pro (Table 1, p. 24). Lowest bits-per-byte on internal docs, code
  and academic text (Fig. 6, p. 25).

### Post-training (SFT → RL → OPD)
- **No new algorithms.** "Essentially all" the gains come from data and environment pipelines (p. 25).
- **Task synthesis.** Each task is a triplet (problem, environment, verifier), scored on difficulty and correctness. The
  model itself is RL-trained to build better tasks (p. 25). There are two pipelines:
  - *General agents:* mocked SaaS and enterprise tools, reconstructed from usage data and failure reports.
  - *Coding agents:* built from internal sessions and starred GitHub repos. Separate agents check the build, set up the
    environment, attempt the task, inspect quality (including hackability) and repair it (pp. 25–26).
- **RL.**
  - Asynchronous and large-scale, scaled on compute and on the number of scaffolds: Claude Code versions, OpenCode, Pi and
    DeepSeek Harness (Figs. 7–8).
  - Successive RL runs are re-initialised by **model merging** (pp. 26–27).
  - Rollouts run on the "DSec" sandbox platform: millions of sandboxes, more than 2,500 containers per node, with
    AppArmor and eBPF containment (pp. 27–29).
- **Reasoning-effort control.**
  - A scalar b between 1 and 100 is placed in the system prompt.
  - The length penalty `k(b) = k0·exp(−(b − b_min)/τ)` is applied inside groups of the same b.
  - The API tiers are max = 100, high = 75, low = 50 (Table 2, p. 30).
  - Going from effort 25 to 100 lifts DeepSWE from 66.0 to 74.2 and Terminal-Bench 2.1 from 82.4 to 90.6, for about
    2.5× the tokens.
  - An effort of **60–80 recovers most of the accuracy at under half the tokens of 100** (p. 34; Appendix C, pp. 49–51).
- **OPD.** The last stage is full-vocabulary on-policy distillation from **more than 40 teachers**, some with different
  architectures (pp. 31–32).
- **Trained-in inference features.**
  - The hierarchical indexer, FP4 KV QAT and decoder bounded replay are all trained or simulated in post-training
    (pp. 12, 14, 20).
  - DSpark is trained alongside the backbone without back-propagating into it (p. 14).
- **Results.** DeepSWE 74.2; Terminal-Bench 2.1 90.6, 3.0 30.0, 4.0 31.2; AutomationBench 54.8 (Table 3, p. 33). Robust
  across scaffolds: Claude Code gives 69.8 on DeepSWE and 88.0 on TB 2.1 (Table 4, p. 35). Multi-agent "Agent Team" mode
  beats single-agent at every deadline (Fig. 10, p. 36).

---

## 5. Comparison with GLM-5.3-Flash (our v1.0 model)

| | GLM-5.3-Flash (PRD §6) | DeepSeek-V4.1-Flash |
|---|---|---|
| Layers | 45: 34 KDA + 11 DSA | 40: 20 encoder + 20 decoder; 2 SWA-only + 38 CSA2 |
| Attention | Rope-free MLA, latent 512, qk head dim 256, 64 heads | MQA over a 512-d latent, head dim 512 with 64 RoPE dims, 64 heads, attention sink, plus SWA 128 in every layer |
| Indexer | 32 × 128, k-pool 4, top-2048, in every DSA layer | 32 × 128, FP4, top-512, in 8 layers only; hierarchical pool in the decoder |
| KV per token | ≈ 11.3 KB (11 layers × 512 × BF16; **inferred**) | **890 B** (FP4, 4 source layers) |
| Per-sequence state | KDA recurrent state | SWA rings (about 2.7 MB) + compressor tails |
| MoE | 288 experts, top-8, H 4096, I 2048, sigmoid, ×2.5 | 384 experts, top-6, H 5120, I 2304, sqrt-softplus, ×1.5 |
| Routed experts | ≈ 304B → 150.6 GiB at 4.25 bpw | **543.6B → 268.9 GiB at 4.25 bpw** |
| mHC | 4 streams, 20 Sinkhorn iterations | Same, but single-pass |
| Speculation | 1 MTP layer | DSpark (3 blocks, 5 drafts, 14.2B parameters) |
| Extra | n/a | Engram, 196B parameters, host memory |
| DeepSWE v1.1 / TB 2.1 | 66.9 / 88.2 | 74.2 / 90.6 (Table 3, p. 33) |

---

## 6. Relationship to TensorQuay Engine

### 6(a) Does it fit 2 × RTX PRO 6000?
**Budget.** 2 × 95.6 GiB = **191.2 GiB**. I reserve 10–12 GiB for runtime:
- CUDA, cuTile and NCCL context: about 1.5 GiB per GPU;
- activations and workspaces (4-stream mHC, MoE dispatch, prefill chunk): about 3 GiB per GPU;
- global KV for 8 × 256K tokens: 1.7 GiB (F17).

This is an inferred estimate, to be measured. It leaves about **179–181 GiB for weights**.

**Parameter inputs** (hdr, inferred sums):
- routed experts **E = 543.58B**;
- non-expert backbone as released = **9.53 GB = 8.88 GiB**:
  - attention 5.15 GB (FP8);
  - shared experts 1.42 GB (FP8);
  - embedding and head 2 × 1.32 GB (BF16);
  - router and mHC 0.31 GB.

**Formula:** expert bytes = E × bpw ÷ 8.

| Expert format (bpw incl. scales) | Experts | + non-expert (8.9 GiB) = backbone | Fits in ≈ 180 GiB? | Engram also in HBM (+188.8 GiB) |
|---|---|---|---|---|
| FP8 with 32 × 32 UE8M0 blocks (≈ 8.01) | 543.58 × 8.01 / 8 = 544.1 GB = **506.7 GiB** | **515.6 GiB** | No (2.9× too big) | 704.5 GiB, no |
| NVFP4, E4M3 per 16 (4.5) | 305.8 GB = **284.8 GiB** | **293.6 GiB** | No (short by about 113 GiB) | 482.5 GiB, no |
| NVFP4 or MXFP4, group 32 (4.25) = **as released** | 288.8 GB = **268.9 GiB** | **277.8 GiB** | No (short by about 97 GiB) | 466.7 GiB, no |
| ≈ 3.0 | 203.8 GB = **189.8 GiB** | **198.7 GiB** | No (short by about 18 GiB) | 387.6 GiB, no |
| ≈ 2.75 | 186.9 GB = **174.0 GiB** | **182.9 GiB** | Marginal: no with a 12 GiB reserve | 371.7 GiB, no |
| ≈ 2.5 | 169.9 GB = **158.2 GiB** | **167.1 GiB** | **Yes**, about 12–14 GiB spare | 355.9 GiB, no |

**Ceiling (inferred).**
- (191.2 − 12 − 8.9) GiB = 170.3 GiB for experts, i.e. **≤ 2.69 bpw average**.
- With a 10 GiB reserve: ≤ 2.72 bpw.
- Moving the BF16 input embedding (a pure lookup) to host RAM and casting the head to FP8 frees about 1.8 GiB more.
- DSpark adds 14.2B parameters: 7.4 GiB as released, or about 4.6 GiB at 2.5 bpw. With DSpark on, the ceiling falls to
  about **2.62 bpw**.

**Engram (F24–F25).**
- 188.8 GiB of FP8 tables, so it **cannot be in HBM on this box at any backbone precision**. It must live in **host RAM**,
  as DeepSeek does (p. 13).
- The host then needs **≥ 256 GB DDR5; 384 GB recommended**. That also leaves room for pinned staging, a host
  prefix-KV tier and the OS (inferred).
- Bandwidth is not a concern: about 12.4 KiB per token (§2.6).
- An **SSD tier is plausible but unproven.** It needs about 96 random 4 KiB reads per token with today's split
  row/scale layout, so a re-layout plus a hot-row RAM cache is needed. DeepSeek does not describe an SSD mode (inferred;
  open question 5).

**Speed sanity check at about 2.5 bpw (inferred, rough, before measurement).**
- Batch of 8, no speculation. With top-6 of 384 and 48 routings per layer, about 45 unique experts per layer are read.
  That is 45 × 40 × 35.4M × 2.5/8 ≈ **19.9 GB** of expert weights per step, split over 2 GPUs.
- Non-expert reads per GPU ≈ 4.3 GB, so each GPU reads about 14.2 GB per step.
- At 70 % of 1.79 TB/s (the PRO 6000 specification, not from the paper) that is about 11 ms per step.
- About 80 PCIe all-reduces add about 2 ms. Result: **about 75–90 tok/s per agent**, well above the 30 tok/s target.
- **So the constraint is capacity, not bandwidth.**
- Caveats:
  - Sub-3-bit codebook formats are harder to decode at roofline speed.
  - With DSpark verifying 6 tokens per sequence, about 200 unique experts per layer are read. That is about 4.5× the
    bytes for about 3× the tokens, which is why DSpark's load-aware verify length matters at batch 8.

**Options (for a PM decision; inferred):**

| Option | What | Pros | Cons |
|---|---|---|---|
| A | **2 × PRO 6000**: experts at about 2.5–2.7 bpw; Engram in host RAM (≥ 256 GB) | Same box as GLM | The experts ship as FP4. Re-quantising to 2.5 bits loses precision on top of that, with **unknown** quality impact (the paper does not say whether the experts had FP4 QAT). Needs a new sub-3-bit MoE kernel. Little headroom |
| B | **4 × PRO 6000 (382 GiB)**: MXFP4 as released (277.8 GiB) + DSpark (7.4 GiB) + reserve, about 73 GiB spare; Engram in host RAM | Native precision; no quality risk | TP = 4 or EP over PCIe; interconnect requirements |
| C | Serve DeepSeek-V4-Flash (284B, Table 1) on 2 × PRO 6000 | About 140 GiB at 4.25 bpw (inferred) | Different, older architecture (CSA + HCA hybrid, p. 4); 3,514 B/token; weaker than GLM-5.3 on agents (DeepSWE 54.4 vs 66.9, Table 3). Not recommended |

RTX 5090 (32 GB) profiles are out of scope for V4.1.

### 6(b) Kernel carry-over

| Our planned kernel | What DeepSeek-V4.1 needs | Verdict |
|---|---|---|
| **Sparse-MLA decode** (TEST-PLAN §4) | 64 heads (32 per GPU) × 512-d queries over K = V = one 512-d latent row per index (a direct MQA form of absorbed MLA) — **same shape as our kernel**. Differences: scale 512^-0.5 (ours is 256^-0.5); a **per-head attention sink** in the softmax denominator; **two KV sources** (128 FP8 SWA slots + ≤ 512 FP4 global entries); **dequantise on gather** (E2M1 + E4M3/16; E4M3 + UE8M0/32); cached rows already RoPE'd, so the output needs **inverse RoPE on the last 64 dims** (fused epilogue); K ≤ 640 (ours 2051); up to **6 query tokens per sequence** for DSpark verification (ours 2) | **Partial.** The gather, online softmax, split-K/LSE merge and paging logic carry over. The contract, dtypes, sink and epilogue are new. Our `lse` output makes merging the SWA and global parts easy |
| **NVFP4 MoE W4A16** (TEST-PLAN §5) | 384 experts, top-6, H 5120, I 2304 (1152 per GPU at TP = 2, or EP with 192 experts per GPU); separate w1 and w3 tensors; same SwiGLU clamp semantics (up in [−10, 10], gate ≤ 10); routing `sqrtsoftplus` + bias + normalise + ×1.5; experts released as **MXFP4 (UE8M0/32, no global scale)** | **Skeleton carries** (routing, selected-expert gather, clamp, weighted sum). Must parameterise H, I, E, k and the scale format. For option A a **new sub-3-bit dequant path** is needed. Note: MXFP4 (unlike NVFP4 group 32 with E4M3 scales) is a native block-scaled tensor-core format on Blackwell, so a W4A8 path like the reference's `fp4_gemm` may be possible (to check in cuTile) |
| **mHC** (Phase 1a) | Same n = 4 and 20 Sinkhorn iterations. Coefficients come from a [24 × 4·5120] FP32 projection. **Single-pass shift**, plus an optional fused Mega-mHC. Engram gates act per stream | **Carries.** Add a "coefficient source = previous sub-layer" mode and the fused variant |
| **MTP** (FR-6) | None. DSpark instead: 3 SWA blocks with 128-expert top-3 MoE; main projection from layers 37–39; 5 drafts per pass; Markov and confidence heads; load-aware verify length | **Does not carry.** Only the verify side (multi-token queries, acceptance) is shared |
| **Indexer** (Phase 1a) | 32 × 128 heads (same as GLM), **FP4 Q/K with UE8M0/32**, ReLU-weighted head sum, top-512 over compressed positions; candidate-pool restriction | **Partial**: same geometry, new precision and top-k sizes |
| **FP8 block GEMM** (Phase 1a) | FP8 with **32 × 32** UE8M0 blocks (GLM uses 128 × 128) | Parameterise the block size and scale format |
| KDA | Not used | n/a |

**New kernels or components V4.1 would need** (inferred sizing):
1. **CSA2 KV writer** (4 layers): projection → gated softmax pooling (ratio 2) or plain projection (ratio 1) → RMSNorm →
   RoPE (θ = 1.6e5 with YaRN) → FP4 pack into the global pool. Also indexer-K projection → norm → RoPE → MXFP4, and the
   per-sequence tail state. Small, fusable.
2. **SWA writer and prefill kernel:** FP8 ring-buffer write, and a windowed MQA flash attention for prefill (head dim 512,
   window 128, with sink).
3. **FP4 / FP8 KV dequant-on-gather** inside sparse attention (see the table above).
4. **Hierarchical indexer:**
   - FP4 scoring over all positions (Full layers) or pool positions only (Reindex layers);
   - a block-max over blocks of 8;
   - **top-2048 blocks** and **top-512** selection kernels (DeepSelect equivalents);
   - newest-block pinning.
5. **Sparse prefill attention** for CSA2 layers (T large, 640 selected entries per query).
6. **Engram:** a CPU-side hash and gather service writing into pinned staging buffers. On the GPU: FP8 dequant, the
   6144 → 25,600 FP8 GEMM, and the per-stream gate plus residual add. Precomputed hash tables ship in the checkpoint.
7. **DSpark:** draft blocks (reusing our attention and MoE kernels at other shapes), the Markov head (a rank-256 GEMV over
   the 129K vocabulary × 5 sequential steps), the confidence head, block sampling and acceptance, and the verify-length
   policy.
8. **Grouped low-rank output projection** (8 groups, block-diagonal 4096 → 1024 each): a batched FP8 GEMM.
9. **Router** with `sqrtsoftplus` and a text/image bias.
10. *(Optional, performance)* fused Mega-mHC; the vision encoder is out of v1 scope.

### 6(c) Architecture and modularity lessons for our engine
- **Memory manager with typed KV classes** instead of "one paged KV pool":
  - `GlobalKv { source_layer, ratio, main_entry_bytes: 288, index_entry_bytes: 68 }`. Paged and prefix-shareable.
    Allocated **per KV-source layer** (4), not per layer.
  - `SwaRing { window: 128, entry_bytes: 528 }` per layer per *active* sequence. Fixed size and recomputable.
  - `SeqState`: compressor tails (DeepSeek) and KDA recurrent state (GLM). Fixed size per sequence.
  - A **step-scoped arena** for transient shared tensors: top-k indices from 8 index sources, the candidate mask.

  GLM needs the same split: pages for DSA latent KV, slots for KDA state. So this generalisation is justified by two real
  users.
- **Layer plan as data.**
  - Build a validated `LayerPlan` at load time from `compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids`
    and `candidate_source_layer_id`. Use an enum such as
    `AttnKind::{Kda, Dsa, SwaOnly, Csa2 { ratio, mode: Full | Reindex | Reuse, kv_src, idx_src }}`.
  - Check every producer→consumer edge before serving ("parse, don't validate"). This mirrors
    `ref: SharedAttentionRuntime` but is typed.
- **Page geometry.** Prefix-cache blocks should align with the compression ratio (2) and the candidate block (8), for
  example 16-token multiples. They should also align with GLM's k-pool of 4 (inferred).
- **Tiered prefix cache.**
  - HBM holds active sequences. Host RAM holds idle sessions' global KV (890 B/token, so 100 sessions × 200K tokens ≈
    17 GiB). SSD is optional.
  - Classify state as **persist** (global KV) or **recompute** (SWA, via a 128-token encoder replay).
  - The resume path: global hit + SWA miss → bounded replay → continue.
  - For GLM, the analogue is snapshotting KDA state at turn boundaries.
- **CED-aware prefill scheduler.**
  - The prefill cost model is encoder-only for most tokens, plus decoder work for the last 128.
  - Chunked prefill must keep L20's KV projection in the encoder pass and run decoder replay once at the end.
- **Speculative-decoding interface.** A `Drafter` trait with two real implementations (GLM MTP, DeepSeek DSpark), so it is
  allowed by DEV-GUIDELINES §1.3:
  - `draft(batch) → {tokens[k], confidence[k]}`;
  - `verify_len(confidence, load)` using a **profiled step-time curve** (DSpark's idea; also useful to switch GLM MTP off
    under heavy load);
  - `commit(accepted)` / `rollback()`, which must restore SWA rings and compressor tails for rejected drafts.
- **Tensor-parallel details** (inferred):
  - The reference splits indexer heads across ranks and then **all-reduces a [T × context] score tensor** per index
    layer (`ref: Indexer.forward`). That is about 4 MB per layer per step at 8 × 128K, and grows linearly.
  - Instead, **replicate the small indexer on both GPUs**. The same applies to GLM's indexer.
  - Global KV is one head (MQA), so it is replicated per GPU anyway; at 890 B/token that doesn't matter.
- **Chat-template layer.** DeepSeek ships no Jinja template. Prompt encoding is in `encoding/` (Python) and
  `deepseek-recipe`, which is Rust (README). Our tokenizer and chat layer should be pluggable per model family.
- **Reasoning effort as an API knob.** It maps to a system-prompt line (p. 29). For agents, an effort of 60–80 gives most
  of the accuracy at under half the tokens of 100 (p. 34). That is a direct lever on our step-time metric M2.

### 6(d) Recommended changes to our documents (for PM approval; not applied)

**PRD**
- §5 "Later": replace "v1.1: DeepSeek, which reuses sparse MLA and MoE" with "v1.1: DeepSeek-V4.1-Flash (MIT): new
  attention family (CSA2 + SWA + FP4 KV) that reuses the sparse-attention core and the MoE skeleton; hardware profile to
  be decided (see research note)".
- §6: add a DeepSeek-V4.1 constraints block:
  - 543.6B routed experts, 277.8 GiB as released;
  - Engram 188.8 GiB, host RAM only;
  - 890 B/token KV;
  - 14.2B-parameter DSpark.
- §11 open decisions: add **"DeepSeek target box: A (2 × PRO 6000 with sub-3-bit experts, after a quality spike), B (4 × PRO 6000),
  or C (none)"**, and **host RAM for the selected hardware (≥ 256 GB; 384 GB if DeepSeek is planned)**.
- FR-5: make prefix caching **tiered (HBM, host, optional SSD) and model-agnostic**, with persist and recompute state
  classes.
- FR-6: "speculative decoding through a common drafter interface (GLM MTP in v1.0; DeepSeek DSpark later)".
- FR-3: add a `reasoning_effort` request field, with per-family mapping. For DeepSeek it is the effort line, 1–100.
- §10 risks: add "DeepSeek-V4.1 does not fit 2 × PRO 6000 at 4-bit" and "Engram needs about 189 GiB of host RAM".

**DEV-PLAN**
- **Phase 0 scope unchanged** (GLM kernels), but write the contracts to be forward-compatible at no extra
  implementation cost:
  - sparse attention with parameters for the softmax scale, an optional per-head sink, a KV element-format enum (only
    BF16 implemented now), an optional inverse-RoPE epilogue on the last r dims, T ≤ 6 queries per sequence, and LSE
    output for merging two sources;
  - MoE with parameters H, I, E, k and the scale format (E4M3 g16/g32 with a global scale; UE8M0 g32).
- **Step 0.5 gather microbenchmark (SMLA-P-005):** add row sizes of **288 B, 528 B and 68 B** as well as 1 KB. The V4.1
  regime has small rows, which is the harder case for cuTile gathers.
- Phase 1a: FP8 block GEMM must support 32 × 32 UE8M0 blocks as well as 128 × 128. The mHC kernel gets a single-pass mode.
- Phase 2 (tq-runtime): build the typed KV-class memory manager, the `LayerPlan`, the step arena and the `Drafter` trait
  from the start. Replicate the indexer under TP rather than splitting its heads (measure both for GLM).
- New **Phase 3-DS outline** (details at that gate):
  - **3-DS.0** Fit and quality spike (cost-capped): quantise V4.1 experts to 2.5/2.75/3.0 bpw and measure KL and QUAL
    against MXFP4 on rented large GPUs. Decide A or B from the result.
  - **3-DS.1** Engram host service (hash parity, gather, prefetch deadline).
  - **3-DS.2** CSA2 kernels (KV writer, FP4 indexer + top-k, FP4/FP8 sparse attention with sink, SWA prefill).
  - **3-DS.3** CED-aware prefill and bounded replay.
  - **3-DS.4** DSpark.
  - **3-DS.5** Mega-mHC.
- Checkpoint conversion (1c analogue): precompute the Engram token map, primes and hash multipliers. Re-lay out the
  Engram rows so each row and its scales sit together (one read per row).

**TEST-PLAN**
- **Oracles for DeepSeek:**
  - O1 written from `inference/model.py` at a pinned commit;
  - O2 the `transformers` `deepseek_v41` model (config lists transformers 5.6.0; availability to verify);
  - a **production-semantics oracle** that adds decoder and encoder bounded replay.

  The PM decides which of these defines "correct" for the engine.
- **REF additions:**
  - FP4 KV codec (E2M1 + E4M3/16, no global scale, range ±2688);
  - MXFP4 codec (E2M1 + UE8M0/32) for the indexer and experts;
  - Engram hash parity on 10K random sequences, including sequence start and image spans (dead tokens).
- **Accounting test:** the engine reports **890 B/token** of global KV for the V4.1 config, and a startup memory plan of
  ≤ 95.6 GiB per GPU (fail fast).
- **CSA2 dependency tests:**
  - Reuse consumes the latest index source;
  - Reindex consumes the latest Full KV;
  - the candidate pool is ≤ 16,384 and always contains the newest block;
  - each Reindex top-512 is a subset of the pool;
  - the ratio-2 tail is handled correctly for odd prompt lengths and across decode steps;
  - the SWA ring wraps correctly at 128.
- **Change the API rule "a prefix-cache hit gives the same output as a miss":**
  - bitwise equal when SWA state is also hit;
  - **KL-bounded** when bounded replay runs, because by design it is not identical (p. 20).
- **DSpark:** draft parity with `forward_spec`; acceptance rate on agent traces; **rollback correctness** (SWA ring and
  compressor tail restored after rejection).
- **QUAL for sub-4-bit experts:** per-token KL against the MXFP4 reference, plus hidden-test coding pass rate
  (gate M3 style).
- **TP:** replicated indexer equals head-split indexer (set equality of the indices).

---

## 7. PM brief (plain language)
- **What it is.** DeepSeek-V4.1-Flash is a free (MIT-licensed) AI model released on 10 Sep 2026. On coding agents it
  matches the best closed models: 74.2 % on DeepSWE, vs 66.9 % for GLM-5.3, our v1.0 model.
- **KV cache** = the model's *notes about the conversation so far*, kept on the GPU so it doesn't re-read everything for
  each new word. Coding agents have very long conversations, so these notes usually fill the GPU.
- **What V4.1 changed:** extreme shorthand notes, 890 bytes per token (word-piece): 1/4 of its predecessor and about
  1/13 of GLM-5.3 (my estimate). Eight agents with 256,000-token histories need under 2 GB of notes. Tricks: 40 layers
  share 4 notebooks, notes are stored at 4-bit, and short-term scratch notes are rebuilt from the last 128 tokens
  instead of being saved.
- **The catch:** the "brain" (weights) is big: 552 billion numbers, about 278 GiB at the usual 4-bit compression, while
  our 2-GPU box has about 190 GiB. A 196-billion-number phrasebook ("Engram") must also sit in main memory.
  *Analogy:* the GPU is a small, fast desk; main memory is a big, slower shelf. The phrasebook stays on the shelf and
  the model fetches only the few lines it needs, just in time.
- **Options:** (A) squeeze the brain to about 2.5 bits per number to fit our box, after a quality test; or (B) evaluate a
  4-GPU configuration. Either way the box needs at least 256 GB of main memory.
- **How it was trained:**
  - **SFT** (supervised fine-tuning): study worked examples with correct answers.
  - **RL** (reinforcement learning): try real tasks in practice sandboxes, get scored, and do more of what worked.
  - **On-policy distillation:** the student solves problems *its own way* while 40+ expert "teacher" models mark each
    step, fixing its real habits rather than textbook ones.
- **Serving parameter:** "reasoning effort" (1–100). Settings of 60–80 give most of the quality at under half the words, so
  agent steps get shorter.
- **Bottom line:** keep GLM-5.3-Flash for v1.0. Before committing to DeepSeek, choose A (after a cheap quality test) or
  B. DeepSeek's engineering ideas (shared notes, cheap rebuilds, pluggable draft-ahead decoding) help GLM too, so build
  them into our engine now.

---

## 8. Open questions and unverified items
1. **Expert QAT.** The paper does not say whether the routed experts were trained with FP4 QAT. The checkpoint ships them
   as MXFP4. This decides how much quality a further cut to 2.5–2.7 bpw loses. Measure it (3-DS.0).
2. **Canonical prefill semantics.** Production uses decoder (and, on a miss, encoder) bounded replay (p. 20). The
   reference `model.py` runs all 40 layers exactly. Which one is our correctness target? Is the `transformers`
   implementation (config says 5.6.0) the same as `model.py`?
3. **DSpark scheduler.** The verify-length policy and the "throughput curves" are not specified. The reference omits the
   loop. Read the DSpark paper (arXiv 2607.05147) before designing the drafter interface.
4. **The 15/11 kernel breakdown** per Reuse layer is not listed; only the fused kernel names are (pp. 18–19).
5. **Engram on SSD.** Not described by DeepSeek. The hit-rate distribution of n-gram rows and the latency budget
   (prefetch before layers 1 and 14) are unmeasured.
6. **SWA KV scale format.** The paper says only "FP8". 528 B per entry assumes the reference's UE8M0 per-32 scales.
7. **Internal inconsistencies in the paper:**
   - Mega-mHC is described against an "original four-kernel implementation" (p. 5) but a "three kernels" multi-pass
     scheme (p. 12);
   - NL2Repo-Bench is 65.4 in Table 3 (p. 33) but 64.0 in the HF README.
8. **sm_120 support** in DeepSeek's kernels (FlashMLA, DeepGEMM, TileKernels, DeepSelect) and in vLLM/SGLang for V4.1 is
   unknown. It is the same opportunity as GLM's (PRD §2), but must be checked.
9. **The runtime reserve (10–12 GiB) and the decode-speed estimate** (§6a) are my estimates. They need a measured memory
   plan and roofline on real hardware.
10. **Long-context indexer cost.** The Full layers still scan the whole context: L20 over N entries, and L2, L8, L14 over
    N/2 each. At 1M tokens this is linear work per decode step, in FP4. It needs measuring for our 8-agent target.
11. **`deepseek-recipe`** (Rust prompt encoding): check its licence and whether it fits as a dependency (DEV-GUIDELINES
    §2.4).
12. **Hierarchical indexer in the encoder.** It is decoder-only by design (p. 11). Whether a similar restriction would be
    safe in the encoder is untested; do not do it.
13. **DeepSeek-V4-Flash (option C)** figures are extrapolated from Table 1 only (284B backbone). Its exact breakdown was
    not checked.
