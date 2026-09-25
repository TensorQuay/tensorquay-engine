# GLM-5 paper review: findings and what they mean for TensorQuay Engine

Public technical extract: private deployment observations and operational plans are omitted.

| | |
|---|---|
| **Paper** | *GLM-5: from Vibe Coding to Agentic Engineering*, GLM-5 Team (Zhipu AI and Tsinghua University) |
| **arXiv** | 2602.15763, **version 2**, dated 24 Feb 2026 (p. 1, margin stamp) |
| **Review date** | 19 Sep 2026 |
| **Our target model** | GLM-5.3-Flash (a later, smaller model; the paper does not describe it) |

**What was read:**
- **The whole paper, 40 pages, including Appendices A (hyper-parameters) and B (evaluation details).** Pages 1–30 and
  36–40 were read as rendered pages, so figures and tables were checked visually. Pages 31–35 (contributors and
  references) were read from the text extraction.
- **GLM-5.3-Flash reference code:** Hugging Face `transformers` `models/glm5_next` at commit 770e4c40d0 (the same pin as
  TEST-PLAN §2), files `modeling_glm5_next.py` (2,444 lines) and `configuration_glm5_next.py` (322 lines).
- **GLM-5.3-Flash `config.json`**, including the full `quantization_config.modules_to_not_convert` list (1,509 entries).
- **IndexCache**, arXiv 2603.12201 v1: the abstract page and a skim of the HTML version (not read page by page).
- **Project documents:** PRD, DEV-PLAN, TEST-PLAN and DEV-GUIDELINES.

**How to read the references:**

| Tag | Meaning |
|---|---|
| **p. N** | PDF page N of the paper. PDF page numbers equal the printed page numbers. |
| **code LN** | Line N of `modeling_glm5_next.py`. |
| **cfg `key`** | A key in the GLM-5.3-Flash `config.json` (text config, unless stated otherwise). |
| **(computed)** | Arithmetic from the config values. The script logic is described next to each number. |
| **(inferred)** | Our interpretation. Not stated in any source; verify before relying on it. |

---

## 1. Key facts (GLM-5, from the paper)

### 1.1 Model and architecture
| Topic | Fact | Page |
|---|---|---|
| Size | 744B total, 40B active parameters. GLM-4.5 was 355B / 32B. The counts include MTP layers but not word embeddings or the output layer | p. 4; Table 10, p. 36 |
| Layers | The text says the layer count was reduced "to 80" to cut expert-parallel communication. Table 10 lists 3 dense layers + 75 MoE layers + 1 MTP layer (see §7, Q4) | p. 4; p. 36 |
| Widths | Hidden 6144. Dense intermediate 12288. MoE intermediate 2048. Vocabulary 154,880 | Table 10, p. 36 |
| Experts | 256 in total, 8 routed per token, 1 shared | Table 10, p. 36 |
| Attention | MLA + DSA in every layer. 64 heads. QK head dim 192, V head dim 256. Q LoRA 2048, KV LoRA 512. Indexer: 32 heads × 128 | Table 10, p. 36 |
| MLA vs GQA under the Muon optimizer | MLA with a 576-d latent cache could not match GQA-8. **Muon Split** fixes this: it orthogonalises each head's slice of W_UQ, W_UK and W_UV separately. It also keeps attention logits stable without clipping | pp. 4–5 |
| Table 1 (MMLU / BBH / GSM8K / HumanEval) | GQA-8: 61.2 / 53.3 / 47.6 / 38.5. MLA: 61.5 / 48.9 / 46.2 / 33.5. MLA + Muon Split: 62.5 / 51.8 / 45.0 / 36.7. MLA-256 + Muon Split: 62.0 / 51.3 / 47.5 / 36.6 | Table 1, p. 5 |
| MLA-256 | Head dim goes from 192 to 256 and head count drops by 1/3. Training compute and parameter count stay the same; decode compute goes down. It matches MLA + Muon Split in quality | p. 5 |
| Decode cost | MLA decode does a 576-d dot product, against 128-d for GQA. DeepSeek-V3's head count was chosen for the H800 roofline, which "is inappropriate for other hardware" | p. 5 |

### 1.2 Multi-token prediction (MTP)
| Topic | Fact | Page |
|---|---|---|
| Design | DeepSeek-V3 trains one MTP layer but uses it for 2 tokens at inference, and that mismatch lowers acceptance of the second token. GLM-5 instead **shares the parameters of 3 MTP layers during training**, so the draft model costs the same memory as DeepSeek-V3's | p. 5 |
| Accept length | GLM-5: 2.76. DeepSeek-V3.2: 2.55. Both with 4 speculative steps, on a private prompt set | Table 2, p. 5 |
| Pipeline placement | The MTP output layer sits with the main output layer on the last pipeline stage, so they can share parameters | p. 9 |
| Where MTP pays off | "Especially effective under the small-batch decoding regime". Used for RL rollouts to cut tail latency | pp. 14–15 |

### 1.3 DSA continued pre-training
| Topic | Fact | Page |
|---|---|---|
| Starting point | The base model at the end of mid-training | p. 6 |
| Warm-up | 1,000 steps. Each step is 14 sequences × 202,752 tokens. Maximum learning rate 5e-3, decaying to 2e-4 | p. 6; p. 36 |
| Sparse adaptation | 20B tokens with the mid-training data and hyper-parameters, at a constant LR of 1e-5. For comparison, DeepSeek-V3.2 used 943.7B tokens | p. 6; Fig. 5, p. 4; p. 36 |
| Long-context results, MLA → DSA | MQ-NIAH-128k 100 → 100. MV-NIAH-128k 95.5 → 97.0. SQuAD-128k 79.7 → 86.0. **HotpotQA-128k 66.3 → 63.0** | Table 3, p. 5 |
| SFT check | MLA and DSA models fine-tuned on the same data "tie" in loss and on benchmarks. The relative loss stays within about ±0.0004 | Fig. 6, p. 6 |
| Speed claim | Attention compute is about 1.5–2× lower for long sequences: "128K contexts at half the GPU cost". The paper also says 90 % of attention entries are redundant, citing DeepSeek-V3.2-Exp | p. 6 |
| DSA in RL | A **deterministic top-k** (`torch.topk`) is required. Non-deterministic CUDA or TileLang top-k "caused drastic performance degradation … after only a few steps". The indexer is frozen during RL. Replaying the indexer's choices is impractical because k = 2048 | p. 12 |

### 1.4 Efficient-attention ablations (GLM-9B)
The baseline is GLM-9B: 40 layers, GQA in every layer, fine-tuned for 128K context.

| Topic | Fact | Page |
|---|---|---|
| SWA pattern search | Beam search with beam 8, changing 2 layers per step; it converges in about 10 steps. Each candidate is scored on RULER at 16K. The resulting S/F pattern is printed on p. 6 | p. 6 |
| SWA without extra training (RULER 128K) | 1:1 full-to-SWA ratio, 4,096-token window. Full attention 75.28; interleaved SWA **6.51**; searched pattern 53.95 | Table 4, p. 7 |
| Continued training | 190B tokens at 64K context, with a 1:1 ratio of efficient to full-attention layers | p. 7 |
| RULER 64K / 128K | GLM-9B 85.35 / 75.28. Interleaved SWA 65.94 / 44.93. SWA pattern 83.72 / 69.59. GDN 76.76 / 64.00. SimpleGDN 81.76 / 67.03 | Table 5, p. 7 |
| RepoQA 128K | GLM-9B 65.83. SimpleGDN 58.50 (−7.33). GDN 56.17. SWA pattern 51.17 | Table 5, p. 7 |
| SimpleGDN | Removes the Conv1d and the explicit gates, and maps the pre-trained Q/K/V weights straight into the linear recurrence | p. 7 |
| Paper's conclusion | Every efficient variant loses on fine-grained retrieval, even when half the layers keep full attention. The authors call DSA "lossless by construction" | p. 7 |
| DSA on GLM-4.7-Flash (RULER 128K) | Warm-up trains only the indexer (1,000 steps, batch 16); joint training then runs for 150B tokens. Baseline 79.21 → warm-up only 71.35 → full DSA 78.86 | Table 6, pp. 7–8 |

### 1.5 Context length and training data
| Topic | Fact | Page |
|---|---|---|
| Context length by stage | Pre-training at 4K (18T general + 9T code and reasoning). Mid-training: 32K (1T tokens) → 128K (500B) → 200K (50B). SFT up to 202,752 tokens | Fig. 5, p. 4; p. 8; p. 11 |
| Evaluation lengths | Up to 131,072 generated tokens. 202,752-token context for HLE with tools | p. 22; p. 36 |
| Token count | 28.5T tokens for the base model (the introduction says "27 trillion token corpus") | p. 3; p. 4 |
| Code data | Unique tokens up 28 % after fuzzy deduplication. About 10M issue–PR pairs, about 160B unique tokens | p. 8 |
| Learning rate | Muon optimizer. LR rises from 0 to 2e-4, decays to 4e-5 during pre-training, then decays linearly from 4e-5 to 1e-5 in mid-training | p. 36 |

### 1.6 Infrastructure
| Topic | Fact | Page |
|---|---|---|
| Training memory techniques | Pipeline ZeRO-2 gradient sharding (1/dp of the gradients per rank, plus 2 rolling full buffers). Muon all-gather limited to owned shards. Activation offload to host. Sequence-chunked output projection | p. 9 |
| INT4 QAT | Quantisation-aware training in the SFT stage. The same quantisation kernel serves training and offline quantisation, "bitwise-identical" between the two | p. 10 |
| RL hyper-parameters | GRPO + IcePop, without the KL term. β = 2, ε_low 0.2, ε_high 0.28. Group size 32, batch size 32 | pp. 11–12 |
| On-policy distillation | Group size 1, batch size 1024 | p. 14 |
| Rollout serving | EP64 and DP64 over 8 nodes. DP-attention "to prevent copying KV across different ranks". FP8 rollouts, MTP, and prefill–decode (PD) disaggregation | pp. 14–15 |
| Orchestrator | More than 1k concurrent rollouts | p. 16 |

### 1.7 Environments and context management
| Topic | Fact | Page |
|---|---|---|
| Environments | SWE: over 10k verifiable environments across 9 languages. Terminal: thousands of Harbor-format tasks, with Docker builds succeeding over 90 % of the time. Search: a web knowledge graph built from more than 2M pages | pp. 17–18 |
| Context management (BrowseComp) | Keep-recent-k with k = 5 raises the score from 55.3 to 62.0. Adding discard-all at T = 32K (hierarchical context management) reaches 75.9 | p. 19; Fig. 8 |

### 1.8 Inference and deployment
| Topic | Fact | Page |
|---|---|---|
| Quantisation on Ascend | W4A8 lets the 750B model fit one Atlas 800T A3: attention and MLP at W8A8 (INT8), experts at W4A8 (INT4). QuaRot and Flex_AWQ_SSZ are used for calibration | p. 21 |
| Fused kernels on Ascend | **Lightning Indexer** fuses score, ReLU and TopK. **Sparse Flash Attention** selects the top-k entries and attends in parallel. **MLAPO** fuses 13 MLA pre-processing operators into one | pp. 21–22 |
| Engine features on Ascend | Async scheduling (overlaps the device-to-host copy of sampled tokens with preparation of the next step). RadixCache and a prefix cache in host RAM. Attention DP + MoE EP + FlashComm (a split all-reduce). MTP | p. 22 |
| Claimed result | One domestic node matches "dual-GPU international clusters", with 50 % lower deployment cost for long sequences | p. 22 |
| NVIDIA numbers | **None.** The paper gives no tok/s, latency or roofline figures for any NVIDIA GPU | whole paper |

### 1.9 Results
| Benchmark | GLM-5 | Reference points | Page |
|---|---|---|---|
| SWE-bench Verified | **77.8** | Claude Opus 4.5 80.9; GLM-4.7 73.8; DeepSeek-V3.2 73.1 | Table 7, p. 23 |
| SWE-bench Multilingual | 73.3 | Claude Opus 4.5 77.5 | p. 23 |
| Terminal-Bench 2.0 (Terminus-2) | **56.2**, or 60.7 on the verified set | Claude Opus 4.5 59.3 | p. 23 |
| Terminal-Bench 2.0 (Claude Code) | 56.2, or 61.1 verified | Claude Opus 4.5 57.9 | p. 23 |
| CyberGym | 43.2 | Claude Opus 4.5 50.6 | p. 23 |
| BrowseComp | 62.0, or 75.9 with context management | | p. 23 |
| Tool use | τ²-Bench 89.7. MCP-Atlas 67.8. Tool-Decathlon 39.2 | | p. 23 |
| Long horizon | Vending-Bench 2: $4,432. GDPval-AA Elo 1,409 | | p. 23 |
| Reasoning | HLE 30.5, or 50.4 with tools. AIME 2026 I 92.7. GPQA-Diamond 86.0. LongBench v2 64.5 | | p. 23 |
| SWE-rebench (Jan 2026) | 42.1 % ± 1.21 (SEM), pass@5 50.0 % | Claude Opus 4.6 52.9 % | Table 9, p. 27 |
| CC-Bench-V2 (internal) | Backend 25.8 (Opus 4.5: 26.9). Repo exploration 65.6 (Opus: 64.5). Chained tasks 52.3 (Opus: 61.6). Frontend build success 95–100 | | Table 8, p. 25 |
| Agent-as-a-Judge validity | Agrees with humans on 94 % of 130 check-items. Spearman 85.7 % against human model rankings | | p. 26 |
| Artificial Analysis Intelligence Index v4.0 | 50, the open-weights leader (GLM-4.7: 42) | | p. 2 |
| Base model (Table 11) | EvalPlus 87.0. LiveCodeBench-Base 34.4. MMLU 88.3. **GSM8K 68.8**, below GLM-4.5-Base at 79.4 | | p. 37 |

**Evaluation settings (useful for our QUAL tests):**
- **SWE-bench:** OpenHands, temperature 0.7, top_p 0.95, max_new_tokens 16,384, 200K context (p. 36).
- **Terminal-Bench 2.0 with Terminus-2:** 2-hour timeout, temperature 0.7, top_p 1.0, max_new_tokens 8,192, 128K
  context, 16 CPUs and 32 GB RAM (p. 37).
- **Terminal-Bench 2.0 with Claude Code:** temperature 1.0, top_p 0.95, max_new_tokens 65,536, averaged over 5 runs
  (p. 37).

---

## 2. Architecture explained

### 2.1 MLA-256
**How MLA stores memory.**
- Multi-head latent attention (MLA) compresses each token's keys and values into one shared latent vector: 512 values,
  plus 64 RoPE values in DeepSeek-style models, so "576-dimension" (p. 4).
- At decode time, each head's key and value up-projections are folded ("absorbed") into the query and the output. Each
  head then compares its query with every cached latent: one 576-wide dot product per head per key (p. 5).

**The consequence (inferred explanation, consistent with p. 5).**
- In absorbed decode, compute grows with the **number of heads**, and bytes grow with the latent width. Head
  *dimension* does not enter.
- Training and prefill run MLA in ordinary multi-head form, where cost grows with heads × head_dim.
- So GLM-5 made heads wider (192 → 256) and fewer (−1/3). Training cost stayed the same while decode FLOPs dropped by
  about a third.
- The paper says this directly for hardware: DeepSeek-V3's head count was tuned to the H800 roofline and does not suit
  other chips (p. 5).
- The quality cost is nil: MLA-256 matches MLA under Muon Split (Table 1, p. 5).

**GLM-5.3-Flash takes this further.**
- 64 heads with a 256-d query/key head, all without positional encoding: cfg `qk_nope_head_dim` 256,
  `qk_rope_head_dim` 0, `v_head_dim` 256.
- The model uses no RoPE at all: `position_embeddings=None`, code L1503–1504. The config validator requires
  `qk_rope_head_dim` to be 0 (configuration file, lines 225–228).
- The absorbed decode dot product is therefore exactly `kv_lora_rank` = 512 wide, not 576.
- The latent row, after `kv_a_layernorm`, serves as both K and V in absorbed form (code L1191–1193, L1166–1167).
- The softmax scale is `qk_head_dim^-0.5` = 256^-0.5 (code L1149). This matches our TEST-PLAN C-008 guard.

### 2.2 DSA: indexer and top-k
**What the paper says.**
- DSA replaces dense O(L²) attention with content-based selection (p. 5). A small "lightning indexer" (32 heads × 128;
  Table 10) scores the cached tokens, keeps the top k = 2048, and attention runs only over those (p. 12).
- DSA is added to an already-trained dense model in two steps:
  1. **dense warm-up:** only the indexer learns;
  2. **sparse adaptation:** everything trains together (pp. 5–6).

**How the GLM-5.3-Flash indexer works** (from the code; the paper gives no formula).
1. **Indexer query.** `q_I = wq_b(q_resid)`, 32 heads × 128. `q_resid = RMSNorm(q_a_proj(x))` is the same low-rank query
   the attention uses (code L835, L1188). The indexer therefore has no `wq_a` of its own.
2. **Indexer key.** One 128-d key per token, `k_I = LayerNorm(wk(x))` (code L836, L801).
3. **Score.** `Σ_h w_h(x) · 32^-0.5 · ReLU(q_I,h · k_I · 128^-0.5)`, computed in FP32 (code L863–868).
4. **k-pool 4 (new in Flash).** Tokens are grouped in fours, counted from the first valid token (code L978–985). Each
   complete group is compressed into one pooled key. The pooled key is a per-channel softmax-weighted mean: the weights
   come from a gate on the hidden state plus a learned per-slot bias called "APE" (code L999–1005). Scores are computed
   against pooled keys.
5. **Selection.** Of the complete pools whose last token the query can see (code L871–877), the top 2048 / 4 = 512 are
   chosen with `torch.topk` (code L885, L890). They are expanded back to 2048 token positions.
6. **Tail.** Up to 3 tokens of the current incomplete pool are always appended (code L904–907, L1012–1062).
7. **Output.** `int32 [T, 2051]`, where −1 means empty (code L901, L910–915). Our SMLA contract (K ≤ 2051, −1 = empty)
   matches this.

**Cost and quality.**
- The paper claims 1.5–2× less attention compute on long inputs, and 128K "at half the GPU cost" (p. 6).
- Quality is near-lossless on most long-context tests (Table 3; Fig. 6). The one clear loss is HotpotQA-128k, 66.3 →
  63.0, so "lossless by construction" (p. 7) is the authors' framing more than a measured fact.

**IndexCache (arXiv 2603.12201 v1, 12 Mar 2026; from the abstract and a skim of the HTML).**
- **Idea:** consecutive layers pick very similar top-k sets. Layers are split into *Full* layers, which run the indexer,
  and *Shared* layers, which reuse the nearest Full layer's indices.
- **How the Full layers are chosen:** either greedily, converting whichever layer least raises calibration loss, or by
  training with a multi-layer distillation loss.
- **Quality:** on GLM-4.7-Flash (30B-A3B, MLA, 47 layers), keeping 1/4 of the indexers costs about −0.3 on the
  long-context average; keeping 1/8 costs −4.1.
- **Speed at 200K on H100 with SGLang (DP-attention, dp_size 8):** prefill up to 1.82× faster (19.5 s → 10.7 s); decode
  up to 1.48× faster (58 → 86 tok/s).
- **GLM-5 (preliminary):** at 1/2, the average is 78.7 against 78.4, with about 1.2× end to end.
- **In GLM-5.3-Flash:** the code implements it as `indexer_types` "full"/"shared", passing the top-k forward
  (code L1102–1155, L1507). But **the GLM-5.3-Flash config sets all 45 entries to "full"**, so the released model does
  *not* share indices across layers.

### 2.3 MTP parameter sharing and speculative decoding
**What MTP does.**
- MTP adds a small extra block that predicts tokens beyond the next one. It improves the base model and works as a
  built-in draft model for speculative decoding (p. 5).
- In speculative decoding, the draft proposes d tokens and the main model checks all of them in one pass. Every token up
  to the first wrong guess is kept.

**GLM-5's twist (p. 5).**
- The model is trained with 3 chained MTP steps that **share one set of weights**.
- At inference the one MTP layer is chained d times, exactly as in training. DeepSeek-V3 instead trained 1 step and used
  2, and that mismatch lowers second-token acceptance.
- Result: accept length 2.76 against 2.55 for DeepSeek-V3.2, at 4 speculative steps (Table 2).
- If "accept length" counts the always-kept bonus token, and each draft token is accepted independently, then 2.76 at
  d = 4 means about **70 % acceptance per draft token** (computed from `(1 − p^(d+1)) / (1 − p)`; the assumption is
  inferred).

**When MTP helps (pp. 14–15).** Most at small batch, because each step's cost is then dominated by weights that every
token shares.

### 2.4 Decode cost and roofline

**What the paper says** (qualitative only):
- **MLA:** the 576-d decode dot product vs 128-d for GQA, and head counts should follow the target chip's roofline
  (p. 5).
- **DSA:** 1.5–2× less attention compute (p. 6).
- **Serving:** FP8 and MTP for latency; DP-attention to avoid duplicated KV; PD disaggregation to protect decodes
  (pp. 14–15).
- **Ascend:** fused indexer, sparse attention and pre-processing kernels (pp. 21–22).

**What it means on sm_120 for GLM-5.3-Flash** (all computed from config values; bandwidth and FLOP peaks must come from
our own measurements, H-002 and the proposed H-006):

| Quantity | Value | Basis |
|---|---|---|
| Sparse-MLA bytes read per query token per DSA layer | 2051 × 512 × 2 B = **2.0 MiB** (BF16 latent) | cfg `kv_lora_rank`, `index_topk`, `index_kpool` |
| Sparse-MLA FLOPs per query token per layer | 134 M at H = 32 (TP = 2); 269 M at H = 64 | 2 × (QK + PV) × 512 × 2051 × H |
| Arithmetic intensity | **64 FLOP/B at H = 32; 128 FLOP/B at H = 64** | ratio of the two rows above |
| Indexer pooled-key bytes per query per layer (BF16) | 2 MiB at 32K context; **8 MiB at 128K** | (ctx / 4) × 128 × 2 B |
| Indexer prefill FLOPs, 11 layers | 48 TFLOP at 64K; 194 TFLOP at 128K. Linear layers need about 2,200 / 4,400 TFLOP | 1024 · L² per layer, vs 2 × 16.7B active × L |

- Because the latent is shared by all heads, arithmetic intensity grows with the number of heads per GPU. At 64–128
  FLOP/B the kernel sits near or below the ridge point of an sm_120 card (inferred; the exact ridge depends on the
  BF16/FP32-accumulate tensor peak, which we haven't measured). A DRAM-roofline gate is right for H = 32. At H = 64, or
  with several query tokens per sequence sharing rows, the kernel may become compute-bound (see §5d).
- With k-pool 4 and only 11 DSA layers, the indexer is a small share of GLM-5.3-Flash's compute. That probably explains
  why the Flash config does not use IndexCache sharing (inferred).

---

## 3. GLM-5 vs GLM-5.3-Flash

| Item | GLM-5 (paper) | GLM-5.3-Flash (code / config) |
|---|---|---|
| Total / active parameters | 744B / 40B, counting MTP but not embeddings or the output layer (Table 10, p. 36) | ≈ 312B main model + 7.4B MTP layer, not counting embeddings and output layer; ≈ 321B with them. Active ≈ 16.1B, plus 0.63B for `lm_head` (computed) |
| Layers | 3 dense + 75 MoE + 1 MTP (Table 10); "80" in the text (p. 4) | 45 (cfg `num_hidden_layers`). Layers 0–2 dense, 3–44 MoE (cfg `mlp_layer_types`). MTP is layer 45 (cfg `num_nextn_predict_layers` 1) |
| Hidden size | 6144 | 4096 (cfg `hidden_size`) |
| Experts | 256 total, 8 routed, 1 shared, MoE intermediate 2048 | **288**, 8 routed, 1 shared, intermediate 2048 (cfg `n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size`) |
| Router | Not described | Sigmoid scores in FP32, `noaux_tc` with `e_score_correction_bias` used only to choose experts, `n_group` 1, normalised top-k × 2.5 (code L159–184; cfg `routed_scaling_factor`) |
| MLP activation | Not described | SwiGLU with the gate clamped to ≤ 10 and the up projection clamped to [−10, 10] (code L99–105, L138–143; cfg `swiglu_limit`) |
| Attention layout | MLA + DSA in every layer | **Hybrid.** 34 KDA linear-attention layers and 11 DSA layers, at indices 3, 7, …, 43, i.e. every fourth layer (cfg `layer_types`, `linear_attn_config.full_attn_layers`) |
| MLA shape | 64 heads, QK 192, V 256, Q LoRA 2048, KV LoRA 512 | 64 heads, QK 256 without RoPE, V 256, Q LoRA **1536**, KV LoRA 512, scale 256^-0.5 (cfg; code L1149) |
| Positional encoding | Not stated. DeepSeek-style MLA uses decoupled RoPE (inferred) | **None.** `qk_rope_head_dim` 0 (code L1503–1504). `indexer_rope_interleave: true` in the config has no effect because the RoPE dimension is 0 (inferred) |
| KDA layers (new) | Not in GLM-5. The paper's GDN/SimpleGDN ablations hurt retrieval at 128K (Table 5, p. 7) | 64 heads × 128. Short causal conv (kernel 4) on q, k, v with SiLU. L2-normalised q and k. Per-channel forget gate bounded to [−5, 0) (code L341–371). Per-head beta. Sigmoid-gated output RMSNorm. FP32 state [64, 128, 128] (code L622–771) |
| Indexer | 32 × 128, top-2048 over individual tokens | 32 × 128, top-2048 as **512 pools of 4 tokens (learned pooling) plus up to 3 tail tokens = 2051** (cfg `index_kpool` 4, `index_kpool_always_select_tail`; code L774–1062). Note the code's default `index_kpool` is 16 (configuration file, line 153): always read the value from the config |
| Cross-layer index sharing | Not in the paper. IndexCache tested GLM-5 at 1/2 as a preliminary result | Supported in code (L1151–1155), **not used**: all `indexer_types` are "full" |
| MTP | 1 layer, trained as 3 chained steps with shared parameters (p. 5) | 1 layer (layer 45). It is a DSA layer with **its own indexer**, plus a 288-expert MoE, `eh_proj`, `enorm`, `hnorm` and `shared_head.norm` (cfg `modules_to_not_convert`). `index_share_for_mtp_iteration: true` (§5a). How it was trained is **unknown**. **The HF code ignores layer 45** (code L1380) |
| Residual path | Standard (the paper mentions no hyper-connections) | **mHC**: 4 residual streams per token. Each block has its own collapse, expand and 4 × 4 mixing weights. The mixing matrix is made doubly stochastic with 20 Sinkhorn iterations. The final output is the unweighted mean of the streams (code L220–338; cfg `hc_mult`, `hc_sinkhorn_iters`) |
| Vocabulary | 154,880 | 154,880 (cfg `vocab_size`) |
| Trained context | 200K (p. 8); SFT 202,752 (p. 11) | `max_position_embeddings` 1,048,576. The training length is unknown |
| Official weights | FP8 rollouts (p. 14); INT4 QAT in SFT (p. 10); W4A8 on Ascend (p. 21) | FP8 E4M3 in 128 × 128 blocks with dynamic activation scales (cfg `quantization_config`). **Kept in BF16/FP32:** all KDA projections, `kv_b_proj`, the indexer, the router, mHC, embeddings and `lm_head` (cfg `modules_to_not_convert`) |
| Modality | Text | `Glm5NextForConditionalGeneration` with a 24-block vision tower (`vision_config`). The PRD's v1.0 is text only |

**Budget checks** (all computed; they agree with PRD §6):

| Item | Size |
|---|---|
| Routed experts | 304.4B parameters: 150.6 GiB at NVFP4 group 32, or 159.5 GiB at group 16 |
| Weights other than experts, in official formats | 14.3 GiB, including **8.7 GiB of BF16 KDA projections** |
| MTP layer | 3.8 GiB with group-32 experts |
| Latent KV | 11,264 B per token (11 layers × 512 × 2 B); +1,024 B per token with the MTP layer |
| Indexer cache, HF layout (k + gate, BF16) | 5,632 B per token |
| Indexer cache, pooled keys only (see §5a) | **704 B per token** |
| KDA state | about 141 MiB per sequence (FP32 recurrent state plus conv state) |

---

## 4. Training and post-training summary

### 4.1 Pre-training and mid-training
- **Pre-training (pp. 4, 8):**
  - 18T general tokens plus 9T code and reasoning tokens, at 4K context (Fig. 5).
  - Web data: a new DCLM-style embedding classifier and a "world knowledge" classifier for long-tail facts.
  - Code: refreshed repository snapshots (+28 % unique tokens), fixed Software Heritage metadata, and classifiers for
    low-resource languages.
  - Maths and science: LLM-scored documents, and no synthetic or AI-generated text.
- **Mid-training (p. 8):**
  - Context grows in three steps: 32K (1T tokens) → 128K (500B) → 200K (50B).
  - Repository-level software-engineering data: about 10M issue–PR pairs, about 160B tokens.
  - Long-context data, natural and synthetic (interleaved packing, MRCR-like data).
  - Finding: the 200K stage improved results even inside 128K.
- **DSA adaptation** follows mid-training (§1.3).

### 4.2 SFT (pp. 10–11)
- **Data:** general chat, reasoning, and coding and agent data (expanded a lot compared with GLM-4.5), up to 202,752
  tokens.
- **Three thinking modes:**
  - *Interleaved thinking:* the model thinks before every reply and tool call.
  - *Preserved thinking:* reasoning blocks are kept across turns in coding agents.
  - *Turn-level thinking:* thinking can be switched on or off per turn (Fig. 7).
- **Agent trajectories:** they come from real execution environments. Erroneous segments stay in the trajectory but are
  **masked out of the loss**, so the model learns to recover from errors without learning the errors themselves.

### 4.3 Reasoning RL (pp. 11–12)
- **Algorithm:** GRPO plus IcePop, without the KL term. Tokens whose train/inference probability ratio falls outside
  [1/β, β] are dropped, with β = 2. Clip range [0.2, 0.28]. Fully on-policy, group size 32, batch size 32.
- **Domains:** maths, science, code and tool-integrated reasoning, mixed in roughly equal shares. Problems are filtered
  to those GLM-4.7 rarely solves but stronger teachers can.
- **DSA-specific rules:** a deterministic `torch.topk`, and the indexer frozen during RL (p. 12).

### 4.4 Agentic RL (pp. 12–13, 15–17)
- **Asynchronous training:** fully asynchronous and decoupled. Inference and training run on separate GPUs, weights are
  pushed every K updates, and the optimizer is reset after each push.
- **Objective:** group-wise, reward minus the group mean. Only model-generated tokens count; environment output is
  excluded from the loss.
- **Stabilisers:**
  - **Token-in-Token-out (TITO) gateway:** training uses the exact token IDs the inference engine produced, never
    re-tokenised text.
  - **Direct double-sided importance sampling:** the ratio is taken against the *rollout* log-probabilities, and tokens
    outside [1 − ε_l, 1 + ε_h] are masked, so no history of old policies is needed.
  - **Staleness filter:** a sample is dropped if its oldest policy version lags by more than τ.
  - **Environment crashes:** samples that failed because the environment crashed are removed. A group is padded only if
    more than half of it survives.
  - **DP-aware routing:** consistent hashing sends each rollout to one DP rank, so its prefix KV is reused (p. 17).

### 4.5 General RL (p. 13)
- Three goals: foundational correctness, emotional intelligence, and task-specific quality.
- A hybrid reward: rules, outcome reward models (ORMs) and generative reward models (GRMs).
- Human-written exemplars act as style anchors against "model-like" verbosity.

### 4.6 On-policy cross-stage distillation (pp. 13–14)
- The final stage. Earlier stage checkpoints (SFT, Reasoning RL, General RL) act as teachers. The advantage is replaced
  by `sg[log π_teacher − log π_train]`. Group size 1, batch size 1024.
- It recovers skills lost during sequential RL. Teacher logits are currently fetched from the inference engine.

### 4.7 RL infrastructure: slime (pp. 14–15)
- **Rollouts:** customisable rollouts, served over HTTP.
- **Tail latency:** avoided with no-queue multi-node serving (EP64/DP64, DP-attention), FP8 rollouts and MTP.
- **Prefill vs decode:** PD disaggregation, so long prefills don't stall decodes.
- **Fault tolerance:** heartbeats, and removal of unhealthy servers from the router.

### 4.8 Environments (pp. 17–19)
- **SWE:** RepoLaunch-based automatic environment setup and LLM-written log parsers for Fail-to-Pass and Pass-to-Pass
  tests. Over 10k environments across 9 languages.
- **Terminal:**
  - from seed tasks: draft, then a construction agent, then a refine agent, in Harbor format, with Docker builds
    succeeding over 90 % of the time;
  - from web pages: a self-verifying construction agent.
- **Search:** a web knowledge graph from more than 2M pages. Multi-hop questions come from subgraphs and pass
  three-stage difficulty and correctness filtering.
- **Slides:** HTML slides with a three-level reward (markup, rendered layout, visual features) and fixes against reward
  hacking. 16:9 compliance rose from 40 % to 92 % (pp. 20–21).

### 4.9 Context management (p. 19)
- **Keep-recent-k:** tool outputs older than the last k = 5 rounds are replaced by a placeholder.
- **Hierarchical context management:** also discard all tool history once the context passes T = 32K.
- The paper notes that accuracy "degrades substantially" beyond about 100K tokens of history.
- BrowseComp: 55.3 → 62.0 → 75.9.

---

## 5. Relationship to TensorQuay Engine

### (a) Kernels and contracts

**Sparse-MLA decode (SMLA).** The paper and the code confirm our contract: latent width 512, K ≤ 2051 with −1 as empty,
scale 256^-0.5, int32 indices. Changes and additions:
- **MTP query count.**
  - C-004 assumes 2 query tokens per sequence.
  - With GLM-5-style chained drafting, the verify pass sends **1 + d** tokens per sequence, each with its own index set
    (a pool counts only if its last token is visible to that query; code L871–877).
  - Generalise the contract to d ∈ {0, 1, 2, 3}. Keep the per-GPU capacity bound (E-015).
- **The MTP layer is a twelfth DSA layer.** It has its own paged latent cache and indexer cache (cfg modules for
  layer 45), so the same kernel serves it. Budget +1,024 B per token of latent (computed).
- **Roofline.** Intensity is 64 FLOP/B at H = 32 and 128 at H = 64 (§2.4). If we move DSA layers to DP-attention
  (§5c), H becomes 64 per GPU and the kernel may be compute-bound. **Gate on the attainable roofline**, the minimum of
  the DRAM and tensor-core limits, and add a measured tensor-core peak (§5d).
- **Optional P1 optimisation:** when a sequence's 1 + d query tokens have heavily overlapping index sets, gather each
  shared row once. This is likely with `index_share_for_mtp_iteration`.

**Indexer (Phase 1a in our plan; the paper argues for treating it with the same rigour as SMLA).**
- **Deterministic top-k is a correctness property, not polish.**
  - The paper saw RL collapse with non-deterministic CUDA or TileLang top-k (p. 12).
  - For us, non-determinism breaks DEV-GUIDELINES §1.2.7 and makes the per-token KL tests noisy.
  - Contract: ties are broken by the lower pool index, compared as index sets. TEST-PLAN §9 already warns that the
    reference `topk` order under ties is undefined.
- **Fuse like the Ascend "Lightning Indexer" (p. 21).**
  - Score, ReLU, head-weighted sum and top-512 in one kernel, so the up to 32K pool scores per query at 128K never go
    to DRAM.
  - The contract should state the FP32 scoring (code L863–868) and the 32^-0.5 head-weight scale (code L867).
- **Cache format (inferred, exact by construction).**
  - A pooled key depends only on its 4 member tokens: the per-channel softmax over gate + APE (code L999–1005). It
    can be computed **once**, when the pool is complete.
  - So the engine can store **pooled keys only** (704 B per token across 11 layers) plus raw `k` and gate values for
    the ≤ 3 tail tokens. The HF layout stores 5,632 B per token and re-pools on every step (code L837–862).
  - An **8× smaller indexer cache** matters under our roughly 24 GiB KV budget.
  - REF test: build index sets from the pooled-only cache and from the HF layout, and check they are identical.
- **Speculative rollback.** If a pool completes with draft tokens that are later rejected, its pooled key must be
  recomputed when those positions are rewritten. Only finalise pools over accepted tokens (inferred).
- **Share of cost.** At batch 8, a whole decode step is dominated by MoE expert reads (§5a, MTP). The indexer mainly
  matters for 128K prefill (194 TFLOP, against 194 TFLOP for sparse-MLA) and for per-layer latency. It does not need
  to join Phase 0; make it the first Phase 1a kernel.

**MTP: how many draft steps.**
- **The paper's point of reference:** 4 speculative steps, 2.76 accepted, with the MTP trained as 3 chained steps
  (p. 5). It says MTP is best at small batch (pp. 14–15).
- **Our cost model (computed; inferred assumptions):**
  - Assumptions: routing is uniform and independent; about 1.6 TB/s effective per GPU; non-expert weights about
    7.2 GiB per GPU per step; 3 ms of TP communication per step; 65 % acceptance; the MTP layer's own cost is ignored,
    so gains are optimistic.
  - Verifying 1 + d tokens per sequence makes the step read the union of their experts. At 8 sequences the union per
    layer grows 58 → 105 → 142 experts for d = 0 → 1 → 2 (out of 288).

  | Active sequences | d = 1 | d = 2 | d = 3 |
  |---|---|---|---|
  | 1 | ×1.44 | ×1.60 | ×1.64 |
  | 2 | ×1.32 | ×1.40 | ×1.38 |
  | 4 | ×1.21 | ×1.23 | ×1.18 |
  | 8 | **×1.14** | **×1.14** | ×1.12 |

  - The same model gives about 56 tok/s per user at 8 sequences **without** MTP. M1 (≥ 30 tok/s) is therefore reachable
    on bandwidth alone if the kernels reach the roofline (inferred).
- **Recommendation:**
  - Choose d **per step from the number of active decoding sequences**, using a cost model calibrated by SYS
    benchmarks. Start with d = 3 for 1–2 sequences, d = 1–2 for 3–8, and 0 above.
  - Agent sessions alternate between model generation and tools. Measure the number of actively decoding sessions
    before choosing an adaptive MTP policy.
  - Keep MTP off by default until SYS-P-003 shows a net gain (PRD §6 already says this).
- **The shared MTP layer and the verify path.**
  - Draft steps chain the *same* layer-45 weights d times. The MTP cache is written at positions t + 1 … t + d and
    overwritten when rejected (inferred from DeepSeek-style MTP and p. 5).
  - The **verify pass on a hybrid model needs KDA state rollback.** KDA's recurrent state (about 141 MiB per sequence,
    computed) has to end at the last accepted token. Options:
    1. the recurrent kernel writes a state per position in the verify pass, costing (1 + d) × state memory;
    2. snapshot the state before the pass and re-apply only the accepted tokens.

    Option 2 is cheaper in memory (inferred). Either way it is a **new KDA kernel requirement** that is not in our
    documents yet. The same applies to the conv state (window 4).
- **`index_share_for_mtp_iteration` (inferred; the HF code does not implement MTP).**
  - Most likely meaning: when the MTP layer is chained, its indexer runs on the first draft step only, and later draft
    steps reuse that top-k. This is IndexCache applied across MTP iterations, which saves d − 1 indexer runs per step.
  - Open detail: whether the reused set gains the new tail tokens (t + 2 …). Pin the behaviour to an official serving
    implementation (zai-org, SGLang or vLLM) before writing the contract (§7, Q1).

**MoE (MOE).** Nothing in the paper changes the W4A16 contract. Two points:
- The paper's Ascend path puts experts at 4 bits while keeping attention at 8 bits (p. 21). This supports our "experts
  at 4 bits, the rest FP8/BF16" split.
- GLM-5 was trained with **INT4 QAT** in SFT, with bitwise-identical train and inference quantisation (p. 10). If an
  official INT4 (QAT) GLM-5.3-Flash checkpoint exists, INT4 W4A16 may be more accurate than NVFP4 converted from FP8.
  Keep the in-register dequant a small codec parameter (E2M1 + E4M3 scale, or INT4 + scale) and decide by QUAL (§7, Q9).

### (b) TEST-PLAN: quality benchmarks and MTP targets

**Benchmarks the paper uses for coding and agents, and what to adopt for QUAL.** Every QUAL test compares our engine with
O3 (the official runtime) on the same prompts, harness and sampling. The paper's absolute numbers are GLM-5's, not
GLM-5.3-Flash's.

| Proposed ID | Benchmark | Why | Settings to copy |
|---|---|---|---|
| QUAL-C1 (P0 for v1.0) | **Terminal-Bench 2.0, verified set** (`zai-org/terminal-bench-2-verified`, Harbor/Docker) | Closest to terminal-based coding agents; GLM-5's headline agentic coding score (p. 23) | Terminus-2: 2 h timeout, temp 0.7, top_p 1.0, max_new_tokens 8,192, 128K context, 16 CPUs / 32 GB (p. 37). ≥ 3 runs (the paper averages 5 for Claude Code runs) |
| QUAL-C2 (P0 as a subset, P1 in full) | **SWE-bench Verified**: a fixed, seeded 100-task subset for regression; all 500 at release | The industry reference; 77.8 for GLM-5 (p. 23) | OpenHands, temp 0.7, top_p 0.95, max_new_tokens 16,384, 200K context (p. 36) |
| QUAL-C3 (P2) | SWE-rebench, latest monthly set | Free of contamination (p. 27) | Official harness |
| QUAL-LC1 (P0) | **RULER at 32K / 64K / 128K** | The most sensitive probe of indexer and KDA errors: RULER@128K moved 79.21 → 71.35 with a weak indexer (Table 6, p. 8) | Standard RULER |
| QUAL-LC2 (P1) | RepoQA at 64K / 128K | Long-context *code* retrieval, where linear attention lost up to 7.33 (Table 5, p. 7) | Standard |
| QUAL-T1 (P2) | τ²-Bench or MCP-Atlas public set | Tool-calling and chat-template correctness (p. 23) | p. 37 |
| Keep | GSM8K and per-token KL (M3) | Cheap and fast | Note: GSM8K is weak for this model family (GLM-5-Base 68.8, p. 37) |

**Statistics.**
- Agentic pass rates are noisy. SWE-rebench SEMs are 0.9–2.1 points (Table 9, p. 27).
- One 500-task run at p ≈ 0.75 has a standard error of about 1.9 points (computed). A "within 2 points" gate (M3)
  cannot be decided from one run.
- Use **paired per-task comparisons** (McNemar, or a paired bootstrap), ≥ 3 runs, and a pre-registered number of runs.
  Treat per-token KL and RULER as the primary *engine-correctness* signals, and agent pass rates as confirmation.

**Diagnostic for KL spikes.** Log the top-k index-set overlap (Jaccard) between our engine and O3 per DSA layer.
Index-set flips at near-ties are the expected source of divergence (p. 12).

**MTP tests (new section, proposed).**
- **MTP-C-001, correctness.** Verify-pass logits at each position match non-speculative decode logits within the §2
  rule; the maths is the same.
- **MTP-C-002, greedy text.**
  - With MTP on, greedy output equals MTP off on at least 95 % of 200 prompts × 512 tokens.
  - Every divergence must start at a near-tie: top-2 logit gap below a threshold fixed in advance.
  - Bitwise equality isn't guaranteed, because the verify pass has a different batch shape (inferred).
- **MTP-E-001, rollback.** After j of d drafts are accepted, the KDA recurrent and conv states, the latent KV, and the
  pooled indexer keys equal those of non-speculative decoding of the same tokens (FP32 state within tolerance).
- **MTP-E-002, pool completion.** A draft completes a k-pool, and the draft is then rejected.
- **MTP-Q-001, acceptance (gate for enabling MTP by default).**
  - Measure per-depth acceptance and mean accept length for d = 1…4 on three prompt sets: code, agent traces and
    prose.
  - Gate: within 2 percentage points (per depth) and 3 % (accept length) of O3 on the same prompts and sampling.
  - Reference points, not gates: GLM-5 reached 2.76 at d = 4 (Table 2, p. 5), about 70 % per token (§2.3), under the stated independence assumption.
  - Provisional floors until O3 values exist (inferred): d = 1 acceptance ≥ 60 % on code; d = 3 accept length ≥ 2.2 on
    code.
- **MTP-P-001, net speed.** Measure tok/s per user at 1, 2, 4 and 8 active sequences for d = 0…3. It sets the adaptive
  policy and must show a net gain wherever the policy enables MTP.

**SYS additions.**
- **Prefill interference.** 7 agents decode while an eighth sends a 16K or 64K prompt. Report the decode stall at
  p50/p99. The paper uses PD disaggregation for this problem (p. 15).
- **Prefix-cache hit rate.** Measure it on reproducible agent sessions, with preserved thinking on and off (see §5c).

### (c) Architecture and modularity lessons

1. **Attention variants as a closed enum, with sharing designed in.**
   - Use `LayerKind::{Kda, Dsa { indexer: Full | Shared }}`. DEV-GUIDELINES §1.3 prefers enums for closed sets.
   - GLM-5.3-Flash uses only `Full`, but the reference code already supports `Shared` (IndexCache), and DeepSeek and
     future GLM checkpoints may use it. Adding it costs one optional input (the previous layer's indices).
   - The indexer → attention interface (`i32 [T, K ≤ 2051]`, −1 = empty) already covers all three cases: full, shared
     and MTP-shared.
2. **Per-layer cache kinds, owned by `tq-runtime`.**
   - **DSA:** a paged latent cache (512 × BF16 per token).
   - **Indexer:** a paged pooled-key cache plus a small tail buffer.
   - **KDA:** a fixed-size recurrent state and conv state per sequence.
   - **MTP layer:** its own latent and indexer caches.
   - Admission control (FR-4) must count all four in bytes.
3. **Prefix caching on a hybrid model needs KDA state checkpoints (inferred).**
   - A recurrent state can't be cut back to an earlier position. Reusing a shared prefix means saving a KDA/conv state
     snapshot at reusable boundaries, such as the end of each agent turn or every N tokens.
   - The paper (a pure DSA model) only needs KV reuse, via DP-aware routing (p. 17). Our equivalent is session
     affinity: all of one agent's requests go to the same cache lineage.
4. **Context management belongs to the harness, but the engine decides what it costs.**
   - Keep-recent-k and discard-all (p. 19) rewrite history, so the prefix cache is invalid from the first folded tool
     output onwards.
   - *Preserved thinking* (p. 11) keeps agent histories append-only. That is the best case for prefix caching and for
     M2 (inferred).
   - The engine should support GLM's preserved-thinking and turn-level thinking template options, and report the
     prefix-cache hit rate per session (FR-8).
5. **Speculative decoding as a runtime module, not model code.** It has four parts:
   - a *draft provider* (MTP chain);
   - a *verifier* (a multi-token forward pass of the main model);
   - an *acceptance rule* (greedy, or rejection sampling for temperature > 0);
   - per-layer-kind *commit/rollback hooks* (KDA state, k-pool finalisation, KV positions).

   An EAGLE-style draft model could later plug into the same interface.
6. **Parallelism: consider DP-attention for the 11 DSA layers (ADR candidate).**
   - The paper uses DP-attention because tensor parallelism *copies the MLA latent to every rank* (p. 14). With TP = 2,
     each GPU needs the full latent for its 32 heads (inferred).
   - Cost of that duplication: about 11 KiB per token per GPU. 8 agents × 128K is about 11 GiB per GPU.
   - With DP-attention on the DSA layers only (KDA and MoE stay TP), each GPU holds its own sequences' latent. That
     roughly halves latent memory, at the cost of about 0.8 GiB of duplicated DSA weights per GPU and an
     all-gather/reduce-scatter around those layers (computed and inferred).
   - It also gives H = 64 per GPU, which doubles the sparse-MLA arithmetic intensity.
   - Weigh it against load imbalance. The paper's other serving ideas map to our PCIe box:
     - split or overlapped all-reduce (FlashComm, p. 22);
     - async D2H sampling overlap (p. 22);
     - host-RAM prefix cache (p. 22) for idle agent sessions.
7. **Token-in-token-out API.** The paper found re-tokenisation mismatches harmful (p. 16). A token-ID completion endpoint
   that returns token IDs and log-probabilities makes QUAL/KL testing and future RL use exact, and costs little (P1).

### (d) Recommended changes

**PRD**
- **§6 constraints:** add the per-token and per-sequence cache budget.
  - Latent 11,264 B per token (+1,024 with MTP). Indexer 704 B per token with the pooled layout. KDA about 141 MiB per
    sequence.
  - Note that TP = 2 replicates the latent (inferred): 8 × 128K ≈ 11 GiB per GPU.
  - State that 8.7 GiB of the non-expert weights are BF16 KDA projections, per the official `modules_to_not_convert`.
- **FR-6 (MTP):** "chained single MTP layer, d ∈ {0…3} chosen per step from the active batch by a measured cost model;
  off by default until SYS shows a net gain at 8 agents." Add KDA-state rollback as a requirement.
- **New FR (determinism):** a deterministic indexer top-k with a defined tie-break (p. 12).
- **New FR (agents):** support the preserved-thinking and turn-level thinking chat-template options (p. 11). Report the
  prefix-cache hit rate per session.
- **FR-1:** the checkpoint includes a vision tower (`vision_config`). The text-only loader must skip `visual.*` and
  verify it skipped them.
- **M3:** name the suites. Terminal-Bench 2.0 (verified) + SWE-bench Verified (fixed subset) + RULER 32K/128K, each
  against O3, with paired statistics and ≥ 3 runs. Keep GSM8K and per-token KL as fast gates.
- **§10 risks, add:**
  - KDA rollback complexity for MTP;
  - a small MTP gain at 8 agents (+10–15 % modelled);
  - prefill interference;
  - the lack of an MTP reference in `transformers` (layer 45 is ignored, code L1380).

**DEV-PLAN**
- **Phase 0:** unchanged. Optionally, run a 15-minute PyTorch timing of the indexer (score + top-512 over 32K pools) in
  the gate session, to size Phase 1a.
- **Phase 1a kernel list, add:**
  - a **fused indexer** (score + ReLU + head-weighted sum + deterministic top-k), like the Ascend Lightning Indexer
    (p. 21);
  - a pooled-key cache writer;
  - a fused MLA prologue (q_a → norm → q_b → absorb, kv_a → norm → cache write), in the spirit of MLAPO (p. 22);
  - a **KDA verify kernel with per-position state output or rollback.**
- **Phase 1:** pin an **MTP oracle.** The official GLM-5.3-Flash serving code (zai-org, SGLang or vLLM) at a fixed
  commit, used to define the layer-45 forward pass and `index_share_for_mtp_iteration`.
- **Phase 2 scheduler:**
  - chunked prefill with decode priority;
  - session affinity and prefix caching with KDA checkpoints;
  - async sampling copy;
  - an adaptive MTP policy;
  - optional host-RAM parking of idle sessions' caches.
- **ADRs to write:**
  1. TP vs DP-attention for the DSA layers;
  2. the indexer cache layout (pooled-only) and precision;
  3. the MTP draft policy;
  4. the 4-bit expert codec (NVFP4 vs INT4, if an official QAT checkpoint exists).

**TEST-PLAN**
- **§2 oracles:** O2 does not cover MTP (code L1380). Add REF-006, "MTP layer vs the pinned MTP oracle", once it exists.
- **§4:** generalise C-004 to 1 + d query tokens, d ≤ 3. Add C-010: identical index rows across a sequence's query
  tokens (the MTP-shared case).
- **§4.3 / §6:**
  - add **H-006, measured BF16 tensor-core peak** (FP32 accumulate);
  - report sparse-MLA arithmetic intensity;
  - gate SMLA-P-001 on the **attainable roofline**, the minimum of the DRAM and compute limits.
- **REF-005:** move it to P0 when the indexer starts. Add:
  - pooled-only cache vs HF layout, giving identical index sets;
  - deterministic tie-break;
  - per-query pool visibility for multi-token queries;
  - tail selection for `kv_len mod 4` ∈ {0, 1, 2, 3};
  - reading `index_kpool` from the config, never from the code default of 16.
- **New MTP section** (MTP-C / E / Q / P) as in §5b, with O3-relative acceptance gates.
- **§9 QUAL:** QUAL-C1, C2, LC1 and LC2 with the paper's settings (pp. 36–37), paired statistics, and index-set overlap
  logging.
- **§9 SYS:** a prefill-interference test, prefix-cache hit rate with preserved thinking, and MTP on/off at 1, 2, 4 and
  8 agents.

---

## 6. PM brief

**What GLM-5 is.** A very large open model from Zhipu AI and Tsinghua: 744B parameters, of which 40B work on each word.
It was built to act as a coding *agent*, planning, editing files, running tests and fixing its own mistakes over many
steps. On SWE-bench Verified (fixing real GitHub bugs) it scores 77.8, against Claude Opus 4.5's 80.9. It got good at
agent work by practising on more than 10,000 real software projects and terminal tasks, graded automatically each time.

**How it was trained, like a doctor's career:**
- **Pre-training** is reading the whole library (28.5 trillion tokens).
- **Mid-training** is reading very long case files: whole codebases, up to 200K tokens at once.
- **SFT** (supervised fine-tuning) is studying worked examples written by experts.
- **RL** (reinforcement learning) is residency: try real tasks, get pass or fail from a grader, and do more of what
  worked.
- **Distillation** is a final refresher from its own earlier specialist versions, so it doesn't forget early lessons.

**Why it is fast.**
- **Sparse attention.** Normally each new word re-reads the whole conversation. Sparse attention uses an index, like
  the index of a book, to pick the 2,048 most relevant passages and read only those. Long sessions stop slowing down;
  the paper says long contexts cost half as much.
- **Multi-token prediction (MTP).** A quick junior guesses the next few words and the senior checks them all in one go.
  Correct guesses are kept: about 2.8 words per check for GLM-5.

**What it means for TensorQuay.**
- Our first kernel is the sparse-attention reader, and MTP is on our plan. The paper confirms these are the right bets.
- MTP helps a lot with 1–2 active users but only about 10–15 % with 8 busy agents, so we'll switch it by load.
- We should test quality on the lab's own public coding benchmarks (Terminal-Bench, SWE-bench), so buyers can compare.
- GLM-5.3-Flash uses a cheaper "running summary" (KDA) in 3 of every 4 layers. That saves memory, but can recall long
  documents less precisely, and undoing wrong MTP guesses takes extra engineering.

## 7. Open questions and unverified items

1. **`index_share_for_mtp_iteration`.** Its exact meaning isn't in any source we have. Our reading: the MTP indexer
   runs once per draft chain and its top-k is reused. Does the reused set gain new tail tokens? Does the MTP indexer ever
   reuse the main model's top-k instead? Resolve from the official serving code before writing contracts.
2. **Does the MTP layer have mHC?** `modules_to_not_convert` lists `hc_*` for layers 0–44 only, so probably not
   (inferred). Confirm from the checkpoint's `model.safetensors.index.json`. Also confirm that the MTP layer shares
   `embed_tokens` and `lm_head` with the main model (the paper does for GLM-5, p. 9).
3. **Definition of "accept length" in Table 2** (p. 5): does it include the bonus token, and what is the prompt mix?
   Our 70 % per-token figure depends on it.
4. **Layer count:** "80" in the text (p. 4), but Table 10 gives 3 + 75 (+1 MTP) (p. 36).
5. **MLA-256 head dimension:** Table 10 lists QK head dim 192, while the text describes raising head dim from 192 to 256
   (p. 5). Possibly 192 + 64 RoPE = 256. Not stated; it doesn't affect GLM-5.3-Flash, which has 256 and no RoPE.
6. **Small inconsistencies in the paper:**
   - the SWA pattern string on p. 6 has 41 characters for a 40-layer model (as extracted);
   - "98.0 % BSR" (p. 26) against Table 8's values, which average 98.75;
   - the Terminal-Bench 2.0 Claude Code row repeats 56.2, the same as Terminus-2 (p. 23).
7. **IndexCache numbers** come from the arXiv abstract and a skim of the HTML v1, not a page-by-page read.
8. **Latent replication under TP = 2** is inferred from MLA's structure (the latent is shared by all heads). Confirm it
   in the parallelism ADR.
9. **INT4 QAT:** is there an official INT4 (QAT) GLM-5.3-Flash checkpoint? The paper describes INT4 QAT for GLM-5 only
   (p. 10). If there is, compare it with NVFP4 group 32 in REF-004 and QUAL.
10. **Unused config keys:** `index_kpool_compress` and `indexer_rope_interleave` are in the config, but the HF config
    class doesn't define the first, and the second has no effect with RoPE dimension 0. Behaviour if either ever
    changes is unknown.
11. **GLM-5.3-Flash training details** (tokens, context length, how the MTP was trained, whether it went through DSA
    adaptation) are not in this paper. `max_position_embeddings` is 1,048,576, but PRD FR-7 only promises 128K.
12. **Quantising the KDA projections:** the official checkpoint keeps them in BF16 (8.7 GiB). Could FP8 work? It would
    save about 4.3 GiB in total, but it is untested and a quality risk. Measure it only if memory forces the question.
13. **MTP gain model** (§5a) assumes uniform routing, 1.6 TB/s effective bandwidth, 3 ms of TP communication and 65 %
    acceptance. Replace it with MOE-P-003 (real routing) and SYS measurements.
14. **Decode-time share of the indexer and KDA at 8 agents** is modelled, not measured. Nsight profiles in Phase 1 should
    confirm that MoE experts dominate the step.
