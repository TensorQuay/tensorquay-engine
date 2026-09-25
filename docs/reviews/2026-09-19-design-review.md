# Review of the TensorQuay Engine design docs (PRD, DEV-PLAN, TEST-PLAN, DEV-GUIDELINES, README)

Public technical extract: internal operational notes are omitted; locations refer to the reviewed revisions.

Independent design review, 19 Sep 2026. Most severe first. File:line references are to the docs in `tensorquay-engine/docs`, the reference `modeling_glm5_next.py` (transformers commit 770e4c40d0), and cutile-rs at `e04245b`. I made no edits to the reviewed documents and posted nothing online.

Findings: 1 critical, 8 high, 22 medium, 10 low (41 total).

## Critical

**1. [critical] TEST-PLAN §3 (REF-001, REF-002, REF-005), DEV-PLAN 0.2: the reference tests cannot pass at `rel_l2 ≤ 1e-10`.**
- **Problem:** the plan treats the `transformers` code (O2) as a float64 oracle, but it is not one. It casts to float32 in fixed places, even when the model runs in float64:
  - RMSNorm, including `q_a_layernorm` and `kv_a_layernorm`: `hidden_states.to(torch.float32)` (lines 77–81).
  - Eager attention: `softmax(..., dtype=torch.float32)` (line 1094). PyTorch casts the input to that dtype first.
  - Router: `F.linear(hidden_states.type(torch.float32), ...)` (line 161).
  - Indexer: `q.float()`, `.float()` (lines 863, 867).
  - mHC `.float()` (302–305) and KDA `.to(torch.float32)` (478, 534).
- **Result:** O1 and O2 will differ by about 1e-7, not 1e-10. The router difference can also flip top-8 choices on near-ties. This blocks step 0.2 and so the whole gate.
- **Fix:**
  - REF-001: compare only the attention core. Feed the same post-norm `q_resid` and `k_pass` into both sides, and run O2 with `attn_implementation="sdpa"`, which stays float64 on CPU. Tolerance 1e-12.
  - REF-002: O1 must copy the float32 router on purpose (`moe_router_dtype: float32` is part of the model). Or feed O2's `topk_indices`/`topk_weights` into O1 and compare only the expert path at 1e-12.
  - REF-005: copy the float32 score casts, and build inputs whose score gaps are larger than 1e-5 relative.
  - Add to §2: "O1 reproduces every dtype cast that the reference code makes on purpose. Each REF test states its own tolerance."

## High

**2. [high] DEV-PLAN 0.3: "test vectors < 50 MB total" does not fit the P0 tests.**
- **Evidence:**
  - MOE-E-003 uses all 288 experts at TP=2 shapes: 288 × 12.6M params × 0.53 B ≈ 1.93 GB.
  - MOE-C-001 at T=1 already needs about 53 MB of weights.
  - SMLA-E-007 needs a KV pool over 8 GB, and C-003 goes up to kv_len 131072.
- **Fix:**
  - Generate inputs on the pod from seeds, using a counter-based RNG implemented the same way in Rust and Python.
  - Run O1 on the pod (PyTorch float64 on CPU or GPU).
  - Store only hashes. Keep committed files for the small cases only.

**3. [high] TEST-PLAN §2 with §4.2: the aggregate tolerance cannot see index bugs.**
- **Problem:** with random data and about 2051 keys, dropping or duplicating one index changes that output row by about 1/2051.
  - The overall `rel_l2` moves by roughly 5e-4 × √(share of rows affected).
  - The baseline error is about 2–3e-3, so the allowed limit is about 5e-3. A broken tail, broken −1 handling or an off-by-one at a page edge in E-003, E-004 or E-008 would still pass.
- **Fix:**
  - Add exact index-set probes: with q = 0 every weight is 1/n. Encode each position into its KV row with exactly representable values, so `out` gives back exactly which positions were used and how often.
  - Compute the §2 metrics per (token, head) row as well as over the whole output.

**4. [high] PRD §2, §4, §10: the problem statement is going stale and M1 is below what is already public.**
- **What upstream has done:**
  - vLLM merged GLM-5.3-Flash support (#53906, 3 Sep).
  - At least four vLLM PRs for rope-free sparse MLA on sm_120 are open: #55277, #53969, #55778 and #54929 (a Triton fallback).
  - FlashInfer merged its sm_120 sparse-MLA refactor (#4802) and a rope-free fix (#4947).
- **What already runs on the same 2 cards:**
  - A vLLM recipe (github.com/1austinanderson/aa-glm53-flash-rtx-pro-6000) gets 107 / 163 / 282 tok/s total at 1 / 2 / 4 streams without speculative decoding, and about 530–560 total at 8 streams with DFlash-2.
  - The SGLang fork's 106 tok/s per request is correct (425.1 ÷ 4). But it is at 4 concurrent requests only, with 3-step MTP, fixed 4096-token `ignore_eos` outputs and about 18K context. It is not evidence for 8 users.
- **Problem:** M1 (≥ 30 tok/s per user at 8 users) is less than half of what is shown publicly.
- **Fix:**
  - Add a risk row: "Upstream vLLM/SGLang/FlashInfer ship sm_120 rope-free sparse MLA before our v1.0."
  - State what makes the engine worth building apart from the missing kernel.
  - Set M1 and M2 relative to the best public setup, measured in a reproducible agent harness on the same box.

**5. [high] PRD §5–§6, FR-4, FR-7: KV and state memory are not budgeted, and the targets do not fit.**
- **Evidence:**
  - The latent KV cache is copied to both GPUs under TP=2 (`kv_a_proj_with_mqa: "mla_kv_a_proj"`, `kv_b_proj: "colwise"`).
    - Latent: 1 KiB per token per DSA layer; with 11 DSA layers plus MTP that is about 12 KiB per token per GPU.
    - The indexer cache adds up to about 6 KiB per token per GPU (I estimated about 18 KiB per token per GPU in total).
  - What is left for KV: the ~24 GiB of headroom, minus context, activations, MTP experts and KDA state, is about 7 GiB per GPU. That is roughly 0.4–0.6M tokens, or about 50–75K per agent at 8 agents.
  - FR-7 × 8 agents (8 × 128K ≈ 1M tokens) does not fit.
- **KDA state:** 64 × 128 × 128 × 4 B per layer × 34 layers ≈ 141 MiB per sequence. The SGLang fork uses 4–5 state slots per live request, so FR-4's 64 sequences alone is 8.8 GiB, or several times that with MTP and prefix caching.
- **What the public stacks do:** both use FP8 KV. The SGLang fork has a 524,288-token pool with 528-byte rows; the vLLM recipe has `fp8_ds_mla` with a 512K pool.
- **Fix:**
  - Add a budget table (tokens, state slots, per-agent context).
  - Define the FP8 latent KV format in the SMLA contract now.
  - Change FR-7 to "128K per request, subject to the pool," and bound FR-4 by memory.

**6. [high] PRD FR-5/FR-6, TEST-PLAN §4 (T per sequence), §9: KDA recurrent state is ignored.**
- **Problem:** 34 of the 45 layers are recurrent.
  - MTP needs per-draft-step state checkpoints or rollback.
  - Prefix caching needs a state snapshot (about 141 MiB each) at every cache point.
- **No oracle for MTP:** `transformers` ignores the MTP layer (`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", ...]`, line 1380).
- **Contract gap:** the SMLA contract fixes "MTP: 2" tokens per sequence, but the public stacks gain from 3 draft steps (about 3 tokens per forward pass).
- **Fix:**
  - Write a decision record for state checkpointing.
  - Pick an MTP oracle (the vLLM #53906 implementation or Z.ai's code).
  - Change the contract to T per sequence ≤ 1 + n_draft (up to 4).
  - Add tests: state rollback after rejected drafts, and a prefix-snapshot restore.

**7. [high] DEV-PLAN 0.4, DEV-GUIDELINES §6: no GPU-free kernel compile in CI, so compile errors are found during hardware sessions.**
- **Evidence:** cutile-rs can compile kernels without a GPU. `KernelCompiler::new(...).target("sm_120").compile()` (cutile-book/guide/jit-compilation.md:343), and the debugging guide says "Requires tileiras … but no GPU." The plan only runs `cargo check`, which never invokes `tileiras`.
- **Fix:** add gate G-11 on the Linux runner: compile every specialisation in the manifest to an sm_120 cubin. Archive registers, spills and shared memory per kernel (from `cuobjdump`/`nvdisasm`).

**8. [high] PRD §6, DEV-PLAN 0.5: the W4A16 path is plausible but has not been tested anywhere.**
- **What cuTile offers:**
  - `mmaf_scaled` needs both operands of the same type (`lhs: Tile<EIn>, rhs: Tile<EIn>`, `_core.rs:2539`), so it cannot take BF16 activations. It is only for FP4×FP4 with E4M3 scales per 16 (or MXFP8 with E8M0 scales per 32).
  - So W4A16 needs `unpack` → `ftof(f4E2M1FN → bf16)` → multiply by the scale → `mma`.
  - The Tile IR `ftof` operation lists fp4e2m1fn among its supported types (NVIDIA Tile IR docs).
  - cuTile's `convert_tile` emits FToF for any pair of float types. Its own comment only names "f16, bf16, f32, f64, tf32, f8e4m3fn, f8e5m2" (`compile_intrinsic.rs:2149`).
  - No example or test in the repo converts fp4 to bf16.
- **Problem:** the planned pod smoke test (the `nvfp4` example) only exercises `mmaf_scaled`, i.e. W4A4. It says nothing about W4A16.
- **Fix:** add a W4A16 dequant microkernel as test H-006. It should compile in CI, run on the 5090, and report dequant throughput in weights per cycle per SM. Name the fallback: decode by bit operations (`andi`/`shri`/`bitcast`).

**9. [high] TEST-PLAN §4, DEV-PLAN risks: the SMLA design ignores sm_120's 99 KB shared-memory limit per block.**
- **Evidence:** SGLang #39302 measured `MaxSharedMemoryPerBlockOptin = 101376` B. The TileLang kernel needed 206 KB and failed.
- **Arithmetic:** for H=32 and D=512, q is 32 KB and one 64-row KV tile is 64 KB, so 96 KB total. That leaves no room for double buffering.
- **Fix:** add perf test P-008, a sweep over tile size (BN ∈ {16, 32, 64}) and over splitting D for P·V, with shared-memory use taken from the cubin. Record this as the main performance risk next to gathers.

## Medium

**10. [medium] TEST-PLAN §4 rules: MTP causality is not in the contract.**
- **Problem:** "valid if 0 ≤ i < kv_len[seq]" is per sequence. When both MTP tokens are already in the cache, token 0 could attend to token 1. The reference relies on the indexer for causality (`get_visible_tokens`).
- **Fix:** add `q_pos: i32 [T]`, with "valid iff 0 ≤ i ≤ q_pos[t] and i < kv_len[seq]". Add test C-010, where token 0's indices include token 1's position and must be ignored.

**11. [medium] TEST-PLAN H-003 cannot pass as written.**
- **Problem:** a buffer larger than L2 is never warm, so cold and warm reads look the same.
- **Fix, replacement text:** "Buffer = ½ L2 (48 MB on the 5090, 64 MB on the PRO 6000). The warm read must be at least 1.5× faster than the cold read. The flush writes 2× L2."

**12. [medium] TEST-PLAN H-002 and §8.3: the roofline is measured with a device-to-device copy.**
- **Problem:** a copy is half reads, half writes. Both SMLA and MoE decode are almost entirely reads, and read-only streaming on GDDR7 usually beats copy bandwidth, so using copy as the peak inflates the percentage.
- **Fix:** use the maximum of a read-only streaming kernel written in cuTile and the copy, and report both.

**13. [medium] DEV-PLAN §3 and SMLA-P-005: the stop threshold does not match the gate.**
- **Problem:** the plan continues if gathers reach 60% of copy bandwidth, but the SMLA gate is 70% of peak. If gathers land at 65%, the plan continues into a gate that cannot pass.
- **Fix:** stop below 80% of the H-002 peak, using the same denominator as the gate.
- **Also:** benchmark both `load_ptr_tko` and the CUDA 13.3 `make_gather_scatter_view` / `load_gather_scatter_view_tko`. The second only supports `padding::None` (`_tileir.rs:72`), so −1 indices would have to be remapped to a zero row.

**14. [medium] TEST-PLAN H-004 cannot be built in cuTile.**
- **Problem:** there is no clock or globaltimer operation in `_core.rs` or `_tileir.rs`.
- **Fix:** time a copy of known size over at least 100 ms against the host clock, or cross-check with Nsight Systems.

**15. [medium] TEST-PLAN §8.6 and DEV-GUIDELINES tier 2: the 5% regression check will be flaky.**
- **Problem:** runs happen on different rented cards with unlocked clocks.
- **Fix:**
  - Use `cutile::bench::do_bench_paired` (`bench.rs:216`), which alternates the two versions on the same pod.
  - Fail only if the slowdown exceeds the larger of 5% and 3× the interquartile range.
  - Reuse `cutile::bench` (CUDA events, L2 clear, quantiles) rather than writing `tq-bench` from scratch.

**16. [medium] TEST-PLAN §4.2: split-K merge edge cases are missing.**
- **Problem:** no test forces some splits to have zero valid indices. Merging with lse = −inf in two splits gives −inf − (−inf) = NaN.
- **Fix:** add E-017, with the valid indices all in the first split and then all in the last. Also state that P-001 timing includes the merge kernel.

**17. [medium] TEST-PLAN §4/§5, PRD M1: CUDA graphs and JIT recompiles are not covered.**
- **Problem:**
  - cuTile graphs fix tensor shapes at capture time ("Dynamic shapes per iteration: No," 10-cuda-graphs.md:378). Decode launch settings therefore cannot depend on kv_len or routing.
  - The JIT cache key includes shape and stride divisibility, so new shapes can trigger a compile in the middle of serving.
- **Fix:**
  - Add a contract clause: "The grid does not depend on data."
  - Add E-018 and MOE-E-011: capture a graph once, then replay it with new kv_len, idx and routing data.
  - Add SYS-R-005: `jit_backend_compile_count()` does not change after warm-up.

**18. [medium] MOE-E-009, SMLA-E-016, TEST-PLAN §1a: device-data checks "before launch" contradict the no-sync rule.**
- **Problem:** checking expert-ID ranges or duplicates in data that lives on the GPU needs a GPU-to-host sync. That breaks DEV-GUIDELINES §1.2.4 and §7, and it is not "pure CPU logic" as §1a claims.
- **Fix:** split the checks. Shape, stride and alignment checks stay on the host. Range and duplicate checks become debug-only checks on the GPU that set an error flag.

**19. [medium] MOE-E-006: E2M1 has no NaN encoding.**
- **Problem:** "weights filled with NaN" cannot be built with 4-bit weights.
- **Fix:** put NaN in the E4M3 scales (0x7F/0xFF) and in the f32 global scale instead.

**20. [medium] MOE-P-002: the proposed baseline is not like-for-like.**
- **Problem:** FlashInfer's CUTLASS NVFP4 MoE is W4A4, and it ignores `swiglu_limit` (SGLang #39939). On sm_120, vLLM picks Marlin NvFp4 (W4A16, group 16) automatically (vLLM #53963), and the SGLang fork already runs W4A16 at group 32.
- **Fix:** use those two as baselines. For SMLA-P-004, add the community sm_120 sparse-MLA kernels as a baseline instead of relying on PyTorch gather.

**21. [medium] TEST-PLAN §10, DEV-PLAN 0.7: the gate card may not be the product card.**
- **Problem:** the 300 W Max-Q and 600 W workstation variants have different power limits. W4A16 dequant is ALU work, so record clocks and power limits in every comparison.
- **Fix:** run the gate on Max-Q if one can be rented. Otherwise add a margin, and log `nvidia-smi -q -d POWER` at the start of each session.

**22. [medium] SMLA-P-007 (P0) and the gate's "cause from Nsight Compute evidence": profiling may be blocked.**
- **Problem:** GPU performance counters are often blocked in containers (`ERR_NVGPUCTRPERM`). Check this before every hardware session.
- **Fix:** check `ncu` in the first 10 minutes of session 1, and have a fallback (Nsight Systems or `do_bench` roofline data only).

**23. [medium] TEST-PLAN §7, "compute-sanitizer clean": cutile-rs itself can trip this.**
- **Evidence:** open cutile-rs #252 (sm_120) reports that `compute-sanitizer` flags a use-after-free in cutile's own host API (`to_host_vec().sync_on`).
- **Fix:** add a triage and suppression policy for faults in upstream code.

**24. [medium] TEST-PLAN §2 baseline and SMLA-C-006.**
- **Problem 1:** where the baseline rounds is not pinned. The reference rounds P to BF16 (line 1094); a "2× baseline" rule means nothing if that point is left open.
- **Fix, replacement text:** "Baseline = scores accumulated in FP32 → softmax in FP32 → P rounded to BF16 → P·V accumulated in FP32 → output rounded to BF16."
- **Problem 2:** C-006's absolute 1e-3 on lse is impossible for E-010's logits of ±1e5, where the FP32 step size is about 0.0078.
- **Fix:** use a relative tolerance for lse.

**25. [medium] PRD §5–§6: two claims are wrong.**
- **"Other weights in the official FP8 block format (128 × 128)" is wrong.** The official `modules_to_not_convert` list keeps these in BF16, about 10 GiB in total:
  - KDA q/k/v/o/f/g/b projections;
  - `kv_b_proj`;
  - indexer `wq_b`, `wk` and `weights_proj`;
  - routers, embeddings, `lm_head` and mHC.
- **Missing kernels:** the Phase 1a list has no BF16 GEMM/GEMV and no per-head batched GEMMs for the W_UK/W_UV absorption.
- **"Group-32 is a requirement" is contradicted.** The vLLM recipe fits group-16 experts (163.27 GiB including the MTP experts, which matches my 163.27), MXFP8 attention and a 512K-token FP8 KV pool, in 174.75 GiB total.
- **Fix:** make group-32 an option chosen by measured KL/QUAL. Quantizing the BF16 KDA weights to FP8 saves about 4.4 GiB.

**26. [medium] DEV-PLAN 1c: quantizing from the FP8 weights means quantizing twice.**
- **Evidence:** `zai-org/GLM-5.3-Flash-BF16` exists (MIT, confirmed via the HF API).
- **Fix:** quantize from the BF16 release. Consider ModelOpt's `W4A16_NVFP4` format with `group_size: 32`, already used by `ormandj/…-K32…`, for interoperability and a cross-check.

**27. [medium] PRD M3 and TEST-PLAN §9 (QUAL): the quality thresholds are inside the noise.**
- **Problem:** GSM8K has 1319 items, so the standard error is about 0.8 points at 90% accuracy. "Within 1.0 point" can fail by chance.
- **Fix:** use a paired comparison with a confidence interval (McNemar test or bootstrap) against a non-inferiority margin. Set the KL ≤ 0.02 limit from a measured noise floor, e.g. official FP8 vs BF16.

**28. [medium] TEST-PLAN §9 API: "prefix-cache hits give the same output as a cache miss" cannot hold exactly.**
- **Problem:** a cache hit changes how prefill is chunked, which changes the KDA chunked-vs-recurrent numerics and the GEMM shapes. DEV-GUIDELINES only promises identical output for the same batch composition.
- **Fix:** define it as greedy-token agreement over N tokens plus a KL bound, or write a decision record for batch-invariant kernels.

**29. [medium] DEV-GUIDELINES §4.2 and G-06: the coverage gate will fail on CI.**
- **Problem:**
  - Host code that drives the GPU (`tq-bench`, launch paths) cannot run on GPU-less CI, so 90% per crate is out of reach there.
  - `--fail-under-*` applies to the combined report, not per crate. (The flags themselves exist, and `--branch` does need nightly; both confirmed in the cargo-llvm-cov README.)
  - The ratchet needs a script that is not listed.
- **Fix:** merge coverage data from the tier-2 GPU runs (`--no-report`, then `report`), and do per-crate thresholds by post-processing `--json`.

**30. [medium] DEV-PLAN 0.4, TEST-PLAN §1a, DEV-GUIDELINES §1.2.2: building on the Mac (partly verified).**
- **Problem:** `cuda-bindings/build.rs` needs a CUDA toolkit: it runs bindgen on `cuda.h` and `curand.h`, and `toolkit_target_dirs` returns nothing for non-Linux. There is no CUDA 13.3 for macOS.
- **Fix:** put the validation types (`SmlaParams::new`, etc.) in a crate or feature with no cutile dependency.

**31. [medium] PRD NFR-5 vs cuTile JIT: "a single binary" is not accurate.**
- **Problem:** kernels are compiled at runtime by `tileiras` from the CUDA 13.3 toolkit. The cache key includes the `tileiras --version` output, and the disk cache is off by default.
- **Fix:** state that the toolkit is a runtime dependency (or at least `tileiras`), pre-warm the cache at install, and warm up before accepting traffic.
- **Unverified:** whether `tileiras` may be redistributed under the CUDA EULA.

## Low

**32. [low] Cross-references between documents.**
- PRD M5 says "batch ≥ 8", but the MOE-P-001 gate is "T ≤ 64", which includes T = 1–4.
- PRD §9 leaves MOE-P-002 out of the gate.
- No QUAL-* IDs exist.
- DEV-PLAN 0.4 uses unprefixed IDs ("E-011/E-012") and leaves out SMLA-E-016, which TEST-PLAN §1a puts on the Mac.
- §10 says "every P0 in §3–§7 passes on an RTX PRO 6000", which includes the CPU-only REF tests.
- Fix: always use prefixed IDs, which the manifest (G-08) needs anyway.

**33. [low] TEST-PLAN §4.**
- E-015 (T = 512, 256 sequences) contradicts FR-4's limit of 64 sequences.
- P-001 does not say whether T means T separate sequences.
- There is no prefill SMLA performance case (a 4096-token chunk of one sequence), which matters for time to first token and so for M2.

**34. [low] DEV-GUIDELINES §2.1–2.2: lint rules.**
- `clippy::cognitive_complexity` is in the `restriction` group. Clippy's own docs say it is "left in restriction so as to not mislead users into using this lint as a measurement tool" (rust-clippy source). Use `excessive_nesting` instead.
- Lint groups in `[workspace.lints]` need `priority = -1`, as cutile-rs does.
- The `cast_*` lints are already part of `pedantic`, so listing them again is redundant.
- The `too_many_arguments` exception does nothing: `#[cutile::module]` already injects `#![allow(clippy::all)]` (`cutile-macro/src/_module.rs:378,392`).
- But pedantic and restriction lints still fire on kernel code (`too_many_lines`, `undocumented_unsafe_blocks`, `multiple_unsafe_ops_per_block` on `load_ptr_tko`). Write a lint policy for each `*_kernel.rs` module, and prefer `#[expect(lint, reason = "…")]`.

**35. [low] DEV-GUIDELINES §2.2: `forbid(unsafe_code)` in `tq-model` rules out loading safetensors via mmap.** `memmap2::Mmap::map` is `unsafe`. Choose streamed reads or an exception, and record it in a decision record.

**36. [low] DEV-GUIDELINES §2.4: the licence allow-list is narrower than cutile-rs's own `deny.toml`.** That file also allows "Apache-2.0 WITH LLVM-exception", Unicode-DFS-2016, BSL-1.0, OpenSSL and CC0-1.0. `cargo deny` may fail on transitive dependencies (unverified which ones).

**37. [low] DEV-GUIDELINES §4.1: "exactly the same coverage report every time" is overstated.** Randomised `HashMap` ordering and thread interleaving change which paths run, so the 0.25-point ratchet can flap.

**38. [low] TEST-PLAN §9.**
- mHC "rows sum to 1 within 1e-5" is not guaranteed. The reference does one column normalisation plus 19 row/column pairs with eps in the denominators, and ends on a column step (lines 323–326). Compare against O2 instead.
- On real data, ReLU plus weighted index scores can produce exact ties at zero, so `torch.topk` tie order will legitimately differ in C-009 and the golden tests. Define our own tie rule.

**39. [low] TEST-PLAN §5 precision.**
- Specify "dequant = e2m1 × e4m3 (exact in BF16); the f32 global scale is applied to the FP32 accumulator."
- MOE-C-007 can only be "exact" with a one-hot x and a power-of-two global scale, run through a test-only entry that shares the dequant helper.
- REF-004 must name the quantizer (scale = amax/6 rounded to E4M3, with saturation).
- A W13 global scale shared across gate and up must be enforced by our quantizer.

**40. [low] DEV-PLAN build and reference prerequisites.**
- Build a pinned CUDA/Rust/PyTorch image before hardware measurement and record its storage requirements.
- Select a reference runner with enough memory for the official checkpoint before claiming model equivalence.

**41. [low] PRD NFR-1.** A team of 8 needs LAN access. When the server binds to anything other than loopback, the API key should be mandatory. Also add request and token limits, and no prompt logging by default.

## Things that are correct and should stay
- **Layer structure:** 45 layers = 34 KDA + 11 DSA (layers 3, 7, …, 43), 42 MoE layers plus 3 dense, 1 MTP layer.
- **Attention facts:**
  - The softmax scale is 256^-0.5, from `qk_head_dim = 256 + 0` (line 1149).
  - `kv_a_layernorm` is applied before caching (lines 1193–1198).
  - Absorbed MLA is exactly equivalent to the expanded form: `kv_b_proj` has no bias, and each head's 512 rows are the first 256 as K and the next 256 as V.
  - K ≤ 2051: the output width is `index_topk + kpool − 1` and the budget is 512 pools (lines 885, 904–907).
  - −1 and indices ≥ kv_len are invalid, matching the reference (line 1253).
  - Indices are unique for this checkpoint: all indexer layers are "full", pools are disjoint, and the tail does not overlap them. Note that the reference silently removes duplicates (`.ne(0)`), so keep the debug check.
- **MoE facts:**
  - Router: sigmoid, bias used only for selection, weights from the unbiased scores, `+1e-20` normalisation, then × 2.5, all in fp32.
  - SwiGLU clamp: `silu(min(g, 10)) · clamp(u, ±10)`, gate first.
  - The shared expert is outside the kernel.
  - MOE-C-004's TP sum test is exactly valid.
- **Memory arithmetic:** 304.4B expert parameters → 159.47 GiB at group 16 and 150.61 GiB at group 32; other weights ≈ 14.5 GiB; about 90 all-reduces per decode step.
- **cuTile facts:** fp4 packing and block-scaled MMA need CUDA 13.3; group-32 E4M3 is not a native tensor-core format; there is no CPU emulator.
- **Cited evidence:** vLLM #53963 and SGLang #37813/#39302 exist and say what the PRD claims.
- **Tooling claims:** the cargo-llvm-cov flag names, `--branch` needing nightly, `#[allow(..., reason)]` on stable, nextest running each test in its own process.
- **Test design worth keeping:**
  - running the gather benchmark first;
  - the scale guard, layout guard and NaN-containment tests;
  - the 64-bit addressing test (cuTile index maths is i32; see cutile-rs #213);
  - the determinism requirement;
  - the split between device code and host code;
  - "contract coverage" for kernels.

## Sources
- [vLLM #53963](https://github.com/vllm-project/vllm/issues/53963), [SGLang #39302](https://github.com/sgl-project/sglang/issues/39302), [SGLang #37813](https://github.com/sgl-project/sglang/issues/37813)
- [ormandj SGLang fork](https://github.com/ormandj/sglang-glm53-flash-sm120), [vLLM 2 × PRO 6000 recipe](https://github.com/1austinanderson/aa-glm53-flash-rtx-pro-6000)
- [NVIDIA Tile IR operations](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html), [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov)
- [RTX PRO 6000 Max-Q](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/)
