# Response to the adversarial review (`2026-09-19-adversarial-review.md`)

Public technical extract: internal operational notes are omitted; locations refer to the reviewed revisions.

Engineering response, 19 Sep 2026. The review report is preserved unchanged. Each finding was checked independently
against pinned primary sources before any document changed. Documents are now **v0.3 drafts**. The technical lead's follow-up (`2026-09-19-v0.3-follow-up.md`) decided the
proposals (§3) and raised R1–R6 (§3a). **No acceptance threshold was lowered.** P4 was not approved, so the existing M3
stays in force.

## 1. Summary
- **Accepted: 15 of 15.** None is rejected. Every finding was reproduced from a primary source or checked as arithmetic.
- **Nuances:**
  - F2: H-002 already defined a read-only peak; the bytes convention was the missing part.
  - F9: extended, because NVIDIA's Tile IR documentation also exempts MMA from bit-identity guarantees when programs
    change, which narrows D0 as well.
- **Still open:** hardware validation. Nothing has run on a GPU; closing findings in the documents proves no contract on
  hardware.

## 2. Findings
| # | Status | Independent evidence | Changed |
|---|---|---|---|
| 1 | Accepted | ModelOpt `nvfp4_tensor.py` @ `b311c054`: global scale `amax / (E2M1_MAX × E4M3_MAX)` = `amax/(6·448)`; block scale `block_amax / (6 × g)`, zero blocks → 1.0, clamp [2⁻⁹, 448] | TEST-PLAN §2 (NVFP4 codec, formats), REF-003/004, MOE-C-007; DEV-PLAN 0.2 |
| 2 | Accepted | NVIDIA CUDA Best Practices: 32-byte sectors, so a cold 68 B row is ≤ 68/96 ≈ 71 % useful. The SMLA-P-005 stop rule covered every row width | TEST-PLAN SMLA-P-005 (fixed distribution, useful vs transferred bytes, matched CUDA gather), H-002 (bytes convention), §10; DEV-PLAN 0.5/0.6 |
| 3 | Accepted | FlashInfer `prepare.py` @ `dc04f50c` L1456–1513: cuTile NVFP4 preparation requires E4M3 scales with a 16-element K group. The MXFP4 path is group-32/E8M0. There is no group-32 E4M3 preparation | TEST-PLAN §2 formats, §5 contract, MOE-P-001/002, new MOE-P-005, §10; PRD §6 and M5; ARCHITECTURE `WeightFormat`; DEV-PLAN 0.5/0.8 |
| 4 | Accepted | `transformers` 770e4c40d0 L1094: the eager softmax is cast to FP32. The O1 rule and REF-001's 1e-12 conflicted | TEST-PLAN §2 (O1-alg vs O1-prod; each test names its oracle), REF-001/002/005/006 |
| 5 | Accepted | Protocol text: one flush before N graph launches leaves N−1 warm; "3 × IQR" had no unit | TEST-PLAN §8 items 3–6 (cold per invocation, working set, steady state separate, dimensionless paired ratio); DEV-GUIDELINES tier 2 |
| 6 | Accepted | vLLM `glm5next/nvidia/attention.py` @ `98ed0856`: `Glm5NextTailCache` holds "the trailing incomplete pool's raw K + gate score" | ARCHITECTURE §4 `SlotPool`, §5 indexer cache (pooled keys plus tail), ADR-0012; TEST-PLAN IDX, new IDX-R, CACHE-R |
| 7 | Accepted | The config sets `qk_rope_head_dim = 0`. The same vLLM file skips rope when `rope_dim = 0`; the `transformers` indexer forward applies no rotation | DEV-PLAN 1a; TEST-PLAN REF-005, IDX (plus a regression case); ARCHITECTURE §5 |
| 8 | Accepted | vLLM `mtp.py` @ `98ed0856`: its own `topk_indices_buffer`; "step 0 computes top-k, steps 1+ reuse"; `compact_topk_indices` | ARCHITECTURE §4 `IndexSrc`; TEST-PLAN SMLA-C-013 (capability only), MTP block, MTP-C, REF-007; DEV-PLAN 1b |
| 9 | Accepted, extended | vLLM speculative-decoding docs (distribution vs greedy guarantees). Tile IR 13.3 stability: "MMA … exempt from the bit-identity guarantees" | DEV-GUIDELINES §1.3 (D0 scoped to the same binaries; D1 later narrowed to fixed partitions, R6; three speculative properties); TEST-PLAN SAMPLE, API-006; ARCHITECTURE §4 |
| 10 | Accepted | SMLA-P-009 (a 4096-token chunk) contradicted the ≤ 4 queries per sequence of the decode contract | TEST-PLAN §4 (decode only), SMLA-P-009 → SMLA-PF-P-001, new SMLA-PF contract (§9); DEV-PLAN 1a |
| 11 | Accepted | 43 × 288 × 3 × 4096 × 2048 × 2 B = 580.5 GiB (recomputed) | TEST-PLAN §9 (KL definition; engine equivalence vs quantization loss; BF16 plan); PRD M3 note (P4 not approved), risks, §11; DEV-PLAN 1b |
| 12 | Accepted | NCCL user guide (CUDA graphs): multi-GPU single-process deadlock warning; ranks must capture and replay consistently | ARCHITECTURE §6 (lifecycle, destruction order, abort); TEST-PLAN new SYS-R-011; DEV-PLAN 1a; PRD §9 Phase 1 |
| 13 | Accepted | Chat template @ `eb9eb208` L2–4: effort ∈ {low, high}, else `max`; `clear_thinking` defaults to false; no thinking-off switch | PRD FR-3, M2; TEST-PLAN API-004; DEV-PLAN phase 2 |
| 14 | Accepted | PRD §9's release row said M1–M6. M7 had no test. The roofline fallback allowed a go without M5. QUAL smoke was undefined | TEST-PLAN new SYS-P-010, QUAL-S, REL-001, §10 fallback, §11 traceability (every metric and requirement mapped); PRD M5/M7/M8, §9 |
| 15 | Accepted | Specification text: no workload manifest; "checkpoint class" was ambiguous | TEST-PLAN new WL-001, QUAL pre-registration; PRD §5 (WL-001, identical checkpoint files); DEV-PLAN phase 2 |

## 3. Proposals P1–P7: the technical lead's decisions (`2026-09-19-v0.3-follow-up.md`)
| # | Decision | Applied at |
|---|---|---|
| P1 | **Accepted.** GLM-only Phase 0 gather scope with the **80 % target kept**; R4 accounting and diagnosis are required. Future row widths are report-only **because of Phase 0 scope**. *Correction:* the earlier claim that small rows "can't physically reach" the target was wrong for 288 B rows (9 sectors = 100 % when 32-byte aligned; 90 % at a 16-byte offset). It holds only as a sector bound for an isolated cold 68 B row | TEST-PLAN SMLA-P-005 (L195); DEV-PLAN 0.6 (L42) |
| P2 | **Accepted.** G16 is the MOE parity format. G32 keeps its correctness tests and measurements, with no demonstrated parity or full-model reference support | TEST-PLAN MOE-P-002 (L252), MOE-P-005; PRD M5 (L84) |
| P3 | **Accepted.** Without a gate-eligible baseline: provisional go only; M5 not satisfied | TEST-PLAN §10 (L484, L489); PRD M5 |
| P4 | **Not approved as written.** The existing M3 stays in force. Before the Phase 1 gate: the EQV-001 prerequisites and a **selected** BF16 plan. This is a replacement or deferral of the kind of evidence required, not only a threshold question | PRD M3 note (L82); TEST-PLAN EQV-001 (L360), BF16 plan (L379); DEV-PLAN 1b (L61) |
| P5 | **Accepted.** M7 is a v1.0 release criterion | PRD §9 (L183) |
| P6 | **Accepted.** Denominator = summed compute-kernel time across both GPUs by implementation origin; NCCL, copies and step time reported separately | TEST-PLAN SYS-P-010 (L449); PRD M7 |
| P7 | **Accepted** as the starting design. Fixtures, task versions and thresholds are frozen before the first qualification run; no M8 pass until then | TEST-PLAN QUAL-S (L389); PRD M8; DEV-PLAN R5 |

## 3a. Follow-up findings R1–R6
Statuses: **Spec** = corrected in the specification. **Contract** = awaiting a defined future contract. **Evidence** =
awaiting reference evidence. **HW** = hardware validation pending.

| # | Status | Resolution | Exact locations |
|---|---|---|---|
| R1 | **Spec** + HW | A baseline is gate-eligible only if it passes the same MOE-C/E cases (including the clamp edges) with the same values, routing, precision contract and timed operation boundary (adapter time included). Otherwise it is context only; with no eligible baseline, P3 applies | TEST-PLAN MOE-P-002 (L252), §10 (L484, L489); PRD M5 (L84) |
| R2 | **Spec** + Contract | M1 is enforced **per session** on eight fixed, paired traces (floor and relative target on each); all eight rates reported; committed emitted tokens counted once, rejected drafts never counted; decode timing defined. The stored fixture is **[Pending]** before any SYS evaluation. Also folded into WL-001 (user request): M2 gates the **full** agent step (request through tool completion, comparable with the historical harness), with model-turn latency and tool time reported separately and never compared across definitions as a speed-up; v0.3 had implicitly excluded tool time | TEST-PLAN WL-001 (L400–L422); PRD M1 (L79), M2 (L80); DEV-PLAN phase 2 |
| R3 | **Spec** | Non-finite input rejected. A zero block inside a non-zero tensor follows the pinned rule. An all-zero tensor gets a canonical encoding (zero codes, unit scales), declared an intentional deviation. The byte-for-byte ModelOpt check covers only the finite, non-zero path. CPU validation follows in REF-004 (Phase 0.2) | TEST-PLAN §2 codec (L64–L77), REF-004 (L108) |
| R4 | **Spec** + HW | Whole-launch useful read bytes (each token-row once across heads), written bytes separate, predicted sector footprint kept separate from measured profiler traffic (or recorded as unmeasured). The matched CUDA gather runs before any compiler attribution. If both miss 80 %, an access-pattern or hardware investigation follows. P1's wording is corrected | TEST-PLAN SMLA-P-005 (L195); DEV-PLAN 0.6 (L42); this response §3 (P1) |
| R5 | **Contract** + Evidence | **EQV-001** is a named future gate with prerequisites (a) to (f): reference runner and format, loader and semantics check, noise-floor experiment, aggregation, zero-floor rule, threshold. Kept distinct from the quantization-quality gate, which needs a selected BF16 plan. No floor is invented, and G32 runner support is not claimed | TEST-PLAN §9 golden tests (L360–L379); PRD M3 note (L82); DEV-PLAN 1b (L61) |
| R6 | **Spec** (narrowed) + Contract | D1 is narrowed to **fixed-partition batch invariance**. Cross-partition cases (chunk boundaries, KDA chunked vs recurrent, MTP verification vs one-token decode, cache-hit resume) are explicitly not promised. The canonical schedule is **deferred**, [Pending] a contract before Phase 1 and built only if approved. Greedy MTP on/off uses agreement and KL. Sampler correctness is tested with identical supplied distributions | DEV-GUIDELINES §1.3 (L36–L53); TEST-PLAN SAMPLE (L338–L344), API-006 (L472) |

**Resolved in the specification now:** R1, R3, R4, the per-session rule in R2, and R6 as narrowed.

**Future contract or evidence work:**
- the R2 fixture;
- R5: EQV-001 and the BF16 plan;
- R6: the canonical schedule, only if needed.

**Hardware validation, all pending:** R1 baseline eligibility, R4 gather measurements, and every Phase 0 gate.

## 4. Remaining open items (owner: the technical lead)
1. **Select** the BF16 acquisition plan (quantization-quality gate).
2. **Define EQV-001** before the Phase 1 gate: runner, format, loader check, noise-floor experiment, aggregation,
   zero-floor rule, threshold.
3. **Freeze** the QUAL-S fixtures and thresholds, and **store** the WL-001 fixture, before their first use.
4. The D1 canonical-schedule contract, only if cross-partition determinism becomes a requirement.


## 5. Checks run for this response
- **Documentation:**
  - every repository file is ≤ 1000 lines;
  - relative links resolve;
  - every test ID referenced outside the TEST-PLAN is defined there;
  - every ADR referenced is defined;
  - every PRD metric and requirement appears in the TEST-PLAN traceability table.
- **Guardrails:** `tools/git-guard.py audit` is clean.
- **Not run:** no engine code, GPU session, PyTorch oracle or benchmark. Nothing was committed or pushed.
