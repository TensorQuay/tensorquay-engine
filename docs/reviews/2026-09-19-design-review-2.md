# Design review, round 2 (v0.2 documents)

Public technical extract: internal operational notes are omitted; locations refer to the reviewed revisions.

Second pass, 19 Sep 2026, by the engineering team:
- **(A)** check that every round-1 finding (`2026-09-19-design-review.md`, 41 findings) is reflected in the v0.2
  documents;
- **(B)** an adversarial read of v0.2 for new problems.

## A. Round-1 findings
- All 41 findings have a corresponding change in the documents, checked by locating each fix in its document.
- The gate thresholds agree in PRD §5/§9, DEV-PLAN §3 and TEST-PLAN §10.
- Finding #40 also covered operational planning; only its technical prerequisites are retained in this extract.

## B. New findings and resolution
| # | Severity | Finding | Resolution |
|---|---|---|---|
| N1 | high | The masked-lane "zero-row convention" contradicted SMLA-E-009, which poisons page 0 row 0, and DEV-GUIDELINES said "never dereferenced" | TEST-PLAN §4 and DEV-GUIDELINES §2.3: a masked lane uses a masked load, **or** reads a reserved zero row **outside the allocatable pool** (zero data, scales 1.0); never page 0 row 0 |
| N2 | high | The TP layout of the fused W13 (gate/up) was not specified. A naive row split sends all gate rows to shard 0, and MOE-C-004 would only catch this if the test built the shards correctly | TEST-PLAN §5 TP-layout clause, plus **MOE-C-008** (a naive split must fail) |
| N3 | medium | The FP8 528-byte row layout was implicit | TEST-PLAN §4: bytes 0–511 are E4M3 values; 512–527 are 4 f32 scales, one per 128 values; the same payload as FlashInfer GLM53_NOPE (flashinfer #5075, verified) |
| N4 | medium | Baseline B was undefined if no official runtime runs on the box yet | PRD §5: fall back to the best public runtime, as a benchmark only, and say so |
| N5 | medium | Build prerequisites were missing: pinned reference dependencies, sufficient image storage, and FlashInfer AOT compilation before timing | DEV-PLAN 0.1 (storage and pinned dependencies) and 0.5 (AOT-built FlashInfer for sm_120 and recorded image digest) |
| N6 | medium | API-006's "greedy agreement ≥ 99 %" was ambiguous | TEST-PLAN API-006: teacher-forced per-token KL within the noise floor (primary); ≥ 95 % of 200 prompts give identical first 256 greedy tokens (secondary) |
| N7 | low | The PRD made FP8 KV the default before its quality ADR | PRD §6: "intended default, subject to ADR-0002" |
| N8 | low | The memory example covered only 8 agents, while the FR-4 target is 16 | PRD §7: a 16-sequence example (it fits, but only just) |
| N9 | low | AGENTS.md and the checker disagreed: IP exceptions, pod/host IDs are not machine-checked, and the `.guard-allow` entry types weren't listed | AGENTS.md updated |
| N10 | low | cudarc binds the NCCL 2.30 API while we need a ≥ 2.31.2 runtime; it wasn't recorded which head-count path the FlashInfer baseline runs at H=32 | DEV-PLAN §2 note (verified by SYS-R-008); SMLA-P-004 records the path used |

## Still open (need measurement or PM input, not document changes)
- **The semantics of `index_share_for_mtp_iteration`.** It needs the MTP oracle (TEST-PLAN §9).
- **Whether FlashInfer's sm_120 kernel runs at H=32** on its runtime-H path (tested in SMLA-P-004).
- **The ADRs (0002–0011),** decided by the measurements listed in ARCHITECTURE §9.
