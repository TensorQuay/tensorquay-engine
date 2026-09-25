# Evaluation reports

Every implementation milestone needs a short, reproducible evaluation report before technical acceptance. The lead
owns acceptance and independently reruns the relevant checks. Developer test results are supporting evidence.
Publication remains a Product Owner decision. The existing PRD and TEST-PLAN thresholds remain in force.

| Milestone | Evidence required | Status |
|---|---|---|
| CPU references | Independent numerical oracles, boundary cases, line/branch coverage | [NVFP4 accepted](2026-09-19-nvfp4-reference.md); [MLA core accepted](2026-09-20-mla-reference.md); [MoE experts accepted](2026-09-20-moe-reference.md); [FP32 router accepted](2026-09-20-router-reference.md); Phase 0 P0 references complete |
| Shared test generator / first Rust crate | Upstream and cross-language exactness, independent coverage, mutation checks | [Philox accepted](2026-09-20-philox-foundation.md); step 0.3 complete, step 0.4 partial |
| Rust host contracts | Layout/bounds acceptance, private API probes, independent coverage and deliberate faults | [Host slice accepted](2026-09-20-host-contracts.md) |
| CPU tooling and CI | Fail-closed checks, independent coverage, failure injection and actual local gates | [Local step 0.4 accepted](2026-09-20-cpu-tooling.md); [publication CPU rerun](2026-09-25-publication.md); hosted results in the CPU workflow |
| CUDA compiler image preparation | Immutable inputs, isolated build context and failure propagation; then actual Linux compilation | [Recipe checks accepted](2026-09-20-cuda-build-preparation.md); image build and CUDA execution unverified |
| GPU kernels | Contract tests, sanitizers, resource limits and eligible same-card kernel baselines | Pending |
| One model layer | Reference outputs and state transitions, including resume and rollback | Pending |
| Full model | EQV-001 prerequisites, quality evaluation and the selected BF16 reference plan | Pending |
| Eight agents | Every session's M1 result, full agent-step M2, context, memory and paired baseline | Pending |
| Sixteen agents | Capacity/latency report, admission behavior and failures at the stated context | Pending |
| Release/new-model support | All applicable release gates and a reproducible release-day rehearsal | Pending |

Each report records:

1. The decision: accepted, rejected or evidence incomplete, and exactly which next step it permits.
2. Code/model revisions, dependency versions, hardware/topology and complete reproduction commands.
3. Workload/fixture versions, context, concurrency, sampling, quantization and speculative-decoding settings.
4. Baseline eligibility and matching inputs, numerical semantics and timed boundaries. For engine claims, use the
   best eligible official vLLM/SGLang configuration on the same hardware, including available host caching. A missing
   baseline makes the result provisional; it is not evidence of superiority.
5. Results against thresholds fixed before the run: per-session values and distributions, failures, raw-data location
   and independent checks. Keep hypotheses and arithmetic estimates separate from hardware measurements.
6. Quality, latency, speed and memory together. Include prefill/TTFT, committed decode tokens, full agent-step time,
   usable context, RAM/VRAM peaks, cache behavior and completion rate where applicable.
7. Limitations and remaining gates. An improvement in one metric must disclose regressions in the others.

Keep reports small; link to sanitized machine-readable results when hardware runs produce them. Do not add an
evaluation service or a dashboard before the existing commands and workload fixtures need one. Passing a foundation
test establishes correctness for that slice; community-facing performance claims need the relevant hardware report.

Reports dated before the public import retain their original test results and file digests. Earlier Git revision
identifiers refer to unpublished development history; that history is not part of this repository. Reproduce the
current checkout with the documented CPU gate rather than assuming an earlier revision is publicly fetchable.
