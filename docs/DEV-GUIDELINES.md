# TensorQuay Engine: Development Guidelines

Status: **draft v0.3 for PM review** (19 Sep 2026). v0.3 applies the adversarial review findings 5 and 9 and the follow-up's R6 (D1 narrowed to fixed partitions; `reviews/2026-09-19-adversarial-review-response.md`). Earlier: v0.2 applies the review findings (#7, #15, #17, #18, #29, #30,
#34–#37) and the research recommendations: cuTile hot-path rules, determinism tiers and GPU-CI security. These rules
apply to every change. Most are enforced in CI (§6); the rest are checked in review (§7). Architecture is in
[ARCHITECTURE](ARCHITECTURE.md). Related: [PRD](PRD.md), [DEV-PLAN](DEV-PLAN.md), [TEST-PLAN](TEST-PLAN.md).

## 1. Design rules
### 1.1 Structure
1. Crates, responsibilities and the **downward-only dependency direction** are in ARCHITECTURE §2 (gate G-07). A new
   crate needs a decision record.
2. **Pure crates** (`tq-core`, `tq-sched`, `tq-server`) have no CUDA or cutile dependency, so they build and test on
   the Mac. The kernel contracts (parameter types and host validation) live in `tq-core`, not in `tq-kernels`.
3. **Device code and host code are in separate files:** `xxx_kernel.rs` (only `#[cutile::module]`) and `xxx_launch.rs`
   (checks and launch).
4. **Parse, don't validate.** Launchers accept only validated parameter types (`SmlaParams::new(..) -> Result`).
5. **Validation is split by where the data lives.**
   - Shapes, strides, alignment and dtypes are checked on the host before launch.
   - Checks on data that lives on the GPU (id ranges, duplicates) are **debug-build device checks** that set an error
     flag. Release builds never add a GPU-to-host sync for validation.

### 1.2 Runtime
6. **No allocation, lock or host sync on the decode path,** apart from the one sync per step. Pools are allocated at
   start-up.
7. **One worker thread per GPU owns its context, stream and graphs.** `cuda-core` owns the CUDA context. `cudarc`
   (NCCL) is used only inside `tq-gpu`, with borrowed handles. Stream-ordered frees happen on the owning stream (see
   cutile-rs #252).
8. **Configuration only in the validated TOML.** No environment-variable tuning knobs; only documented debug flags.
9. **Observability:** `tracing` spans at request and step boundaries; Prometheus metrics. No `println!` in libraries.

### 1.3 Determinism (ADR-0006)
- **D0 (always):** bitwise-identical output for the same inputs, seed and batch composition, **on the same compiled
  binaries, GPU model, driver and toolchain.** No floating-point atomics; fixed reduction order.
  - NVIDIA's Tile IR exempts MMA from bit-identity guarantees when tile sizes, the target or the toolchain change.
  - So results are not promised identical across those.
- **D1 (opt-in mode): fixed-partition batch invariance.**
  - **Promise:** a sequence's output does not depend on which other sequences share the batch, **provided its own
    execution partition is the same.** The partition means its prefill chunk boundaries, verification token grouping,
    kernel specialisations and tile shapes.
  - **Means:** batch-independent specialisations and tile shapes; fixed split sizes; no batch-dependent split-K.
  - It costs speed (20–60 % in published work), so it is never the default.
  - **Not promised:** invariance across different partitions. That covers different chunk boundaries; KDA chunked
    prefill vs recurrent decode; multi-token MTP verification vs one-token decode; and a cache-hit resume vs a full
    prefill. A canonical recurrence/reduction schedule that would extend D1 across partitions is **deferred**. It is
    **[Pending]** a contract before Phase 1, and it is not built unless that contract is approved.
- **Speculative decoding has three separate properties,** each tested on its own (TEST-PLAN §9 SAMPLE):
  1. deterministic forward computation (D0; D1 within a fixed partition);
  2. **greedy** MTP on vs off: compared by agreement and KL, **not bitwise**, because the partitions differ;
  3. **sampler correctness:** distribution preservation, tested with identical supplied target and draft
     distributions, separately from forward differences.

  **Sampled tokens are not promised equal with MTP on vs off.**
- "Cache hit ≡ miss" is bitwise only under D1 when both use the same partition. Otherwise it is tested with tolerances
  (TEST-PLAN API-006).

### 1.4 cuTile rules (JIT, graphs, limits)
- **Nothing varies per step in integer scalar kernel arguments.** Values such as `kv_len` and `T` are passed through
  device tensors, because cuTile specialises on scalar divisibility and would JIT-compile in the middle of serving.
- Pad T to fixed buckets, or set `max_divisibility`.
- **Warm up every specialisation before graph capture.** A compile counter must not change after "ready" (H-008,
  SYS-R-005).
- **The launch grid never depends on the data** (graph-replayable).
- Use `deny_in_kernel_checks = true` on hot kernels. Never build a view with more than 2^31 elements in a dimension.
  Byte offsets are 64-bit.
- sm_120 limits: **99 KB shared memory per block**; no tcgen05, TMEM or multicast. The resource gate checks shared memory
  and spills (G-11).
- **Offline autotuning only.** Tables are committed, keyed by (kernel, specialisation, GPU, `tileiras` fingerprint,
  cutile-rs version, **schema version**). Runtime loads the table or falls back to the default. The table must pass the
  correctness gate before it is accepted.

### 1.5 Patterns
| Use | Avoid |
|---|---|
| Newtypes for ids and units | Bare integers in public APIs |
| `enum`s for closed sets (layers, formats, drafters) | Trait objects and registries for closed sets |
| A trait only with ≥ 2 real implementations (today: `Comm`) | Speculative abstraction and frameworks |
| RAII for GPU resources and pool leases | Manual frees |
| Typed error enums in libraries | `anyhow` in libraries; `unwrap`/`expect`/`panic` on inputs |
| Plain functions and data | Deep layering; macros other than cuTile's |

## 2. Code style
### 2.1 Size limits
| Item | Limit | Enforced by |
|---|---|---|
| Any source file | **≤ 1000 lines, hard limit** (aim for ≤ 500) | `scripts/check-lengths.sh` (G-03) |
| Kernel file (`*_kernel.rs`) | ≤ 400 lines | Same script |
| Function | ≤ 100 lines | `clippy::too_many_lines` |
| Nesting depth | ≤ 5 | `clippy::excessive_nesting` |
| PR | Aim for ≤ 400 changed lines of code | Review |

### 2.2 Formatting and lints
- `rustfmt`: edition 2024, `max_width = 100`, stable options only. Python: `ruff format`, `ruff check`.
- `[workspace.lints]` (lint groups get `priority = -1`, as in cutile-rs):

| Lint | Level |
|---|---|
| `clippy::all`, `clippy::pedantic` (which includes the `cast_*` lints) | deny, with a reviewed allow-list |
| `clippy::undocumented_unsafe_blocks`, `clippy::multiple_unsafe_ops_per_block` | deny |
| `unsafe_op_in_unsafe_fn` (rustc lint, under `workspace.lints.rust`) | deny |
| `clippy::unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `dbg_macro`, `print_stdout` | deny in libraries |
| `clippy::excessive_nesting` (threshold 5) | deny |
| `missing_docs` | deny for public items |
| `unsafe_code` | **forbid** except in `tq-gpu` and `tq-kernels` (mmap lives in `tq-gpu`; ADR-0007) |

- **Kernel modules:** `#[cutile::module]` already injects `allow(clippy::all)`. Pedantic and restriction lints still
  apply. Each `*_kernel.rs` states its lint policy at the top, using `#[expect(lint, reason = "…")]` (preferred over
  `allow`).
- Exceptions use the smallest scope, with a reason.

**Current CPU lint scope:** the runner checks all `reference/` and root `tests/` Python, both new
CPU tooling modules and `tools/check-philox-upstream.py`. The existing `tools/git-guard.py` is
deliberately outside the new Ruff scope: reformatting it would mix a frozen guard change into
step 0.4. Its existing style findings remain outstanding and must be handled in a separate
reviewed guard change. This exemption does not cover new tooling or changes to guard behavior.

### 2.3 Naming, comments, safety
- Follow the Rust API Guidelines. Put units in names (`kv_len_tokens`, `pool_bytes`, `latency_us`).
- Comments explain **why**. Each kernel file starts with its contract summary and the TEST-PLAN IDs that cover it. No
  commented-out code. `TODO(#issue)` only.
- **Every `load_ptr_tko` / `int_to_ptr` `// SAFETY:`** comment states the index-range invariant and the masked-lane
  convention: a masked lane uses a masked load, or reads the reserved zero row outside the allocatable pool. It is never
  clamped to page 0 / row 0.

### 2.4 Dependencies and pins
- Few and justified; decision records for large ones.
- Pinned: `rust-toolchain.toml`; committed `Cargo.lock`; `--locked`; cutile-rs commit (with a monthly upgrade PR that
  runs the full suite); CUDA 13.3; **NCCL ≥ 2.31.2**, checked at start-up.
- **Relationship to NVIDIA's cutile-rs (ADR-0016).**
  - Consume the pinned upstream release unchanged whenever possible.
  - Put our extensions in our own crates on top of its public API.
  - If a change inside cutile-rs is unavoidable, carry it as a small, documented patch in a TensorQuay fork. Apache-2.0
    requires keeping `LICENSE` and `NOTICE`, and a notice in every modified file. The fork is rebased on every upstream
    release through the monthly upgrade PR, and each patch lists the upstream version it applies to and why it exists.
  - The goal is **zero carried patches**. The patch count is reported in each upgrade PR.
  - No upstream code contributions are planned.
- Licence allow-list, aligned with cutile-rs `deny.toml`: Apache-2.0, Apache-2.0 WITH LLVM-exception, MIT, BSD-2/3,
  ISC, Unicode-3.0, Unicode-DFS-2016, Zlib, BSL-1.0, CC0-1.0. Anything else needs a decision record.

### 2.5 Implementation provenance
- Study published papers, architecture designs and reference behavior; cite the sources that inform a contract or
  design. Write TensorQuay's runtime, kernels and model integration independently. Do not copy or line-by-line port
  implementation code from vLLM or other inference engines.
- Pinned external implementations may run separately as correctness or benchmark references. Keep their source
  outside this repository and record versions and fixture provenance. Approved dependencies, including the NVIDIA
  toolchain described in §2.4, remain part of the build.
- Demonstrate an improvement with independent correctness tests and comparable measurements. A different language
  or architecture alone is not evidence of better performance.

## 3. Git and review
- **Git guardrails are in `AGENTS.md`** and are enforced by hooks (`tools/git-guard.py`; install with
  `tools/install-guardrails.sh`). They cover one identity, no AI attribution, forbidden content and files, and no force-push.
- `main` is protected: PRs only, required gates green, squash-merge, Conventional Commits, TensorQuay identity.
- No personal names, IPs, local paths or credentials in the repository (G-10).
- The PR states: what and why; test evidence (IDs); for kernels, the benchmark JSON diff and a profile summary.
- Design-level changes (contracts, formats, architecture, technical gates) need technical-lead approval within the
  Product Owner's agreed direction. Product scope, spending and publication remain Product Owner decisions.

## 4. Tests and coverage
### 4.1 Coverage tool
- **`cargo-llvm-cov`** (pinned; LLVM source-based instrumentation) run through `cargo-nextest`.
- Instrumentation counts are exact. With deterministic inputs and tests the report is stable, so it can be ratcheted.
- Tests must not depend on `HashMap` order or thread timing (use `BTreeMap` or a fixed hasher in tests).
- Python: `coverage.py` via `pytest-cov`.

### 4.2 Thresholds
| Scope | Gate |
|---|---|
| Pure crates (`tq-core`, `tq-sched`, `tq-server`), **per crate** (a script reads `--json`) | Lines, regions and functions: **100 %** |
| Ratchet vs `main` | A drop of more than 0.25 pts fails |
| GPU-host crates (`tq-gpu`, launchers, `tq-engine`) | Coverage from tier-2 GPU runs is merged (`--no-report`, then `report`) and **reported**; it becomes a gate once the GPU runner exists |
| Python `reference/` | Lines and branches: **100 %** |
| Branch and MC/DC | Weekly on nightly; reported, not gated |

### 4.3 GPU kernel code: contract coverage
- `*_kernel.rs` is excluded from line coverage.
- `tests/manifest.toml` maps every contract clause and every specialisation to test IDs. G-08 fails if any are missing.
- GPU runs record the specialisations actually launched, which must match the manifest.

### 4.4 Determinism of tests
1. Fixed seeds via the shared Philox stream. `proptest` uses a fixed seed and saved regressions.
2. No wall clock, network or order dependence (nextest runs one process per test).
3. **No retries** (`retries = 0`). A flaky test is a bug and is fixed or quarantined with an issue the same day.
4. **GPU tests fail when no GPU is present;** they never skip as a pass (nextest `gpu` profile).
5. Tolerances follow TEST-PLAN §2 and are never tuned to pass.

### 4.5 Test quality
- `cargo-mutants`, weekly, on `tq-core`, `tq-sched` and launch validation. Target ≥ 80 % of mutants caught.
- `proptest` state-machine tests for `tq-sched`: no double free, correct reference counts, radix consistency, budget
  never exceeded, no admission deadlock.
- Miri on pure-Rust `unsafe` code.
- Every kernel suite includes "a broken kernel is caught" (H-005).

**Independent acceptance:** the reviewing lead writes the acceptance tests and reruns them independently of the
implementing engineer. Developer checks support implementation but do not certify acceptance. The engineer must not
weaken those tests or add coverage exclusions to obtain a pass. Coverage is required alongside numerical-oracle,
contract and performance evidence; it is not a substitute for them.

## 5. Test levels
Test levels and where they run are defined in TEST-PLAN §1a:
- Mac: pure crates and REF tests.
- CI: compile-only kernels.
- RTX 5090: kernel tests.
- RTX PRO 6000: gate.
- 2 × RTX PRO 6000: end to end.

## 6. CI gates
**Tier 1: every push and PR** (GitHub-hosted Linux, no GPU, CUDA 13.3 toolkit installed). All are required.

The current [CPU foundation](CPU-CHECKS.md) supplies G-01…G-09 and G-10 through
`scripts/check-cpu.sh`. G-11/G-12 and the CUDA build image belong to step 0.5; this CPU workflow
does not claim them. The workflow has not run on GitHub while the repository has no remote.
Fork PRs run the public checks without secrets and report the private identifier check as pending.
Maintainers must run the strict local guard on the reviewed revision before merging. A separate
trusted-main push job requires the private denylist secret and fails when it is absent. Never
expose that secret to a fork PR or use `pull_request_target` to run its code.

| Gate | Check |
|---|---|
| G-01 | `cargo fmt --check`, `ruff format --check` |
| G-02 | `cargo clippy --workspace --all-targets --locked -- -D warnings`, `ruff check` |
| G-03 | File size limits (`scripts/check-lengths.sh`) |
| G-04 | Fresh advisory database fetch, recorded revision, then `cargo deny --locked --offline check` (licences, advisories, sources) |
| G-05 | `cargo build --workspace --locked` (debug and release) |
| G-06 | `cargo llvm-cov nextest` for the pure crates with per-crate thresholds; `pytest --cov` including REF tests |
| G-07 | Dependency direction (`scripts/check-deps.sh` reads `cargo metadata`) |
| G-08 | Contract coverage manifest (`scripts/check-manifest.sh`) |
| G-09 | `RUSTDOCFLAGS=-D warnings cargo doc --no-deps` |
| G-10 | `python3 tools/git-guard.py audit` plus `gitleaks`; private denylist required locally and on trusted-main CI, pending on secret-free PR CI |
| **G-11** | **Compile every manifest specialisation for sm_120** (`KernelCompiler` + `tileiras`, no GPU). Resource gate from `cuobjdump -res-usage`: shared memory ≤ 99 KB per block, zero spills on hot kernels, register deltas reported |
| G-12 | GPU test binaries build; the `gpu` profile is marked "requires GPU" (it fails, never skips) |

**Tier 2: GPU gate** before merging any change to kernels, `tq-gpu` or `tq-engine`.
- Until a runner exists, this is a scripted pod run. A future GPU runner must be a **just-in-time, ephemeral,
  maintainer-triggered** self-hosted runner:
  - no fork-PR triggers and no `pull_request_target` with a checkout;
  - no secrets;
  - a clean workspace per job;
  - a JIT cache writable only by the runner.
- Contents:
  - all P0 kernel C and E tests;
  - sanitizers clean (with the triage policy in TEST-PLAN §7);
  - a paired A/B regression check using the rule in TEST-PLAN §8 (a dimensionless ratio);
  - launched specialisations match the manifest;
  - no JIT after warm-up.

**Tier 3: phase and release gate** on 2 × RTX PRO 6000: end-to-end suites, QUAL and SYS vs baseline B; PM sign-off.

**Weekly (reported):** branch coverage, `cargo-mutants`, Miri, advisory refresh, a cutile-rs upstream diff and
changelog review.

## 7. Review checklist
- Does the contract match the pinned reference code, including the dtype casts?
- Are all invalid inputs rejected (on the host) or flagged (on the device, in debug)? Is every `unsafe` justified,
  including the masked-lane convention?
- Could any read touch memory it shouldn't: −1, stale padding, out-of-range values, 64-bit offsets, slot-0 clamps?
- Any integer scalar argument that varies per step, any data-dependent grid, or any allocation, lock or sync on the
  decode path?
- Is the tolerance from the baseline rule? Are index semantics covered by exact probes?
- Is this the simplest version? Could a file, type or dependency be removed?
- Are numbers reported with their configuration and raw data, and is external work credited?
