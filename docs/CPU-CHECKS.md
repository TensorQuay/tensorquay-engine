# Phase 0 CPU tooling contract

Status: technical-lead contract, frozen before implementation on 20 Sep 2026.
This slice completes the local CPU tooling in step 0.4. It supplies a CPU CI workflow;
hosted Linux execution is tracked in the CPU workflow and evaluation reports. Kernel compilation,
G-11/G-12, ModelSpec parsing and GPU execution are separate work.

## Scope and interfaces

Use Python 3.12 standard-library checkers in `tools/cpu_checks.py`, with thin shell
entry points `scripts/check-lengths.sh`, `check-deps.sh`, `check-manifest.sh` and
`check-coverage.sh`. A small `scripts/check-cpu.sh` runs the checks in order and
stops on failure. No new runtime crate, framework or Python runtime dependency.

The independently tested Python API is:

```python
class CheckError(ValueError): ...
def check_lengths(root: Path, paths: list[str]) -> int: ...
def check_dependencies(metadata: dict) -> int: ...
def check_manifest(manifest: dict, required: set[str], available: set[str]) -> int: ...
def check_coverage(report: dict, root: Path, crates: set[str]) -> dict: ...
def main(argv: list[str] | None = None) -> int: ...
```

The first three return the number of source files, declared dependency edges and
mapped clauses respectively. Coverage returns per-crate `lines`, `regions` and
`functions`, each with integer `count` and `covered`. Invalid check inputs raise
`CheckError` containing `source`, `dependency`, `manifest` or `coverage` respectively.
No input is modified. Extra fields in upstream JSON are ignored; required fields
used by a check must be present and well formed. CLI usage errors exit nonzero.

CLI subcommands are `lengths`, `deps`, `manifest`, and `coverage REPORT`. They take
an optional `--root PATH` (default: the repository containing the script). Wrappers
work from any current directory, use the pinned reference environment and forward
arguments. Missing files/tools, failed subprocesses and malformed input fail the
check. The log reports what was examined; empty evidence never becomes a pass.

## G-03: source length

Enumerate tracked and non-ignored untracked files with Git, using NUL separators.
Select `.rs`, `.py` and `.sh`: at most 1,000 physical lines, or 400 for
`*_kernel.rs`. Count a final unterminated line; CRLF is one line ending. Ignore
data fixtures, lockfiles and documents for this source-code limit. The 500-line
aim is advisory. At least one source file must be checked.

Paths must name readable UTF-8 files within the root; missing files, absolute or
escaping paths and external symlinks fail. Repeated paths are counted once.
Generated and ignored files are excluded by Git, rather than a broad `target*`
directory rule that could conceal an ordinary source directory. A tracked source
deletion is reconciled with the index before this check passes.

## G-07: dependency direction

Read `cargo metadata --format-version 1 --all-features --locked --offline` without
`--no-deps` or a platform filter. Package IDs are opaque; actual package names,
dependency kinds and resolved package IDs determine the check, never aliases.

The approved bootstrap members are exactly `tq-core` and `tq-testkit`. The only
permitted workspace edge is `tq-core --dev--> tq-testkit`. A new member or edge
needs an explicit policy update with its architectural review; do not pre-create
the future crate graph. Ordinary external CPU dependencies are permitted here
and are independently checked by cargo-deny.

Check every declared dependency, including optional, target-specific and build
dependencies. The kind is null (normal), `dev` or `build`. Check the full resolved
closure reachable from both workspace members, including dev/build edges, for
GPU packages: family names `cuda`, `cutile`, `cudarc`, `nccl`, `cust`, `rustacuda`,
`cudnn`, `cublas`, `cufft`, `cusparse`, `nvrtc`, `nvtx` and `nvml` are forbidden,
including suffixes separated by `-` or `_`. Do not reject unrelated names such as
`custom` or `curl`. Thus an alias or CPU wrapper cannot hide
a known CUDA dependency. This name policy is not a proof about arbitrary native
code; unfamiliar dependencies still require review.

Member IDs and names must be unique; all members and resolved dependency IDs
must resolve to packages/nodes. An absent resolve graph fails. An empty
dependency list is valid for a real member, but an empty workspace is not.
Return the total number of dependency declarations across workspace members.
Scanning all features and platforms is intentionally stricter than the default
build.

## G-08: contract-to-test mapping

`tests/manifest.toml` has exactly these top-level fields:

```toml
schema = 1
scope = "cpu-foundation"
kernel_specializations = []
[clauses]
# Each catalogue ID below maps to a nonempty array of test selectors.
```

The `CPU clause catalogue` below is the required-ID inventory, independent of
the manifest. Parse its table, rejecting duplicate IDs or malformed rows; do not
duplicate its ID list in implementation code. Compare whole IDs, reject missing/extra IDs, empty mappings,
duplicate selectors within a clause and unknown tests. A selector is either
`rust:<nextest binary-id>::<test name>` or `python:<root-relative pytest nodeid>`.
Match a complete node ID; a Python function selector also matches its collected
parameter instances through an immediately following `[`. No wildcards, file-only
selectors or substring matches. One test may cover several clauses.

Build the available inventory from successful actual nextest and pytest
collection, not source-code regular expressions. Require nonempty Rust and Python
inventories. Ignored Rust tests and Python skip/skipif/xfail-marked tests cannot
satisfy a mapping. Use a small pytest collection hook for node IDs and markers;
all three marker kinds are excluded even when a skipif condition is false.
Collection is traceability, not a claim that a test passed;
the same CPU command must also execute the suites successfully.

The empty specialization list explicitly means **no kernel evidence**. A nonempty
list or any tracked/non-ignored `*_kernel.rs` fails this CPU-only checker until the
GPU manifest contract is implemented. No guessed tile variants or GPU passes.

## G-06: per-crate coverage

The CPU runner deletes its previous output file, performs a fresh whole-workspace
`cargo llvm-cov nextest` run and checks that newly written JSON. It cannot accept
a caller-supplied old report. The standalone `coverage REPORT` checks the supplied
file's contents, not its age. Reports are checked in the checkout that built them.
Use metadata to supply the expected crate set, so removing a crate from a report
cannot hide it. There are no source exclusions in this slice.

For each crate, aggregate the reports under `crates/<crate>/src/`. Every reported
source must exist; normalize paths before assigning or detecting duplicates.
Every `.rs` source below that directory except `lib.rs` must appear. The current
`lib.rs` files contain only attributes/module declarations and have no LLVM
regions; any reported root-file regions are included. Review must continue to
ensure unwired or uninstrumented source is not mistaken for tested behavior.

For lines, regions and functions, counts must be nonnegative integers (not booleans),
`0 <= covered <= count`, each crate's total count must be positive, and
`covered == count`. Check raw counts, not rounded percentages or global totals.
Reject missing/duplicate files, missing crates, malformed/empty file data and any
deficit. Extra report files outside the expected source directories do not help
their totals. Rust branch coverage is not claimed by this stable-toolchain gate.

Python reference lines/branches remain 100%. The lead's independent tests must
also reach 100% lines/branches of the new checker module, without exclusions.
The shell orchestration is verified with executable failure-injection tests.

## Runner, dependency audit and CI

- Preserve existing Rust/Python pins and accepted sources/tests/fixtures. Add exact
  tool-version pins for cargo-deny, gitleaks and workflow validation; retain
  cargo-llvm-cov 0.9.1 and cargo-nextest 0.9.145. Install from official releases,
  verifying published checksums. Pin Actions to full commit SHAs. Use the gitleaks
  CLI binary, not the organization-licence-dependent action. Pin uv to the locally
  exercised version, 0.10.12; do not upgrade the Python dependency environment here.
- `deny.toml` uses the DEV-GUIDELINES licence allow-list, checks unpublished
  workspace crates, denies duplicate versions and unknown registries/Git sources,
  and has no advisory ignores. Advisory refresh failure is fatal. Record the
  actual advisory database revision. Zero external dependencies is reported as
  such, not as a security assessment of future dependencies.
- The CPU runner executes G-01…G-09 and G-10's public checks: formatting/lints,
  lengths, cargo-deny, debug/release builds and independent Rust tests, per-crate
  coverage, independent and full Python suites, root harness/tooling acceptance,
  dependency/manifest checks, rustdoc, git guard and gitleaks history plus tree.
  Use locked dependencies, no retries, no coverage reuse, no ignored failures.
- Default local mode also requires the real private denylist and runs the full
  guard. An explicit `--public-ci` mode uses no secrets and clearly reports the
  private-identifier check as pending. It never claims full G-10 acceptance.
- One unprivileged `pull_request`/`push` CPU workflow: read-only contents token,
  full history, no persisted checkout credentials, no path filters, pinned tools,
  no `pull_request_target`, self-hosted runner, publication or GPU provisioning.
  Test the PR head, avoiding synthetic merge identities in the history audit.
- A separate trusted-main push job requires the private denylist; a missing secret
  fails. Fork code never receives it. Before merge, maintainers still run the full
  local guard on the reviewed revision. This is the private part of G-10, not an
  optional warning. Remote configuration and a hosted run are reported separately.
- Linux workflow validation plus actual local CPU execution are required here.
  G-11 is a future **CPU compile-only** CUDA gate; G-12 builds GPU test binaries.
  Neither needs a GPU to compile, and neither is implemented by this CPU slice.

## CPU clause catalogue

These IDs identify accepted CPU obligations only. GPU portions of similarly named
TEST-PLAN cases remain pending. Individual test assertions supply the evidence;
this catalogue does not make a mechanical mapping prove semantic completeness.

| ID | Obligation |
|---|---|
| REF-001 | MLA scale, projection/masking, numerical boundaries and upstream oracle |
| REF-002 | Routed experts, layout/clamps, overflow and upstream oracle |
| REF-003 | Exact quantized decode tables |
| REF-004 | Quantization, boundaries, shared scales and upstream byte agreement |
| REF-006 | FP32 router selection, normalization and upstream comparison |
| RNG-001 | Primitive words and known answers |
| RNG-002 | Streams, offsets, carry and exhaustion |
| RNG-003 | Partition invariance and atomic rejection |
| RNG-004 | Exact uniform float32 mapping |
| RNG-005 | Frozen fixtures, cross-language execution and oracle-checker corruption |
| HOST-001 | Dtype/rank/shape and all tensor-field validation |
| HOST-002 | Checked 64-bit spans, capacity and address ends |
| HOST-003 | Strides, alignment, padding and dimension bounds |
| HOST-004 | Empty batches and empty cache/index cases |
| HOST-005 | SMLA geometry, row formats and exact scale |
| HOST-006 | Host query-plan per-sequence and total limits |
| HOST-007 | MoE geometry, formats, shards and scale layouts |
| HOST-008 | Output aliases and unused allocation tails |
| HOST-009 | Typed errors, private parameters and borrowed-plan lifetime |
| HOST-010 | Independent wide-arithmetic boundary sweeps |
| CPU-001 | Source length boundaries and inventory failures |
| CPU-002 | Dependency direction, kinds, aliases and transitive GPU leakage |
| CPU-003 | Manifest completeness and actual test resolution |
| CPU-004 | Per-crate nonempty exact coverage and missing evidence |

Sources: [Cargo metadata](https://doc.rust-lang.org/cargo/commands/cargo-metadata.html),
[nextest inventory](https://nexte.st/docs/machine-readable/list/),
[coverage JSON](https://github.com/taiki-e/cargo-llvm-cov),
[cargo-deny configuration](https://embarkstudios.github.io/cargo-deny/checks/cfg.html),
and [GitHub workflow security](https://docs.github.com/en/actions/reference/security/secure-use).
