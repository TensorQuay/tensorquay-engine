# Contributing

Help make new open models useful on consumer and workstation NVIDIA GPUs. Start with a concrete model, a correctness
issue, a measured bottleneck or a deployment problem. Prefer a small change that can be checked independently.

The working slices are CPU references for NVFP4 quantization, MLA attention, routed MoE experts and the FP32 router.
Rust `tq-testkit` and Python also share a deterministic test generator; `tq-core` validates attention and MoE host
metadata under [its contract](docs/HOST-CONTRACTS.md). GPU kernels and model
serving are still planned; the design documents are not evidence of a benchmarked engine.

## Getting started

Read [AGENTS.md](AGENTS.md) and [the engineering rules](docs/DEV-GUIDELINES.md). Install the repository guardrails with
`tools/install-guardrails.sh` and configure the private local denylist described in AGENTS.md. Keep credentials,
personal information and internal deployment data out of patches and benchmark fixtures.

For a new model or a change to a kernel contract, agree on the scope with the maintainers before implementing it.
Small fixes and tests can use the existing contracts. Commit and push rules are in AGENTS.md.

Learn from published designs and cite the sources, then write the implementation independently. Do not copy or
line-by-line port another inference engine's code. Pinned external runners remain useful for independent validation;
see [implementation provenance](docs/DEV-GUIDELINES.md#25-implementation-provenance).

## Running the current tests

Install `uv`, then run:

```sh
cd reference
uv sync --locked
uv run --locked pytest --cov --cov-branch
uv run --locked ruff format --check .
uv run --locked ruff check .
```

Python 3.12 and dependencies are pinned. Tests run offline after setup; no GPU or model weights are needed. The CPU
reference package must reach 100 % line and branch coverage. Run the independently maintained acceptance suites alone:

```sh
uv run --locked pytest tests/test_nvfp4_acceptance.py tests/test_mla_acceptance.py \
  tests/test_mla_upstream_acceptance.py tests/test_moe_acceptance.py \
  tests/test_moe_upstream_acceptance.py tests/test_router_acceptance.py \
  tests/test_router_upstream_acceptance.py tests/test_philox_acceptance.py --cov --cov-branch
```

The [reference guide](reference/README.md) explains how to regenerate and check the pinned upstream fixtures. Those
separate checks use isolated PyTorch environments and may need the network; ordinary tests use the stored fixtures.

From the repository root, run the complete CPU gate after setup:

```sh
uv sync --project reference --locked --all-groups
./scripts/check-cpu.sh
```

The runner requires Rust/Cargo 1.95.0 (including rustfmt, clippy and llvm-tools), uv 0.10.12,
`cargo-llvm-cov` 0.9.1, `cargo-nextest` 0.9.145, `cargo-deny` 0.20.2, `gitleaks` 8.30.1 and
`actionlint` 1.7.12 on PATH. Install binaries from their official releases and verify the published
checksums; the [CPU workflow](.github/workflows/cpu.yml) records the pinned Linux setup. Ordinary
tests are offline after setup; the dependency audit deliberately requires a fresh online advisory fetch.

This command stops on the first failure. It runs debug/release Rust tests, independent and full
Python reference coverage, the root acceptance suites, per-crate Rust coverage, dependency and
manifest checks, formatting/lints, documentation, secret scanning and the strict local git guard.
The [CPU contract](docs/CPU-CHECKS.md) explains the boundaries. New checkers and the collection hook
must reach 100% line and branch coverage; all current Rust library lines, regions and functions
must be covered in each crate. Coverage alone does not establish test quality.

Fork PRs use `./scripts/check-cpu.sh --public-ci`, which runs without secrets and reports the private
identifier check as pending. A maintainer runs the full local guard on the reviewed revision before
merge. Trusted-main CI also requires the private denylist. A public-mode pass is not full G-10 acceptance.

For focused work, the individual entry points are `scripts/check-lengths.sh`, `check-deps.sh`,
`check-manifest.sh` and `check-coverage.sh REPORT`. `tests/manifest.toml` maps the CPU obligations
in the contract to actually collected, runnable tests; skipped or expected-failure cases cannot
supply that evidence. Kernel compilation and GPU gates require the separate step 0.5 setup.

The [host-contract evaluation](docs/evals/2026-09-20-host-contracts.md) records metadata acceptance
and deliberate fault probes. Device allocation binding and GPU execution require later evidence.

## Adding a model

1. Identify the official checkpoint, configuration, tokenizer/template and reference implementation; pin their
   revisions. Check memory fit on the intended card before a full-model run.
2. Reuse the supported architecture and kernels where possible. List only the missing behavior; a new model does not
   require a new framework or registry.
3. Add small reference and configuration tests for that behavior before changing the implementation. Test model
   semantics, including cache updates and quantization, against the pinned reference.
4. Implement the smallest supported path. Unsupported configurations should fail clearly.
5. Provide a reproducible run and measurements on the named hardware. Report context, concurrency, speed and quality
   together, with the checkpoint, runtime versions and workload. Separate measured results from estimates.

See [the development plan](docs/DEV-PLAN.md) for the model-support workflow and [the test plan](docs/TEST-PLAN.md) for
reference and benchmark requirements.

## Submitting a change

Fork the repository, create a branch and open a pull request against `main`. Public visibility does not grant write
access. Maintainers review changes and required checks before merging. Keep the TensorQuay commit identity and
guardrails described in AGENTS.md.

Describe the problem, resulting behavior and relevant checks. Include a small reproducer for a bug. For performance
work, include paired measurements with equivalent inputs and semantics; publish sanitized fixtures and reproduction
instructions rather than private traces.

Keep unrelated refactoring out of the change. Host-only tests should be runnable without a GPU; state which hardware
checks remain outstanding. Maintainers review the implementation and arrange the required target-hardware checks
before merging. A passing CPU check does not stand in for a GPU result.

The main branch is the shared implementation. Maintainers keep new-model work compatible with its existing contracts
and tests, so improvements remain available to everyone rather than accumulating in separate deployment forks.
