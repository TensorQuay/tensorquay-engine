# Rust host contracts and input validation

Gate date: 20 Sep 2026. **Decision: accepted for the host-contract slice of Phase 0 step 0.4.**
The new `tq-core` crate validates metadata for sparse MLA and grouped NVFP4 experts. The remaining
step 0.4 tooling and CPU CI are still pending. No GPU kernel, model execution, inference speed,
quality or concurrency result follows from this gate.

## Contract and implementation

The lead wrote the [contract](../HOST-CONTRACTS.md) and independent tests before implementation.
The first acceptance command failed because `tq-core` did not exist. The developer then implemented
the crate without modifying those tests; subsequent boundary and regression tests also belong to
the lead. The crate is `no_std`, inherits `forbid(unsafe_code)`, has no runtime dependency, and uses
only stack data and a borrowed host query plan. `tq-testkit` is a test dependency only.

- SMLA: exact model dimensions and softmax scale; BF16 and FP8 KV payloads; both 528- and
  544-byte FP8 strides; padded q/idx/out; host counts of at most four queries per sequence.
- MoE: both G16 and G32 E4M3 layouts; I=1024/2048 shards; packed weight and scale-array shapes.
- Common: dtypes, strides, alignment, checked 64-bit spans and address ends, sufficient capacity,
  conservative output overlap checks, and well-formed empty launches without backing storage.
- Validated fields are private. Accessors return metadata copies and an immutable borrowed plan;
  callers cannot mutate a validated value through them or construct it through `Default`.

The [Tile IR 13.3 types](https://docs.nvidia.com/cuda/tile-ir/13.3/sections/types.html) and
[operations](https://docs.nvidia.com/cuda/tile-ir/13.3/sections/operations.html) informed the stride,
packing, alignment and arithmetic review. TensorQuay's stricter model geometry and KV policies
come from TEST-PLAN. No inference-engine source was copied.

## Independent results

| Check | Result |
|---|---|
| Host acceptance, debug and release | 44 tests pass in each profile |
| Valid SMLA combinations | 256 combinations of heads, host plans, index counts and row layouts |
| Valid MoE combinations | 32 combinations of batch size, shard size and format |
| Deterministic boundary sweep | 640 cases across five Philox seeds, using an independent u128 span oracle |
| `tq-core` LLVM coverage through nextest | 295/295 lines, 456/456 regions, 22/22 functions |
| Whole Rust workspace | 51 independent tests pass; `tq-testkit` retains 54/54 lines, 102/102 regions, 9/9 functions |
| External Rust consumer probes | Seven pass: public API control and six privacy/lifetime/default rejection probes |
| All root Python harness checks | 29 pass, including cross-language and checker regressions |
| Deliberate implementation faults | All 19 caught after a clean positive control; every mutant compiled |
| Format, clippy, rustdoc, content, source caps and diff checks | Clean |
| Earlier accepted artifacts | All 56 recorded files unchanged byte for byte |

Coverage uses the independent suites alone, with no product-source exclusions. These percentages
cover library lines, regions and functions. Stable Rust branch coverage is not asserted; the
nightly measurement remains a later tooling task. No ordinary test skips a missing compiler or
fixture. Python references were preserved unchanged; their earlier accepted numerical results
remain in the preceding reports.

Large-span cases allocate no device memory: the FP8 pool exceeds 8 GiB, and W13 at I=2048 requires
exactly 2,415,919,104 bytes. The TP=2 I=1024 bank alone does not exercise the latter boundary.
The tests provide sufficient capacity at the forbidden 2^31 dimension so rejection cannot be
explained by a second capacity or stride error.

Two implementation defects were corrected. The prewritten padded-layout case caught an outer
stride check that used the ideal packed size instead of the declared inner stride. The lead also
independently reproduced an overlap check using allocation capacity rather than the tensor span;
this rejected valid disjoint views in a common allocation. Permanent tests now cover both unused
allocation tails and the deliberate policy of including internal padding in overlap checks.

The 19 fault probes changed the softmax scale, head/index/query limits, query totals, format group,
FP8 payload, dtype and dimension validation, KV alignment, capacity, 64-bit span arithmetic,
padded strides, overlap and adjacency, empty launches, and W13 scale rows. Each ran in a fresh
source copy and target directory under the release profile. The live product files remained
unchanged. This is targeted fault detection, not an exhaustive mutation score. The small
[result artifact](2026-09-20-host-contracts.json) records exact source hashes, replacements and counts.

## Reproduction

Environment: macOS ARM64; Rust/Cargo 1.95.0, LLVM 22.1.2, cargo-llvm-cov 0.9.1,
cargo-nextest 0.9.145; Python 3.12.4, pytest 8.4.2, Ruff 0.13.1.
Revision: uncommitted working tree based on `d7f25560d4767e72eb6542b1d9e22489f3a58a45`.
The artifact pins source and acceptance hashes. From the repository root, after tool setup:

```sh
cargo test --workspace --locked --offline
cargo test --workspace --locked --offline --release
cargo llvm-cov nextest --workspace --locked --offline \
  --test host_acceptance --test philox_acceptance \
  --json --output-path target/host-coverage.json \
  --fail-under-lines 100 --fail-under-regions 100 --fail-under-functions 100
uv run --project reference --locked --offline pytest tests/ -W error
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked --offline
python3 tools/git-guard.py audit
```

For each recorded fault, apply its one exact replacement in an isolated copy, run
`cargo test -p tq-core --test host_acceptance --release --locked --offline`, then discard that copy.
Use a fresh target directory for each probe and first verify the unmodified control passes.

## Limits and next step

An accepted parameter object proves metadata consistency, not device allocation ownership,
lifetime, device identity, physical aliasing or memory fit. The future binding must supply live
owned buffers and enforce its actual vector alignment and read-only alias assumptions. It must
also derive GPU sequence IDs from the checked host plan and verify that relationship in debug.
IDs, page mappings, duplicate indices, masking, checkpoint row contents and numerical scales
still require device and numerical tests. Empty descriptors use canonical positive strides;
adapters must normalise a foreign library's representation before constructing them.

This supplies host evidence for SMLA-E-011/012/020, MOE-E-008/009, and the span calculations for
SMLA-E-007/MOE-E-012. Their GPU portions remain pending. No kernel tile, swizzle, graph replay,
sanitizer run or baseline comparison has been validated. The metadata permits natural alignment
and read-only input aliasing; that is not a promise that every future lowering supports them.

Next: finish the bounded step 0.4 dependency, length, manifest and CPU CI checks. SPEC-001/002
model configuration parsing remains a separate pending slice. Step 0.5 compiler and kernel work
needs its own contract and evidence. Nothing was committed or pushed by this acceptance.
