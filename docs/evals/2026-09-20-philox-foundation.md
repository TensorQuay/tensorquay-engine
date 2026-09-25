# Shared Philox generator and first Rust crate

Gate date: 20 Sep 2026. **Decision: accepted for RNG-001…005 and Phase 0 step 0.3.**
Rust `tq-testkit` and the Python reference produce the same frozen Philox4x32-10 stream.
The minimal workspace starts step 0.4; host contracts, broader tooling and CPU CI remain next.
No GPU kernel, inference speed, model quality or concurrency result follows from this gate.

## Contract and implementation

The lead froze the [contract](../../reference/contracts/philox.md), fixtures and acceptance tests
before implementation. The oracle/manifest controls passed; Python acceptance then failed because
the module was absent, and the Rust command failed because the workspace did not yet exist.
The developer implemented the public APIs independently and did not change the lead's tests.

- Primitive: Philox4x32-10, ten rounds, counter `[u32; 4]`, key `[u32; 2]`.
- Stream v1: seed supplies the key; block index and numeric stream ID occupy separate counter
  halves. Offsets count words, including unaligned starts and the last legal u64 word.
- A nonempty range that exceeds u64 fails before any Rust output is written. Empty requests
  remain valid at the maximum offset. Request partition and order do not affect results.
- Uniform conversion is exactly `(word >> 8) * 2^-24`, compared by float32 bits.
- Rust is `no_std`, forbids unsafe code, allocates nothing and has no external dependencies.
  Both implementations compute each distinct four-word block once per request. There are no
  placeholder runtime crates or production sampler interfaces.

Source oracle: original
[Random123 v1.14.0](https://github.com/DEShawResearch/random123/tree/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13),
commit `726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13`. Its header and official known-answer file
are SHA-256-pinned. A separately compiled C++ API caller executes that original implementation;
its source checkout stays outside this repository. No upstream algorithm source is copied here.
[NumPy's built-in Philox](https://numpy.org/doc/2.0/reference/random/bit_generators/philox.html)
uses 4x64 and different seeding, so it is not this stream's oracle.

The [fixture manifest](../../tests/fixtures/philox-v1/manifest.json) records five seeds, explicit
stream assignments and canonical little-endian input/output hashes. Its small TSVs contain:
199 raw blocks, including all three official ten-round known answers and individual high-bit
probes; 160 stream requests, including carry, overlap boundaries and empty requests; and 96
uniform conversions. Total fixture size is about 42 KB. The expected blocks came from the
original upstream code and agree with a separately written 16-bit-limb arithmetic oracle.
Inverse-round checks recover the original counters. Uniform expected bits come from integer
significand/exponent arithmetic, checked separately by exact power-of-two conversion.

## Independent results

| Check | Result |
|---|---|
| Python Philox acceptance alone | 581 passed; 62 statements and 16 branches, 100% |
| Rust acceptance alone, debug and release | Seven tests pass in each profile; fixtures checked inside those tests |
| Rust LLVM coverage through nextest | 54/54 library lines, 102/102 regions, 9/9 functions |
| Execute Rust against Python and frozen outputs | Two tests pass: all 455 rows in both debug and release |
| Fresh original Random123 execution | All 199 block cases and 160 stream requests agree exactly |
| Independent checker corruption regressions | 20 passed |
| Additional checker/source/protocol probes | 12 rejected cases and two compiled positive controls pass |
| Deliberate implementation mutations | All 18 caught: nine Python and nine Rust |
| All independent Python reference suites | 1,263 passed; 470 statements and 150 branches, 100% |
| Full Python suite, warnings treated as errors | 1,522 passed; 100% package lines and branches |
| Formatting, lints, documentation, source caps, guard and diff checks | Clean |
| Earlier accepted references, fixtures and evaluation reports | All 36 recorded files unchanged byte for byte |

Coverage uses the independent suites alone and no product-source exclusions. Rust's reported
scope is library lines, regions and functions; stable Rust branch coverage is not asserted.
Python coverage covers `reference/src/tq_reference`. Neither percentage measures external code,
the upstream-checker script, test harnesses or a future runtime. Checker tests are separate
behavioral evidence. No ordinary test silently skips a missing fixture, compiler or other tool.

The mutations changed rounds, key scheduling, output lanes, counter carry, stream high bits,
range rejection and uniform conversion. Further Python faults accepted booleans or lost the
zero-dimensional ndarray result; Rust faults wrote before rejection or used the wrong initial
lane. Each mutation ran against the unmodified acceptance suite in a separate source copy. The
mutated-source controls passed before faults were introduced, and actual library files stayed
unchanged. This is targeted fault detection, not an exhaustive mutation score.

Review simplified the Rust fill loop and removed an unreachable conversion-error path, rather
than excluding it from coverage. Python initially recomputed a block for every requested word;
it now computes distinct blocks, flattens the lanes and slices the requested words.

The lead also demonstrated an upstream-checker false pass after removing all cases while
keeping the original manifest. The checker now verifies required manifest fields, file hashes,
case counts, source-hash membership and row structure before compiling. Permanent regressions
cover missing/empty/rehashed corrupt data and redirected metadata. Additional probes rejected
an actual dirty transitive header, an untracked source file, a wrong source hash, a simulated
wrong checkout revision, a changed expected word despite a matching file hash, malformed output
and malformed caller input. The original checkout remained clean. A final formatting correction
used the repository's explicit Ruff configuration and preserved the complete Python syntax tree.

## Reproduction

Environment: macOS ARM64; Python 3.12.4, NumPy 2.3.3, pytest 8.4.2, pytest-cov 7.0.0,
Ruff 0.13.1; Rust 1.95.0 (`59807616e`, LLVM 22.1.2), Cargo 1.95.0 (`f2d3ce0bd`),
cargo-llvm-cov 0.9.1, cargo-nextest 0.9.145, Apple clang 17.0.0.
Revision: uncommitted working tree based on `d7f25560d4767e72eb6542b1d9e22489f3a58a45`.
The workspace pins Rust and declares its LLVM tools, rustfmt and clippy components.
Install the two pinned coverage tools before the coverage command; official installation
instructions are in [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) and
[nextest](https://nexte.st/docs/installation/pre-built-binaries/).

From the repository root, after Python/toolchain setup:

```sh
cargo test --workspace --locked --offline
cargo test --workspace --locked --offline --release
cargo llvm-cov nextest --workspace --locked --offline --test philox_acceptance \
  --json --output-path target/philox-coverage.json \
  --fail-under-lines 100 --fail-under-regions 100 --fail-under-functions 100
uv run --project reference --locked --offline pytest tests/test_philox_cross_language.py \
  tests/test_philox_checker.py -W error
```

The [contribution guide](../../CONTRIBUTING.md#running-the-current-tests) gives formatting,
clippy, documentation and independent/full Python coverage commands. Fresh upstream verification
requires a separate clean checkout and C++ compiler. For example, from the repository root:

```sh
git clone --depth 1 --branch v1.14.0 https://github.com/DEShawResearch/random123 ../random123
python3 tools/check-philox-upstream.py --source ../random123
```

The checker enforces the exact revision and hashes above, so another checkout revision fails.
It never downloads or rewrites fixtures. It validates uniform-file structure; the exact uniform
math is checked by the independent integer oracle and cross-language tests, not attributed to
an upstream wrapper. Ordinary tests use the stored data and run offline after setup.

This gate covers test infrastructure on the recorded host. It does not establish cross-platform
execution on untested hosts, cryptographic suitability, a production sampler, GPU correctness,
or superiority over vLLM/SGLang. Next is `tq-core` host validation and the remaining step 0.4
checks, using small approved contracts and independent tests.

## Reviewed artifact snapshot

SHA-256, paths relative to the repository root. The fixture manifest also hashes each TSV and
its canonical input/output bytes. The contract's acceptance status was updated after review;
its API, arithmetic, thresholds and pre-implementation tests stayed unchanged.

```text
1f4846e107a9f19caee3ea731141c085b47b3667750e50fc117d1f74a211e6f7  Cargo.toml
a4de6600dd0a177433c5c3005bc2510f93531e61af607ec8d7071c2953b7374b  Cargo.lock
d08727f290757f5f96277d71dcfb88a3f790be9141288143cbdb5c5569d47659  rust-toolchain.toml
8061ea763de633b75d9f50dc499cdde3aa35a3d6b7d42171182db4a9f6097c74  crates/tq-testkit/Cargo.toml
1725d53b9e5a618b625f64ad82451bb02af848ddc471f51450f9380360d27f36  crates/tq-testkit/src/lib.rs
48f59a5db890bbea22ae4304e37bdc17d7ff4bc3d17e3ede88a836eb9b9914a0  crates/tq-testkit/src/philox.rs
39d8a64ba572b38f4e21892bf5f9f0d45bdc97fd3617ee682422830239286380  reference/contracts/philox.md
85918a5906495766bef3e2e2da2a71f43cdf1339e700cfeb70c4731e10c58b67  reference/src/tq_reference/philox.py
86f224bc383a63cba45785649bc72d3f43044a72d0e12ce17046f9746107f71e  reference/tests/test_philox_acceptance.py
f3f1cb93b9eef38d83219c22e7728c4666a14017fe55546d90df6601de46f210  crates/tq-testkit/tests/philox_acceptance.rs
5c8f2f57fcd07db0673140cb7b885f363848651a4a9b3f0afe3541bda79cfec1  crates/tq-testkit/examples/philox_bridge.rs
8f2ff5fe5cccab06eb7efcf18c5e2662a1b7fd1cdbacce1992a1b1f801f43adf  tests/test_philox_cross_language.py
b077578c7b032e248d774402b3516c094d756012a73a1bb35dc8a0e151476a58  tests/test_philox_checker.py
b9a07429b0c6a1bbc94c08590dc900b5cc2b95c1b9de0cae03ea364b8e347f6d  tools/check-philox-upstream.py
842df29c49c35b338da52ccab24e1d3779b656f55367ee856c883103488b1fc6  tests/fixtures/philox-v1/manifest.json
```
