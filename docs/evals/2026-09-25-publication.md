# Public foundation import: CPU acceptance

Date: 25 Sep 2026. Decision: the CPU foundation passes local publication checks. This is an early source release,
not acceptance of a serving engine. CUDA compilation, GPU kernels, model quality and agent performance remain pending.
Hosted Linux evidence is available in the [CPU workflow](https://github.com/TensorQuay/tensorquay-engine/actions/workflows/cpu.yml).

## Scope

The public import includes the numerical references, Rust host contracts, independent acceptance tests, build recipes
and technical documentation. Internal operational material and prior private Git history are excluded. The original
source and history were preserved privately before preparing this import. Research and design reviews that omitted
operational notes are marked as public technical extracts.

Accepted runtime/reference sources, numerical fixtures and numerical/host acceptance tests are unchanged. The audit
suite now derives the original host-test revision from the file's first addition in the public history, then verifies
its recorded SHA-256. It no longer requires a commit from unpublished history. The exact secret-scanner exception and
its adversarial tests are retained unchanged. Guard scripts and hooks are unchanged.

## Reproduction and results

Environment: macOS ARM64, Python 3.12.4, Rust/Cargo 1.95.0. Tool and dependency pins are listed in CONTRIBUTING.md,
`rust-toolchain.toml`, `reference/uv.lock` and the workflow. Install the guardrails and configure the private local
identifier denylist as described in AGENTS.md, then run:

```sh
uv sync --project reference --locked --all-groups
./scripts/check-cpu.sh
```

| Check | Result |
|---|---|
| Independent Python reference suites | 1,263 passed; subset of the full reference suite |
| Full Python reference suite | 1,522 passed; 470 statements and 150 branches, 100% covered |
| Root acceptance suite | 292 passed; includes live scanner checks and failure injection |
| CPU checker and collection hook | 307 statements and 156 branches, 100% covered |
| Rust acceptance | 51 passed; debug, release and coverage runs pass |
| `tq-core` library coverage | 295 lines, 456 regions, 22 functions; all covered |
| `tq-testkit` library coverage | 54 lines, 102 regions, 9 functions; all covered |
| Manifest | 24 CPU clauses resolve to collected, runnable tests |
| Dependencies | One workspace dev dependency; no external Rust dependencies |
| Formatting, lints, source lengths, rustdoc, workflow syntax | Pass |
| Strict identifier guard and Gitleaks history/tree scans | No findings |

The online dependency audit refreshed RustSec to `593df8c1b5ed0bcde9dddadfeeead776fa514ff8` and passed without advisory
ignores. Stable Rust line/region/function coverage is recorded; Rust branch coverage is not claimed.

## Publication boundary

The public history begins with the reviewed import. It contains no private handover, credentials, model weights,
binary downloads, local environments or build caches. Ignore rules, hooks, the private identifier denylist and the
independent secret scanner provide complementary checks; none alone proves the absence of sensitive information.
Document links and the complete candidate-file inventory were checked before publication.

The CPU suites do not demonstrate CUDA compiler execution, GPU performance, a usable context window or concurrency.
G-11/G-12, the hardware gates, EQV-001, QUAL-S and WL-001 remain open at their specified phases. Historical reports
retain their own dates and results; this rerun does not claim to regenerate every external PyTorch oracle fixture.

The hosted workflow installs Gitleaks and actionlint directly from their pinned official release archives, with
SHA-256 checks before extraction. They are Go tools and are not supported by the Rust tool installer. Audit-tool versions
and all acceptance thresholds remain unchanged.

The publication environment pins pytest 9.0.3 to address
[GHSA-6w46-j5rx-g56g](https://github.com/advisories/GHSA-6w46-j5rx-g56g), a temporary-directory handling issue in
earlier pytest releases. This changes a development dependency only. The complete CPU gate passed again with this
version at the test counts and coverage shown above. Earlier reports retain their
original dependency versions. NumPy, numerical sources, fixtures and test assertions are unchanged.
