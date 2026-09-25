# CUDA build preparation evaluation, 20 Sep 2026

**Decision: the recipe and local orchestration checks are accepted for a first Linux
build. Execution evidence is incomplete.** Step 0.5, G-11/G-12 and H-006 remain open.
No image has been built and no CUDA kernel has compiled or run in this increment.

## Scope and revisions

The [contract](../CUDA-BUILD.md) pins the Linux amd64 CUDA 13.3.1 and Rust 1.95.0
images by manifest digest, and cuTile release 0.3.1 by its full commit. The lead
retrieved both registry manifests and configurations and verified their content
hashes independently. The upstream compiler API and test were inspected at that
immutable revision, rather than inferred from the current main branch.

Implementation: [Dockerfile](../../docker/cuda/Dockerfile),
[context exclusions](../../docker/cuda/.dockerignore) and
[build instructions](../../docker/cuda/README.md). The compiler image is separate
from the later inference-baseline image; this does not waive any baseline gate.
The CUDA base's NCCL 2.30.7 does not satisfy the engine's multi-GPU runtime floor.

The checkout baseline is `28377aad3a51536455a070dfca7d980c702005cd`, together with
the still-uncommitted [accepted step 0.4 tooling](2026-09-20-cpu-tooling.md).
The accepted CPU implementation, its acceptance tests, manifests, lockfiles and
hooks were preserved. Only status/navigation documents from the preceding working
tree changed. This increment adds configuration, documentation and acceptance tests;
it adds no product code, kernel, crate or workflow.

Reviewed working-tree fingerprints:

| File | SHA-256 |
|---|---|
| `docker/cuda/Dockerfile` | `03615b6408092fa1c07199371c0e234d2abd2e0e517fb317737c7f95f2150be2` |
| `docker/cuda/.dockerignore` | `939d607afc3ddf200fd56cf6a2169d6e9b8e80f16cfe0777f9e5c5dfbcb873d4` |
| `docker/cuda/README.md` | `47b4983118ee9330bda90350d9dc54fe159af97d3c311ea86f41b2ffa22b9c29` |
| `docs/CUDA-BUILD.md` | `454fd50fc2f9311a64d1206103b27f77bc3e5922e939c6868e9e2d932932b2c9` |
| `tests/test_cuda_build_acceptance.py` | `56a7c327937ef4dde6453e1127cbb5ac73521b7faf3c290952af5a93c9ec8f5d` |

## Independent results

The lead authored [the acceptance suite](../../tests/test_cuda_build_acceptance.py)
before implementation, then independently reviewed the recipe and reran the checks
on macOS ARM64 with Python 3.12.4 and pytest 8.4.2.

| Check | Result |
|---|---|
| Image pins, context isolation, toolchain selection, failure settings, upstream scope and shell syntax | 6 passed |
| Recording-tool positive control | 1 passed |
| Failed Git command, wrong revision, failed revision lookup, failed status lookup, dirty checkout | 5 rejected as required |
| Failed, empty, renamed or expanded test inventory; failing test execution | 5 rejected as required |
| Full root acceptance suite, including the new cases | 292 passed |
| Existing CPU tooling coverage | 307/307 statements and 156/156 branches, no exclusions |
| Existing CPU dependency and manifest checks | 1 declaration; all 24 clauses mapped |
| Working-tree content scan and secret scan | 127 candidates clean; no secret findings |
| Guard audit, source limits and local documentation links | Passed |

The recording tools deliberately substitute for Git and Cargo. They establish
shell control flow and diagnostic capture; they are **not** successful CUDA or
upstream-compilation evidence. No new production module was added, so there is no
new production coverage percentage. Existing Rust and reference implementation
hashes are unchanged; their accepted numerical and coverage evidence remains in
the previous reports. Those suites were not rerun merely for this recipe change.

## Findings fixed before acceptance

Two fault cases initially failed the lead's tests. Putting `git` inside a string
comparison discarded its exit status: a failing revision lookup that still printed
the expected revision, and a failing status lookup that printed nothing, both let
the build continue. The recipe now assigns each result in a separate checked
command before comparing it. Both regressions are covered independently.

The review also required fetching the immutable upstream commit directly, capturing
stderr as well as stdout, explicit amd64 build commands, and using the local image
config ID for an unpushed build. Indexing an empty registry-digest list would have
broken the evidence-extraction instructions. Outputs go under ignored `target/cuda/`.

Source review corrected two assumptions in feasibility research: release 0.3.1
already contains the public compile-only API, and NVIDIA's CUDA 13.3 compiler
package declares tileiras as a dependency. The recipe explicitly enforces the
tileiras package version and checks that the executable runs; metadata alone does
not establish its presence or usability in an actual build.

## Reproduction and next evidence

With the existing CPU development tools installed as documented in
[CONTRIBUTING](../../CONTRIBUTING.md):

```sh
uv run --project reference --locked --offline pytest tests/test_cuda_build_acceptance.py
uv run --project reference --locked --offline pytest tests \
  --cov --cov-config=tools/cpu-tools.coveragerc --cov-branch
scripts/check-deps.sh
scripts/check-manifest.sh
```

Run the [two image targets and extract their evidence](../../docker/cuda/README.md)
on a Linux x86_64 builder with sufficient free storage. Record the image identity,
installed package and tool versions, one actual passing upstream test, and measured
build-space use. Host packages are recorded but not snapshot-pinned, so bitwise
image reproducibility is not claimed.

The upstream smoke test produces IR and serialized bytecode only. Actual sm_120
cubins, specialization coverage, resource/spill evidence, GPU-test builds, GPU
correctness and performance measurements follow under their own contracts. A
successful smoke build will not complete those gates. CUDA execution is deferred
until a Linux builder is available; no paid compute or model download was used.
