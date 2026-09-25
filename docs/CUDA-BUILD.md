# CUDA compiler image: first step 0.5 increment

Technical contract, 20 Sep 2026. This increment prepares a Linux compiler image and a
pinned upstream smoke check. It does not introduce a product kernel or change the
CPU workspace, manifest or acceptance gates. Execution evidence is required before
calling the image usable; a recipe review alone does not complete step 0.5.

## Inputs and scope

Use separate compiler and benchmark images. Building our kernels should not require
installing an inference server or its Python environment. The later benchmark image
must still contain the eligible, pinned FlashInfer and vLLM baselines required by
DEV-PLAN 0.5. Its construction, production specializations, H-006 and G-11/G-12 remain
separate increments; none is waived by this split.

The initial recipe uses complete official images rather than assembling individual
CUDA packages. Reducing image size can follow a measured working build.

| Input | Immutable Linux amd64 pin |
|---|---|
| `nvidia/cuda:13.3.1-devel-ubuntu24.04` | `sha256:03c372fd9c65fe7739279f8c65473b315dc61efaaffab03e1e65bc7be7aee61e` |
| `rust:1.95.0-slim-bookworm` | `sha256:6f9e63259f12e1e599296f5ecfed2bae46de4af0ee0525dd8b89c046e236d5c5` |
| `NVlabs/cutile-rs`, release `v0.3.1` | `cdc69c13a7529552a26d9941893f047970d8e95f` |

Both registry manifests and their configurations were retrieved and hash-verified on
20 Sep 2026. The CUDA image layers total 4,120,505,252 compressed bytes; the Rust image
layers total 314,228,904 bytes. These are download sizes, not build-space estimates.
Reserve at least 20 GiB on a Linux x86_64 builder for the first attempt and measure
the actual peak. Do not pull these images on a nearly full development machine.

The CUDA base includes NCCL 2.30.7, below the engine's required 2.31.2 minimum. This
compiler-only image is not qualified for multi-GPU inference or performance gates.

## Recipe contract

Files: `docker/cuda/Dockerfile`, `.dockerignore` and `README.md`. Build context is
`docker/cuda`, not the repository root. No repository source, credentials, private
denylist or home directory is copied into the image. No new scripts, crates or
workflow are needed in this increment.

1. Pin both official base images by their amd64 manifest digest. Copy the Rust
   toolchain from the official Rust stage. Keep a `toolchain` target and a subsequent
   `compile-smoke` target; neither requires a GPU or a driver mount.
2. Install only the required host build packages, including Git, a C compiler,
   libc headers and `libclang-18-dev` for bindgen, with
   `LIBCLANG_PATH=/usr/lib/llvm-18/lib`. Enforce `cuda-tileiras-13-3=13.3.36-1`
   explicitly whether it is already present or not. Use noninteractive package installation,
   no recommended packages, and clean package indexes in the same layer. Ubuntu
   package resolution is not yet snapshot-pinned: record the installed versions and
   the resulting image identity, rather than claiming bitwise reproducibility.
3. Select Rust 1.95.0 explicitly. Set `CUDA_HOME` and `CUDA_TOOLKIT_PATH` to
   `/usr/local/cuda`, `CUTILE_TILEIRAS_PATH` to its `bin/tileiras`, and
   `CUTILE_BYTECODE_VERSION=13.3`. Check readable `cuda.h` and `curand.h` headers.
   Record Rust, Cargo, tileiras and cuobjdump versions, and installed package versions,
   under `/opt/tq-build-evidence`. A missing tool or failed command fails the build.
4. Fetch the exact cuTile commit above into the image, verify checkout identity, and
   leave its source unchanged. Execute only the upstream `cutile` package's
   `compile_only` integration test with `--locked`. Confirm its inventory is exactly
   `smoke_compile_only` before execution, so a renamed or missing test cannot pass
   through a zero-test result. Record the inventory and actual test output. Do not
   enable all workspace members, all features or the optional LLVM-building wrapper.
5. Use shell failure propagation, including `pipefail` when recording output through
   `tee`. Capture both stdout and stderr. Check command exit status separately from
   the text it produces; a failed Git command must not become a passing string test.
   No success fallback, ignored test exit status, GPU access or publication.
6. Exclude build-context files by default. Document local checks, exact build commands,
   evidence extraction and what remains unverified. Image publication is separate.

The upstream test checks Rust-to-Tile-IR and bytecode serialization. It does **not**
assemble an sm_120 cubin, execute a GPU kernel, or establish resource usage. Version
output from tileiras is not compilation evidence. G-11 later requires actual cubins
for every manifest specialization, plus its resource checks; G-12 requires the GPU
test binaries. `cuobjdump` STACK/LOCAL fields must not be relabeled as measured spill
loads/stores without validated compiler evidence.

## Acceptance and sources

The lead's `tests/test_cuda_build_acceptance.py` checks the recipe's dependency and
build-context boundaries, shell syntax and failure propagation using recording tools.
These checks do not execute the image or establish kernel coverage. There is no new
product code to claim 100% coverage of. The
existing independent CPU coverage gates remain required and unchanged.

Actual acceptance requires a Linux build of both targets, the image identity and
recorded versions, and one passing upstream test. Save the result in an evaluation
report. Until those exist, status is **recipe prepared; execution unverified**.

- [Released compiler API](https://github.com/NVlabs/cutile-rs/blob/cdc69c13a7529552a26d9941893f047970d8e95f/cutile-compiler/src/compile_api.rs)
- [Released compile-only test](https://github.com/NVlabs/cutile-rs/blob/cdc69c13a7529552a26d9941893f047970d8e95f/cutile/tests/compile_only.rs)
- [Released CUDA binding build](https://github.com/NVlabs/cutile-rs/blob/cdc69c13a7529552a26d9941893f047970d8e95f/cuda-bindings/build.rs)
- [CUDA 13.3.1 binary utilities](https://docs.nvidia.com/cuda/archive/13.3.1/cuda-binary-utilities/index.html)
- [Official CUDA image](https://hub.docker.com/r/nvidia/cuda)
- [Official Rust image](https://hub.docker.com/_/rust)
