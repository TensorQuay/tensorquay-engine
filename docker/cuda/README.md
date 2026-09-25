# CUDA compiler image

Status: **recipe prepared; execution unverified.** This image has never been built. Every
version, digest and dependency in it comes from registry and archive metadata, not a build.
Nothing here is evidence that cuTile compiles on Linux; that is the open gate. Contract,
pins and rationale: [docs/CUDA-BUILD.md](../../docs/CUDA-BUILD.md), not repeated here.

## What this is, and what it is not

It prepares a Linux x86_64 image that compiles cuTile Rust kernels to Tile IR and bytecode,
and runs one pinned upstream test end to end. It is **compiler-only**: the base carries NCCL
2.30.7, below the engine's 2.31.2 floor, so it is not qualified for multi-GPU inference or
any performance gate, and it needs no GPU or driver mount because nothing executes a kernel.

When the build runs it will prove Rust-to-Tile-IR and bytecode serialization, and nothing
more: no sm_120 cubin, no kernel execution, no resource usage. `tileiras --version` is a
tool inventory, not compilation evidence. G-11 separately requires real cubins for every
manifest specialization plus its resource checks, and G-12 the GPU test binaries. `cuobjdump`
STACK/LOCAL must not be relabelled as measured spill loads and stores without validated
compiler evidence.

## Building

The context is this directory and `.dockerignore` excludes all of it, so no repository
source, credential, denylist or home directory enters the image.

```sh
docker build --platform linux/amd64 --target toolchain \
  -t tq-cuda-toolchain:local docker/cuda
docker build --platform linux/amd64 --target compile-smoke \
  -t tq-cuda-compile-smoke:local docker/cuda
```

Reserve at least 20 GiB on the builder and measure the actual peak. The download alone is
about 4.1 GB for the CUDA base plus 314 MB for the Rust stage. **Do not build this on a
nearly full machine.**

## Extracting the evidence

The build writes `/opt/tq-build-evidence`: rustc, cargo, tileiras and cuobjdump versions,
`installed-packages.tsv`, `cutile-commit.txt`, and the compile-only inventory and output.

```sh
mkdir -p target/cuda
id=$(docker create tq-cuda-compile-smoke:local)
docker cp "$id:/opt/tq-build-evidence" target/cuda/tq-build-evidence
docker rm "$id"
docker image inspect --format '{{.Id}}' tq-cuda-compile-smoke:local
```

`target/cuda/` is already git-ignored. `.Id` is the **local** image config digest, available
before publication; it is not interchangeable with a registry manifest digest.
Record a registry digest separately if the image is ever pushed.

## How failure is forced

Both stages run under `/bin/bash -euo pipefail`, so an unset variable, a failed command or a
failure anywhere in a `tee` pipeline stops the build. There is no fallback and no ignored
exit status. The upstream revision and worktree cleanliness are each assigned to a variable before being
tested. Written as `test "$(git ...)" = ...` the substitution's exit status is discarded, so
a `git` invocation that failed while still printing the expected revision would pass the
comparison unnoticed. The lead's failure-injection tests cover that case. The toolchain
stage checks `cuda.h` and `curand.h` are readable — they are headers, not programs — and
that `tileiras` is executable, before the next stage depends on them.

The smoke stage requires the inventory to be exactly `smoke_compile_only` and
`1 test, 0 benchmarks` before the test body runs. `--list` is not a static read: it compiles
and runs the harness to enumerate its tests, so the check lands after the harness builds and
before the test executes. A renamed, deleted or added test cannot pass as a zero-test
success. Only `-p cutile --test compile_only` is built, with `--locked`; all workspace
members, all features and the optional LLVM-building wrapper crate are not enabled.

## Local checks

Configuration only; these neither build nor run the image. The suite covers image and
context boundaries, `RUN` shell syntax, and failure propagation through substitute tools:

```sh
uv run --project reference --locked --offline pytest tests/test_cuda_build_acceptance.py
```

## What remains unverified

- The image has never been built. No target has ever run.
- Whether `libclang-18-dev` with `LIBCLANG_PATH=/usr/lib/llvm-18/lib` satisfies bindgen, and
  whether that package is reachable from the base image's enabled apt components.
- Whether the Rust toolchain from Debian bookworm (glibc 2.36) runs unmodified on Ubuntu
  24.04 (glibc 2.39). The usual direction, but an assumption.
- Peak build disk, build duration and final image size are unmeasured.
- No cubin, kernel execution, resource measurement, or G-11/G-12 claim. The benchmark image
  with the pinned FlashInfer and vLLM baselines is a separate increment, not started.
