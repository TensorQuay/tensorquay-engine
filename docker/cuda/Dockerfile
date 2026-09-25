# TensorQuay CUDA compiler image. Contract: docs/CUDA-BUILD.md.
# Compiler-only. The base carries NCCL 2.30.7, below the engine's 2.31.2 floor, so this
# image is not qualified for multi-GPU inference or any performance gate.
# It needs no GPU and no driver mount: nothing here executes a kernel.
# Nothing is copied from the build context; the context exists only to be empty.

FROM rust:1.95.0-slim-bookworm@sha256:6f9e63259f12e1e599296f5ecfed2bae46de4af0ee0525dd8b89c046e236d5c5 AS rust-toolchain

FROM nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:03c372fd9c65fe7739279f8c65473b315dc61efaaffab03e1e65bc7be7aee61e AS toolchain

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo

ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo RUSTUP_TOOLCHAIN=1.95.0
ENV PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda CUDA_TOOLKIT_PATH=/usr/local/cuda
ENV CUTILE_TILEIRAS_PATH=/usr/local/cuda/bin/tileiras CUTILE_BYTECODE_VERSION=13.3
ENV LIBCLANG_PATH=/usr/lib/llvm-18/lib

# git fetches the pinned upstream; libclang-18-dev backs bindgen in cuda-bindings;
# build-essential is already pulled in by cuda-nvcc but is named so the need is not implicit.
# cuda-tileiras-13-3 should already be present: this base installs cuda-minimal-build-13-3,
# which depends on cuda-compiler-13-3, which depends on cuda-tileiras-13-3 (>= 13.3.36) in
# both 13.3.0-1 and 13.3.1-1. It is named at an exact version anyway, so the pin is enforced
# whether or not the base supplies it. Other package versions are recorded rather than
# pinned, because Ubuntu and NVIDIA apt resolution is not snapshot-pinned in this increment.
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cuda-tileiras-13-3=13.3.36-1 \
      git \
      libclang-18-dev \
 && rm -rf /var/lib/apt/lists/*

# Every tool the later stage depends on is proved present and runnable here, so a missing
# one fails this build rather than surfacing as a confusing compile error.
RUN mkdir -p /opt/tq-build-evidence \
 && test -r "$CUDA_HOME/include/cuda.h" \
 && test -r "$CUDA_HOME/include/curand.h" \
 && test -x "$CUTILE_TILEIRAS_PATH" \
 && rustc --version 2>&1 | tee /opt/tq-build-evidence/rustc-version.txt \
 && cargo --version 2>&1 | tee /opt/tq-build-evidence/cargo-version.txt \
 && "$CUTILE_TILEIRAS_PATH" --version 2>&1 | tee /opt/tq-build-evidence/tileiras-version.txt \
 && cuobjdump --version 2>&1 | tee /opt/tq-build-evidence/cuobjdump-version.txt \
 && dpkg-query -W -f='${binary:Package}\t${Version}\n' 2>&1 | sort \
      | tee /opt/tq-build-evidence/installed-packages.tsv

FROM toolchain AS compile-smoke

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

WORKDIR /opt/cutile-rs

# The immutable commit is fetched directly, so a deleted or moved v0.3.1 tag cannot change
# or break this build. Each command substitution is its own &&-chained command: writing
# test "$(git ...)" would discard git's exit status and compare an empty string instead.
# The checkout is left exactly as upstream published it.
RUN git init --quiet /opt/cutile-rs \
 && git -C /opt/cutile-rs fetch --depth 1 https://github.com/NVlabs/cutile-rs.git cdc69c13a7529552a26d9941893f047970d8e95f \
 && git -C /opt/cutile-rs checkout --detach FETCH_HEAD \
 && revision=$(git -C /opt/cutile-rs rev-parse HEAD) \
 && test "$revision" = "cdc69c13a7529552a26d9941893f047970d8e95f" \
 && worktree=$(git -C /opt/cutile-rs status --porcelain) \
 && test -z "$worktree" \
 && echo "$revision" 2>&1 | tee /opt/tq-build-evidence/cutile-commit.txt

# The inventory is checked before execution, so a renamed or deleted test cannot pass as a
# zero-test success. Only the one upstream integration test runs, on the committed lockfile.
RUN cargo test -p cutile --test compile_only --locked -- --list 2>&1 \
      | tee /opt/tq-build-evidence/compile-only-inventory.txt \
 && grep -qx 'smoke_compile_only: test' /opt/tq-build-evidence/compile-only-inventory.txt \
 && grep -qx '1 test, 0 benchmarks' /opt/tq-build-evidence/compile-only-inventory.txt \
 && cargo test -p cutile --test compile_only --locked -- --nocapture 2>&1 \
      | tee /opt/tq-build-evidence/compile-only-output.txt
