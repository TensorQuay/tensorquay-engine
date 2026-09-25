# TensorQuay Engine

**Run new open models quickly on consumer and workstation NVIDIA GPUs.** TensorQuay is an open-source LLM
inference engine written in Rust, with GPU kernels in NVIDIA cuTile Rust.

- **Hardware:** RTX 5090 and RTX PRO 6000 (Blackwell, sm_120), with models sized to the available memory.
- **First deployment target:** GLM-5.3-Flash on 2 × RTX PRO 6000, starting with 8 concurrent coding agents, then
  evaluating 16. Report usable context per agent, generation speed and agent-step latency together.
- **Main priority:** support new model releases quickly by reusing working kernels and adding only what a model needs.
- **Community:** contributions to model support, correctness, performance and reproducible deployment are welcome.
  See [CONTRIBUTING.md](CONTRIBUTING.md).

**Status: Phase 0 CPU references, shared test generator and Rust host contracts accepted.**
[NVFP4, MLA attention, routed MoE and FP32 router references](reference/README.md) pass independent acceptance.
The first Rust crate, `tq-testkit`, generates the same test inputs as Python; see its
[evaluation](docs/evals/2026-09-20-philox-foundation.md). `tq-core` now validates attention and MoE metadata;
see the [host-contract evaluation](docs/evals/2026-09-20-host-contracts.md). The local
[CPU gate](docs/evals/2026-09-20-cpu-tooling.md) checks these foundations together. See the
[CPU workflow](https://github.com/TensorQuay/tensorquay-engine/actions/workflows/cpu.yml) for Linux run results. GPU kernels, model execution and the serving API are still planned; the deployment targets
above have not been demonstrated by this engine.

We build small working increments and measure them on the target hardware. More GPU layouts, host-memory tiers and
compression are later work when a concrete deployment needs them.

The planned production runtime is independent of vLLM and SGLang. They serve as external correctness and performance
references; Python and PyTorch are test tools. None is a dependency of the shipped engine.

| Document | Content |
|---|---|
| [docs/PRD.md](docs/PRD.md) | What we build and why; success metrics; pending technical decisions |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Modular design: crates, runtime, interfaces, memory, TP, model families |
| [docs/TEST-PLAN.md](docs/TEST-PLAN.md) | Kernel contracts, correctness, edge-case and performance tests, gates |
| [docs/HOST-CONTRACTS.md](docs/HOST-CONTRACTS.md) | Rust metadata API, input validation and its device-binding boundary |
| [docs/CPU-CHECKS.md](docs/CPU-CHECKS.md) | CPU tooling contract, test manifest, coverage and CI boundaries |
| [docs/CUDA-BUILD.md](docs/CUDA-BUILD.md) | Pinned compiler-image preparation; Linux execution still pending |
| [docs/DEV-PLAN.md](docs/DEV-PLAN.md) | Phase 0 kernel spike and later phases |
| [docs/DEV-GUIDELINES.md](docs/DEV-GUIDELINES.md) | Design rules, code style, coverage, CI gates |
| [docs/research/](docs/research/) | Paper reports (GLM-5, DeepSeek-V4.1-Flash), architecture research, and their impact on our design |
| [docs/reviews/](docs/reviews/) | Independent design reviews |
| [docs/evals/](docs/evals/README.md) | Reproducible acceptance reports at implementation milestones |

Licence: [Apache-2.0](LICENSE). NVIDIA, RTX and CUDA are trademarks of NVIDIA Corporation. TensorQuay is not
affiliated with or endorsed by NVIDIA. GLM-5.3-Flash is by Z.ai (MIT licence).
