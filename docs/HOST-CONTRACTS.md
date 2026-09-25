# Phase 0 Rust host contracts

Status: technical-lead approved for step 0.4, 20 Sep 2026. Contract and independent tests precede implementation.
This is the small, dependency-free `tq-core` crate already planned in [ARCHITECTURE](ARCHITECTURE.md) §2.
It follows [DEV-GUIDELINES](DEV-GUIDELINES.md); `unsafe_code` remains forbidden.

## Boundary

Validate metadata for the SMLA and MoE contracts in [TEST-PLAN](TEST-PLAN.md) §§4–5. No GPU allocation, pointer
dereference, tensor arithmetic, CUDA dependency, model registry or runtime is added. Validation uses stack data and
borrowed host metadata; it does not allocate or synchronise a device. Numeric device addresses are declarations,
not proof of allocation ownership, lifetime, device identity or physical aliasing.

The future `tq-gpu` binding must obtain these declarations from live owned buffers on the correct device, preserve
them through execution, and enforce any additional alignment required by its selected instructions. An accepted
parameter object alone is **not a safe device-memory capability**. Device-content checks and GPU execution remain
unimplemented. No numerical, graph, memory-sanitizer or performance gate is satisfied by this slice.
The binding must explicitly verify its lowering's read-only alias assumptions and any vector alignment for `q`,
`out`, `idx` and `lse`; cuTile Python's array-argument restrictions do not establish the Rust lowering's contract.

## API

Public items live in `tq_core::contracts`. Plain metadata types have public fields and derive `Clone`, `Copy`,
`Debug`, `PartialEq` (`Eq` where no floating-point field occurs). No external dependencies are needed.

```rust
pub struct Bytes(pub u64);
pub enum DType { Bf16, I32, F32, U8 }
pub struct TensorSpec<const N: usize> {
    pub shape: [u64; N],
    pub strides_bytes: [Bytes; N],
    pub dtype: DType,
    pub device_address: u64,
    pub buffer_bytes: Bytes,
}
pub enum KvFormat { Bf16, Fp8E4m3 }
pub enum Nvfp4Format { G16, G32E4m3 }
// group() -> u64; try_from_group(u64) -> Result<Self, ContractError>
pub enum Violation {
    Shape, Dtype, Dimension, Stride, Alignment, Address, Capacity, Overflow,
    QueryCount, Scale, Group, Alias,
}
pub struct ContractError { pub field: &'static str, pub kind: Violation }
```

`ContractError` implements `std::error::Error`; display is `"{field}: {kind:?}"`. Field names below are stable.
`try_from_group` accepts exactly 16 and 32; otherwise it returns `format: Group`.

```rust
pub struct SmlaSpec {
    pub q: TensorSpec<3>, pub kv: TensorSpec<3>, pub page_table: TensorSpec<2>,
    pub seq_id: TensorSpec<1>, pub q_pos: TensorSpec<1>, pub kv_len: TensorSpec<1>,
    pub idx: TensorSpec<2>, pub k_len: TensorSpec<1>, pub out: TensorSpec<3>,
    pub lse: TensorSpec<2>, pub kv_format: KvFormat, pub softmax_scale: f32,
}
pub struct MoeSpec {
    pub x: TensorSpec<2>, pub topk_ids: TensorSpec<2>, pub topk_w: TensorSpec<2>,
    pub w13: TensorSpec<3>, pub w2: TensorSpec<3>,
    pub w13_scales: TensorSpec<3>, pub w2_scales: TensorSpec<3>,
    pub w13_global: TensorSpec<1>, pub w2_global: TensorSpec<1>,
    pub y: TensorSpec<2>, pub format: Nvfp4Format,
}
// Fields of validated types are private. No Default, unchecked constructor, or mutable accessor.
// SmlaParams<'a>::new(SmlaSpec, &'a [u8]) -> Result<Self, ContractError>
// MoeParams::new(MoeSpec) -> Result<Self, ContractError>
// Both: spec() returns a COPY; should_launch() returns T != 0.
// SmlaParams: query_counts() -> &'a [u8], immutable host batch plan borrowed by new().
```

The rank parameter only describes the existing rank-1/2/3 inputs. It introduces no tensor operations or extensible
layout framework. Validated types derive `Debug`; other derives are optional and must preserve their invariants.

## Common layout and range rules

- Every declared dimension is **strictly below 2^31**. Zero dimensions are allowed where the shapes below permit
  them. This is our conservative Phase 0 index policy from DEV-GUIDELINES, not a universal CUDA dimension limit.
- Shapes and dtypes must match exactly. Dtypes describe storage; packed weights, E4M3 scale codes, and the mixed KV
  payload use `U8`. Their semantic interpretation comes from the format enum.
- Strides are positive byte counts. Innermost stride is the storage element size: 1, 2 or 4 bytes. Contiguous layout
  requires `stride[i] = stride[i+1] * max(shape[i+1], 1)`. SMLA `q`, `idx` and `out` instead allow `>=` at outer
  dimensions, with each stride divisible by the element size. Transposes, broadcasts and overlapping rows are
  unsupported. The KV rule below replaces the generic outer-stride rule.
  Producers must synthesise these canonical positive strides for empty arrays; library-reported zero strides are
  not this representation. A subview's declared address is checked directly, irrespective of allocator alignment.
- For a nonempty tensor the exact required bounding span is
  `element_bytes + sum((shape[i]-1) * stride_bytes[i])`. An empty tensor needs zero bytes. Last-row trailing padding
  is **not** required. Compute products, sums and address intervals with checked 64-bit arithmetic; overflow is an
  error, never a wrapped value. A tensor may occupy more than 2 GiB or 8 GiB without violating the dimension policy.
- When launching, each nonempty tensor needs a nonzero numeric address aligned to its element size (KV: 16 bytes),
  `buffer_bytes >= required_span`, and a representable exclusive `device_address + buffer_bytes`. No power-of-two,
  16-byte or 32-byte base alignment is imposed on other buffers beyond their natural element alignment.
- Each output's bounding address interval must be disjoint from every input and other output; read-only inputs
  may overlap. Bounding intervals conservatively include internal padding. Adjacent intervals are valid. This
  checks numeric virtual ranges only; the device owner must prevent distinct virtual mappings aliasing physical
  memory. An `Alias` error names the conflicting output (`out`, `lse` or `y`).
- Empty launch: **T=0 still validates geometry, dtype, stride and the host query plan**, then `should_launch()` is
  false. Storage address, capacity and alias checks are skipped for the whole invocation because nothing is read or
  written. Well-formed empty metadata may therefore describe absent weights and a null cache. Malformed geometry
  is still rejected. Empty tensors inside a nonempty invocation also need no address or capacity checks.
- Errors name the offending tensor field. Global errors use `query_counts`, `softmax_scale` or `format`. For a
  single invalid property the indicated kind is required; precedence between multiple invalid properties is not
  an API promise. Shape mismatch, dimension overflow, bad stride, address misalignment, null address, insufficient
  capacity and checked arithmetic overflow use their corresponding enum variants.

## SMLA layout

Derive T, H from `q.shape[0..2]`, K from `idx.shape[1]`, S and L from `page_table.shape`, and P from `kv.shape[0]`.
H is 32 or 64; D is 512; K is 0 through 2051 inclusive. S must be positive when T is positive. There is no arbitrary
64-token ceiling: T=256 remains a valid P1 stress geometry. `softmax_scale` must be exactly `0.0625f32`; reject NaN,
infinities, negative and neighbouring floats with `softmax_scale: Scale`.

| Field | Dtype | Shape | Layout |
|---|---|---|---|
| q, out | Bf16 | [T, H, 512] | Contiguous inner dimension; padded outer strides allowed |
| kv | U8 | [P, 64, payload_bytes] | Strides [64*r, r, 1]; r >= payload_bytes and divisible by 16 |
| page_table | I32 | [S, L] | Contiguous |
| seq_id, q_pos, k_len | I32 | [T] | Contiguous |
| kv_len | I32 | [S] | Contiguous |
| idx | I32 | [T, K] | Contiguous inner dimension; padded row stride allowed |
| lse | F32 | [T, H] | Contiguous |

KV payload is 1024 bytes for BF16 or 528 bytes for FP8 E4M3. FP8 bytes 0–511 hold the codes and 512–527 hold four
f32 scales, one per 128 values. **Both r=528 and r=544 are valid.** The 544-byte stride used in the gather benchmark
is a performance choice. A BF16 row touches only 32 sectors when its start is 32-byte aligned; that benchmark
footprint is not promised for every valid layout. Pages are back to back; page padding is outside this slice.
P and L may be zero for batches whose KV is empty, even with K>0 if every lane is invalid; no page may be read in
that case. K=0 still requires the future launcher
to produce zero output and negative-infinity lse. Never construct a zero-dimension device view merely because this
host representation permits empty tensors.

`query_counts` is host scheduler metadata: length exactly S, each entry 0–4, sum exactly T. Reject violations with
`query_counts: QueryCount`. Borrow it immutably so a successful constructor retains the checked plan. This proves
the **host plan**, not the distribution in GPU `seq_id`. The future batch builder must produce `seq_id` from that
plan, and a device debug check must detect a mismatch. Scalar maxima or T <= 4*S alone are insufficient evidence.
Actual IDs, page mappings, lengths, duplicate sparse indices, causality, masking and FP8 scale values remain device
content responsibilities; there is no GPU readback here.

## MoE layout

Derive T from `x.shape[0]` and I from `2 * w2.shape[2]`. E=288, hidden width=4096, top-k=8, I=1024 or 2048.
G comes from the format enum and is 16 or 32. All tensors are contiguous in this slice.

| Field | Dtype | Shape |
|---|---|---|
| x, y | Bf16 | [T, 4096] |
| topk_ids | I32 | [T, 8] |
| topk_w | F32 | [T, 8] |
| w13 | U8 | [288, 2*I, 2048] |
| w2 | U8 | [288, 4096, I/2] |
| w13_scales | U8 | [288, 2*I, 4096/G] |
| w2_scales | U8 | [288, 4096, I/G] |
| w13_global, w2_global | F32 | [288] |

These are the canonical, unswizzled reference layouts, with the lower-indexed FP4 value in the low nibble. W13
orders gate rows before up rows **within each shard** and shares one global scale per expert. Metadata cannot prove
that the checkpoint loader filled those rows correctly; MOE-C-004/006/008 still must do so. There is no invented tile
size or backend swizzle requirement: the accepted fixed shapes are all group-divisible, and an actual kernel's tile,
packing and alignment constraints must be specified and compiled before its launcher is accepted. Expert IDs,
duplicates, routing weights and numerical scale validity are device contents, not checked by this constructor.

## Independent acceptance and remaining work

`crates/tq-core/tests/host_acceptance.rs` and its small support module belong to the independent reviewer. Test valid
and invalid forms of every tensor; exact byte ends and one-byte-short storage; u64 overflow; >8 GiB KV and >2 GiB
expert banks without allocations; all formats and shard sizes; empty launches; padded SMLA layouts; scale bits;
per-sequence plans; output aliasing and adjacency; copy isolation. Use deterministic Philox inputs and a u128
arithmetic oracle for boundary sweeps, then mutation probes in isolated copies. Validate private-field enforcement
with external-consumer compile failures and positive controls. Debug and release must agree. Required per-crate LLVM
line/region/function coverage is 100%; nightly branch measurement remains a later tooling task, as in DEV-GUIDELINES.

This supplies host evidence for SMLA-E-011/012/020 and MOE-E-008/009 and the span calculations needed by
SMLA-E-007/MOE-E-012. Their GPU portions remain pending. ModelSpec parsing (SPEC-001/002), remaining step 0.4 tooling
and CI, kernel compilation and instruction-specific launch binding follow in separate increments.

Sources reviewed 20 Sep 2026: [Tile IR 13.3 types](https://docs.nvidia.com/cuda/tile-ir/13.3/sections/types.html)
for strided views and nibble order; [Tile IR 13.3 operations](https://docs.nvidia.com/cuda/tile-ir/13.3/sections/operations.html)
for natural alignment, bounds and arithmetic assumptions. These justify checked address math; the stricter shape and
KV policies are TensorQuay's existing TEST-PLAN choices. No inference-engine implementation code is copied.
