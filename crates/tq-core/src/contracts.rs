//! Validated host metadata for the Phase 0 SMLA and `MoE` kernel contracts.
//!
//! Contract: `docs/HOST-CONTRACTS.md`. Everything here is metadata. No device memory is allocated,
//! no address is dereferenced and nothing is read back from a GPU, so an accepted parameter object
//! is a checked description, not a safe device-memory capability. Device content, such as expert
//! ids, page mappings, sparse indices and scale values, stays the device's responsibility.
//!
//! All address and span arithmetic is checked 64-bit; a computation that would wrap is an error.

use core::error::Error;
use core::fmt;

/// Dimensions stay strictly below this, the Phase 0 index policy.
const DIMENSION_LIMIT: u64 = 1 << 31;
/// Rows per paged KV block.
const PAGE_ROWS: u64 = 64;
/// Required address and stride granularity for the paged KV pool.
const KV_GRANULARITY: u64 = 16;
/// `256^-0.5`, the GLM softmax scale, exact in `f32`.
const SOFTMAX_SCALE: f32 = 0.0625;
/// Experts in the GLM `MoE` layer.
const EXPERTS: u64 = 288;
/// Model hidden width.
const HIDDEN: u64 = 4096;
/// Routed experts per token.
const TOP_K: u64 = 8;
/// Largest sparse-index row capacity.
const MAX_INDICES: u64 = 2051;
/// Largest query tokens per sequence, one plus three MTP drafts.
const MAX_QUERIES_PER_SEQUENCE: u8 = 4;

/// This crate supports 32- and 64-bit pointer widths, so a slice length always fits in `u64`.
const _: () = assert!(usize::BITS <= u64::BITS);

/// A byte count or byte stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bytes(pub u64);

/// Storage element type. The semantic interpretation comes from the format enums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// Two-byte bfloat16.
    Bf16,
    /// Four-byte signed integer.
    I32,
    /// Four-byte IEEE-754 binary32.
    F32,
    /// One raw byte: packed weights, E4M3 scale codes, or the mixed KV payload.
    U8,
}

impl DType {
    /// Bytes per stored element.
    const fn size(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::Bf16 => 2,
            Self::I32 | Self::F32 => 4,
        }
    }
}

/// Why a contract was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Violation {
    /// A shape disagrees with the contract or with a dimension derived from another tensor.
    Shape,
    /// A storage dtype disagrees with the contract.
    Dtype,
    /// A dimension reaches the index limit.
    Dimension,
    /// A byte stride is not the required, or a permitted padded, value.
    Stride,
    /// An address is not aligned for its element, or for the KV rule.
    Alignment,
    /// A required address is null.
    Address,
    /// The declared buffer is smaller than the span the tensor must cover.
    Capacity,
    /// Checked 64-bit arithmetic would wrap.
    Overflow,
    /// The host query plan is not a valid per-sequence partition of the tokens.
    QueryCount,
    /// The softmax scale is not the exact model constant.
    Scale,
    /// A group size is neither 16 nor 32.
    Group,
    /// An output's address range meets another declared range.
    Alias,
}

/// A rejected contract, naming the offending field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContractError {
    /// The stable field name.
    pub field: &'static str,
    /// What was wrong with it.
    pub kind: Violation,
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {:?}", self.field, self.kind)
    }
}

impl Error for ContractError {}

/// Declared metadata for one device tensor.
///
/// `device_address` and `buffer_bytes` are numeric declarations. They prove nothing about
/// allocation ownership, lifetime, device identity or physical aliasing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorSpec<const N: usize> {
    /// Extent of each dimension.
    pub shape: [u64; N],
    /// Byte stride of each dimension.
    pub strides_bytes: [Bytes; N],
    /// Storage element type.
    pub dtype: DType,
    /// Numeric device address of the first element.
    pub device_address: u64,
    /// Bytes the caller declares reachable from `device_address`.
    pub buffer_bytes: Bytes,
}

/// Payload format of one paged KV row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFormat {
    /// 512 bfloat16 latent values, 1024 bytes.
    Bf16,
    /// 512 E4M3 codes then four f32 tile scales, 528 bytes.
    Fp8E4m3,
}

impl KvFormat {
    /// Bytes in one row's payload.
    const fn payload_bytes(self) -> u64 {
        match self {
            Self::Bf16 => 1024,
            Self::Fp8E4m3 => 528,
        }
    }
}

/// NVFP4 weight format. The two are distinct formats, not a tunable group size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Format {
    /// Standard NVFP4: one E4M3 scale per 16 values.
    G16,
    /// The TensorQuay/ModelOpt variant: one E4M3 scale per 32 values.
    G32E4m3,
}

impl Nvfp4Format {
    /// Values sharing one E4M3 scale.
    #[must_use]
    pub const fn group(self) -> u64 {
        match self {
            Self::G16 => 16,
            Self::G32E4m3 => 32,
        }
    }

    /// The format with this group size.
    ///
    /// # Errors
    ///
    /// Returns `format: Group` for any size other than 16 or 32.
    pub const fn try_from_group(group: u64) -> Result<Self, ContractError> {
        match group {
            16 => Ok(Self::G16),
            32 => Ok(Self::G32E4m3),
            _ => Err(ContractError {
                field: "format",
                kind: Violation::Group,
            }),
        }
    }
}

/// Declared metadata for one sparse-MLA decode launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmlaSpec {
    /// Absorbed queries, `[T, H, 512]`.
    pub q: TensorSpec<3>,
    /// Paged latent pool, `[P, 64, payload]`.
    pub kv: TensorSpec<3>,
    /// Page indirection, `[S, L]`.
    pub page_table: TensorSpec<2>,
    /// Sequence of each query token, `[T]`.
    pub seq_id: TensorSpec<1>,
    /// Position of each query token, `[T]`.
    pub q_pos: TensorSpec<1>,
    /// Cached length of each sequence, `[S]`.
    pub kv_len: TensorSpec<1>,
    /// Sparse indices, `[T, K]`.
    pub idx: TensorSpec<2>,
    /// Live index count of each query token, `[T]`.
    pub k_len: TensorSpec<1>,
    /// Attention output, `[T, H, 512]`.
    pub out: TensorSpec<3>,
    /// Natural log-sum-exp, `[T, H]`.
    pub lse: TensorSpec<2>,
    /// Row payload format of the pool.
    pub kv_format: KvFormat,
    /// Softmax scale, which must be the exact model constant.
    pub softmax_scale: f32,
}

/// Declared metadata for one grouped-expert launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeSpec {
    /// Token activations, `[T, 4096]`.
    pub x: TensorSpec<2>,
    /// Routed expert ids, `[T, 8]`.
    pub topk_ids: TensorSpec<2>,
    /// Routing weights, `[T, 8]`.
    pub topk_w: TensorSpec<2>,
    /// Packed gate and up weights, `[288, 2I, 2048]`.
    pub w13: TensorSpec<3>,
    /// Packed output weights, `[288, 4096, I/2]`.
    pub w2: TensorSpec<3>,
    /// E4M3 scales for `w13`, `[288, 2I, 4096/G]`.
    pub w13_scales: TensorSpec<3>,
    /// E4M3 scales for `w2`, `[288, 4096, I/G]`.
    pub w2_scales: TensorSpec<3>,
    /// One f32 global scale per expert for `w13`, shared by gate and up.
    pub w13_global: TensorSpec<1>,
    /// One f32 global scale per expert for `w2`.
    pub w2_global: TensorSpec<1>,
    /// Layer output, `[T, 4096]`.
    pub y: TensorSpec<2>,
    /// Weight format, which fixes the group size.
    pub format: Nvfp4Format,
}

/// How a tensor's byte strides must relate to its shape.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// Exactly packed: each stride is the product of the extents inside it.
    Packed,
    /// Packed innermost; outer strides at least packed and a multiple of the element.
    PaddedOuter,
    /// Strides were validated by the caller, as the paged KV rows are.
    Checked,
}

/// A declared address range, used only to keep outputs apart from other buffers.
#[derive(Clone, Copy)]
struct Extent {
    field: &'static str,
    start: u64,
    end: u64,
}

const fn fail<T>(field: &'static str, kind: Violation) -> Result<T, ContractError> {
    Err(ContractError { field, kind })
}

/// Checks one tensor and returns its address range, or `None` when no storage is touched.
fn check<const N: usize>(
    field: &'static str,
    tensor: &TensorSpec<N>,
    shape: [u64; N],
    dtype: DType,
    layout: Layout,
    launching: bool,
) -> Result<Option<Extent>, ContractError> {
    if tensor.dtype != dtype {
        return fail(field, Violation::Dtype);
    }
    if tensor.shape != shape {
        return fail(field, Violation::Shape);
    }
    if tensor.shape.iter().any(|&dim| dim >= DIMENSION_LIMIT) {
        return fail(field, Violation::Dimension);
    }
    let element = dtype.size();
    if layout != Layout::Checked {
        check_strides(field, tensor, layout, element)?;
    }
    if !launching || tensor.shape.contains(&0) {
        return Ok(None);
    }

    let mut span = element;
    for (&dim, stride) in tensor.shape.iter().zip(tensor.strides_bytes) {
        let Some(reach) = (dim - 1)
            .checked_mul(stride.0)
            .and_then(|r| span.checked_add(r))
        else {
            return fail(field, Violation::Overflow);
        };
        span = reach;
    }
    if tensor.device_address == 0 {
        return fail(field, Violation::Address);
    }
    let granularity = if layout == Layout::Checked {
        KV_GRANULARITY
    } else {
        element
    };
    if !tensor.device_address.is_multiple_of(granularity) {
        return fail(field, Violation::Alignment);
    }
    if tensor.buffer_bytes.0 < span {
        return fail(field, Violation::Capacity);
    }
    if tensor
        .device_address
        .checked_add(tensor.buffer_bytes.0)
        .is_none()
    {
        return fail(field, Violation::Overflow);
    }
    // Aliasing is about the bytes the tensor covers, not the capacity behind it, so unused tail
    // never collides. The sum cannot wrap: span is at most buffer_bytes, whose end just fitted.
    Ok(Some(Extent {
        field,
        start: tensor.device_address,
        end: tensor.device_address + span,
    }))
}

/// The innermost stride is the element size; each outer stride spans the dimension inside it.
///
/// The span uses the neighbour's declared stride, not an idealised packed size, so padding at one
/// level pushes every level outside it along by the same amount.
fn check_strides<const N: usize>(
    field: &'static str,
    tensor: &TensorSpec<N>,
    layout: Layout,
    element: u64,
) -> Result<(), ContractError> {
    if tensor.strides_bytes[N - 1].0 != element {
        return fail(field, Violation::Stride);
    }
    for axis in (0..N - 1).rev() {
        let inner = tensor.strides_bytes[axis + 1].0;
        let Some(needed) = inner.checked_mul(tensor.shape[axis + 1].max(1)) else {
            return fail(field, Violation::Overflow);
        };
        let stride = tensor.strides_bytes[axis].0;
        let fits = if layout == Layout::Packed {
            stride == needed
        } else {
            stride >= needed && stride.is_multiple_of(element)
        };
        if !fits {
            return fail(field, Violation::Stride);
        }
    }
    Ok(())
}

/// Paged KV rows are `[64 * r, r, 1]` with `r` at least the payload and a multiple of 16.
fn check_kv_strides(tensor: &TensorSpec<3>, payload: u64) -> Result<(), ContractError> {
    let row = tensor.strides_bytes[1].0;
    if tensor.strides_bytes[2].0 != 1 || row < payload || !row.is_multiple_of(KV_GRANULARITY) {
        return fail("kv", Violation::Stride);
    }
    let Some(page) = PAGE_ROWS.checked_mul(row) else {
        return fail("kv", Violation::Overflow);
    };
    if tensor.strides_bytes[0].0 != page {
        return fail("kv", Violation::Stride);
    }
    Ok(())
}

/// Checks an exactly packed tensor.
fn packed<const N: usize>(
    field: &'static str,
    tensor: &TensorSpec<N>,
    shape: [u64; N],
    dtype: DType,
    live: bool,
) -> Result<Option<Extent>, ContractError> {
    check(field, tensor, shape, dtype, Layout::Packed, live)
}

/// Checks a tensor whose outer strides may carry padding.
fn padded<const N: usize>(
    field: &'static str,
    tensor: &TensorSpec<N>,
    shape: [u64; N],
    dtype: DType,
    live: bool,
) -> Result<Option<Extent>, ContractError> {
    check(field, tensor, shape, dtype, Layout::PaddedOuter, live)
}

/// Checks the paged pool, whose strides were validated by [`check_kv_strides`].
fn rows(
    field: &'static str,
    tensor: &TensorSpec<3>,
    shape: [u64; 3],
    dtype: DType,
    live: bool,
) -> Result<Option<Extent>, ContractError> {
    check(field, tensor, shape, dtype, Layout::Checked, live)
}

/// Each output's range must be disjoint from every input and from every earlier output.
fn check_aliases(
    inputs: &[Option<Extent>],
    outputs: &[Option<Extent>],
) -> Result<(), ContractError> {
    for (position, output) in outputs.iter().enumerate() {
        let Some(out) = output else { continue };
        let earlier = outputs.iter().take(position);
        for other in inputs.iter().chain(earlier).flatten() {
            if out.start < other.end && other.start < out.end {
                return fail(out.field, Violation::Alias);
            }
        }
    }
    Ok(())
}

/// Validated sparse-MLA decode parameters, holding the host batch plan it was checked against.
#[derive(Clone, Copy, Debug)]
pub struct SmlaParams<'a> {
    spec: SmlaSpec,
    query_counts: &'a [u8],
}

impl<'a> SmlaParams<'a> {
    /// Validates one decode launch against the host query plan.
    ///
    /// # Errors
    ///
    /// Returns the first rejected field and reason. Precedence between independently invalid
    /// properties is not part of the contract.
    pub fn new(spec: SmlaSpec, query_counts: &'a [u8]) -> Result<Self, ContractError> {
        let [tokens, heads, width] = spec.q.shape;
        if (heads != 32 && heads != 64) || width != 512 {
            return fail("q", Violation::Shape);
        }
        let indices = spec.idx.shape[1];
        if indices > MAX_INDICES {
            return fail("idx", Violation::Shape);
        }
        if spec.softmax_scale.to_bits() != SOFTMAX_SCALE.to_bits() {
            return fail("softmax_scale", Violation::Scale);
        }
        let [sequences, slots] = spec.page_table.shape;
        let launching = tokens != 0;
        let payload = spec.kv_format.payload_bytes();
        check_kv_strides(&spec.kv, payload)?;

        let query = [tokens, heads, 512];
        let pool = [spec.kv.shape[0], PAGE_ROWS, payload];
        let inputs = [
            padded("q", &spec.q, query, DType::Bf16, launching)?,
            rows("kv", &spec.kv, pool, DType::U8, launching)?,
            packed(
                "page_table",
                &spec.page_table,
                [sequences, slots],
                DType::I32,
                launching,
            )?,
            packed("seq_id", &spec.seq_id, [tokens], DType::I32, launching)?,
            packed("q_pos", &spec.q_pos, [tokens], DType::I32, launching)?,
            packed("kv_len", &spec.kv_len, [sequences], DType::I32, launching)?,
            padded("idx", &spec.idx, [tokens, indices], DType::I32, launching)?,
            packed("k_len", &spec.k_len, [tokens], DType::I32, launching)?,
        ];
        let outputs = [
            padded("out", &spec.out, query, DType::Bf16, launching)?,
            packed("lse", &spec.lse, [tokens, heads], DType::F32, launching)?,
        ];

        // Sequences is now known to be below the index limit, so the total cannot overflow.
        if query_counts.len() as u64 != sequences {
            return fail("query_counts", Violation::QueryCount);
        }
        let mut planned = 0;
        for &count in query_counts {
            if count > MAX_QUERIES_PER_SEQUENCE {
                return fail("query_counts", Violation::QueryCount);
            }
            planned += u64::from(count);
        }
        if planned != tokens {
            return fail("query_counts", Violation::QueryCount);
        }
        check_aliases(&inputs, &outputs)?;
        Ok(Self { spec, query_counts })
    }

    /// A copy of the accepted metadata. Mutating the copy cannot affect this object.
    #[must_use]
    pub const fn spec(&self) -> SmlaSpec {
        self.spec
    }

    /// The host batch plan this object was checked against.
    #[must_use]
    pub const fn query_counts(&self) -> &'a [u8] {
        self.query_counts
    }

    /// Whether any token is present. A launch with no token does nothing.
    #[must_use]
    pub const fn should_launch(&self) -> bool {
        self.spec.q.shape[0] != 0
    }
}

/// Validated grouped-expert parameters.
#[derive(Clone, Copy, Debug)]
pub struct MoeParams {
    spec: MoeSpec,
}

impl MoeParams {
    /// Validates one grouped-expert launch.
    ///
    /// # Errors
    ///
    /// Returns the first rejected field and reason. Precedence between independently invalid
    /// properties is not part of the contract.
    pub fn new(spec: MoeSpec) -> Result<Self, ContractError> {
        let tokens = spec.x.shape[0];
        let Some(intermediate) = spec.w2.shape[2].checked_mul(2) else {
            return fail("w2", Violation::Shape);
        };
        if intermediate != 1024 && intermediate != 2048 {
            return fail("w2", Violation::Shape);
        }
        let group = spec.format.group();
        let launching = tokens != 0;
        let fused = 2 * intermediate;
        let half = intermediate / 2;
        let w13_scale_shape = [EXPERTS, fused, HIDDEN / group];
        let w2_scale_shape = [EXPERTS, HIDDEN, intermediate / group];

        let tokens_hidden = [tokens, HIDDEN];
        let inputs = [
            packed("x", &spec.x, tokens_hidden, DType::Bf16, launching)?,
            packed(
                "topk_ids",
                &spec.topk_ids,
                [tokens, TOP_K],
                DType::I32,
                launching,
            )?,
            packed(
                "topk_w",
                &spec.topk_w,
                [tokens, TOP_K],
                DType::F32,
                launching,
            )?,
            packed(
                "w13",
                &spec.w13,
                [EXPERTS, fused, HIDDEN / 2],
                DType::U8,
                launching,
            )?,
            packed(
                "w2",
                &spec.w2,
                [EXPERTS, HIDDEN, half],
                DType::U8,
                launching,
            )?,
            packed(
                "w13_scales",
                &spec.w13_scales,
                w13_scale_shape,
                DType::U8,
                launching,
            )?,
            packed(
                "w2_scales",
                &spec.w2_scales,
                w2_scale_shape,
                DType::U8,
                launching,
            )?,
            packed(
                "w13_global",
                &spec.w13_global,
                [EXPERTS],
                DType::F32,
                launching,
            )?,
            packed(
                "w2_global",
                &spec.w2_global,
                [EXPERTS],
                DType::F32,
                launching,
            )?,
        ];
        let outputs = [packed("y", &spec.y, tokens_hidden, DType::Bf16, launching)?];
        check_aliases(&inputs, &outputs)?;
        Ok(Self { spec })
    }

    /// A copy of the accepted metadata. Mutating the copy cannot affect this object.
    #[must_use]
    pub const fn spec(&self) -> MoeSpec {
        self.spec
    }

    /// Whether any token is present. A launch with no token does nothing.
    #[must_use]
    pub const fn should_launch(&self) -> bool {
        self.spec.x.shape[0] != 0
    }
}
