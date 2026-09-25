//! Independently computed metadata, with no tensor allocations or product layout helpers.

use tq_core::contracts::{Bytes, DType, KvFormat, MoeSpec, Nvfp4Format, SmlaSpec, TensorSpec};

pub fn element_bytes(dtype: DType) -> u64 {
    match dtype {
        DType::U8 => 1,
        DType::Bf16 => 2,
        DType::I32 | DType::F32 => 4,
    }
}

pub fn wide_span<const N: usize>(t: &TensorSpec<N>) -> u128 {
    if t.shape.contains(&0) {
        return 0;
    }
    // Deliberately wider than the product's required u64 arithmetic.
    u128::from(element_bytes(t.dtype))
        + t.shape
            .iter()
            .zip(t.strides_bytes)
            .map(|(&dim, stride)| u128::from(dim - 1) * u128::from(stride.0))
            .sum::<u128>()
}

pub fn size_to_span<const N: usize>(t: &mut TensorSpec<N>) {
    t.buffer_bytes = Bytes(u64::try_from(wide_span(t)).expect("test span fits u64"));
}

pub fn tensor<const N: usize>(shape: [u64; N], dtype: DType, slot: u64) -> TensorSpec<N> {
    let mut strides_bytes = [Bytes(0); N];
    let mut stride = element_bytes(dtype);
    for i in (0..N).rev() {
        strides_bytes[i] = Bytes(stride);
        stride *= shape[i].max(1);
    }
    let mut t = TensorSpec {
        shape,
        strides_bytes,
        dtype,
        device_address: slot << 48,
        buffer_bytes: Bytes(0),
    };
    size_to_span(&mut t);
    t
}

pub fn smla(tokens: u64, heads: u64, indices: u64, sequences: u64) -> SmlaSpec {
    SmlaSpec {
        q: tensor([tokens, heads, 512], DType::Bf16, 1),
        kv: tensor([17, 64, 1024], DType::U8, 2),
        page_table: tensor([sequences, 16], DType::I32, 3),
        seq_id: tensor([tokens], DType::I32, 4),
        q_pos: tensor([tokens], DType::I32, 5),
        kv_len: tensor([sequences], DType::I32, 6),
        idx: tensor([tokens, indices], DType::I32, 7),
        k_len: tensor([tokens], DType::I32, 8),
        out: tensor([tokens, heads, 512], DType::Bf16, 9),
        lse: tensor([tokens, heads], DType::F32, 10),
        kv_format: KvFormat::Bf16,
        softmax_scale: 0.0625,
    }
}

pub fn fp8(s: &mut SmlaSpec, row_stride: u64) {
    s.kv_format = KvFormat::Fp8E4m3;
    s.kv.shape[2] = 528;
    s.kv.strides_bytes = [Bytes(row_stride * 64), Bytes(row_stride), Bytes(1)];
    size_to_span(&mut s.kv);
}

pub fn moe(tokens: u64, intermediate: u64, format: Nvfp4Format) -> MoeSpec {
    // Expected group mapping must not use the product's group() method.
    let group = match format {
        Nvfp4Format::G16 => 16,
        Nvfp4Format::G32E4m3 => 32,
    };
    MoeSpec {
        x: tensor([tokens, 4096], DType::Bf16, 1),
        topk_ids: tensor([tokens, 8], DType::I32, 2),
        topk_w: tensor([tokens, 8], DType::F32, 3),
        w13: tensor([288, 2 * intermediate, 2048], DType::U8, 4),
        w2: tensor([288, 4096, intermediate / 2], DType::U8, 5),
        w13_scales: tensor([288, 2 * intermediate, 4096 / group], DType::U8, 6),
        w2_scales: tensor([288, 4096, intermediate / group], DType::U8, 7),
        w13_global: tensor([288], DType::F32, 8),
        w2_global: tensor([288], DType::F32, 9),
        y: tensor([tokens, 4096], DType::Bf16, 10),
        format,
    }
}

pub fn wrong_dtype(dtype: DType) -> DType {
    if dtype == DType::U8 {
        DType::Bf16
    } else {
        DType::U8
    }
}
