//! Independent acceptance for HOST-CONTRACTS; written before product implementation.
#![allow(
    clippy::expect_used,
    reason = "Acceptance assertions must fail on rejection"
)]

mod support;

use support::{element_bytes, fp8, moe, size_to_span, smla, tensor, wide_span, wrong_dtype};
use tq_core::contracts::{
    Bytes, ContractError, DType, KvFormat, MoeParams, MoeSpec, Nvfp4Format, SmlaParams, SmlaSpec,
    Violation,
};
use tq_testkit::philox::block;

fn reject_s(s: &SmlaSpec, counts: &[u8], field: &'static str, kind: Violation) {
    assert_eq!(
        SmlaParams::new(*s, counts).expect_err("invalid SMLA accepted"),
        ContractError { field, kind }
    );
}

fn reject_m(m: &MoeSpec, field: &'static str, kind: Violation) {
    assert_eq!(
        MoeParams::new(*m).expect_err("invalid MoE accepted"),
        ContractError { field, kind }
    );
}

#[test]
fn formats_are_distinct_and_group_parser_is_closed() {
    for (group, format) in [(16, Nvfp4Format::G16), (32, Nvfp4Format::G32E4m3)] {
        assert_eq!(Nvfp4Format::try_from_group(group), Ok(format));
        assert_eq!(format.group(), group);
    }
    for group in [0, 1, 8, 15, 17, 31, 33, 64, u32::MAX.into(), u64::MAX] {
        assert_eq!(
            Nvfp4Format::try_from_group(group),
            Err(ContractError {
                field: "format",
                kind: Violation::Group
            })
        );
    }
}

#[test]
fn all_error_kinds_are_structured_and_displayed() {
    for (kind, name) in [
        (Violation::Shape, "Shape"),
        (Violation::Dtype, "Dtype"),
        (Violation::Dimension, "Dimension"),
        (Violation::Stride, "Stride"),
        (Violation::Alignment, "Alignment"),
        (Violation::Address, "Address"),
        (Violation::Capacity, "Capacity"),
        (Violation::Overflow, "Overflow"),
        (Violation::QueryCount, "QueryCount"),
        (Violation::Scale, "Scale"),
        (Violation::Group, "Group"),
        (Violation::Alias, "Alias"),
    ] {
        let err = ContractError { field: "q", kind };
        let dynamic: &dyn std::error::Error = &err;
        assert_eq!(dynamic.to_string(), format!("q: {name}"));
        assert!(dynamic.source().is_none());
    }
}

#[test]
fn smla_supported_shapes_formats_and_capacity() {
    for heads in [32, 64] {
        for counts in [vec![1], vec![0, 1, 2, 3, 4], vec![4; 16], vec![4; 64]] {
            let tokens = counts.iter().map(|&n| u64::from(n)).sum();
            for indices in [0, 1, 15, 16, 17, 2048, 2049, 2051] {
                for (format, row_stride, payload) in [
                    (KvFormat::Bf16, 1024, 1024),
                    (KvFormat::Fp8E4m3, 528, 528),
                    (KvFormat::Fp8E4m3, 544, 528),
                    (KvFormat::Fp8E4m3, 560, 528),
                ] {
                    let mut s = smla(tokens, heads, indices, counts.len() as u64);
                    s.kv_format = format;
                    s.kv.shape[2] = payload;
                    s.kv.strides_bytes = [Bytes(row_stride * 64), Bytes(row_stride), Bytes(1)];
                    size_to_span(&mut s.kv);
                    let p = SmlaParams::new(s, &counts).expect("supported SMLA");
                    assert!(p.should_launch());
                    assert_eq!(p.spec(), s);
                    assert_eq!(p.query_counts(), counts);
                }
            }
        }
    }
}

#[test]
fn smla_fixed_geometry_rejections() {
    for heads in [0, 1, 31, 33, 48, 63, 65] {
        reject_s(&smla(1, heads, 7, 1), &[1], "q", Violation::Shape);
    }
    for dim in [0, 256, 511, 513, 1024] {
        let mut s = smla(1, 32, 7, 1);
        s.q.shape[2] = dim;
        reject_s(&s, &[1], "q", Violation::Shape);
    }
    for k in [2052, 4096] {
        reject_s(&smla(1, 32, k, 1), &[1], "idx", Violation::Shape);
    }
    let mut s = smla(1, 32, 7, 1);
    s.kv.shape[1] = 65;
    reject_s(&s, &[1], "kv", Violation::Shape);
    for payload in [512, 528, 544, 1023, 1025] {
        s = smla(1, 32, 7, 1);
        s.kv.shape[2] = payload;
        reject_s(&s, &[1], "kv", Violation::Shape);
    }
    s = smla(1, 32, 7, 1);
    s.kv_format = KvFormat::Fp8E4m3;
    reject_s(&s, &[1], "kv", Violation::Shape);
}

#[test]
fn softmax_scale_is_the_exact_model_constant() {
    for scale in [
        0.0,
        -0.0,
        -0.0625,
        1.0,
        512_f32.sqrt().recip(),
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0.0625_f32.to_bits() - 1),
        f32::from_bits(0.0625_f32.to_bits() + 1),
    ] {
        let mut s = smla(1, 32, 7, 1);
        s.softmax_scale = scale;
        reject_s(&s, &[1], "softmax_scale", Violation::Scale);
    }
}

#[test]
fn host_query_plan_checks_each_sequence_and_total() {
    for counts in [
        vec![],
        vec![4],
        vec![1, 1, 2],
        vec![0, 3],
        vec![1, 4],
        vec![5, 0],
        vec![255, 0],
    ] {
        reject_s(
            &smla(4, 32, 7, 2),
            &counts,
            "query_counts",
            Violation::QueryCount,
        );
    }
    // Both totals obey T <= 4*S; only the distribution distinguishes these plans.
    let s = smla(6, 32, 7, 2);
    assert!(SmlaParams::new(s, &[3, 3]).is_ok());
    reject_s(&s, &[5, 1], "query_counts", Violation::QueryCount);
    let s = smla(1, 32, 7, 0);
    assert!(SmlaParams::new(s, &[]).is_err());
}

// Generate a test for each public tensor, rather than exercising only q and x.
macro_rules! smla_tensor_cases {
    ($($field:ident),+ $(,)?) => { $(
        #[test]
        fn $field() {
            let original = smla(4, 32, 7, 2);
            let field = stringify!($field);
            let mut s = original;
            s.$field.dtype = wrong_dtype(s.$field.dtype);
            reject_s(&s, &[2, 2], field, Violation::Dtype);
            s = original;
            let last = s.$field.strides_bytes.len() - 1;
            s.$field.strides_bytes[last] = Bytes(0);
            reject_s(&s, &[2, 2], field, Violation::Stride);
            s = original;
            s.$field.strides_bytes[last] = Bytes(2 * element_bytes(s.$field.dtype));
            reject_s(&s, &[2, 2], field, Violation::Stride);
            s = original;
            s.$field.device_address = 0;
            reject_s(&s, &[2, 2], field, Violation::Address);
            s = original;
            s.$field.buffer_bytes.0 -= 1;
            reject_s(&s, &[2, 2], field, Violation::Capacity);
            s = original;
            s.$field.buffer_bytes = Bytes(u64::MAX);
            reject_s(&s, &[2, 2], field, Violation::Overflow);
            s = original;
            s.$field.device_address += 1;
            if element_bytes(s.$field.dtype) > 1 || field == "kv" {
                reject_s(&s, &[2, 2], field, Violation::Alignment);
            } else {
                assert!(SmlaParams::new(s, &[2, 2]).is_ok());
            }
            s = original;
            s.$field.shape[0] = 1 << 31;
            assert!(SmlaParams::new(s, &[2, 2]).is_err());
            for axis in 0..s.$field.shape.len() {
                s = original;
                s.$field.shape[axis] = u64::MAX;
                assert!(SmlaParams::new(s, &[2, 2]).is_err());
            }
        }
    )+ };
}

mod smla_tensor_validation {
    use super::*;
    smla_tensor_cases!(
        q, kv, page_table, seq_id, q_pos, kv_len, idx, k_len, out, lse
    );
}

#[test]
fn smla_dependent_shapes_are_not_inferred_away() {
    macro_rules! bad {
        ($field:ident, $axis:expr) => {{
            let mut s = smla(4, 32, 7, 2);
            s.$field.shape[$axis] += 1;
            reject_s(&s, &[2, 2], stringify!($field), Violation::Shape);
        }};
    }
    bad!(seq_id, 0);
    bad!(q_pos, 0);
    bad!(kv_len, 0);
    bad!(k_len, 0);
    bad!(idx, 0);
    bad!(out, 0);
    bad!(out, 1);
    bad!(out, 2);
    bad!(lse, 0);
    bad!(lse, 1);
}

#[test]
fn fp8_row_stride_and_exact_last_byte() {
    for stride in [528, 544, 560, 1024] {
        let mut s = smla(1, 32, 7, 1);
        fp8(&mut s, stride);
        assert_eq!(s.kv.buffer_bytes.0, (17 * 64 - 1) * stride + 528);
        assert!(SmlaParams::new(s, &[1]).is_ok());
        s.kv.buffer_bytes.0 -= 1;
        reject_s(&s, &[1], "kv", Violation::Capacity);
    }
    for stride in [0, 16, 512, 527, 529, 543, 545] {
        let mut s = smla(1, 32, 7, 1);
        fp8(&mut s, stride);
        reject_s(&s, &[1], "kv", Violation::Stride);
    }
    let mut s = smla(1, 32, 7, 1);
    fp8(&mut s, 544);
    s.kv.strides_bytes[0].0 += 16;
    reject_s(&s, &[1], "kv", Violation::Stride);
    s = smla(1, 32, 7, 1);
    s.kv.strides_bytes[1] = Bytes(1008);
    reject_s(&s, &[1], "kv", Violation::Stride);
}

#[test]
fn padded_q_idx_out_are_valid_with_natural_alignment() {
    let mut s = smla(4, 32, 7, 2);
    for t in [&mut s.q, &mut s.out] {
        t.strides_bytes = [Bytes(32 * 1030 + 6), Bytes(1030), Bytes(2)];
        t.device_address += 2;
        size_to_span(t);
    }
    s.idx.strides_bytes[0] = Bytes(36);
    s.idx.device_address += 4;
    size_to_span(&mut s.idx);
    s.kv.device_address += 16;
    assert!(SmlaParams::new(s, &[2, 2]).is_ok());
    let good = s;
    s.q.strides_bytes[0] = Bytes(32 * 1030 - 2);
    reject_s(&s, &[2, 2], "q", Violation::Stride);
    s = good;
    s.q.strides_bytes[1] = Bytes(1023);
    reject_s(&s, &[2, 2], "q", Violation::Stride);
    s = good;
    s.out.strides_bytes[1] = Bytes(1031);
    reject_s(&s, &[2, 2], "out", Violation::Stride);
    s = good;
    s.idx.strides_bytes[0] = Bytes(24);
    reject_s(&s, &[2, 2], "idx", Violation::Stride);
    s = good;
    s.lse.strides_bytes[0].0 += 4;
    reject_s(&s, &[2, 2], "lse", Violation::Stride);
    s = good;
    s.page_table.strides_bytes[0].0 += 4;
    reject_s(&s, &[2, 2], "page_table", Violation::Stride);
}

#[test]
fn empty_launch_skips_all_storage_but_keeps_metadata_checks() {
    let mut s = smla(0, 32, 0, 0);
    macro_rules! absent { ($obj:ident; $($field:ident),+) => { $(
        $obj.$field.device_address = 0;
        $obj.$field.buffer_bytes = Bytes(0);
    )+ }; }
    absent!(s; q, kv, page_table, seq_id, q_pos, kv_len, idx, k_len, out, lse);
    let p = SmlaParams::new(s, &[]).expect("empty SMLA");
    assert!(!p.should_launch());
    assert_eq!(p.spec(), s);
    assert_eq!(p.query_counts(), &[]);
    s.q.shape[2] = 513;
    reject_s(&s, &[], "q", Violation::Shape);
    s = smla(0, 32, 0, 1);
    reject_s(&s, &[1], "query_counts", Violation::QueryCount);
    let mut m = moe(0, 1024, Nvfp4Format::G16);
    absent!(m; x, topk_ids, topk_w, w13, w2, w13_scales, w2_scales, w13_global, w2_global, y);
    let p = MoeParams::new(m).expect("empty MoE");
    assert!(!p.should_launch());
    assert_eq!(p.spec(), m);
    m.w13.dtype = DType::Bf16;
    reject_m(&m, "w13", Violation::Dtype);
}

#[test]
fn empty_cache_and_indices_inside_nonempty_batch() {
    for k in [0, 7, 2051] {
        let mut s = smla(1, 32, k, 1);
        s.kv = tensor([0, 64, 1024], DType::U8, 0);
        s.page_table = tensor([1, 0], DType::I32, 0);
        if k == 0 {
            s.idx.device_address = 0;
        }
        assert!(SmlaParams::new(s, &[1]).expect("empty KV").should_launch());
    }
}

#[test]
fn bf16_kv_padding_is_a_correctness_layout_not_a_sector_promise() {
    let mut s = smla(1, 32, 7, 1);
    s.kv.strides_bytes = [Bytes(1040 * 64), Bytes(1040), Bytes(1)];
    s.kv.device_address += 16;
    size_to_span(&mut s.kv);
    assert!(SmlaParams::new(s, &[1]).is_ok());
}

#[test]
fn wide_spans_are_not_truncated_to_32_bits() {
    let mut s = smla(1, 32, 7, 1);
    fp8(&mut s, 544);
    s.kv.shape[0] = 1 << 19;
    size_to_span(&mut s.kv);
    assert!(wide_span(&s.kv) > (1_u128 << 33));
    assert!(SmlaParams::new(s, &[1]).is_ok());
    s.kv.buffer_bytes.0 &= u64::from(u32::MAX);
    reject_s(&s, &[1], "kv", Violation::Capacity);
    let mut m = moe(1, 2048, Nvfp4Format::G16);
    assert_eq!(wide_span(&m.w13), 2_415_919_104);
    assert!(MoeParams::new(m).is_ok());
    m.w13.buffer_bytes = Bytes(2_147_483_647);
    reject_m(&m, "w13", Violation::Capacity);
}

#[test]
fn dimension_boundary_is_per_axis_not_total_elements() {
    let mut s = smla(1, 32, 7, 1);
    s.kv.shape[0] = (1 << 31) - 1;
    size_to_span(&mut s.kv);
    assert!(SmlaParams::new(s, &[1]).is_ok());
    s.kv.shape[0] += 1;
    size_to_span(&mut s.kv);
    reject_s(&s, &[1], "kv", Violation::Dimension);
    s = smla(1, 32, 7, 1);
    s.page_table = tensor([1, (1 << 31) - 1], DType::I32, 3);
    assert!(SmlaParams::new(s, &[1]).is_ok());
    s.page_table = tensor([1, 1 << 31], DType::I32, 3);
    reject_s(&s, &[1], "page_table", Violation::Dimension);
    let m = moe((1 << 31) - 1, 1024, Nvfp4Format::G32E4m3);
    assert!(MoeParams::new(m).is_ok());
}

#[test]
fn arithmetic_overflow_is_rejected_in_debug_and_release() {
    let mut s = smla(4, 32, 7, 1);
    s.q.strides_bytes[0] = Bytes(1 << 63);
    assert!(wide_span(&s.q) > u128::from(u64::MAX));
    s.q.buffer_bytes = Bytes(u64::MAX - s.q.device_address);
    reject_s(&s, &[4], "q", Violation::Overflow);
    s = smla(1, 32, 7, 1);
    s.kv.strides_bytes[1] = Bytes(1 << 62);
    reject_s(&s, &[1], "kv", Violation::Overflow);
    s = smla(1, 32, 7, 1);
    s.q.strides_bytes[1] = Bytes(1 << 63);
    reject_s(&s, &[1], "q", Violation::Overflow);
    s = smla(2, 32, 7, 1);
    s.idx.strides_bytes[0] = Bytes(u64::MAX - 3);
    assert!(wide_span(&s.idx) > u128::from(u64::MAX));
    reject_s(&s, &[2], "idx", Violation::Overflow);
}

#[test]
fn numeric_address_end_and_capacity_are_separate_checks() {
    let mut s = smla(1, 32, 7, 1);
    s.q.device_address = u64::MAX - s.q.buffer_bytes.0 + 1;
    reject_s(&s, &[1], "q", Violation::Overflow);
    s = smla(1, 32, 7, 1);
    s.q.device_address = u64::MAX - s.q.buffer_bytes.0 - 1;
    assert!(SmlaParams::new(s, &[1]).is_ok());
    s.q.buffer_bytes.0 += 2;
    reject_s(&s, &[1], "q", Violation::Overflow);
    s = smla(1, 32, 7, 1);
    s.q.buffer_bytes.0 += 1;
    assert!(SmlaParams::new(s, &[1]).is_ok());
}

#[test]
fn smla_output_aliases_and_adjacent_ranges() {
    let good = smla(4, 32, 7, 1);
    macro_rules! overlap { ($($input:ident),+) => { $(
        let mut s = good;
        s.out.device_address = s.$input.device_address;
        reject_s(&s, &[4], "out", Violation::Alias);
        s = good;
        s.lse.device_address = s.$input.device_address;
        reject_s(&s, &[4], "lse", Violation::Alias);
    )+ }; }
    overlap!(q, kv, page_table, seq_id, q_pos, kv_len, idx, k_len);
    let mut s = good;
    s.lse.device_address = s.out.device_address;
    reject_s(&s, &[4], "lse", Violation::Alias);
    s = good;
    s.out.device_address = s.q.device_address + s.q.buffer_bytes.0;
    assert!(SmlaParams::new(s, &[4]).is_ok());
    s.out.device_address -= 2;
    reject_s(&s, &[4], "out", Violation::Alias);
    s = good;
    s.out.device_address = s.q.device_address - s.out.buffer_bytes.0;
    assert!(SmlaParams::new(s, &[4]).is_ok());
    s.out.device_address += 2;
    reject_s(&s, &[4], "out", Violation::Alias);
    s = good;
    s.q_pos.device_address = s.seq_id.device_address;
    assert!(
        SmlaParams::new(s, &[4]).is_ok(),
        "read-only virtual alias is allowed here"
    );
}

#[test]
fn unused_allocation_capacity_does_not_create_a_tensor_alias() {
    let mut s = smla(1, 32, 7, 1);
    s.out.device_address = s.q.device_address + s.q.buffer_bytes.0;
    s.q.buffer_bytes.0 += s.out.buffer_bytes.0;
    assert!(
        SmlaParams::new(s, &[1]).is_ok(),
        "disjoint views in one allocation"
    );
    let mut m = moe(1, 1024, Nvfp4Format::G16);
    m.y.device_address = m.x.device_address + m.x.buffer_bytes.0;
    m.x.buffer_bytes.0 += m.y.buffer_bytes.0;
    assert!(
        MoeParams::new(m).is_ok(),
        "unused input capacity is not a read"
    );
    m = moe(1, 1024, Nvfp4Format::G16);
    m.y.device_address = m.x.device_address - m.y.buffer_bytes.0;
    m.y.buffer_bytes.0 += m.x.buffer_bytes.0;
    assert!(
        MoeParams::new(m).is_ok(),
        "unused output capacity is not a write"
    );
}

#[test]
fn alias_policy_conservatively_includes_internal_padding() {
    let mut s = smla(2, 32, 7, 1);
    s.q.strides_bytes[0] = Bytes(100_000);
    size_to_span(&mut s.q);
    s.lse.device_address = s.q.device_address + 40_000;
    reject_s(&s, &[2], "lse", Violation::Alias);
}

#[test]
fn moe_all_formats_shards_and_batch_sizes() {
    for format in [Nvfp4Format::G16, Nvfp4Format::G32E4m3] {
        for i in [1024, 2048] {
            for t in [1, 2, 8, 16, 36, 64, 256, 4096] {
                let m = moe(t, i, format);
                let p = MoeParams::new(m).expect("supported MoE");
                assert!(p.should_launch());
                assert_eq!(p.spec(), m);
            }
        }
    }
}

macro_rules! moe_tensor_cases {
    ($($field:ident),+ $(,)?) => { $(
        #[test]
        fn $field() {
            let original = moe(4, 1024, Nvfp4Format::G16);
            let field = stringify!($field);
            let mut m = original;
            m.$field.dtype = wrong_dtype(m.$field.dtype);
            reject_m(&m, field, Violation::Dtype);
            m = original;
            let last = m.$field.strides_bytes.len()-1;
            m.$field.strides_bytes[last] = Bytes(0);
            reject_m(&m, field, Violation::Stride);
            m = original;
            m.$field.strides_bytes[last] = Bytes(2 * element_bytes(m.$field.dtype));
            reject_m(&m, field, Violation::Stride);
            m = original;
            m.$field.device_address = 0;
            reject_m(&m, field, Violation::Address);
            m = original;
            m.$field.buffer_bytes.0 -= 1;
            reject_m(&m, field, Violation::Capacity);
            m = original;
            m.$field.buffer_bytes = Bytes(u64::MAX);
            reject_m(&m, field, Violation::Overflow);
            m = original;
            m.$field.device_address += 1;
            if element_bytes(m.$field.dtype) > 1 {
                reject_m(&m, field, Violation::Alignment);
            } else {
                assert!(MoeParams::new(m).is_ok());
            }
            m = original;
            m.$field.strides_bytes[0].0 += element_bytes(m.$field.dtype);
            reject_m(&m, field, Violation::Stride);
            m = original;
            m.$field.shape[last] += 1;
            assert!(MoeParams::new(m).is_err());
            m = original;
            m.$field.shape[0] = 1 << 31;
            assert!(MoeParams::new(m).is_err());
            for axis in 0..m.$field.shape.len() {
                m = original;
                m.$field.shape[axis] = u64::MAX;
                assert!(MoeParams::new(m).is_err());
            }
        }
    )+ };
}

mod moe_tensor_validation {
    use super::*;
    moe_tensor_cases!(
        x, topk_ids, topk_w, w13, w2, w13_scales, w2_scales, w13_global, w2_global, y
    );
}

#[test]
fn moe_model_geometry_and_scale_layout() {
    for i in [2, 512, 1022, 1026, 2046, 2050, 4096] {
        assert!(MoeParams::new(moe(1, i, Nvfp4Format::G16)).is_err());
    }
    let mut m = moe(1, 1024, Nvfp4Format::G16);
    m.w13.shape[0] = 287;
    reject_m(&m, "w13", Violation::Shape);
    m = moe(1, 1024, Nvfp4Format::G16);
    m.w2.shape[0] = 289;
    reject_m(&m, "w2", Violation::Shape);
    m = moe(1, 1024, Nvfp4Format::G16);
    m.w13_global.shape = [576];
    reject_m(&m, "w13_global", Violation::Shape);
    m = moe(1, 1024, Nvfp4Format::G16);
    m.format = Nvfp4Format::G32E4m3;
    assert!(
        MoeParams::new(m).is_err(),
        "G16 scales cannot silently become G32"
    );
    m = moe(1, 1024, Nvfp4Format::G32E4m3);
    m.format = Nvfp4Format::G16;
    assert!(
        MoeParams::new(m).is_err(),
        "G32 scales cannot silently become G16"
    );
}

#[test]
fn moe_output_cannot_overlap_any_input() {
    let good = moe(4, 1024, Nvfp4Format::G16);
    macro_rules! overlap { ($($input:ident),+) => { $(
        let mut m = good;
        m.y.device_address = m.$input.device_address;
        reject_m(&m, "y", Violation::Alias);
    )+ }; }
    overlap!(
        x, topk_ids, topk_w, w13, w2, w13_scales, w2_scales, w13_global, w2_global
    );
    let mut m = good;
    m.y.device_address = m.x.device_address + m.x.buffer_bytes.0;
    assert!(MoeParams::new(m).is_ok());
    m.y.device_address -= 2;
    reject_m(&m, "y", Violation::Alias);
    m = good;
    m.w13_global.device_address = m.w2_global.device_address;
    assert!(MoeParams::new(m).is_ok());
}

#[test]
fn returned_copies_cannot_mutate_validated_parameters() {
    let s = smla(1, 32, 7, 1);
    let counts = [1];
    let p = SmlaParams::new(s, &counts).expect("valid SMLA");
    let mut copy = p.spec();
    copy.q.shape[1] = 48;
    assert!(SmlaParams::new(copy, &counts).is_err());
    assert_eq!(p.spec(), s);
    assert_eq!(p.query_counts().as_ptr(), counts.as_ptr());
    let m = moe(1, 1024, Nvfp4Format::G16);
    let p = MoeParams::new(m).expect("valid MoE");
    let mut copy = p.spec();
    copy.x.shape[1] = 4095;
    assert!(MoeParams::new(copy).is_err());
    assert_eq!(p.spec(), m);
}

#[test]
fn philox_boundary_sweep_uses_an_independent_wide_oracle() {
    for seed in [0, 1, 7, 42, u32::MAX] {
        for case in 0..128 {
            let r = block([case, 0, 0, 0], [seed, 0]);
            let counts = r.map(|x| u8::try_from(x % 5).expect("0 through 4"));
            let tokens = counts.iter().map(|&x| u64::from(x)).sum();
            let mut s = smla(tokens, 32, u64::from(r[0] % 2052), 4);
            fp8(&mut s, 528 + 16 * u64::from(r[1] % 33));
            s.kv.shape[0] = 1 + u64::from(r[2] % 500_000);
            size_to_span(&mut s.kv);
            let head_stride = 1024 + 2 * u64::from(r[3] % 99);
            s.q.strides_bytes = [Bytes(32 * head_stride + 6), Bytes(head_stride), Bytes(2)];
            size_to_span(&mut s.q);
            assert!(
                SmlaParams::new(s, &counts).is_ok(),
                "seed {seed} case {case}"
            );
            if tokens > 0 {
                s.kv.buffer_bytes.0 -= 1;
                reject_s(&s, &counts, "kv", Violation::Capacity);
                s.kv.buffer_bytes.0 += 1;
                s.q.buffer_bytes.0 -= 1;
                reject_s(&s, &counts, "q", Violation::Capacity);
            }
        }
    }
}
