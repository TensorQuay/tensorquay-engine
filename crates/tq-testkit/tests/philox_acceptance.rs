//! Independent RNG-001…005 acceptance, frozen before the implementation.
#![allow(
    clippy::expect_used,
    reason = "Malformed frozen fixtures must fail acceptance"
)]

use tq_testkit::philox::{RangeError, block, fill_words, uniform_f32};

const BLOCKS: &str = include_str!("../../../tests/fixtures/philox-v1/blocks.tsv");
const STREAMS: &str = include_str!("../../../tests/fixtures/philox-v1/streams.tsv");
const UNIFORMS: &str = include_str!("../../../tests/fixtures/philox-v1/uniform.tsv");

fn rows(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines()
        .filter(|s| !s.starts_with('#'))
        .map(|s| s.split_whitespace().collect())
}

fn hex32(text: &str) -> u32 {
    u32::from_str_radix(text, 16).expect("uint32 fixture")
}

fn hex64(text: &str) -> u64 {
    u64::from_str_radix(text, 16).expect("uint64 fixture")
}

#[test]
fn official_known_answers() {
    assert_eq!(
        block([0; 4], [0; 2]),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
    );
    assert_eq!(
        block([u32::MAX; 4], [u32::MAX; 2]),
        [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
    );
    assert_eq!(
        block(
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0]
        ),
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
    );
}

#[test]
fn all_upstream_blocks() {
    let mut count = 0;
    for fields in rows(BLOCKS) {
        assert_eq!(fields.len(), 10);
        let values: Vec<u32> = fields.iter().map(|s| hex32(s)).collect();
        let counter = values[..4].try_into().expect("four counter words");
        let key = values[4..6].try_into().expect("two key words");
        assert_eq!(block(counter, key), values[6..], "case {count}");
        count += 1;
    }
    assert_eq!(count, 199);
}

#[test]
fn all_upstream_streams_and_float_bits() {
    let mut count = 0;
    for fields in rows(STREAMS) {
        let n: usize = fields[3].parse().expect("word count");
        let expected: Vec<u32> = fields[4..].iter().map(|s| hex32(s)).collect();
        assert_eq!(expected.len(), n);
        let mut actual = vec![0xdead_beef; n];
        fill_words(
            hex64(fields[0]),
            hex64(fields[1]),
            hex64(fields[2]),
            &mut actual,
        )
        .expect("valid frozen range");
        assert_eq!(actual, expected, "case {count}");
        count += 1;
    }
    assert_eq!(count, 160);
    let mut count = 0;
    for fields in rows(UNIFORMS) {
        assert_eq!(fields.len(), 2);
        assert_eq!(uniform_f32(hex32(fields[0])).to_bits(), hex32(fields[1]));
        count += 1;
    }
    assert_eq!(count, 96);
}

#[test]
fn partition_overlap_and_no_global_state() {
    for seed in [0, 1, 0x0123_4567_89ab_cdef, 1 << 63, u64::MAX] {
        for start in [0, 1, 2, 3, (1 << 34) - 7, u64::MAX - 63] {
            let mut whole = [0; 64];
            fill_words(seed, u64::MAX, start, &mut whole).expect("valid whole range");
            for split in [0, 1, 2, 3, 4, 17, 31, 63, 64] {
                let mut partitioned = [0; 64];
                let right_start = start + u64::try_from(split.min(63)).expect("small split");
                fill_words(seed, u64::MAX, right_start, &mut partitioned[split..])
                    .expect("valid suffix");
                fill_words(seed, u64::MAX, start, &mut partitioned[..split]).expect("valid prefix");
                assert_eq!(whole, partitioned);
            }
            let mut overlap = [0; 17];
            fill_words(seed, u64::MAX, start + 5, &mut overlap).expect("valid overlap");
            assert_eq!(overlap, whole[5..22]);
            let mut again = [0; 64];
            fill_words(seed, u64::MAX, start, &mut again).expect("repeat");
            assert_eq!(again, whole);
        }
    }
}

#[test]
fn exhausted_range_is_atomic() {
    for (start, count) in [(u64::MAX, 2), (u64::MAX - 3, 5)] {
        let mut out = vec![0x1234_5678; count];
        assert_eq!(fill_words(0, 0, start, &mut out), Err(RangeError));
        assert_eq!(out, vec![0x1234_5678; count]);
    }
    for start in [0, 1, u64::MAX] {
        assert_eq!(fill_words(u64::MAX, u64::MAX, start, &mut []), Ok(()));
    }
    let mut last = [0; 1];
    fill_words(u64::MAX, u64::MAX, u64::MAX, &mut last).expect("last legal word");
    assert_eq!(
        last[0],
        block([u32::MAX, 0x3fff_ffff, u32::MAX, u32::MAX], [u32::MAX; 2])[3]
    );
}

#[test]
fn error_is_a_small_typed_value() {
    fn copy_value<T: Copy + core::error::Error + PartialEq>(value: T) -> (T, T) {
        (value, value)
    }
    let (first, second) = copy_value(RangeError);
    assert_eq!(first, second);
    assert_eq!(first.to_string(), "word range exceeds u64");
    assert_eq!(format!("{second:?}"), "RangeError");
    assert_eq!(core::mem::size_of::<RangeError>(), 0);
    assert!(core::error::Error::source(&first).is_none());
}

#[test]
fn uniform_endpoints_and_low_byte_invariance() {
    for (prefix, bits) in [
        (0, 0),
        (0x8000_0000, 0x3f00_0000),
        (0xffff_ff00, 0x3f7f_ffff),
    ] {
        for tail in 0..256 {
            let value = uniform_f32(prefix | tail);
            assert_eq!(value.to_bits(), bits);
            assert!((0.0..1.0).contains(&value));
        }
    }
}
