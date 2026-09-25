//! Philox4x32-10 and the `tq-philox-v1` stream mapping.
//!
//! Contract: `reference/contracts/philox.md`. The generator is a counter-based bijection, so it
//! keeps no state: the same `(counter, key)` always gives the same block, and a stream is only an
//! assignment of counters. Written independently from the published algorithm.
//!
//! One round multiplies `M0 * c0` and `M1 * c2` as full 64-bit products, splits each into a high
//! and a low half, and emits `(hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)`. Round `r` uses the key
//! `(k0 + r*W0, k1 + r*W1)` modulo `2^32`, so round zero uses the key exactly as supplied.
//!
//! The word at logical offset `i` of a stream is lane `i % 4` of the block whose key is the seed
//! split low word first, and whose counter is `[(i/4) low, (i/4) high, stream low, stream high]`.
//! Every stream therefore owns `2^64` words and never carries into another stream's domain.

use core::error::Error;
use core::fmt;

/// Multiplier applied to counter word 0.
const M0: u64 = 0xd251_1f53;
/// Multiplier applied to counter word 2.
const M1: u64 = 0xcd9e_8d57;
/// Key increment for word 0, the golden ratio.
const W0: u32 = 0x9e37_79b9;
/// Key increment for word 1, `sqrt(3) - 1`.
const W1: u32 = 0xbb67_ae85;
/// Rounds in Philox4x32-10.
const ROUNDS: usize = 10;
/// `2^-24`, exact in `f32` because both the numerator and the denominator are powers of two.
const UNIFORM_SCALE: f32 = 1.0 / 16_777_216.0;

/// This slice supports 32- and 64-bit pointer widths, so a slice length always fits in `u64`.
const _: () = assert!(usize::BITS <= u64::BITS);

/// The requested words do not all fit in the `2^64` addressable words of a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeError;

impl fmt::Display for RangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("word range exceeds u64")
    }
}

impl Error for RangeError {}

/// Splits the full product of a 32-bit value and a multiplier into its high and low halves.
#[expect(
    clippy::cast_possible_truncation,
    reason = "The shift and the implicit mask each select exactly one 32-bit half of the product"
)]
const fn halves(multiplier: u64, value: u32) -> (u32, u32) {
    let product = multiplier * value as u64;
    ((product >> 32) as u32, product as u32)
}

/// Returns the four output words of one Philox4x32-10 block.
#[must_use]
pub fn block(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = counter;
    let [mut k0, mut k1] = key;
    for _ in 0..ROUNDS {
        let (hi0, lo0) = halves(M0, c0);
        let (hi1, lo1) = halves(M1, c2);
        (c0, c1, c2, c3) = (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0);
        // The final bump is never used; keeping it unconditional keeps the round body uniform.
        k0 = k0.wrapping_add(W0);
        k1 = k1.wrapping_add(W1);
    }
    [c0, c1, c2, c3]
}

/// Writes the stream's words at logical offsets `start ..= start + out.len() - 1`.
///
/// # Errors
///
/// Returns [`RangeError`] if the last requested offset would leave `u64`. The buffer is then
/// left exactly as the caller passed it; nothing is wrapped and nothing is partially filled.
pub fn fill_words(seed: u64, stream: u64, start: u64, out: &mut [u32]) -> Result<(), RangeError> {
    let Some(last) = out.len().checked_sub(1) else {
        return Ok(()); // An empty request is valid at every start, including u64::MAX.
    };
    start.checked_add(length_as_u64(last)).ok_or(RangeError)?;

    let key = [low(seed), high(seed)];
    let mut group = start / 4;
    let mut lane = lane_of(start);
    let mut rest = out;
    while !rest.is_empty() {
        let words = block([low(group), high(group), low(stream), high(stream)], key);
        let taken = (4 - lane).min(rest.len());
        rest[..taken].copy_from_slice(&words[lane..lane + taken]);
        rest = &mut rest[taken..];
        // Only the first block can start mid-way, and the validated range keeps group below 2^62.
        lane = 0;
        group += 1;
    }
    Ok(())
}

/// Maps a word onto `[0, 1)` as `(word >> 8) * 2^-24`, a grid of `2^24` float32 values.
///
/// The top 24 bits are exactly representable in float32 and the scale is a power of two, so the
/// product is exact. The largest result is `1 - 2^-24`, never `1`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "The shifted word is below 2^24, so every value converts to f32 exactly"
)]
pub fn uniform_f32(word: u32) -> f32 {
    (word >> 8) as f32 * UNIFORM_SCALE
}

/// The lane a logical word offset occupies inside its block, always `0..=3`.
const fn lane_of(index: u64) -> usize {
    (index % 4) as usize
}

/// Widens a slice length, lossless on every supported pointer width per the assertion above.
const fn length_as_u64(length: usize) -> u64 {
    length as u64
}

/// The low 32 bits of a 64-bit value.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Truncation to the low half is the intended split"
)]
const fn low(value: u64) -> u32 {
    value as u32
}

/// The high 32 bits of a 64-bit value.
const fn high(value: u64) -> u32 {
    (value >> 32) as u32
}
