# RNG-001…005: shared Philox test stream

Status: technical-lead contract approved before implementation on 20 Sep 2026; implementation
accepted the same day. See the [evaluation](../../docs/evals/2026-09-20-philox-foundation.md).
This supplies reproducible test inputs for Phase 0 step 0.3. It is test infrastructure, not the
production token sampler. Implement independently from the published algorithm. Keep one
dependency-free `tq-testkit` Rust crate and one NumPy reference module; no RNG framework or
placeholder runtime crates. Previously accepted references and fixtures remain unchanged.

## Sources and exact primitive

Use **Philox4x32-10**, a 128-bit counter and 64-bit key with ten rounds. Source:
[Random123 v1.14.0](https://github.com/DEShawResearch/random123/tree/726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13),
commit `726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13`. Pin `include/Random123/philox.h` and
`tests/kat_vectors` by SHA-256 in the fixture manifest. Run that upstream implementation
separately as an oracle; do not store its source here.
[NumPy's built-in Philox](https://numpy.org/doc/2.0/reference/random/bit_generators/philox.html)
uses 4x64 and a different seed expansion, so it is not this stream's oracle.

Let `M0=0xd2511f53`, `M1=0xcd9e8d57`, `W0=0x9e3779b9`, `W1=0xbb67ae85`.
For counter words `(c0,c1,c2,c3)` and key words `(k0,k1)`, one round multiplies `M0*c0`
and `M1*c2` as full 64-bit unsigned products, split into `(hi0,lo0)` and `(hi1,lo1)`.
Its output is `(hi1 XOR c1 XOR k0, lo1, hi0 XOR c3 XOR k1, lo0)`.
Round `r` (zero-based, 0…9) uses key `(k0+r*W0, k1+r*W1)` modulo `2^32`.
All outputs are unsigned 32-bit words. Round zero uses the original key. No entropy, global
state, platform-native byte order or library seed expansion participates.

## API and stream mapping v1

```rust
// tq_testkit::philox, safe Rust, no library dependencies
pub fn block(counter: [u32; 4], key: [u32; 2]) -> [u32; 4];
pub fn fill_words(seed: u64, stream: u64, start: u64, out: &mut [u32])
    -> Result<(), RangeError>;
pub fn uniform_f32(word: u32) -> f32;
```

`RangeError` is a public, copyable, comparable unit struct implementing `Debug`, `Display`
and `core::error::Error`. Its display text is `word range exceeds u64`. This slice supports
32- and 64-bit pointer widths. The Rust crate has no allocation, unsafe code or external
dependencies; integration tests may use the standard library.

```python
# tq_reference.philox
class PhiloxError(ValueError): ...

def block(counter: np.ndarray, key: np.ndarray) -> np.ndarray: ...
def words(seed, stream, start, count) -> np.ndarray: ...
def uniform_f32(words: np.ndarray) -> np.ndarray: ...
```

Python `block` requires uint32 arrays of shapes `[4]` and `[2]`, and returns a new uint32 `[4]`
array. `uniform_f32` accepts a uint32 ndarray of any shape and returns a new float32 ndarray
of the same shape, including a zero-dimensional ndarray for scalar-shaped input. Read-only,
non-contiguous and negative-stride input arrays are supported without mutation or aliasing.
Returned arrays are C-contiguous. No implicit dtype conversion is allowed.

Python `seed`, `stream`, `start` and `count` accept Python/NumPy integer scalars, excluding
booleans and arrays, each in `[0, 2^64-1]`. `words` returns a new uint32 `[count]` array.
Resource exhaustion allocating an otherwise valid large result is an ordinary allocation
failure, not a different interpretation of the stream. Tests never request huge allocations.

The **word** at logical offset `i` is lane `i % 4` of a block with:

- key words `[seed low32, seed high32]`;
- counter words `[(i//4) low32, (i//4) high32, stream low32, stream high32]`;
- output lanes in order `0,1,2,3`, with no initial counter increment or discarded block.

`start` is a word offset, not a block offset. Each stream has `2^64` addressable words; the
upper counter words identify its domain and never receive a carry from another stream.
For a nonempty request the final index `start+count-1` must fit in u64. Empty requests are
valid at every valid start, including `2^64-1`. Reject an overflowing range before producing
output; Rust must leave the caller's entire buffer unchanged on rejection. Do not wrap or
partially fill it. Repeated, overlapping and differently partitioned requests give the same
words, independent of request order. Rust fills should reuse each four-word block.

Fixtures assign numeric stream IDs explicitly to test IDs in their manifest. No unstable
language hash or implicit string-to-seed function is allowed. The manifest checks unique test
IDs and unique assigned stream IDs. These are independent test-data domains, not an assertion
that individual random words cannot coincide. Five frozen seeds cover zero, one, a mixed-bit
value, the top bit and the maximum u64 value.

Uniform conversion is exactly `(word >> 8) * 2^-24` as float32, giving `[0,1)` on a grid of
`2^24` values. The top 24 bits are exactly representable in float32. Compare IEEE-754 bits,
never a tolerance. In particular, the largest result is `1-2^-24`, not `1`. Normal transforms,
unbiased bounded integers, production sampling and compatibility with other RNG wrappers are
outside this slice.

## Diagnostics and acceptance

Python invalid inputs raise `PhiloxError` containing `numpy.ndarray`, `uint32`, `shape`, or the
invalid scalar's name (`seed`, `stream`, `start`, `count`), as applicable. An overflowing request
contains `word range`. Scalar diagnostic formatting must also handle arbitrarily large integers
without changing Python's global formatting limit. Missing arguments retain normal `TypeError`.

| ID | Required evidence |
|---|---|
| RNG-001 | All three official ten-round known-answer vectors; asymmetric and high-bit counters/keys; an independently written scalar oracle and inverse-round checks |
| RNG-002 | Frozen seed/stream/word mapping, five seeds, zero and maximum domains, lane offsets, low-counter carry and last legal word; Rust and Python match exactly |
| RNG-003 | Partition and overlap invariance, zero length, atomic exhaustion rejection, typed input boundaries and deterministic repeated calls |
| RNG-004 | Exact float32 bit mapping, endpoint and shift-boundary cases, scalar-shaped and strided Python arrays |
| RNG-005 | Small frozen fixture files, input/output hashes, manifest validation and direct cross-language comparison; no skipped missing-tool or missing-fixture cases |

The lead owns the contract, fixtures, `reference/tests/test_philox_acceptance.py`,
`crates/tq-testkit/tests/philox_acceptance.rs`, the test-only Rust comparison bridge and the
cross-language acceptance runner. Developer tests are additional evidence. Freeze tests before
implementation and show they fail when the implementation is absent. The first Rust workspace
uses edition 2024, a pinned installed stable release, a lockfile and the engineering lints.

The independent Python suite must achieve 100% module line and branch coverage. Independent
Rust acceptance must achieve 100% library line, region and function coverage, without source
exclusions. Run debug and release tests, rustfmt, clippy and documentation checks. Use the
standard LLVM coverage tooling; record exact compiler and tool versions in the evaluation.
Deliberate round, key-schedule, lane, carry, exhaustion and uniform-mapping mutations must fail.

Step 0.3 acceptance requires a fresh comparison with the pinned Random123 implementation,
not merely agreement between two local implementations. The small fixture manifest records
the source revision/hashes, exact encoding and input/output SHA-256 values. Existing large
reference cases are not regenerated with this RNG. Step 0.4's broader host contracts, CI and
dependency/manifest tooling remain separate work; a minimal workspace does not finish them.
