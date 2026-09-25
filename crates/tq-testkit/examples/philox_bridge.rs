//! Test-only bridge: reads frozen inputs and prints actual Rust outputs for Python comparison.
#![allow(
    clippy::expect_used,
    reason = "Invalid frozen fixture inputs fail acceptance"
)]
#![allow(
    clippy::print_stdout,
    reason = "This executable is the acceptance output protocol"
)]

use tq_testkit::philox::{block, fill_words, uniform_f32};

fn rows(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines()
        .filter(|s| !s.starts_with('#'))
        .map(|s| s.split_whitespace().collect())
}

fn hex32(text: &str) -> u32 {
    u32::from_str_radix(text, 16).expect("uint32 fixture input")
}

fn hex64(text: &str) -> u64 {
    u64::from_str_radix(text, 16).expect("uint64 fixture input")
}

fn main() {
    let blocks = include_str!("../../../tests/fixtures/philox-v1/blocks.tsv");
    for (index, fields) in rows(blocks).enumerate() {
        let counter = [
            hex32(fields[0]),
            hex32(fields[1]),
            hex32(fields[2]),
            hex32(fields[3]),
        ];
        let key = [hex32(fields[4]), hex32(fields[5])];
        let out = block(counter, key);
        println!(
            "B {index} {:08x} {:08x} {:08x} {:08x}",
            out[0], out[1], out[2], out[3]
        );
    }
    let streams = include_str!("../../../tests/fixtures/philox-v1/streams.tsv");
    for (index, fields) in rows(streams).enumerate() {
        let mut out = vec![0; fields[3].parse().expect("word count")];
        fill_words(
            hex64(fields[0]),
            hex64(fields[1]),
            hex64(fields[2]),
            &mut out,
        )
        .expect("valid frozen range");
        print!("S {index}");
        for value in out {
            print!(" {value:08x}");
        }
        println!();
    }
    let uniforms = include_str!("../../../tests/fixtures/philox-v1/uniform.tsv");
    for (index, fields) in rows(uniforms).enumerate() {
        println!("U {index} {:08x}", uniform_f32(hex32(fields[0])).to_bits());
    }
}
