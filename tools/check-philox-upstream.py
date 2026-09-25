#!/usr/bin/env python3
"""Check the frozen Philox fixtures against the original Random123 implementation.

This runs the upstream code itself: it compiles a small C++ caller of `r123::Philox4x32_R<10>`
against a clean checkout and compares the words that implementation actually returns with the
expected outputs already frozen in `tests/fixtures/philox-v1`.

The approved revision, algorithm, mapping and required fixture set are constants here, so a
changed manifest cannot redirect the check or empty it. Every fixture file is hashed and counted
against the manifest, and every row is shape- and range-checked, before anything is compiled.

It deliberately imports nothing from `tq_reference` and no local test oracle, and it never
reimplements or copies the Philox algorithm. The only arithmetic here is the `tq-philox-v1`
counter mapping, which is ours and not upstream's: word `i` is lane `i % 4` of the block whose
counter is `[(i//4) low, (i//4) high, stream low, stream high]`.

The checkout stays outside the repository and is never written to or downloaded.

    tools/check-philox-upstream.py --source /path/to/random123
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/philox-v1"
MASK32 = (1 << 32) - 1
MAX64 = (1 << 64) - 1

REVISION = "726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13"
ALGORITHM = "Philox4x32-10"
MAPPING = "tq-philox-v1"
UNIFORM = "high24-times-2^-24"
SCHEMA_VERSION = 1
REQUIRED_FILES = ("blocks.tsv", "streams.tsv", "uniform.tsv")
REQUIRED_SOURCES = ("include/Random123/philox.h", "tests/kat_vectors")
WORD = re.compile(r"[0-9a-f]{8}\Z")
LONG_WORD = re.compile(r"[0-9a-f]{16}\Z")

CALLER = r"""
#include <cstdio>
#include <cstdint>
#include <cctype>
#include <Random123/philox.h>

int main() {
    long long expected = 0;
    if (scanf("%lld", &expected) != 1) {
        fprintf(stderr, "missing input count\n");
        return 2;
    }
    if (expected < 0) {
        fprintf(stderr, "negative input count %lld\n", expected);
        return 5;
    }
    r123::Philox4x32_R<10> generator;
    for (long long i = 0; i < expected; i++) {
        uint32_t c0, c1, c2, c3, k0, k1;
        if (scanf("%u %u %u %u %u %u", &c0, &c1, &c2, &c3, &k0, &k1) != 6) {
            fprintf(stderr, "malformed input row %lld\n", i);
            return 3;
        }
        r123::Philox4x32_R<10>::ctr_type counter = {{c0, c1, c2, c3}};
        r123::Philox4x32_R<10>::key_type key = {{k0, k1}};
        r123::Philox4x32_R<10>::ctr_type out = generator(counter, key);
        printf("%08x %08x %08x %08x\n", out[0], out[1], out[2], out[3]);
    }
    int trailing;
    while ((trailing = getchar()) != EOF) {
        if (!isspace(trailing)) {
            fprintf(stderr, "unexpected trailing input\n");
            return 4;
        }
    }
    return 0;
}
"""


def fail(message: str) -> None:
    sys.exit(f"check-philox-upstream: {message}")


def rows(name: str) -> list[list[str]]:
    lines = (FIXTURES / name).read_text().splitlines()
    return [line.split() for line in lines if line and line[0] != "#"]


def word(text: str, name: str) -> int:
    if not WORD.match(text):
        fail(f"{name}: {text!r} is not eight lowercase hexadecimal digits")
    return int(text, 16)


def long_word(text: str, name: str) -> int:
    if not LONG_WORD.match(text):
        fail(f"{name}: {text!r} is not sixteen lowercase hexadecimal digits")
    return int(text, 16)


def load_manifest() -> dict:
    """Pin the manifest against this tool's own constants, then against the files on disk."""
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for field, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("algorithm", ALGORITHM),
        ("mapping", MAPPING),
        ("uniform", UNIFORM),
    ):
        if manifest.get(field) != expected:
            fail(f"manifest {field} is {manifest.get(field)!r}, not the approved {expected!r}")
    sources = manifest.get("source", {}).get("sha256", {})
    if tuple(sorted(sources)) != tuple(sorted(REQUIRED_SOURCES)):
        fail(f"manifest pins source files {sorted(sources)}, not {sorted(REQUIRED_SOURCES)}")
    if manifest.get("source", {}).get("revision") != REVISION:
        fail(
            f"manifest pins revision {manifest.get('source', {}).get('revision')!r}, not {REVISION}"
        )
    if tuple(sorted(manifest.get("files", {}))) != tuple(sorted(REQUIRED_FILES)):
        fail(
            f"manifest lists files {sorted(manifest.get('files', {}))}, "
            f"not {sorted(REQUIRED_FILES)}"
        )
    for name in REQUIRED_FILES:
        entry = manifest["files"][name]
        digest = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            fail(f"{name} hashes to {digest}, not the manifest's {entry['sha256']}")
        if not isinstance(entry["cases"], int) or entry["cases"] < 1:
            fail(f"{name} declares {entry['cases']!r} cases; a frozen set must not be empty")
        if len(rows(name)) != entry["cases"]:
            fail(f"{name} holds {len(rows(name))} rows, not the manifest's {entry['cases']}")
    return manifest


def verify_source(source: Path, manifest: dict) -> None:
    """The checkout must be a clean git tree at the approved revision with the pinned contents."""
    if not (source / ".git").exists():
        fail(f"{source} is not a git checkout")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source), *arguments], capture_output=True, text=True, check=True
        ).stdout.strip()

    revision = git("rev-parse", "HEAD")
    if revision != REVISION:
        fail(f"checkout is at {revision}, not the approved {REVISION}")
    dirty = git("status", "--porcelain")
    if dirty:
        fail(f"checkout has uncommitted changes:\n{dirty}")
    for relative, digest in manifest["source"]["sha256"].items():
        actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        if actual != digest:
            fail(f"{relative} hashes to {actual}, not the manifest's {digest}")


def block_cases() -> tuple[list[tuple[int, ...]], list[list[int]]]:
    inputs, expected = [], []
    for index, fields in enumerate(rows("blocks.tsv")):
        if len(fields) != 10:
            fail(f"blocks.tsv row {index} has {len(fields)} fields, expected 10")
        values = [word(field, f"blocks.tsv row {index}") for field in fields]
        inputs.append(tuple(values[:6]))
        expected.append(values[6:])
    return inputs, expected


def stream_cases() -> tuple[list[tuple[int, ...]], list[tuple[int, int, list[int]]]]:
    """Map each frozen request onto the distinct blocks it needs, using the v1 mapping."""
    inputs, wanted = [], []
    for index, fields in enumerate(rows("streams.tsv")):
        where = f"streams.tsv row {index}"
        if len(fields) < 4:
            fail(f"{where} has {len(fields)} fields, expected at least 4")
        seed, stream, start = (long_word(fields[position], where) for position in range(3))
        if not fields[3].isdigit():
            fail(f"{where}: count {fields[3]!r} is not a decimal number")
        count = int(fields[3])
        if len(fields) != 4 + count:
            fail(f"{where} declares {count} words but lists {len(fields) - 4}")
        if count and start + count - 1 > MAX64:
            fail(f"{where}: words {start}..{start + count - 1} leave the u64 range")
        expected = [word(field, where) for field in fields[4:]]
        offset = len(inputs)
        if count:  # An empty request needs no block at all, whatever its starting lane.
            first = start // 4
            for group in range(first, (start + count - 1) // 4 + 1):
                inputs.append(
                    (
                        group & MASK32,
                        group >> 32,
                        stream & MASK32,
                        stream >> 32,
                        seed & MASK32,
                        seed >> 32,
                    )
                )
            wanted.append((offset, start - first * 4, expected))
        else:
            wanted.append((offset, 0, expected))
    return inputs, wanted


def check_uniform_rows() -> None:
    """Uniform rows feed no upstream call, but a frozen set must still be well formed."""
    for index, fields in enumerate(rows("uniform.tsv")):
        if len(fields) != 2:
            fail(f"uniform.tsv row {index} has {len(fields)} fields, expected 2")
        for field in fields:
            word(field, f"uniform.tsv row {index}")


def run_upstream(source: Path, counters: list[tuple[int, ...]]) -> list[list[int]]:
    """Compile and run the C++ caller, returning the blocks upstream actually produced."""
    with tempfile.TemporaryDirectory() as directory:
        caller = Path(directory) / "caller.cpp"
        binary = Path(directory) / "caller"
        caller.write_text(CALLER)
        built = subprocess.run(
            [
                "clang++",
                "-O2",
                "-std=c++17",
                f"-I{source / 'include'}",
                str(caller),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if built.returncode:
            fail(f"compiling the upstream caller failed:\n{built.stderr}")
        stdin = f"{len(counters)}\n" + "".join(
            " ".join(str(value) for value in row) + "\n" for row in counters
        )
        result = subprocess.run(
            [str(binary)], input=stdin, capture_output=True, text=True, check=False, timeout=300
        )
    if result.returncode:
        fail(f"the upstream caller exited {result.returncode}: {result.stderr.strip()}")
    produced = [[int(value, 16) for value in line.split()] for line in result.stdout.splitlines()]
    if len(produced) != len(counters):
        fail(f"upstream returned {len(produced)} blocks for {len(counters)} inputs")
    for index, block in enumerate(produced):
        if len(block) != 4:
            fail(f"upstream block {index} has {len(block)} words, expected 4")
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path, help="clean Random123 checkout")
    source = parser.parse_args().source

    manifest = load_manifest()
    verify_source(source, manifest)
    check_uniform_rows()
    block_inputs, block_expected = block_cases()
    stream_inputs, stream_wanted = stream_cases()
    produced = run_upstream(source, block_inputs + stream_inputs)

    blocks = produced[: len(block_inputs)]
    for index, (actual, expected) in enumerate(zip(blocks, block_expected, strict=True)):
        if actual != expected:
            fail(f"block case {index}: upstream gave {actual}, fixture expects {expected}")

    tail = produced[len(block_inputs) :]
    for index, (offset, lane, expected) in enumerate(stream_wanted):
        needed = 0 if not expected else (lane + len(expected) + 3) // 4
        flat = [value for group in tail[offset : offset + needed] for value in group]
        actual = flat[lane : lane + len(expected)]
        if actual != expected:
            fail(f"stream case {index}: upstream gave {actual}, fixture expects {expected}")

    print(
        f"upstream {REVISION[:12]} reproduced {len(block_expected)} blocks and "
        f"{len(stream_wanted)} stream requests"
    )


if __name__ == "__main__":
    main()
