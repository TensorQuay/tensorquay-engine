"""RNG-005: compare executed Rust and Python, not two fixture-only assertions."""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/philox-v1"
sys.path.insert(0, str(ROOT / "reference/src"))


def rows(name):
    return [line.split() for line in (FIXTURES / name).read_text().splitlines() if line[0] != "#"]


def comparison_lines():
    philox = importlib.import_module("tq_reference.philox")
    actual, expected = [], []
    for index, fields in enumerate(rows("blocks.tsv")):
        value = np.array([int(v, 16) for v in fields[:6]], dtype=np.uint32)
        output = philox.block(value[:4], value[4:])
        actual.append(f"B {index}" + "".join(f" {v:08x}" for v in output))
        expected.append(f"B {index} " + " ".join(fields[6:]))
    for index, fields in enumerate(rows("streams.tsv")):
        seed, stream, start = (int(v, 16) for v in fields[:3])
        output = philox.words(seed, stream, start, int(fields[3]))
        actual.append(f"S {index}" + "".join(f" {v:08x}" for v in output))
        expected.append(f"S {index}" + "".join(f" {v}" for v in fields[4:]))
    for index, fields in enumerate(rows("uniform.tsv")):
        output = philox.uniform_f32(np.array(int(fields[0], 16), dtype=np.uint32))
        actual.append(f"U {index} {output.view(np.uint32).item():08x}")
        expected.append(f"U {index} {fields[1]}")
    assert len(actual) == len(expected) == 455
    assert actual == expected
    return actual


@pytest.mark.parametrize("profile", ["debug", "release"])
def test_executed_cross_language_stream(profile):
    command = [
        "cargo",
        "run",
        "--locked",
        "--offline",
        "--quiet",
        "-p",
        "tq-testkit",
        "--example",
        "philox_bridge",
    ]
    if profile == "release":
        command.append("--release")
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=120, check=True
    )
    assert result.stdout.splitlines() == comparison_lines()
