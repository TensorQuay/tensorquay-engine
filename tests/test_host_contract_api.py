"""External Rust consumers check the validated types' privacy and borrowed plan."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = "use tq_core::contracts::{ContractError, MoeParams, MoeSpec, SmlaParams, SmlaSpec};\n"


@pytest.fixture(scope="module")
def host_library(tmp_path_factory):
    target = tmp_path_factory.mktemp("host-api-build")
    built = subprocess.run(
        [
            "cargo",
            "build",
            "-p",
            "tq-core",
            "--locked",
            "--offline",
            "--target-dir",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    library = target / "debug/libtq_core.rlib"
    assert library.is_file()
    return library, target / "debug/deps"


def compile_consumer(host_library, tmp_path, body):
    library, dependencies = host_library
    source = tmp_path / "consumer.rs"
    source.write_text(IMPORTS + body)
    # Run from the repository so rustup resolves the committed compiler pin.
    result = subprocess.run(
        [
            "rustc",
            "--edition=2024",
            "--crate-type=lib",
            "--emit=metadata",
            "--error-format=json",
            "--extern",
            f"tq_core={library}",
            "-L",
            f"dependency={dependencies}",
            "--out-dir",
            str(tmp_path),
            str(source),
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    diagnostics = [json.loads(line) for line in result.stderr.splitlines()]
    codes = {
        item["code"]["code"]
        for item in diagnostics
        if item.get("level") == "error" and item.get("code")
    }
    return result, codes


def test_positive_public_api_consumer(host_library, tmp_path):
    result, codes = compile_consumer(
        host_library,
        tmp_path,
        """
pub fn smla(s: SmlaSpec, counts: &[u8]) -> Result<SmlaSpec, ContractError> {
    let p = SmlaParams::new(s, counts)?;
    let _ = (p.should_launch(), p.query_counts());
    let mut copy = p.spec();
    copy.q.shape[1] = 48;
    Ok(copy)
}
pub fn moe(s: MoeSpec) -> Result<MoeSpec, ContractError> {
    let p = MoeParams::new(s)?;
    let _ = p.should_launch();
    Ok(p.spec())
}
""",
    )
    assert result.returncode == 0, result.stderr
    assert not codes


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "pub fn break_it(p: &mut SmlaParams<'_>) { p.spec.q.shape[1] = 48; }",
            "E0616",
        ),
        (
            "pub fn break_it(p: &mut MoeParams) { p.spec.x.shape[1] = 1; }",
            "E0616",
        ),
        (
            """
pub fn break_it(s: SmlaSpec) -> Result<(), ContractError> {
    let mut counts = [1];
    let p = SmlaParams::new(s, &counts)?;
    counts[0] = 5;
    let _ = p.query_counts();
    Ok(())
}
""",
            "E0506",
        ),
        (
            """
pub fn break_it(s: SmlaSpec) -> Result<SmlaParams<'static>, ContractError> {
    let counts = [1];
    SmlaParams::new(s, &counts)
}
""",
            "E0515",
        ),
        ("pub fn break_it() -> SmlaParams<'static> { Default::default() }", "E0277"),
        ("pub fn break_it() -> MoeParams { Default::default() }", "E0277"),
    ],
)
def test_rejects_invariant_bypasses(host_library, tmp_path, body, expected):
    result, codes = compile_consumer(host_library, tmp_path, body)
    assert result.returncode != 0, "invalid external consumer compiled"
    assert expected in codes, result.stderr
