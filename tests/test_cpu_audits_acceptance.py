"""Exercise the actual audit tools, including the single reviewed checksum exception."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = Path("docs/evals/2026-09-20-host-contracts.json")
TEST_SOURCE = "tests/test_host_contract_api.py"


def checksum_line():
    # Verify the historical evidence without preventing future edits to the host tests.
    additions = subprocess.check_output(
        ["git", "log", "--diff-filter=A", "--format=%H", "--reverse", "--", TEST_SOURCE],
        cwd=ROOT,
        text=True,
        timeout=20,
    ).splitlines()
    assert additions, "The acceptance source must have a recorded initial revision"
    original = subprocess.check_output(
        ["git", "show", f"{additions[0]}:{TEST_SOURCE}"], cwd=ROOT, timeout=20
    )
    digest = hashlib.sha256(original).hexdigest()
    report = (ROOT / EVALUATION).read_text()
    lines = [line for line in report.splitlines() if f'"{TEST_SOURCE}"' in line]
    assert len(lines) == 1
    assert json.loads("{" + lines[0] + "}")[TEST_SOURCE] == digest
    return lines[0], digest


@pytest.mark.parametrize("case", ["clean", "exact", "changed", "other-path", "nearby-secret"])
def test_actual_gitleaks_exception_is_exact(tmp_path, case):
    line, digest = checksum_line()
    path = tmp_path / EVALUATION
    if case == "clean":
        line = '  "status": "clean control"'
    elif case == "changed":
        changed = ("0" if digest[0] != "0" else "1") + digest[1:]
        line = line.replace(digest, changed)
    elif case == "other-path":
        path = tmp_path / "different-report.json"
    elif case == "nearby-secret":
        synthetic = "ghp_" + hashlib.sha256(b"synthetic-only-cpu-acceptance").hexdigest()[:36]
        line += ',\n  "access_token": "' + synthetic + '"'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{\n" + line + "\n}\n")
    done = subprocess.run(
        [
            "gitleaks",
            "dir",
            str(tmp_path),
            "--config",
            str(ROOT / ".gitleaks.toml"),
            "--redact",
            "--no-banner",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    expected = 0 if case in {"clean", "exact"} else 1
    assert done.returncode == expected, done.stderr


@pytest.mark.parametrize("license_name,allowed", [("Apache-2.0", True), ("GPL-3.0-only", False)])
def test_actual_dependency_audit_checks_unpublished_crate_license(tmp_path, license_name, allowed):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/lib.rs").write_text("pub fn control() -> u32 { 1 }\n")
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname="audit-probe"\nversion="0.1.0"\nedition="2024"\n'
        f'publish=false\nlicense="{license_name}"\n'
    )
    subprocess.run(
        ["cargo", "generate-lockfile", "--offline"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    done = subprocess.run(
        [
            "cargo",
            "deny",
            "--config",
            str(ROOT / "deny.toml"),
            "--locked",
            "--offline",
            "check",
            "licenses",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if allowed:
        assert done.returncode == 0, done.stderr
    else:
        assert done.returncode != 0
        assert "error[rejected]: failed to satisfy license requirements" in done.stderr
        assert license_name in done.stderr
