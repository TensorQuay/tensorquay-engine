"""Independent collection evidence: real test names, markers and lossless transport."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collection_hook_preserves_names_and_excludes_all_markers(tmp_path, monkeypatch):
    hook = load("tq_collection_hook", ROOT / "tools/tq_pytest_inventory.py")
    output = tmp_path / "inventory.json"
    monkeypatch.setenv("TQ_INVENTORY_ROOT", str(tmp_path))
    monkeypatch.setenv("TQ_INVENTORY_OUT", str(output))
    items = []
    names = ["tests/test_same.py", "reference/tests/test_same.py", "tests/newline\nfile.py"]
    node = 'test_values[space, quote" and newline\nvalue]'
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        items.append(
            SimpleNamespace(
                path=path, nodeid=f"test_same.py::{node}", get_closest_marker=lambda _: None
            )
        )
    for marker in [pytest.mark.skip(), pytest.mark.skipif(False), pytest.mark.xfail()]:
        item = SimpleNamespace(
            path=tmp_path / names[0],
            nodeid=f"test_same.py::test_{marker.name}",
            get_closest_marker=lambda key, mark=marker.mark: mark if key == mark.name else None,
        )
        items.append(item)
    hook.pytest_collection_modifyitems(None, None, items)
    assert set(json.loads(output.read_text())) == {f"{name}::{node}" for name in names}
    hook.pytest_collection_modifyitems(None, None, [])
    assert json.loads(output.read_text()) == []


@pytest.fixture
def collected_project(tmp_path, monkeypatch):
    root = tmp_path / "collected project"
    for directory in ["docs", "tests", "reference/tests", "tools", "bin"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / ".gitignore").write_text("target/\n")
    (root / "docs/CPU-CHECKS.md").write_text(
        "## CPU clause catalogue\n\n| ID | Obligation |\n|---|---|\n"
        "| REF-001 | Collected Python behavior |\n| HOST-001 | Collected Rust behavior |\n"
    )
    (root / "tests/test_sample.py").write_text("""import pytest

@pytest.mark.parametrize("value", [1, 2], ids=["one", "two"])
def test_run(value):
    assert value > 0

@pytest.mark.skip(reason="not evidence")
def test_skip():
    pass

@pytest.mark.skipif(False, reason="conservatively excluded")
def test_skipif():
    pass

@pytest.mark.xfail(reason="not evidence")
def test_xfail():
    pass

@pytest.mark.skip(reason="inherited marker")
class TestMarked:
    def test_inherited(self):
        pass
""")
    (root / "reference/tests/test_second.py").write_text("def test_other():\n    assert True\n")
    shutil.copy2(ROOT / "tools/tq_pytest_inventory.py", root / "tools/tq_pytest_inventory.py")
    listing = {
        "rust-suites": {
            "binary": {
                "binary-id": "tq-core::host_acceptance",
                "testcases": {"test_run": {"ignored": False}, "test_ignored": {"ignored": True}},
            }
        },
    }
    (root / "listing.json").write_text(json.dumps(listing))
    cargo = root / "bin/cargo"
    cargo.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nprint(Path('listing.json').read_text())\n"
    )
    uv = root / "bin/uv"
    uv.write_text(
        f"#!{sys.executable}\nimport subprocess, sys\n"
        "args = sys.argv[sys.argv.index('pytest') + 1:]\n"
        "sys.exit(subprocess.run([sys.executable, '-m', 'pytest', *args]).returncode)\n"
    )
    cargo.chmod(0o755)
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", str(root / "bin") + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    cli = load("tq_inventory_cli", ROOT / "tools/cpu_checks.py")
    return root, cli


def manifest(root, python="tests/test_sample.py::test_run", rust="test_run"):
    (root / "tests/manifest.toml").write_text(
        'schema = 1\nscope = "cpu-foundation"\nkernel_specializations = []\n[clauses]\n'
        f'"REF-001" = [{json.dumps("python:" + python)}]\n'
        f'"HOST-001" = [{json.dumps("rust:tq-core::host_acceptance::" + rust)}]\n'
    )


def test_manifest_cli_accepts_real_collection(collected_project):
    root, cli = collected_project
    manifest(root)
    assert cli.main(["manifest", "--root", str(root)]) == 0


@pytest.mark.parametrize(
    "test",
    ["test_skip", "test_skipif", "test_xfail", "TestMarked::test_inherited", "test_nonexistent"],
)
def test_manifest_cli_rejects_non_evidence_python(collected_project, test):
    root, cli = collected_project
    manifest(root, python="tests/test_sample.py::" + test)
    assert cli.main(["manifest", "--root", str(root)]) != 0


def test_manifest_cli_rejects_ignored_rust(collected_project):
    root, cli = collected_project
    manifest(root, rust="test_ignored")
    assert cli.main(["manifest", "--root", str(root)]) != 0


@pytest.mark.parametrize(
    "case",
    [
        "rust-empty",
        "python-error",
        "bad-toml",
        "kernel",
        "no-catalogue",
        "empty-catalogue",
        "duplicate-id",
        "bad-row",
    ],
)
def test_manifest_cli_collection_and_catalogue_failures(collected_project, case):
    root, cli = collected_project
    manifest(root)
    catalogue = root / "docs/CPU-CHECKS.md"
    if case == "rust-empty":
        (root / "listing.json").write_text('{"rust-suites": {}}')
    elif case == "python-error":
        (root / "tests/test_sample.py").write_text("raise RuntimeError('collection failed')\n")
    elif case == "bad-toml":
        (root / "tests/manifest.toml").write_text("not = [valid\n")
    elif case == "kernel":
        (root / "unsupported_kernel.rs").write_text("// not implemented\n")
    elif case == "no-catalogue":
        catalogue.write_text("No catalogue here.\n")
    elif case == "empty-catalogue":
        catalogue.write_text("## CPU clause catalogue\n\n| ID | Obligation |\n|---|---|\n")
    elif case == "duplicate-id":
        catalogue.write_text(catalogue.read_text() + "| REF-001 | Duplicate |\n")
    else:
        catalogue.write_text(catalogue.read_text() + "| broken-id | Incomplete row | extra |\n")
    assert cli.main(["manifest", "--root", str(root)]) != 0
