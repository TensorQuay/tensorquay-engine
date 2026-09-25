"""Independent RNG-005 regressions: corrupted evidence must fail before running an oracle."""

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checker(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "philox_checker", ROOT / "tools/check-philox-upstream.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = tmp_path / "fixtures"
    shutil.copytree(ROOT / "tests/fixtures/philox-v1", destination)
    module.FIXTURES = destination
    monkeypatch.setattr(sys, "argv", ["checker", "--source", "unused-for-invalid-fixtures"])
    monkeypatch.setattr(module, "verify_source", lambda *_: None)

    def must_not_run(*_):
        pytest.fail("Corrupted fixture reached the upstream compiler/runner")

    monkeypatch.setattr(module, "run_upstream", must_not_run)
    return module


def change_manifest(checker, transform):
    path = checker.FIXTURES / "manifest.json"
    value = json.loads(path.read_text())
    transform(value)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("name", ["blocks.tsv", "streams.tsv", "uniform.tsv"])
def test_deleted_cases_with_original_manifest_are_rejected(checker, name):
    (checker.FIXTURES / name).write_text("# all cases removed\n")
    with pytest.raises(SystemExit):
        checker.main()


@pytest.mark.parametrize("name", ["blocks.tsv", "streams.tsv", "uniform.tsv"])
def test_empty_rehashed_fixture_is_still_rejected(checker, name):
    path = checker.FIXTURES / name
    path.write_text("# all cases removed\n")
    change_manifest(
        checker,
        lambda m: m["files"][name].update(
            cases=0, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    with pytest.raises(SystemExit):
        checker.main()


@pytest.mark.parametrize("name", ["blocks.tsv", "streams.tsv", "uniform.tsv"])
def test_declared_count_must_match_file(checker, name):
    change_manifest(checker, lambda m: m["files"][name].update(cases=12345))
    with pytest.raises(SystemExit):
        checker.main()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("algorithm", "Philox4x64-10"),
        ("mapping", "changed"),
        ("uniform", "closed-closed"),
    ],
)
def test_manifest_cannot_select_another_contract(checker, field, value):
    change_manifest(checker, lambda m: m.update({field: value}))
    with pytest.raises(SystemExit):
        checker.main()


def test_manifest_cannot_redirect_revision(checker):
    change_manifest(checker, lambda m: m["source"].update(revision="0" * 40))
    with pytest.raises(SystemExit):
        checker.main()


@pytest.mark.parametrize("extra", [False, True])
def test_required_fixture_set_is_exact(checker, extra):
    def change(manifest):
        if extra:
            manifest["files"]["extra.tsv"] = manifest["files"]["blocks.tsv"]
        else:
            del manifest["files"]["blocks.tsv"]

    change_manifest(checker, change)
    with pytest.raises(SystemExit):
        checker.main()


@pytest.mark.parametrize("name", ["blocks.tsv", "streams.tsv", "uniform.tsv"])
def test_rehashed_malformed_row_cannot_reach_upstream(checker, name):
    path = checker.FIXTURES / name
    lines = path.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    lines[first] = "invalid row"
    path.write_text("\n".join(lines) + "\n")
    change_manifest(
        checker,
        lambda m: m["files"][name].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    with pytest.raises(SystemExit):
        checker.main()


def test_source_hashes_cannot_be_omitted(checker):
    change_manifest(checker, lambda m: m["source"].update(sha256={}))
    with pytest.raises(SystemExit):
        checker.main()
