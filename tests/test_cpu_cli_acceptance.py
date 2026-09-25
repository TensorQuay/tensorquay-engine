"""Independent executable checks for CPU gate entry points and failure handling."""

import importlib.util
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("tq_cpu_cli", ROOT / "tools/cpu_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(cli, args):
    try:
        result = cli.main(args)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else int(error.code is not None)
    assert type(result) is int
    return result


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / ".gitignore").write_text("ignored/\n")
    (root / "tracked.rs").write_text("// tracked\n")
    subprocess.run(["git", "add", ".gitignore", "tracked.rs"], cwd=root, check=True)
    (root / "untracked space.py").write_text("# untracked\n")
    (root / "newline\nsource.sh").write_text("# NUL inventory boundary\n")
    (root / "ignored").mkdir()
    (root / "ignored/too_long.py").write_text("# ignored\n" * 1001)
    return root


def test_length_cli_reads_git_nul_inventory(cli, checkout, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert invoke(cli, ["lengths", "--root", str(checkout)]) == 0
    (checkout / "untracked space.py").write_text("# too long\n" * 1001)
    assert invoke(cli, ["lengths", "--root", str(checkout)]) != 0


def test_length_cli_does_not_prune_target_named_source(cli, checkout):
    (checkout / "target_sources").mkdir()
    (checkout / "target_sources/check.py").write_text("# ordinary source\n" * 1001)
    assert invoke(cli, ["lengths", "--root", str(checkout)]) != 0


def test_length_cli_git_failure_is_not_success(cli, tmp_path):
    (tmp_path / "file.py").write_text("# no repository\n")
    assert invoke(cli, ["lengths", "--root", str(tmp_path)]) != 0


@pytest.mark.parametrize("args", [[], ["unknown"], ["coverage"], ["lengths", "--unknown"]])
def test_cli_invalid_usage_fails(cli, args):
    assert invoke(cli, args) != 0


def test_dependency_cli_checks_actual_workspace(cli, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert invoke(cli, ["deps", "--root", str(ROOT)]) == 0


@pytest.mark.parametrize("output,exit_code", [("", 47), ("not-json", 0), ("{}", 0)])
def test_dependency_cli_subprocess_and_bad_data_fail(cli, tmp_path, monkeypatch, output, exit_code):
    executable = tmp_path / "cargo"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stdout.write({output!r})\nsys.exit({exit_code})\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    assert invoke(cli, ["deps", "--root", str(ROOT)]) != 0


def coverage_payload():
    files = []
    for crate, module in [("tq-core", "contracts"), ("tq-testkit", "philox")]:
        files.append(
            {
                "filename": str(ROOT / f"crates/{crate}/src/{module}.rs"),
                "summary": {
                    name: {"count": 3, "covered": 3, "percent": 100.0}
                    for name in ["lines", "regions", "functions"]
                },
            }
        )
    return {"data": [{"files": files}]}


def test_coverage_cli_raw_counts_control_exit(cli, tmp_path):
    path = tmp_path / "coverage.json"
    payload = coverage_payload()
    path.write_text(json.dumps(payload))
    assert invoke(cli, ["coverage", str(path), "--root", str(ROOT)]) == 0
    payload["data"][0]["files"][0]["summary"]["functions"]["covered"] = 2
    path.write_text(json.dumps(payload))
    assert invoke(cli, ["coverage", str(path), "--root", str(ROOT)]) != 0


@pytest.mark.parametrize("contents", [None, "not json", "{}", '{"data": []}'])
def test_coverage_cli_missing_or_invalid_file(cli, tmp_path, contents):
    path = tmp_path / "report.json"
    if contents is not None:
        path.write_text(contents)
    assert invoke(cli, ["coverage", str(path), "--root", str(ROOT)]) != 0


def test_coverage_cli_cannot_forget_a_missing_workspace_crate(cli, tmp_path, monkeypatch):
    report = tmp_path / "coverage.json"
    payload = coverage_payload()
    metadata = {"workspace_members": ["a", "b"], "packages": []}
    for identity, entry, crate in zip(
        ["a", "b"], payload["data"][0]["files"], ["tq-core", "tq-testkit"], strict=True
    ):
        path = tmp_path / "crates" / crate / "src" / Path(entry["filename"]).name
        path.parent.mkdir(parents=True)
        path.write_text("// coverage fixture\n")
        entry["filename"] = str(path)
        metadata["packages"].append({"id": identity, "name": crate})
    executable = tmp_path / "cargo"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stdout.write({json.dumps(metadata)!r})\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    report.write_text(json.dumps(payload))
    assert invoke(cli, ["coverage", str(report), "--root", str(tmp_path)]) == 0
    first = Path(payload["data"][0]["files"][0]["filename"])
    first.unlink()
    first.parent.rmdir()
    payload["data"][0]["files"].pop(0)
    report.write_text(json.dumps(payload))
    assert invoke(cli, ["coverage", str(report), "--root", str(tmp_path)]) != 0


def test_script_main_entry_point(checkout, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cpu_checks.py", "lengths", "--root", str(checkout)])
    try:
        runpy.run_path(str(ROOT / "tools/cpu_checks.py"), run_name="__main__")
    except SystemExit as error:
        assert error.code == 0


@pytest.mark.parametrize("gate", ["lengths", "deps", "manifest", "coverage"])
def test_shell_wrappers_forward_arguments_and_exit_codes(tmp_path, gate):
    script = ROOT / f"scripts/check-{gate}.sh"
    assert script.is_file()
    assert os.access(script, os.X_OK)
    log = tmp_path / "calls.jsonl"
    executable = tmp_path / "uv"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['TQ_TEST_CALL_LOG'], 'a') as f:\n"
        "    f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n"
        "sys.exit(int(os.environ['TQ_TEST_EXIT']))\n"
    )
    executable.chmod(0o755)
    args = (["a report.json"] if gate == "coverage" else []) + ["--root", "literal space & value"]
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    env.update(TQ_TEST_CALL_LOG=str(log), TQ_TEST_EXIT="0")
    passed = subprocess.run(
        [str(script), *args], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert passed.returncode == 0, passed.stderr
    env["TQ_TEST_EXIT"] = "47"
    failed = subprocess.run(
        [str(script), *args], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert failed.returncode == 47, failed.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    for call in calls:
        assert Path(call["cwd"]).resolve() == ROOT.resolve()
        command = call["args"]
        assert command[0] == "run"
        assert "--locked" in command and "--offline" in command and "--project" in command
        position = next(i for i, value in enumerate(command) if value.endswith("cpu_checks.py"))
        assert command[position + 1 :] == [gate, *args]
