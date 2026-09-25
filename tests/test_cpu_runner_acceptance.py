"""Execute the CPU shell runner against recording tools and inject gate failures."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Runner:
    checkout: Path
    environment: dict = field(repr=False)


# These stand-ins test orchestration only. Actual tools and product tests run
# separately for acceptance; a mocked green result is never their evidence.
TOOL = r"""
import json, os, pathlib, subprocess, sys

tool = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if tool.startswith('cargo-'):
    args = [tool.removeprefix('cargo-'), *args]
    tool = 'cargo'
joined = ' '.join(args)
key = tool
if tool == 'cargo':
    key += ':' + (args[0] if args else '')
    if args and args[0] == 'build':
        key += ':release' if '--release' in args else ':debug'
elif tool == 'uv':
    if 'cpu_checks.py' in joined:
        index = next(i for i, a in enumerate(args) if a.endswith('cpu_checks.py'))
        key += ':checks:' + args[index + 1]
    elif 'ruff' in args:
        key += ':ruff:' + args[args.index('ruff') + 1]
    elif 'pytest' in args:
        key += ':pytest'
elif tool in ('python', 'python3') and 'git-guard.py' in joined:
    key = 'guard'
elif tool == 'gitleaks':
    key += ':' + (args[0] if args else '')

record = {'key': key, 'tool': tool, 'args': args, 'cwd': os.getcwd(),
          'require_private': os.environ.get('TQ_GUARD_REQUIRE_DENYLIST'),
          'rustdocflags': os.environ.get('RUSTDOCFLAGS')}
with open(os.environ['TQ_TEST_TRACE'], 'a') as stream:
    stream.write(json.dumps(record) + '\n')

version = '--version' in args or '-version' in args or args == ['version']
if version:
    versions = {'cargo': 'cargo 1.95.0', 'uv': 'uv 0.10.12',
                'gitleaks': '8.30.1', 'actionlint': '1.7.12', 'rustc': 'rustc 1.95.0'}
    if tool == 'cargo' and args[0] in ('deny', 'llvm-cov', 'nextest'):
        versions['cargo'] = {'deny': 'cargo-deny 0.20.2',
                             'llvm-cov': 'cargo-llvm-cov 0.9.1',
                             'nextest': 'cargo-nextest 0.9.145'}[args[0]]
    reported = versions.get(tool, '1.0.0')
    if os.environ.get('TQ_TEST_PIN_DRIFT') == tool:
        reported += '0'
    print(reported)
    sys.exit(0)

if key == os.environ.get('TQ_TEST_FAIL_AT') and (
    not os.environ.get('TQ_TEST_FAIL_PHASE') or os.environ['TQ_TEST_FAIL_PHASE'] in args
):
    sys.exit(47)

if key == 'guard':
    if os.environ.get('TQ_GUARD_REQUIRE_DENYLIST') == '1':
        path = pathlib.Path(os.environ['TQ_GUARD_DENYLIST'])
        if not path.is_file() or not path.read_text().strip():
            sys.exit(48)
elif tool in ('python', 'python3'):
    sys.exit(subprocess.run([os.environ['TQ_TEST_REAL_PYTHON'], *args]).returncode)
elif tool == 'git':
    if 'rev-parse' in args:
        if os.environ.get('TQ_TEST_BAD_DB_REVISION') == '1':
            sys.exit(46)
        print('a' * 40)
        sys.exit(0)
    if 'ls-files' in args:
        sys.stdout.buffer.write(pathlib.Path(os.environ['TQ_TEST_FILES']).read_bytes())
        sys.exit(0)
elif key == 'cargo:metadata':
    print(pathlib.Path(os.environ['TQ_TEST_METADATA']).read_text())
elif key == 'cargo:llvm-cov' and '--output-path' in args:
    path = pathlib.Path(args[args.index('--output-path') + 1])
    if os.environ.get('TQ_TEST_NO_COVERAGE') != '1':
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
elif key == 'uv:checks:coverage':
    index = next(i for i, a in enumerate(args) if a.endswith('cpu_checks.py'))
    path = pathlib.Path(args[index + 2])
    if not path.is_file():
        sys.exit(49)

print('recorded tool completed')
"""


def test_recording_tool_positive_and_failure_controls(tmp_path):
    executable = tmp_path / "cargo"
    executable.write_text(f"#!{sys.executable}\n" + TOOL)
    executable.chmod(0o755)
    trace = tmp_path / "trace.jsonl"
    env = dict(os.environ, TQ_TEST_TRACE=str(trace))
    good = subprocess.run([str(executable), "fmt"], env=env, capture_output=True, text=True)
    assert good.returncode == 0
    env["TQ_TEST_FAIL_AT"] = "cargo:fmt"
    bad = subprocess.run([str(executable), "fmt"], env=env, capture_output=True, text=True)
    assert bad.returncode == 47
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [record["key"] for record in records] == ["cargo:fmt", "cargo:fmt"]


@pytest.fixture
def runner(tmp_path):
    assert (ROOT / "scripts/check-cpu.sh").is_file()
    checkout = tmp_path / "runner checkout"
    checkout.mkdir()
    # Copy only repository inputs; never the environment or build directories.
    names = (
        subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT)
        .decode()
        .strip("\0")
        .split("\0")
    )
    for name in set(names):
        original = ROOT / name
        if original.is_file():
            destination = checkout / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in [
        "cargo",
        "cargo-deny",
        "cargo-nextest",
        "cargo-llvm-cov",
        "rustc",
        "uv",
        "python",
        "python3",
        "git",
        "gitleaks",
        "actionlint",
    ]:
        executable = binary / name
        executable.write_text(f"#!{sys.executable}\n" + TOOL)
        executable.chmod(0o755)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "workspace_members": ["core", "kit"],
                "packages": [
                    {
                        "id": "core",
                        "name": "tq-core",
                        "manifest_path": str(checkout / "crates/tq-core/Cargo.toml"),
                    },
                    {
                        "id": "kit",
                        "name": "tq-testkit",
                        "manifest_path": str(checkout / "crates/tq-testkit/Cargo.toml"),
                    },
                ],
            }
        )
    )
    private = tmp_path / "private-denylist"
    private.write_text("synthetic-denied-identifier\n")
    inventory = tmp_path / "files.nul"
    inventory.write_bytes("\0".join(sorted(set(names))).encode() + b"\0")
    cargo_home = tmp_path / "cargo-cache"
    (cargo_home / "advisory-dbs" / "synthetic-db").mkdir(parents=True)
    env = dict(os.environ)
    env.update(
        PATH=str(binary) + os.pathsep + os.environ["PATH"],
        TQ_TEST_TRACE=str(tmp_path / "trace.jsonl"),
        TQ_TEST_METADATA=str(metadata),
        TQ_TEST_FILES=str(inventory),
        TQ_TEST_REAL_PYTHON=sys.executable,
        TQ_GUARD_DENYLIST=str(private),
        CARGO_HOME=str(cargo_home),
    )
    return Runner(checkout, env)


def execute(runner, *arguments, **changes):
    checkout = runner.checkout
    env = dict(runner.environment, **changes)
    Path(env["TQ_TEST_TRACE"]).unlink(missing_ok=True)
    done = subprocess.run(
        [str(checkout / "scripts/check-cpu.sh"), *arguments],
        cwd=checkout.parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    path = Path(env["TQ_TEST_TRACE"])
    trace = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return done, trace


def test_runner_positive_control_contains_all_cpu_gates(runner):
    done, trace = execute(runner)
    assert done.returncode == 0, done.stdout + done.stderr
    keys = {item["key"] for item in trace}
    required = {
        "cargo:fmt",
        "cargo:clippy",
        "cargo:deny",
        "cargo:build:debug",
        "cargo:build:release",
        "cargo:nextest",
        "cargo:llvm-cov",
        "cargo:doc",
        "uv:ruff:format",
        "uv:ruff:check",
        "uv:pytest",
        "uv:checks:lengths",
        "uv:checks:deps",
        "uv:checks:manifest",
        "uv:checks:coverage",
        "guard",
        "gitleaks:git",
        "gitleaks:dir",
        "actionlint",
    }
    assert required <= keys
    assert any(item["key"] == "guard" and item["require_private"] == "1" for item in trace)
    assert any(
        item["key"] == "cargo:doc" and "warnings" in (item["rustdocflags"] or "") for item in trace
    )
    for item in trace:
        if item["key"] in {
            "cargo:build:debug",
            "cargo:build:release",
            "cargo:clippy",
            "cargo:llvm-cov",
        }:
            assert "--locked" in item["args"]
    executed_release = [
        item
        for item in trace
        if item["tool"] == "cargo"
        and "--release" in item["args"]
        and ("test" in item["args"] or "nextest" in item["args"])
    ]
    assert executed_release, "A release build alone does not execute the Rust acceptance tests."
    pytest_runs = [item for item in trace if item["key"] == "uv:pytest"]
    reference_runs = [
        item
        for item in pytest_runs
        if Path(item["cwd"]).name == "reference"
        or any(arg.startswith("reference/tests") for arg in item["args"])
    ]
    assert len(reference_runs) >= 2, "Run independent reference acceptance and the full suite."
    for item in reference_runs:
        assert any(arg == "--cov" or arg.startswith("--cov=") for arg in item["args"])
        assert "--cov-branch" in item["args"]
    root_runs = [item for item in pytest_runs if item not in reference_runs]
    assert any(
        any(arg == "--cov" or arg.startswith("--cov=") for arg in item["args"])
        for item in root_runs
    ), "The new checker and inventory hook need their independent Python coverage gate."


@pytest.mark.parametrize(
    "gate",
    [
        "cargo:fmt",
        "cargo:clippy",
        "cargo:deny",
        "cargo:build:debug",
        "cargo:build:release",
        "cargo:nextest",
        "cargo:llvm-cov",
        "cargo:doc",
        "uv:ruff:format",
        "uv:ruff:check",
        "uv:pytest",
        "uv:checks:lengths",
        "uv:checks:deps",
        "uv:checks:manifest",
        "uv:checks:coverage",
        "guard",
        "gitleaks:git",
        "gitleaks:dir",
        "actionlint",
    ],
)
def test_runner_stops_at_each_failed_gate(runner, gate):
    done, trace = execute(runner, TQ_TEST_FAIL_AT=gate)
    assert done.returncode != 0, done.stdout + done.stderr
    assert trace and trace[-1]["key"] == gate, (gate, trace, done.stdout, done.stderr)


def test_runner_stops_at_advisory_check_after_successful_fetch(runner):
    done, trace = execute(runner, TQ_TEST_FAIL_AT="cargo:deny", TQ_TEST_FAIL_PHASE="check")
    assert done.returncode != 0
    assert any(item["key"] == "cargo:deny" and "fetch" in item["args"] for item in trace)
    assert trace[-1]["key"] == "cargo:deny" and "check" in trace[-1]["args"]


def test_runner_default_requires_private_list(runner):
    Path(runner.environment["TQ_GUARD_DENYLIST"]).unlink()
    done, trace = execute(runner)
    assert done.returncode != 0
    assert any(item["key"] == "guard" and item["require_private"] == "1" for item in trace)


def test_runner_public_mode_reports_private_gap(runner):
    Path(runner.environment["TQ_GUARD_DENYLIST"]).unlink()
    done, trace = execute(runner, "--public-ci")
    assert done.returncode == 0, done.stdout + done.stderr
    text = (done.stdout + done.stderr).lower()
    assert "private" in text and "pending" in text
    assert any(item["key"] == "guard" and item["require_private"] != "1" for item in trace)


def test_runner_cannot_reuse_previous_coverage(runner):
    passed, trace = execute(runner)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    generated = [
        item
        for item in trace
        if item["key"] == "cargo:llvm-cov" and "--output-path" in item["args"]
    ]
    assert generated
    done, _ = execute(runner, TQ_TEST_NO_COVERAGE="1")
    assert done.returncode != 0, "A tool that wrote no new coverage reused old evidence."


def test_runner_rejects_unknown_options(runner):
    done, trace = execute(runner, "--ignore-failures")
    assert done.returncode != 0
    assert not trace


def test_runner_rejects_version_substring_match(runner):
    done, trace = execute(runner, TQ_TEST_PIN_DRIFT="uv")
    assert done.returncode != 0, "uv 0.10.120 is not the pin 0.10.12."
    assert not any(item["key"] == "cargo:fmt" for item in trace)


def test_runner_requires_resolvable_advisory_revision(runner):
    done, trace = execute(runner, TQ_TEST_BAD_DB_REVISION="1")
    assert done.returncode != 0, "An unresolved database revision cannot certify G-04."
    assert not any(item["key"] == "cargo:build:debug" for item in trace)


def test_runner_requires_a_fetched_advisory_database(runner):
    shutil.rmtree(Path(runner.environment["CARGO_HOME"]) / "advisory-dbs")
    done, trace = execute(runner)
    assert done.returncode != 0
    assert not any(item["key"] == "cargo:build:debug" for item in trace)


def test_runner_does_not_guess_between_advisory_databases(runner):
    cache = Path(runner.environment["CARGO_HOME"]) / "advisory-dbs"
    (cache / "another-db").mkdir()
    done, trace = execute(runner)
    assert done.returncode != 0, "An arbitrary first cache cannot identify the database checked."
    assert not any(item["key"] == "cargo:build:debug" for item in trace)
