"""Lead-owned configuration boundaries; these do not execute a CUDA image or kernel."""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "docker" / "cuda"
CUDA = (
    "nvidia/cuda:13.3.1-devel-ubuntu24.04@"
    "sha256:03c372fd9c65fe7739279f8c65473b315dc61efaaffab03e1e65bc7be7aee61e"
)
RUST = (
    "rust:1.95.0-slim-bookworm@"
    "sha256:6f9e63259f12e1e599296f5ecfed2bae46de4af0ee0525dd8b89c046e236d5c5"
)
CUTILE = "cdc69c13a7529552a26d9941893f047970d8e95f"


def instructions():
    """Read the simple Dockerfile subset approved for this small recipe."""
    text = (CONTEXT / "Dockerfile").read_text()
    assert "<<" not in text, "Keep this recipe simple; heredocs need a separate review"
    joined = re.sub(r"\\\n[ \t]*", " ", text)
    return [
        tuple(line.strip().split(maxsplit=1))
        for line in joined.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_only_the_two_approved_external_images_are_used():
    sources, stages = [], set()
    for operation, arguments in instructions():
        if operation.upper() == "FROM":
            tokens = shlex.split(arguments)
            source = next(token for token in tokens if not token.startswith("--"))
            if source not in stages:
                sources.append(source)
            assert tokens[-2].lower() == "as", "Name the build stages explicitly"
            stages.add(tokens[-1])
    assert sources == [RUST, CUDA]
    assert {"toolchain", "compile-smoke"} <= stages


def test_no_repository_content_or_remote_add_is_copied_into_the_image():
    rust_stage = next(
        shlex.split(arguments)[-1]
        for operation, arguments in instructions()
        if operation.upper() == "FROM" and RUST in shlex.split(arguments)
    )
    copies = []
    for operation, arguments in instructions():
        assert operation.upper() != "ADD", "Remote ADD bypasses the source pin"
        if operation.upper() == "COPY":
            tokens = shlex.split(arguments)
            assert tokens[0].startswith("--from="), "Only toolchain-stage copies are approved"
            assert tokens[0].removeprefix("--from=") == rust_stage
            copies.append(tokens[-2:])
    assert sorted(copies) == [
        ["/usr/local/cargo", "/usr/local/cargo"],
        ["/usr/local/rustup", "/usr/local/rustup"],
    ]
    patterns = [
        line.strip()
        for line in (CONTEXT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert patterns[0] in ("*", "**"), "Build-context input must be excluded by default"
    assert set(patterns[1:]) <= {"!Dockerfile", "!.dockerignore"}


def test_toolchain_selection_does_not_depend_on_the_host_environment():
    environment = {}
    for operation, arguments in instructions():
        if operation.upper() == "ENV":
            for entry in shlex.split(arguments):
                name, value = entry.split("=", maxsplit=1)
                environment[name] = value
    assert environment["CUDA_HOME"] == "/usr/local/cuda"
    assert environment["CUDA_TOOLKIT_PATH"] == "/usr/local/cuda"
    assert environment["CUTILE_TILEIRAS_PATH"] == "/usr/local/cuda/bin/tileiras"
    assert environment["CUTILE_BYTECODE_VERSION"] == "13.3"
    assert environment["RUSTUP_TOOLCHAIN"] in ("1.95.0", "1.95.0-x86_64-unknown-linux-gnu")
    assert environment["CARGO_HOME"] == "/usr/local/cargo"
    assert environment["RUSTUP_HOME"] == "/usr/local/rustup"


def test_recording_does_not_hide_build_or_test_failures():
    shells = [json.loads(value) for op, value in instructions() if op.upper() == "SHELL"]
    assert shells, "Recording build output needs an explicit fail-closed shell"
    for shell in shells:
        assert shell[0] == "/bin/bash"
        flags = "".join(token[1:] for token in shell[1:] if token.startswith("-"))
        assert "pipefail" in shell and "e" in flags and "u" in flags
    commands = "\n".join(value for op, value in instructions() if op.upper() == "RUN")
    assert not re.search(r"\|\|\s*(?:true|:|exit\s+0)\b", commands)
    assert "--locked" in commands
    assert "--all-features" not in commands and "--workspace" not in commands
    assert "--gpus" not in commands and "nvidia-smi" not in commands


def test_the_only_compilation_scope_is_the_pinned_upstream_smoke():
    commands = "\n".join(value for op, value in instructions() if op.upper() == "RUN")
    assert "https://github.com/NVlabs/cutile-rs" in commands
    assert CUTILE in commands
    assert "rev-parse" in commands, "Verify the checkout, not only the fetch argument"
    assert "--test compile_only" in commands and "-p cutile" in commands
    assert "smoke_compile_only: test" in commands
    assert "1 test, 0 benchmarks" in commands
    assert "--list" in commands
    assert "--nocapture" in commands
    assert "/opt/tq-build-evidence" in commands
    assert "dpkg-query" in commands


def test_run_instructions_have_valid_shell_syntax():
    for operation, arguments in instructions():
        if operation.upper() == "RUN":
            result = subprocess.run(
                ["/bin/bash", "-n"], input=arguments, text=True, capture_output=True, check=False
            )
            assert result.returncode == 0, result.stderr


# These substitutes exercise shell failure propagation only. They are never evidence
# that Rust, CUDA or an upstream test compiled or ran successfully.
RECORDING_TOOL = r"""
import json, os, pathlib, sys
tool = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
case = os.environ['TQ_TEST_CASE']
trace = pathlib.Path(os.environ['TQ_TEST_TRACE'])
history = [json.loads(row) for row in trace.read_text().splitlines()] if trace.exists() else []
with open(os.environ['TQ_TEST_TRACE'], 'a') as output:
    output.write(json.dumps({'tool': tool, 'args': args}) + '\n')
if tool == 'git':
    if case == 'git-failure':
        sys.exit(41)
    if 'rev-parse' in args:
        print('0' * 40 if case == 'checkout-mismatch' else os.environ['TQ_TEST_CUTILE'])
        if case == 'revision-failure' and not any('rev-parse' in row['args'] for row in history):
            sys.exit(41)
    if 'status' in args:
        if case == 'status-failure':
            sys.exit(41)
        if case == 'dirty-checkout':
            print(' M Cargo.lock')
    if args and args[0] in ('clone', 'init'):
        pathlib.Path(args[-1]).mkdir(parents=True, exist_ok=True)
elif tool == 'cargo':
    print('simulated compiler diagnostic', file=sys.stderr)
    if '--list' in args:
        if case != 'inventory-empty':
            name = 'renamed' if case == 'inventory-renamed' else 'smoke_compile_only'
            print(name + ': test')
            if case == 'inventory-extra':
                print('another_test: test\n\n2 tests, 0 benchmarks')
            else:
                print('\n1 test, 0 benchmarks')
        sys.exit(41 if case == 'inventory-failure' else 0)
    print('simulated upstream test output')
    sys.exit(41 if case == 'execution-failure' else 0)
"""


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "git-failure",
        "checkout-mismatch",
        "revision-failure",
        "status-failure",
        "dirty-checkout",
        "inventory-failure",
        "inventory-empty",
        "inventory-renamed",
        "inventory-extra",
        "execution-failure",
    ],
)
def test_smoke_stage_rejects_failed_or_missing_evidence(tmp_path, case):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("cargo", "git"):
        executable = binaries / name
        executable.write_text(f"#!{sys.executable}\n" + RECORDING_TOOL)
        executable.chmod(0o755)
    trace = tmp_path / "trace.jsonl"
    env = dict(
        os.environ,
        PATH=f"{binaries}:/usr/bin:/bin",
        TQ_TEST_CASE=case,
        TQ_TEST_TRACE=str(trace),
        TQ_TEST_CUTILE=CUTILE,
    )
    (tmp_path / "opt" / "tq-build-evidence").mkdir(parents=True)
    active = False
    cwd = tmp_path
    result = None
    for operation, arguments in instructions():
        operation = operation.upper()
        if operation == "FROM":
            active = shlex.split(arguments)[-1] == "compile-smoke"
        if not active:
            continue
        arguments = arguments.replace("/opt/", f"{tmp_path}/opt/")
        if operation == "WORKDIR":
            cwd = Path(arguments)
            assert cwd.is_relative_to(tmp_path)
            cwd.mkdir(parents=True, exist_ok=True)
        elif operation == "RUN":
            result = subprocess.run(
                ["/bin/bash", "-euo", "pipefail", "-c", arguments],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode:
                break
    assert result is not None, "The smoke target must actually invoke commands"
    assert (result.returncode == 0) == (case == "success"), result.stdout + result.stderr
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    executions = [row for row in records if row["tool"] == "cargo" and "--list" not in row["args"]]
    if case in ("success", "execution-failure"):
        assert len(executions) == 1
        assert "--locked" in executions[0]["args"]
    else:
        assert not executions, "Do not execute after failed source or inventory verification"
    if case == "success":
        evidence = tmp_path / "opt" / "tq-build-evidence"
        execution_logs = [
            file.read_text()
            for file in evidence.iterdir()
            if file.is_file() and "simulated upstream test output" in file.read_text()
        ]
        assert execution_logs and all(
            "simulated compiler diagnostic" in log for log in execution_logs
        ), "Save compiler diagnostics as well as test stdout"
