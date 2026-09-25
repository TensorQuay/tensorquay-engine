#!/usr/bin/env python3
"""CPU gate checkers for G-03, G-06, G-07 and G-08.

Contract: `docs/CPU-CHECKS.md`. Each checker decides over evidence it is handed, so gathering it
(Git, Cargo, nextest, pytest) stays in thin helpers and the decisions stay independently testable.
Nothing here modifies its input, and empty evidence is a failure rather than a pass.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

SOURCE_SUFFIXES = (".rs", ".py", ".sh")
SOURCE_LIMIT = 1000
KERNEL_SUFFIX = "_kernel.rs"
KERNEL_LIMIT = 400

MEMBERS = frozenset({"tq-core", "tq-testkit"})
APPROVED_EDGES = frozenset({("tq-core", "tq-testkit", "dev")})
DEPENDENCY_KINDS = (None, "dev", "build")
GPU_FAMILIES = (
    "cuda",
    "cutile",
    "cudarc",
    "nccl",
    "cust",
    "rustacuda",
    "cudnn",
    "cublas",
    "cufft",
    "cusparse",
    "nvrtc",
    "nvtx",
    "nvml",
)

MANIFEST_KEYS = frozenset({"schema", "scope", "kernel_specializations", "clauses"})
MANIFEST_SCHEMA = 1
MANIFEST_SCOPE = "cpu-foundation"
SELECTOR_PREFIXES = ("rust:", "python:")

METRICS = ("lines", "regions", "functions")

CATALOGUE_HEADING = "## CPU clause catalogue"
CATALOGUE_HEADER = ("ID", "Obligation")
CLAUSE_ID = re.compile(r"[A-Z]+-[0-9]{3}")
RULE_CHARACTERS = set("-:")


class CheckError(ValueError):
    """Evidence that does not support a pass."""


def _fail(check, detail):
    raise CheckError(f"{check}: {detail}")


def _line_count(text):
    """Physical lines, counting a final unterminated line and treating CRLF as one ending."""
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def check_lengths(root, paths):
    """Enforce the source-length caps over the given repository-relative paths."""
    root = Path(root).resolve()
    selected = {}
    for given in paths:
        if not given.endswith(SOURCE_SUFFIXES):
            continue
        if Path(given).is_absolute():
            _fail("source", f"{given!r} is absolute")
        resolved = (root / given).resolve()
        if not resolved.is_relative_to(root):
            _fail("source", f"{given!r} leaves the repository")
        selected.setdefault(resolved, given)
    if not selected:
        _fail("source", "no source file was checked")
    for resolved, given in selected.items():
        if not resolved.is_file():
            _fail("source", f"{given!r} is not a readable file")
        try:
            text = resolved.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            _fail("source", f"{given!r} is not UTF-8")
        limit = KERNEL_LIMIT if given.endswith(KERNEL_SUFFIX) else SOURCE_LIMIT
        lines = _line_count(text)
        if lines > limit:
            _fail("source", f"{given!r} has {lines} lines, over the {limit} limit")
    return len(selected)


def _is_gpu(name):
    """A GPU family name exactly, or that family with a `-` or `_` separated suffix."""
    return any(
        name == family or name.startswith((f"{family}-", f"{family}_")) for family in GPU_FAMILIES
    )


def _declared_edges(members):
    """Declared workspace edges and the total declaration count across the members."""
    edges, declared = set(), 0
    for package in members:
        for dependency in package["dependencies"]:
            name = dependency["name"]
            if dependency["kind"] not in DEPENDENCY_KINDS:
                _fail("dependency", f"{name} has the unknown kind {dependency['kind']!r}")
            if _is_gpu(name):
                _fail("dependency", f"{package['name']} declares the GPU package {name}")
            declared += 1
            if name in MEMBERS:
                edges.add((package["name"], name, dependency["kind"]))
    if not edges <= APPROVED_EDGES:
        _fail(
            "dependency", f"the workspace edges {sorted(edges - APPROVED_EDGES)} are not approved"
        )
    return declared


def _walk_resolved(roots, nodes, by_id):
    """Walk the resolved graph from the workspace members, over every dependency kind."""
    seen, pending = set(), list(roots)
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current not in nodes or current not in by_id:
            _fail("dependency", f"the resolved id {current} has no node or package")
        if _is_gpu(by_id[current]["name"]):
            _fail("dependency", f"the GPU package {by_id[current]['name']} is reachable")
        for resolved in nodes[current]["deps"]:
            if not resolved["dep_kinds"]:
                _fail("dependency", f"the resolved edge to {resolved['pkg']} declares no kind")
            pending.append(resolved["pkg"])


def _dependencies(metadata):
    identifiers = metadata["workspace_members"]
    if not identifiers:
        _fail("dependency", "the workspace declares no member")
    if len(set(identifiers)) != len(identifiers):
        _fail("dependency", "a workspace member is listed twice")
    # Package ids are unique; names are not, because one graph can hold several versions of the
    # same external crate. Duplicate-version policy belongs to cargo-deny, not to this check.
    by_id = {}
    for package in metadata["packages"]:
        if package["id"] in by_id:
            _fail("dependency", f"the package id {package['id']} appears twice")
        by_id[package["id"]] = package
    members = []
    for identifier in identifiers:
        if identifier not in by_id:
            _fail("dependency", f"the workspace member {identifier} has no package")
        members.append(by_id[identifier])
    # A list, not a set: two members under distinct ids may not share one name.
    if sorted(package["name"] for package in members) != sorted(MEMBERS):
        _fail("dependency", f"the workspace members are not exactly {sorted(MEMBERS)}")
    declared = _declared_edges(members)
    nodes = {}
    for node in metadata["resolve"]["nodes"]:
        if node["id"] in nodes:
            _fail("dependency", f"the resolve graph lists {node['id']} twice")
        nodes[node["id"]] = node
    if not nodes:
        _fail("dependency", "the resolve graph is empty")
    _walk_resolved(identifiers, nodes, by_id)
    return declared


def check_dependencies(metadata):
    """Enforce the approved crate graph and the absence of GPU packages."""
    try:
        return _dependencies(metadata)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        _fail("dependency", f"malformed cargo metadata ({error})")


def _selector_is_well_formed(selector):
    if not selector.startswith(SELECTOR_PREFIXES):
        return False
    return "::" in selector.split(":", 1)[1]


def _selector_resolves(selector, available):
    if selector in available:
        return True
    return selector.startswith("python:") and any(
        entry.startswith(f"{selector}[") for entry in available
    )


def check_manifest(manifest, required, available):
    """Map every catalogue clause onto tests the collectors actually reported."""
    if set(manifest) != set(MANIFEST_KEYS):
        _fail("manifest", f"the top-level fields are {sorted(manifest)}")
    schema = manifest["schema"]
    if not isinstance(schema, int) or isinstance(schema, bool) or schema != MANIFEST_SCHEMA:
        _fail("manifest", f"the schema is {schema!r}")
    if manifest["scope"] != MANIFEST_SCOPE:
        _fail("manifest", f"the scope is {manifest['scope']!r}")
    kernels = manifest["kernel_specializations"]
    if not isinstance(kernels, list) or kernels:
        _fail("manifest", f"kernel specializations must be an empty list, got {kernels!r}")
    if not required:
        _fail("manifest", "the clause catalogue is empty")
    for prefix in SELECTOR_PREFIXES:
        if not any(entry.startswith(prefix) for entry in available):
            _fail("manifest", f"the {prefix.rstrip(':')} inventory is empty")
    clauses = manifest["clauses"]
    if set(clauses) != set(required):
        _fail("manifest", f"the mapped clauses {sorted(clauses)} are not the catalogue")
    for clause, selectors in sorted(clauses.items()):
        if not isinstance(selectors, list) or not selectors:
            _fail("manifest", f"{clause} maps to {selectors!r}")
        if len(set(selectors)) != len(selectors):
            _fail("manifest", f"{clause} repeats a selector")
        for selector in selectors:
            if not isinstance(selector, str) or not _selector_is_well_formed(selector):
                _fail("manifest", f"{clause} has the malformed selector {selector!r}")
            if not _selector_resolves(selector, available):
                _fail("manifest", f"{clause} names the uncollected test {selector}")
    return len(clauses)


def _metric_counts(summary, where):
    """Raw count/covered pairs for one file, rejecting anything but a sane integer."""
    counts = {}
    for metric in METRICS:
        item = summary[metric]
        count, covered = item["count"], item["covered"]
        for value in (count, covered):
            if isinstance(value, bool) or not isinstance(value, int):
                _fail("coverage", f"{where} reports {metric} as {value!r}")
        if count < 0 or covered < 0 or covered > count:
            _fail("coverage", f"{where} reports {metric} as {covered} of {count}")
        counts[metric] = (count, covered)
    return counts


def _coverage(report, root, crates):
    if not crates:
        _fail("coverage", "no crate was expected")
    directories = {crate: (root / "crates" / crate / "src").resolve() for crate in crates}
    totals = {crate: {metric: [0, 0] for metric in METRICS} for crate in crates}
    files = report["data"][0]["files"]
    if not files:
        _fail("coverage", "the report names no file")
    seen = set()
    for entry in files:
        name = entry["filename"]
        path = Path(name)
        resolved = (path if path.is_absolute() else root / path).resolve()
        if resolved in seen:
            _fail("coverage", f"{name} is reported twice")
        seen.add(resolved)
        if not resolved.is_file():
            _fail("coverage", f"{name} is not a file")
        owner = next(
            (crate for crate, at in directories.items() if resolved.is_relative_to(at)), None
        )
        if owner is None:
            continue
        for metric, (count, covered) in _metric_counts(entry["summary"], name).items():
            totals[owner][metric][0] += count
            totals[owner][metric][1] += covered
    for crate in sorted(crates):
        at = directories[crate]
        if not at.is_dir():
            _fail("coverage", f"{crate} has no source directory")
        for source in sorted(at.rglob("*.rs")):
            if source.name != "lib.rs" and source.resolve() not in seen:
                _fail("coverage", f"{crate} never reports {source.name}")
        for metric, (count, covered) in totals[crate].items():
            if count <= 0:
                _fail("coverage", f"{crate} reports no {metric}")
            if covered != count:
                _fail("coverage", f"{crate} covers {covered} of {count} {metric}")
    return {
        crate: {
            metric: {"count": count, "covered": covered} for metric, (count, covered) in by.items()
        }
        for crate, by in totals.items()
    }


def check_coverage(report, root, crates):
    """Aggregate raw per-crate counts and require every one of them to be complete."""
    try:
        return _coverage(report, Path(root).resolve(), crates)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        _fail("coverage", f"malformed coverage report ({error})")


def run(root, command, env=None):
    """Run a gathering command, failing the check rather than the interpreter."""
    try:
        done = subprocess.run(
            command, cwd=root, env=env, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail("evidence", f"{command[0]} failed ({error})")
    return done.stdout


def git_sources(root):
    """Tracked and non-ignored untracked paths, NUL separated so odd names survive."""
    listing = run(root, ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    return [entry for entry in listing.split("\0") if entry]


def cargo_metadata(root):
    """The full resolved workspace graph, over all features and platforms."""
    arguments = ["--format-version", "1", "--all-features", "--locked", "--offline"]
    return json.loads(run(root, ["cargo", "metadata", *arguments]))


def workspace_crates(metadata):
    """The expected crate set, taken from metadata so a thinned report cannot hide a crate."""
    by_id = {package["id"]: package["name"] for package in metadata["packages"]}
    return {by_id[identifier] for identifier in metadata["workspace_members"]}


def clause_catalogue(root):
    """The required clause IDs, read from the contract's own catalogue table."""
    contract = (root / "docs/CPU-CHECKS.md").read_text(encoding="utf-8")
    _, heading, section = contract.partition(CATALOGUE_HEADING)
    if not heading:
        _fail("manifest", "the contract has no clause catalogue")
    identifiers = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if cells == CATALOGUE_HEADER or all(set(cell) <= RULE_CHARACTERS for cell in cells):
            continue
        if len(cells) != 2 or not CLAUSE_ID.fullmatch(cells[0]) or not cells[1]:
            _fail("manifest", f"the catalogue row {line!r} is malformed")
        if cells[0] in identifiers:
            _fail("manifest", f"the catalogue lists {cells[0]} twice")
        identifiers.add(cells[0])
    if not identifiers:
        _fail("manifest", "the clause catalogue is empty")
    return identifiers


def test_inventory(root, out):
    """Tests the collectors actually reported: nextest, then pytest through the marker hook."""
    arguments = ["--workspace", "--locked", "--offline", "--message-format", "json"]
    listing = json.loads(run(root, ["cargo", "nextest", "list", *arguments]))
    available = {
        f"rust:{suite['binary-id']}::{name}"
        for suite in listing["rust-suites"].values()
        for name, case in suite["testcases"].items()
        if not case["ignored"]
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    environment = dict(
        os.environ,
        PYTHONPATH=str(root / "tools"),
        TQ_INVENTORY_ROOT=str(root),
        TQ_INVENTORY_OUT=str(out),
    )
    collect = ["--collect-only", "-q", "-p", "tq_pytest_inventory", "tests", "reference/tests"]
    run(
        root,
        ["uv", "run", "--project", "reference", "--locked", "--offline", "pytest", *collect],
        env=environment,
    )
    available |= {f"python:{nodeid}" for nodeid in json.loads(out.read_text(encoding="utf-8"))}
    return available


def _manifest_evidence(root):
    """Load the manifest, refusing any kernel evidence this CPU-only checker cannot judge."""
    kernels = [path for path in git_sources(root) if path.endswith(KERNEL_SUFFIX)]
    if kernels:
        _fail("manifest", f"{kernels} need the GPU manifest contract")
    with (root / "tests/manifest.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    inventory = test_inventory(root, root / "target/cpu/pytest-inventory.json")
    return manifest, clause_catalogue(root), inventory


def main(argv=None):
    """Run one check from the command line; a failed check returns a nonzero status."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description="TensorQuay CPU gate checks")
    checks = parser.add_subparsers(dest="check", required=True)
    for gate in ("lengths", "deps", "manifest"):
        checks.add_parser(gate, parents=[common])
    checks.add_parser("coverage", parents=[common]).add_argument("report", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        if arguments.check == "lengths":
            print(f"source files within limits: {check_lengths(root, git_sources(root))}")
        elif arguments.check == "deps":
            print(f"dependency declarations checked: {check_dependencies(cargo_metadata(root))}")
        elif arguments.check == "manifest":
            print(f"clauses mapped to collected tests: {check_manifest(*_manifest_evidence(root))}")
        else:
            report = json.loads(arguments.report.read_text(encoding="utf-8"))
            crates = workspace_crates(cargo_metadata(root))
            for crate, counts in sorted(check_coverage(report, root, crates).items()):
                totals = ", ".join(f"{counts[metric]['count']} {metric}" for metric in METRICS)
                print(f"{crate}: {totals} fully covered")
    except (CheckError, OSError, ValueError, KeyError) as error:
        print(f"check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
