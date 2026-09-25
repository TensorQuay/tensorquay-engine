"""Independent CPU gate acceptance: broken evidence must never become a pass."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def checks():
    path = ROOT / "tools/cpu_checks.py"
    spec = importlib.util.spec_from_file_location("tq_cpu_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(root, name, lines=1, ending="\n", final=True):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ending.join("# source line" for _ in range(lines))
    if final and lines:
        text += ending
    path.write_bytes(text.encode())
    return name


@pytest.mark.parametrize(
    "name,limit", [("a.rs", 1000), ("a.py", 1000), ("a.sh", 1000), ("x_kernel.rs", 400)]
)
@pytest.mark.parametrize("ending,final", [("\n", True), ("\n", False), ("\r\n", True)])
def test_source_length_boundaries(checks, tmp_path, name, limit, ending, final):
    source(tmp_path, name, limit, ending, final)
    assert checks.check_lengths(tmp_path, [name]) == 1
    source(tmp_path, name, limit + 1, ending, final)
    with pytest.raises(checks.CheckError, match="source"):
        checks.check_lengths(tmp_path, [name])


def test_source_inventory_is_unique_and_data_is_not_source(checks, tmp_path):
    paths = [
        source(tmp_path, "nested space/check.py", 0),
        source(tmp_path, "notes.md", 2000),
    ]
    paths += [source(tmp_path, "vectors.json", 3000), "nested space/check.py"]
    before = copy.deepcopy(paths)
    assert checks.check_lengths(tmp_path, paths) == 1
    assert paths == before


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "only-data",
        "missing",
        "absolute",
        "escape",
        "symlink",
        "utf8",
        "directory",
    ],
)
def test_source_inventory_failures(checks, tmp_path, case):
    good = source(tmp_path, "ok.py")
    paths = [good]
    if case == "empty":
        paths = []
    elif case == "only-data":
        paths = [source(tmp_path, "data.json")]
    elif case == "missing":
        paths.append("missing.py")
    elif case == "absolute":
        paths.append(str(tmp_path / good))
    elif case == "escape":
        paths.append("../outside.py")
    elif case == "symlink":
        outside = tmp_path.parent / "outside.py"
        outside.write_text("# outside\n")
        (tmp_path / "linked.py").symlink_to(outside)
        paths.append("linked.py")
    elif case == "utf8":
        (tmp_path / "bad.py").write_bytes(b"\xff\xfe")
        paths.append("bad.py")
    else:
        (tmp_path / "folder.py").mkdir()
        paths.append("folder.py")
    with pytest.raises(checks.CheckError, match="source"):
        checks.check_lengths(tmp_path, paths)


def metadata():
    """Opaque IDs deliberately contain no package names."""
    return {
        "version": 1,
        "workspace_members": ["opaque-a", "opaque-b"],
        "packages": [
            {
                "id": "opaque-a",
                "name": "tq-core",
                "dependencies": [dependency("tq-testkit", "dev")],
            },
            {"id": "opaque-b", "name": "tq-testkit", "dependencies": []},
        ],
        "resolve": {
            "nodes": [
                {
                    "id": "opaque-a",
                    "deps": [edge("opaque-b", "dev", "renamed_test_support")],
                },
                {"id": "opaque-b", "deps": []},
            ]
        },
    }


def dependency(name, kind=None, **kwargs):
    return {
        "name": name,
        "kind": kind,
        "rename": None,
        "optional": False,
        "target": None,
        **kwargs,
    }


def edge(package, kind=None, name="opaque_alias"):
    return {"name": name, "pkg": package, "dep_kinds": [{"kind": kind, "target": None}]}


def add_external(data, name="serde", kind=None, **kwargs):
    data["packages"][0]["dependencies"].append(dependency(name, kind, **kwargs))
    data["packages"].append({"id": "opaque-external", "name": name, "dependencies": []})
    data["resolve"]["nodes"][0]["deps"].append(edge("opaque-external", kind))
    data["resolve"]["nodes"].append({"id": "opaque-external", "deps": []})


def test_dependency_bootstrap_and_external_cpu_are_valid(checks):
    data = metadata()
    before = copy.deepcopy(data)
    assert checks.check_dependencies(data) == 1
    assert data == before
    add_external(data, rename="cuda_looking_alias")
    assert checks.check_dependencies(data) == 2


@pytest.mark.parametrize("name", ["custom", "curl", "nv-helper"])
def test_dependency_unrelated_cpu_names_remain_allowed(checks, name):
    data = metadata()
    add_external(data, name)
    assert checks.check_dependencies(data) == 2


def test_dependency_real_empty_lists_are_not_an_empty_workspace(checks):
    data = metadata()
    data["packages"][0]["dependencies"] = []
    data["resolve"]["nodes"][0]["deps"] = []
    assert checks.check_dependencies(data) == 0


def test_dependency_gpu_declaration_is_checked_even_without_a_resolved_edge(checks):
    data = metadata()
    data["packages"][0]["dependencies"].append(
        dependency("cudarc", optional=True, target="cfg(any())", rename="innocent_alias")
    )
    with pytest.raises(checks.CheckError, match="dependency"):
        checks.check_dependencies(data)


def test_dependency_package_ids_disambiguate_external_versions(checks):
    data = metadata()
    add_external(data, "serde", rename="serde_one")
    data["packages"][0]["dependencies"].append(dependency("serde", rename="serde_two"))
    data["packages"].append({"id": "opaque-second-version", "name": "serde", "dependencies": []})
    data["resolve"]["nodes"][0]["deps"].append(edge("opaque-second-version", name="serde_two"))
    data["resolve"]["nodes"].append({"id": "opaque-second-version", "deps": []})
    # Duplicate-version policy belongs to cargo-deny; this is a valid Cargo graph.
    assert checks.check_dependencies(data) == 3


def test_dependency_workspace_names_cannot_be_duplicated_under_distinct_ids(checks):
    data = metadata()
    data["workspace_members"].append("opaque-third")
    data["packages"].append({"id": "opaque-third", "name": "tq-core", "dependencies": []})
    data["resolve"]["nodes"].append({"id": "opaque-third", "deps": []})
    with pytest.raises(checks.CheckError, match="dependency"):
        checks.check_dependencies(data)


def test_dependency_duplicate_resolve_node_cannot_hide_a_gpu_edge(checks):
    data = metadata()
    add_external(data)
    data["packages"].append({"id": "opaque-gpu", "name": "cudarc", "dependencies": []})
    data["resolve"]["nodes"] += [
        {"id": "opaque-external", "deps": [edge("opaque-gpu")]},
        {"id": "opaque-gpu", "deps": []},
        {"id": "opaque-external", "deps": []},
    ]
    with pytest.raises(checks.CheckError, match="dependency"):
        checks.check_dependencies(data)


@pytest.mark.parametrize("kind", [None, "dev", "build"])
@pytest.mark.parametrize(
    "name",
    [
        "cuda",
        "cuda-core",
        "cudarc",
        "cutile",
        "cutile-rs",
        "nccl-sys",
        "cust",
        "cust_raw",
        "cust-derive",
        "rustacuda",
        "cudnn",
        "cublas",
        "cufft",
        "cusparse",
        "nvrtc",
        "nvtx",
        "nvml-wrapper",
    ],
)
def test_dependency_gpu_declarations_cannot_hide_by_alias_or_target(checks, kind, name):
    data = metadata()
    add_external(
        data,
        name,
        kind,
        rename="ordinary_cpu_helper",
        optional=True,
        target="cfg(linux)",
    )
    with pytest.raises(checks.CheckError, match="dependency"):
        checks.check_dependencies(data)


@pytest.mark.parametrize("kind", [None, "dev", "build"])
def test_dependency_transitive_gpu_leakage(checks, kind):
    data = metadata()
    add_external(data, "innocent-wrapper")
    data["packages"][-1]["dependencies"] = [dependency("cudarc", kind)]
    data["packages"].append({"id": "opaque-hidden", "name": "cudarc", "dependencies": []})
    data["resolve"]["nodes"][-1]["deps"] = [edge("opaque-hidden", kind)]
    data["resolve"]["nodes"].append({"id": "opaque-hidden", "deps": []})
    with pytest.raises(checks.CheckError, match="dependency"):
        checks.check_dependencies(data)


@pytest.mark.parametrize("kind", [None, "build"])
def test_dependency_testkit_must_stay_dev_only(checks, kind):
    data = metadata()
    data["packages"][0]["dependencies"][0]["kind"] = kind
    data["resolve"]["nodes"][0]["deps"][0]["dep_kinds"][0]["kind"] = kind
    with pytest.raises(checks.CheckError, match="dependency"):
        checks.check_dependencies(data)


def test_dependency_reverse_edge_is_not_approved(checks):
    data = metadata()
    data["packages"][1]["dependencies"].append(dependency("tq-core", "dev"))
    data["resolve"]["nodes"][1]["deps"].append(edge("opaque-a", "dev"))
    with pytest.raises(checks.CheckError, match="dependency"):
        checks.check_dependencies(data)


@pytest.mark.parametrize(
    "case",
    [
        "empty-members",
        "missing-member",
        "duplicate-member",
        "duplicate-package-id",
        "duplicate-name",
        "unknown-member",
        "missing-package",
        "no-resolve",
        "empty-nodes",
        "dangling-edge",
        "unknown-kind",
        "missing-dependencies",
        "missing-kind",
        "missing-dep-kinds",
        "empty-dep-kinds",
        "missing-nodes",
    ],
)
def test_dependency_broken_metadata_fails_closed(checks, case):
    data = metadata()
    if case == "empty-members":
        data["workspace_members"] = []
    elif case == "missing-member":
        data["workspace_members"].pop()
    elif case == "duplicate-member":
        data["workspace_members"].append("opaque-a")
    elif case == "duplicate-package-id":
        data["packages"].append(copy.deepcopy(data["packages"][0]))
    elif case == "duplicate-name":
        data["packages"][1]["name"] = "tq-core"
    elif case == "unknown-member":
        data["packages"][1]["name"] = "tq-surprise"
    elif case == "missing-package":
        data["packages"].pop()
    elif case == "no-resolve":
        data["resolve"] = None
    elif case == "empty-nodes":
        data["resolve"]["nodes"] = []
    elif case == "dangling-edge":
        data["resolve"]["nodes"][0]["deps"][0]["pkg"] = "not-a-package"
    elif case == "unknown-kind":
        data["packages"][0]["dependencies"][0]["kind"] = "optional"
    elif case == "missing-dependencies":
        del data["packages"][0]["dependencies"]
    elif case == "missing-kind":
        del data["packages"][0]["dependencies"][0]["kind"]
    elif case == "missing-dep-kinds":
        del data["resolve"]["nodes"][0]["deps"][0]["dep_kinds"]
    elif case == "empty-dep-kinds":
        data["resolve"]["nodes"][0]["deps"][0]["dep_kinds"] = []
    else:
        del data["resolve"]["nodes"]
    with pytest.raises(checks.CheckError, match="dependency"):
        checks.check_dependencies(data)


def manifest_inputs():
    rust = "rust:tq-core::host_acceptance::wide_spans_are_not_truncated_to_32_bits"
    python = "python:reference/tests/test_sample.py::test_table"
    manifest = {
        "schema": 1,
        "scope": "cpu-foundation",
        "kernel_specializations": [],
        "clauses": {"HOST-002": [rust], "REF-003": [python]},
    }
    return manifest, {"HOST-002", "REF-003"}, {rust, python + "[a]", python + "[b]"}


def test_manifest_exact_and_parameterized_selectors(checks):
    manifest, required, available = manifest_inputs()
    before = copy.deepcopy((manifest, required, available))
    assert checks.check_manifest(manifest, required, available) == 2
    assert (manifest, required, available) == before
    available.add("python:reference/tests/test_sample.py::test_table")
    assert checks.check_manifest(manifest, required, available) == 2
    manifest["clauses"]["REF-003"] = ["python:reference/tests/test_sample.py::test_table[a]"]
    assert checks.check_manifest(manifest, required, available) == 2


def test_manifest_one_test_can_cover_multiple_clauses(checks):
    manifest, required, available = manifest_inputs()
    manifest["clauses"]["REF-003"] = manifest["clauses"]["HOST-002"].copy()
    assert checks.check_manifest(manifest, required, available) == 2


@pytest.mark.parametrize(
    "field,value",
    [("schema", 1.0), ("kernel_specializations", ""), ("kernel_specializations", False)],
)
def test_manifest_toml_scalar_types_are_not_coerced(checks, field, value):
    manifest, required, available = manifest_inputs()
    manifest[field] = value
    with pytest.raises(checks.CheckError, match="manifest"):
        checks.check_manifest(manifest, required, available)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "schema",
        "bool-schema",
        "scope",
        "extra-key",
        "missing-key",
        "kernel",
        "missing-id",
        "extra-id",
        "empty-selectors",
        "duplicate",
        "missing-test",
        "prefix",
        "wildcard",
        "file-only",
        "bad-selector",
        "wrong-selector-type",
        "wrong-list-type",
        "no-required",
        "no-available",
        "no-rust",
        "no-python",
    ],
)
def test_manifest_missing_or_false_evidence(checks, case):
    manifest, required, available = manifest_inputs()
    selectors = manifest["clauses"]["REF-003"]
    if case == "empty":
        manifest = {}
    elif case == "schema":
        manifest["schema"] = 2
    elif case == "bool-schema":
        manifest["schema"] = True
    elif case == "scope":
        manifest["scope"] = "gpu"
    elif case == "extra-key":
        manifest["typo"] = []
    elif case == "missing-key":
        del manifest["kernel_specializations"]
    elif case == "kernel":
        manifest["kernel_specializations"] = ["imaginary-sm120"]
    elif case == "missing-id":
        del manifest["clauses"]["HOST-002"]
    elif case == "extra-id":
        manifest["clauses"]["REF-0030"] = selectors.copy()
    elif case == "empty-selectors":
        selectors.clear()
    elif case == "duplicate":
        selectors.append(selectors[0])
    elif case == "missing-test":
        selectors[0] += "_removed"
    elif case == "prefix":
        selectors[0] = selectors[0].removesuffix("_table")
    elif case == "wildcard":
        selectors[0] += "*"
    elif case == "file-only":
        selectors[0] = selectors[0].split("::")[0]
    elif case == "bad-selector":
        selectors[0] = "reference/tests/test_sample.py::test_table"
    elif case == "wrong-selector-type":
        selectors[0] = 7
    elif case == "wrong-list-type":
        manifest["clauses"]["REF-003"] = selectors[0]
    elif case == "no-required":
        required = set()
    elif case == "no-available":
        available = set()
    elif case == "no-rust":
        available = {s for s in available if not s.startswith("rust:")}
    else:
        available = {s for s in available if not s.startswith("python:")}
    with pytest.raises(checks.CheckError, match="manifest"):
        checks.check_manifest(manifest, required, available)


def coverage_inputs(root):
    files = []
    for crate, module, counts in [
        ("tq-core", "contracts", (11, 19, 3)),
        ("tq-testkit", "philox", (7, 13, 2)),
    ]:
        relative = f"crates/{crate}/src/{module}.rs"
        source(root, relative)
        source(root, f"crates/{crate}/src/lib.rs")
        summary = {
            metric: {"count": count, "covered": count, "percent": 100.0}
            for metric, count in zip(("lines", "regions", "functions"), counts, strict=True)
        }
        files.append({"filename": str(root / relative), "summary": summary})
    report = {"data": [{"files": files, "totals": {"percent": 100.0}}]}
    return report, {"tq-core", "tq-testkit"}


def test_coverage_per_crate_counts_and_unchanged_inputs(checks, tmp_path):
    report, crates = coverage_inputs(tmp_path)
    before = copy.deepcopy((report, crates))
    result = checks.check_coverage(report, tmp_path, crates)
    assert result == {
        "tq-core": {
            "lines": {"count": 11, "covered": 11},
            "regions": {"count": 19, "covered": 19},
            "functions": {"count": 3, "covered": 3},
        },
        "tq-testkit": {
            "lines": {"count": 7, "covered": 7},
            "regions": {"count": 13, "covered": 13},
            "functions": {"count": 2, "covered": 2},
        },
    }
    assert (report, crates) == before


def test_coverage_relative_paths_root_module_and_extra_files(checks, tmp_path):
    report, crates = coverage_inputs(tmp_path)
    files = report["data"][0]["files"]
    files[0]["filename"] = "crates/tq-core/src/contracts.rs"
    root_module = copy.deepcopy(files[0])
    root_module["filename"] = "crates/tq-core/src/lib.rs"
    files.append(root_module)
    external = copy.deepcopy(files[0])
    external["filename"] = source(tmp_path, "elsewhere/not_product.rs")
    files.append(external)
    result = checks.check_coverage(report, tmp_path, crates)
    assert result["tq-core"]["lines"] == {"count": 22, "covered": 22}
    assert result["tq-testkit"]["lines"] == {"count": 7, "covered": 7}


@pytest.mark.parametrize("metric", ["lines", "regions", "functions"])
@pytest.mark.parametrize(
    "case",
    ["deficit", "zero", "negative", "overcount", "bool", "float", "missing", "rounded"],
)
def test_coverage_raw_metric_boundaries(checks, tmp_path, metric, case):
    report, crates = coverage_inputs(tmp_path)
    summary = report["data"][0]["files"][0]["summary"]
    item = summary[metric]
    if case == "deficit":
        item["covered"] -= 1
    elif case == "zero":
        item.update(count=0, covered=0, percent=100.0)
    elif case == "negative":
        item.update(count=-1, covered=-1)
    elif case == "overcount":
        item["covered"] = item["count"] + 1
    elif case == "bool":
        item.update(count=True, covered=True)
    elif case == "float":
        item.update(count=1.0, covered=1.0)
    elif case == "missing":
        del summary[metric]
    else:
        item.update(count=1000000, covered=999999, percent=100.0)
    with pytest.raises(checks.CheckError, match="coverage"):
        checks.check_coverage(report, tmp_path, crates)


@pytest.mark.parametrize(
    "case",
    [
        "empty-report",
        "empty-data",
        "empty-files",
        "missing-crate",
        "unknown-crate",
        "empty-crates",
        "missing-source",
        "unreported-module",
        "duplicate",
        "normalized-duplicate",
        "missing-summary",
        "missing-filename",
        "wrong-count-type",
        "missing-count",
        "missing-covered",
    ],
)
def test_coverage_absent_and_inconsistent_evidence(checks, tmp_path, case):
    report, crates = coverage_inputs(tmp_path)
    files = report["data"][0]["files"]
    if case == "empty-report":
        report = {}
    elif case == "empty-data":
        report["data"] = []
    elif case == "empty-files":
        files.clear()
    elif case == "missing-crate":
        files.pop()
    elif case == "unknown-crate":
        crates.add("tq-extra")
    elif case == "empty-crates":
        crates = set()
    elif case == "missing-source":
        Path(files[0]["filename"]).unlink()
    elif case == "unreported-module":
        source(tmp_path, "crates/tq-core/src/forgotten.rs")
    elif case == "duplicate":
        files.append(copy.deepcopy(files[0]))
    elif case == "normalized-duplicate":
        duplicate = copy.deepcopy(files[0])
        duplicate["filename"] = "crates/tq-core/src/../src/contracts.rs"
        files.append(duplicate)
    elif case == "missing-summary":
        del files[0]["summary"]
    elif case == "missing-filename":
        del files[0]["filename"]
    elif case == "wrong-count-type":
        files[0]["summary"]["lines"]["covered"] = "11"
    elif case == "missing-count":
        del files[0]["summary"]["lines"]["count"]
    else:
        del files[0]["summary"]["lines"]["covered"]
    with pytest.raises(checks.CheckError, match="coverage"):
        checks.check_coverage(report, tmp_path, crates)
