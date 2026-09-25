# CPU tooling and CI foundation

Gate date: 20 Sep 2026. **Decision: local CPU tooling for Phase 0 step 0.4 accepted.**
The lead independently ran the strict CPU gate successfully. The Linux workflow is implemented
and passes actionlint; its hosted execution remains unverified because no remote exists.
This closes the local tooling work after the accepted Rust host contracts. Step 0.5 still needs
its own CUDA build-image and compile-only contract. ModelSpec parsing remains a separate slice.

## What was built

The lead wrote the [contract](../CPU-CHECKS.md) and acceptance tests before implementation;
the first test failed because the checker did not exist. The developer implemented four
standard-library checks, a small pytest collection hook, shell entry points and the CPU workflow.
No runtime crate or runtime dependency was added. Earlier Rust/Python implementations, tests,
fixtures, dependency pins, hooks and evaluation reports remain unchanged.

The checks enforce source limits, dependency direction and GPU dependency isolation, exact
per-crate coverage, and mappings from 24 CPU obligations to actually collected tests. Skipped
and expected-failure cases cannot supply manifest evidence. The runner executes debug and release
Rust tests, independent and full reference suites, tooling acceptance, audits and documentation.
Its version checks use complete tokens; failed gates stop execution and coverage is regenerated.

## Independent results

| Check | Result |
|---|---|
| New tooling acceptance | 246 pass: 171 API, 22 CLI, 16 inventory, 30 runner, 7 actual-audit cases |
| All root acceptance and harness tests | 275 pass |
| Checker coverage | 295/295 statements and 156/156 branches |
| Collection-hook coverage | 12/12 statements; no branch points |
| Python reference independent / full suites | 1,263 / 1,522 pass; 470/470 statements and 150/150 branches |
| Rust workspace, debug and release | 51 tests pass in each profile; none skipped |
| `tq-core` LLVM coverage | 295/295 lines, 456/456 regions, 22/22 functions |
| `tq-testkit` LLVM coverage | 54/54 lines, 102/102 regions, 9/9 functions |
| Deliberately broken checker variants | All 12 caught after an unmodified positive control |
| Dependency/manifest/source checks | One permitted internal dependency, 24 mapped clauses, 54 source files within limits |
| Strict local runner | Exit 0, including fresh advisory audit, private git guard, gitleaks and actionlint |

The [result artifact](2026-09-20-cpu-tooling.json) records versions, hashes, raw coverage counts,
the advisory revision and the exact fault replacements. The new checker and collection hook
have no excluded source lines. This is not a coverage claim for the older guard or upstream
fixture generators. Stable Rust branch coverage is not asserted. The existing guard also remains
outside the new Ruff scope, as deliberately recorded in DEV-GUIDELINES §2.2; its pre-existing style
findings need a separate guard change. All new Python and the reference/test suites are linted.

The 30 runner cases execute the shell against recording tools. They check ordering, failures
at each gate, release-test execution, required Python coverage, exact version pins, stale-report
rejection, private/public modes, advisory-fetch/check failures and ambiguous database caches.
Those stand-ins test orchestration; the separate strict run uses the actual tools and tests.

Seven audit cases run actual gitleaks and cargo-deny. They distinguish clean and forbidden
inputs, including an unpublished crate with an unapproved licence. The workspace currently has
zero external Rust dependencies; its own licences are checked. This is not a security assessment
of dependencies a future change may add.

## Corrections found during review

Independent checks caught omitted release-test execution and Python coverage, substring version
matching, a catalogue parser that failed when its heading started the document, and dependency
handling that confused repeated external names with repeated package IDs. Further probes rejected
duplicate workspace names, overwritten graph nodes, malformed manifest types, swallowed advisory
revision failures and arbitrary selection among multiple advisory caches.

The first complete run stopped at a secret-scanner false positive: an existing evaluation stores
the SHA-256 of a test whose filename contains `api`. The lead recomputed that digest. The reviewed
[scanner configuration](../../.gitleaks.toml) inherits the default rules and exempts only that
exact line in that exact report under the generic-key rule. A first global-path exception failed
acceptance: the scanner skipped the entire file. The rule-specific correction passes controls
that still reject a changed digest, the same entry elsewhere and a nearby synthetic credential.
Neither the old report nor the hooks were changed. See the upstream
[configuration reference](https://github.com/gitleaks/gitleaks#configuration).

The fault probes cover source caps, unterminated lines, declared and transitive GPU dependencies,
workspace direction, missing clauses, partial test-name matching, zero and incomplete coverage,
unreported sources, duplicate coverage and skipped-test evidence. One initially surviving fault
showed a gap in the lead's tests: declared GPU dependencies were caught only through the resolved
graph. A new case checks an unresolved optional declaration directly. The implementation already
rejected it. All 12 variants then failed assertions in isolated copies; none failed to parse.
This is targeted fault detection, not an exhaustive mutation score.

## Reproduction

Platform: macOS ARM64. Source: uncommitted changes on the baseline revision recorded in the JSON.
Use the exact versions and setup in [CONTRIBUTING](../../CONTRIBUTING.md#running-the-current-tests).
After materialising the pinned environments, from the repository root:

```sh
./scripts/check-cpu.sh
```

The command requires the private local denylist and a successful online advisory refresh.
It records the database revision and checks it offline, avoiding a second fetch. If multiple
database caches make that identity ambiguous, this bounded single-source runner fails explicitly.
For individual checks, use `scripts/check-lengths.sh`, `check-deps.sh`, `check-manifest.sh` and
`check-coverage.sh target/cpu/coverage.json`. The last command checks report contents; the complete
runner owns report freshness.

To reproduce a fault probe, copy `tools/cpu_checks.py`, `tools/tq_pytest_inventory.py`,
`tests/test_cpu_checks_acceptance.py` and `tests/test_cpu_inventory_acceptance.py` into an isolated
directory with the same relative layout. Run the following with the pinned Python environment,
first unchanged, then once per exact replacement in the artifact:

```sh
python -m pytest -q tests/test_cpu_checks_acceptance.py \
  tests/test_cpu_inventory_acceptance.py::test_collection_hook_preserves_names_and_excludes_all_markers
```

The control passes 172 tests. Every mutated run exits 1 with failed assertions. Discard each copy.

## CI boundary and next step

The workflow pins Action commits and tool versions, uses a read-only token, tests the PR head,
fetches full history and persists no checkout credentials. Fork PRs receive no secrets. Their
public-mode result explicitly leaves private identifier checking pending. Maintainers must run
the strict local guard on the reviewed revision before merge; a separate trusted-main job also
requires the denylist. Missing secrets fail that job. No hosted run or branch protection is claimed.

The existing guard scans tracked working-tree files and commit identities/messages. Acceptance
also scanned all tracked and non-ignored untracked candidate files against its content rules;
gitleaks scanned history and the Git-selected candidate tree. Untracked-file guard checks remain
a required part of local review before staging a change.

Next: define step 0.5's pinned CUDA build image and compile-only sm_120 checks (G-11/G-12).
No GPU, full-model, speed, quality, context or concurrency gate is closed by this CPU result.
Nothing was committed or pushed, and no model download or GPU spending occurred.
