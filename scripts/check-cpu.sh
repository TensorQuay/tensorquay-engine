#!/usr/bin/env bash
# The CPU gate runner: G-01…G-09 and the public part of G-10, in one order, stopping at the
# first failure. Options are parsed before any tool runs, every tool version is matched as a
# complete token before any gate runs, and the coverage report is deleted and regenerated so a
# stale file can never stand in as evidence.
#
# Default (local) mode requires the private identifier denylist and is the only mode that can
# support full G-10. `--public-ci` runs without secrets and reports that check as pending.
set -euo pipefail

mode=local
for argument in "$@"; do
  case "$argument" in
    --public-ci) mode=public ;;
    *)
      printf 'check-cpu: unknown option %s\n' "$argument" >&2
      exit 2
      ;;
  esac
done

cd -- "$(dirname -- "$0")/.."
root=$(pwd)
coverage=$root/target/cpu/coverage.json
candidates=

cleanup() {
  [ -n "$candidates" ] && rm -rf -- "$candidates"
}
trap cleanup EXIT

step() {
  printf '\n== %s ==\n' "$1"
}

# Pass through only complete NUL-terminated entries that still exist, so a truncated or
# non-NUL listing yields nothing to scan rather than a path built from arbitrary output.
select_existing() {
  local path
  while IFS= read -r -d '' path; do
    if [ -e "$path" ]; then
      printf '%s\0' "$path"
    fi
  done
  return 0
}

# A complete whitespace-separated token, never a substring: 0.10.120 must not satisfy 0.10.12.
require_version() {
  local label=$1 expected=$2
  shift 2
  local reported token
  reported=$("$@")
  for token in $reported; do
    if [ "$token" = "$expected" ]; then
      printf '%-16s %s\n' "$label" "$expected"
      return 0
    fi
  done
  printf 'check-cpu: %s reported "%s", which does not contain the pinned token %s\n' \
    "$label" "$reported" "$expected" >&2
  exit 3
}

step "pinned tool versions"
require_version rustc 1.95.0 rustc --version
require_version cargo 1.95.0 cargo --version
require_version uv 0.10.12 uv --version
require_version cargo-deny 0.20.2 cargo deny --version
require_version cargo-nextest 0.9.145 cargo nextest --version
# `--locked` here as well as on the gate: every coverage invocation resolves the pinned lock.
require_version cargo-llvm-cov 0.9.1 cargo llvm-cov --locked --version
require_version gitleaks 8.30.1 gitleaks version
require_version actionlint 1.7.12 actionlint -version

uv_run=(uv run --project reference --locked --offline)
checker=("${uv_run[@]}" python tools/cpu_checks.py)

step "G-01/G-02 formatting and lints"
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
# The reference package is checked from its own directory, as CONTRIBUTING documents, so ruff
# resolves `tq_reference` as first-party and judges its import order the same way everywhere.
(
  cd reference
  uv run --locked --offline ruff format --check --config pyproject.toml .
  uv run --locked --offline ruff check --config pyproject.toml .
)
# Root-level Python. tools/git-guard.py is absent on purpose: it is a frozen accepted source
# written before this config covered `tools/`, and reformatting it is a separate change.
"${uv_run[@]}" ruff format --check --config reference/pyproject.toml \
  tests tools/cpu_checks.py tools/tq_pytest_inventory.py tools/check-philox-upstream.py
"${uv_run[@]}" ruff check --config reference/pyproject.toml \
  tests tools/cpu_checks.py tools/tq_pytest_inventory.py tools/check-philox-upstream.py

step "G-03 source length"
"${checker[@]}" lengths

step "G-04 dependency audit"
# Refreshing the advisory database must succeed; a stale or unfetchable database is a failure,
# never a quiet pass. The revision actually used is reported.
cargo deny --locked fetch db
# The fetch above must succeed, deny.toml configures the single default advisory source, and the
# check below is offline, so it cannot reach a different database than the one identified here.
# One cache directory per configured source therefore means exactly one directory; anything else
# is ambiguous and fails rather than being guessed at.
caches=${CARGO_HOME:-$HOME/.cargo}/advisory-dbs
database=
found=0
for entry in "$caches"/*/; do
  if [ -d "$entry" ]; then
    database=$entry
    found=$((found + 1))
  fi
done
if [ "$found" -ne 1 ]; then
  printf 'check-cpu: expected exactly one advisory database under %s, found %s\n' \
    "$caches" "$found" >&2
  exit 4
fi
# Assigned on its own line: inside printf's arguments a failed lookup would be swallowed.
revision=$(git -C "$database" rev-parse HEAD)
printf 'advisory database %s at %s\n' "$database" "$revision"
# Offline, so the check uses the database just fetched and reported, not a second fetch.
cargo deny --locked --offline check

step "G-05 debug and release builds, with the release tests actually executed"
cargo build --workspace --all-targets --locked
cargo build --workspace --all-targets --release --locked
cargo nextest run --workspace --all-features --release --locked

step "G-06 per-crate Rust coverage"
rm -f -- "$coverage"
mkdir -p -- "$(dirname -- "$coverage")"
cargo llvm-cov nextest --workspace --all-features --locked --json --summary-only \
  --output-path "$coverage"
if [ ! -s "$coverage" ]; then
  printf 'check-cpu: the coverage run wrote no report to %s\n' "$coverage" >&2
  exit 5
fi
"${checker[@]}" coverage "$coverage"

step "G-06 Python coverage: independent reference suites, full reference suite, root tooling"
(
  cd reference
  uv run --locked --offline pytest --cov --cov-branch \
    tests/test_nvfp4_acceptance.py tests/test_mla_acceptance.py \
    tests/test_mla_upstream_acceptance.py tests/test_moe_acceptance.py \
    tests/test_moe_upstream_acceptance.py tests/test_router_acceptance.py \
    tests/test_router_upstream_acceptance.py tests/test_philox_acceptance.py
  uv run --locked --offline pytest --cov --cov-branch
)
"${uv_run[@]}" pytest tests --cov --cov-config=tools/cpu-tools.coveragerc --cov-branch

step "G-07/G-08 dependency direction and contract mapping"
"${checker[@]}" deps
"${checker[@]}" manifest

step "G-09 rustdoc"
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked

step "G-10 git guard and secret scan"
if [ "$mode" = public ]; then
  TQ_GUARD_REQUIRE_DENYLIST=0 python3 tools/git-guard.py audit
else
  TQ_GUARD_REQUIRE_DENYLIST=1 python3 tools/git-guard.py audit
fi
# The repository configuration is named explicitly for both scans: the candidate tree below is
# outside the repository, so gitleaks would otherwise discover no configuration there at all.
gitleaks git "$root" --no-banner --redact --log-opts=--all --config "$root/.gitleaks.toml"
# Scan the tree Git itself selects, so ignored virtual environments and build output are not
# mistaken for project source, and nothing tracked or newly added escapes the scan.
candidates=$(mktemp -d)
git ls-files -z --cached --others --exclude-standard \
  | select_existing \
  | tar -cf - --null -T - \
  | tar -xf - -C "$candidates"
gitleaks dir "$candidates" --no-banner --redact --config "$root/.gitleaks.toml"

step "workflow validation"
actionlint

if [ "$mode" = public ]; then
  printf '\ncheck-cpu: public CI mode passed. The private identifier check is PENDING: it needs\n'
  printf 'the local denylist, so this run is not full G-10 acceptance.\n'
else
  printf '\ncheck-cpu: all CPU gates passed, including the private identifier check.\n'
fi
