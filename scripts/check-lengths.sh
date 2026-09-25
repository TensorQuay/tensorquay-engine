#!/usr/bin/env bash
# Thin entry point for the G-0x CPU gate: run the checker from the repository root, in the
# pinned reference environment, forwarding every argument and its exit status unchanged.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
exec uv run --project reference --locked --offline python tools/cpu_checks.py lengths "$@"
