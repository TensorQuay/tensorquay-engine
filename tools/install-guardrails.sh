#!/bin/sh
# One-time setup per clone: company identity + guardrail hooks (AGENTS.md → Git guardrails).
set -e
cd "$(git rev-parse --show-toplevel)"
git config user.name "TensorQuay"
git config user.email "admin@tensorquay.com"
git config core.hooksPath .githooks
chmod +x .githooks/* tools/git-guard.py
deny="${TQ_GUARD_DENYLIST:-$HOME/.config/tensorquay/guard-denylist}"
[ -s "$deny" ] || echo "warning: create $deny (one personal identifier per line; never commit it)"
python3 tools/git-guard.py audit
