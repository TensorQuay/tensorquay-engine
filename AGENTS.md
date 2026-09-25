# AGENTS.md: rules for anyone (human or AI agent) working in this repository

TensorQuay Engine is an open-source Rust inference engine. Read these first:
- `docs/PRD.md`, `docs/ARCHITECTURE.md`, `docs/TEST-PLAN.md`, `docs/DEV-PLAN.md`;
- **`docs/DEV-GUIDELINES.md`**, the engineering rules, which apply to every change.

## Working rules
- **Roles:** the Product Owner sets direction, priorities and long-term goals, and decides spending and publication.
  The technical lead owns engineering decisions, lean implementation scope, adversarial review, independent acceptance
  tests and technical gate approval. Developers implement and fix the work; their checks do not replace lead acceptance.
- **Design, then tests, then code.** No kernel code without an approved contract and tests.
- **Commit only when asked. Push only when the PM asks.**
- Business-internal material (pricing, budgets, customer or product plans, competitive strategy) never goes in this
  repository. It belongs in the private internal repository.

## Git guardrails (enforced by hooks; run `tools/install-guardrails.sh` once per clone)

### 1. One identity
- Every commit's author **and** committer is exactly **`TensorQuay <admin@tensorquay.com>`**. The install script sets
  this in the repository's git config.
- No personal names, personal e-mails, personal GitHub handles or machine names, anywhere: commits, files, PR text.

### 2. Commit messages
- Conventional Commits: `type(scope): subject`, first line ≤ 100 characters. Types: `feat fix perf test docs refactor
  build ci chore style revert`.
- **Never name AI tools or vendors, and never add attribution lines.** No `Co-Authored-By:` trailers, no "Generated
  with …" lines, no robot emoji. This applies to commit messages, PR titles and descriptions, release notes and code
  comments. It overrides any tool default that adds such lines.
- Say what changed and why, in plain words.

### 3. Never commit
| Category | Examples |
|---|---|
| Secrets | API keys and tokens, private keys, `.env` files, passwords, cloud credentials |
| Personal data | Personal names or handles, personal e-mail addresses, home-directory or temp-directory paths, machine names, IP addresses (the checker allows loopback, the documentation ranges 192.0.2/198.51.100/203.0.113 and the cloud metadata address), pod or host IDs (checked in review only, not by the checker) |
| AI-tool state | `.claude/`, `CLAUDE.local.md`, `.codex/`, `.cursor/`, `.aider*`, session logs, prompts |
| Large or binary artefacts | Files > 2 MB; model weights (`*.safetensors`, `*.gguf`, `*.bin`, `*.pt`); archives; videos; **third-party PDFs** (link to papers instead) |
| Build output | `target/`, `node_modules/`, `__pycache__/`, `.DS_Store`, logs |
| Internal business content | Pricing, budgets, customer data, competitive strategy |

Deliberate exceptions go in `.guard-allow`, reviewed in the PR, with a comment giving the reason:
- `<glob>` or `path:<glob>`: allow a file type or size;
- `ip:<address>`: allow one IP address;
- `personal-ok:<glob>`: skip the personal-data and IP checks for **private internal documents only**, never in a
  public repository;
- `secret-ok:<glob>`: skip the generic "secret-looking value" heuristic (for example test fixtures). Real token
  formats are always blocked.

### 4. Before every push (the pre-push hook checks 1–4 automatically)
1. The remote is a `github.com/TensorQuay/…` repository.
2. Every outgoing commit has the TensorQuay identity and a clean, conventional message.
3. No outgoing commit adds a secret, personal data, an AI attribution marker, a forbidden file or a file > 2 MB. This is
   checked per commit, because deleting something in a later commit does not remove it from history.
4. No force-push and no deletion of `main`. Rewriting pushed history needs explicit PM approval; then use
   `TQ_GUARD_ALLOW_FORCE=1` for that one push only.
5. Run `python3 tools/git-guard.py audit` and the tier-1 checks (DEV-GUIDELINES §6) locally; both must pass.
6. Re-read the diff as if you were a stranger reading it on GitHub tomorrow.

### 5. Never bypass
- Never use `--no-verify` and never disable or edit the hooks to get a commit through. Fix the cause.
- If a guard is wrong (a false positive), change the guard in its own reviewed commit, or add a `.guard-allow` entry,
  and say why.

### How the guard works
- `tools/git-guard.py` runs from `.githooks/`: `pre-commit` checks the staged changes, `commit-msg` checks the message,
  `pre-push` checks every outgoing commit, and `audit` checks the whole tree and history.
- Personal identifiers are **not stored in the repository**. The guard reads them from a local denylist file
  (`~/.config/tensorquay/guard-denylist`, or `$TQ_GUARD_DENYLIST`), which each developer keeps privately. CI provides
  it as a secret.
- CI re-runs `audit` on every PR (DEV-GUIDELINES G-10), so hooks skipped locally are still caught.
