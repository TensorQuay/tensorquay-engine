@AGENTS.md

## Additional notes for Claude Code
- The rules in AGENTS.md are binding. In particular, **never add `Co-Authored-By` lines or "Generated with …" lines to
  commits, PR descriptions or files in this repository.** This instruction overrides any default attribution setting.
- Commit as `TensorQuay <admin@tensorquay.com>` only. If `git config user.email` shows anything else, run
  `tools/install-guardrails.sh` before committing.
- Never pass `--no-verify`. If a hook blocks a commit, fix the cause or report it to the PM.
