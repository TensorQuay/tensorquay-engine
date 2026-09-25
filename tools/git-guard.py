#!/usr/bin/env python3
"""TensorQuay git guardrails. Rules and rationale: AGENTS.md ("Git guardrails").

Subcommands (wired by .githooks/*):
  pre-commit            identity + staged paths, sizes and added lines
  commit-msg <file>     message format and banned words
  pre-push <remote> <url>   (ref updates on stdin) remote, force-push, every outgoing commit
  audit                 whole tree + full history; exit 1 on any violation

Personal identifiers are never stored in the repository. They are read from a local denylist file
(default ~/.config/tensorquay/guard-denylist, or $TQ_GUARD_DENYLIST), one case-insensitive literal per line.
"""
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

IDENTITY = ("TensorQuay", "admin@tensorquay.com")
REMOTE_OK = re.compile(r"github\.com[:/]TensorQuay/", re.I)
MAX_BYTES = 2 * 1024 * 1024
ZERO = "0" * 40
PROTECTED_BRANCHES = {"refs/heads/main", "refs/heads/master"}

# Commit messages must never name AI tools/vendors or carry attribution trailers.
MSG_BANNED = re.compile(
    r"claude|anthropic|codex|openai|chatgpt|copilot|gemini|\baider\b|co-authored-by|generated (with|by)|\U0001F916",
    re.I,
)
# File contents must never carry AI attribution markers (model names in research text are allowed).
FILE_ATTRIBUTION = re.compile(
    r"co-authored-by:|generated (with|by) \[?(claude|codex|chatgpt|copilot|gemini|an? ai)"
    r"|noreply@anthropic\.com|\U0001F916 generated",
    re.I,
)
CONVENTIONAL = re.compile(
    r"^(feat|fix|perf|test|docs|refactor|build|ci|chore|style|revert)(\([a-z0-9._/-]+\))?!?: \S.{0,98}$"
)
SECRETS = [
    re.compile(p)
    for p in (
        r"hf_[A-Za-z0-9]{30,}",
        r"gh[pousr]_[A-Za-z0-9]{36,}",
        r"github_pat_[A-Za-z0-9_]{50,}",
        r"sk-(ant-)?[A-Za-z0-9_-]{32,}",
        r"AKIA[0-9A-Z]{16}",
        r"xox[abprs]-[A-Za-z0-9-]{10,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    )
]
# Heuristic: a quoted value of 12+ chars assigned to a secret-like name. Skipped when the value is plainly a test value.
GENERIC_SECRET = re.compile(r"(?i)(api[_-]?key|secret|passw(or)?d|token)[\"']?\s*[:=]\s*[\"']([^\"'\s]{12,})[\"']")
FAKE_VALUE = re.compile(r"(?i)fixture|test|fake|dummy|example|placeholder|changeme|redacted")
PERSONAL = [
    re.compile(p, re.I)
    for p in (
        r"(?<![\w/.-])/Users/[A-Za-z0-9._-]+",
        r"(?<![\w/.-])/home/(?!runner\b)[A-Za-z0-9._-]+",
        r"(?<![\w/.-])/private/(tmp|var)/",
        r"C:\\+Users\\+",
        r"[A-Za-z0-9._%+-]+@(gmail|googlemail|qq|163|126|hotmail|outlook|live|icloud|yahoo)\.com",
    )
]
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IP_OK = {"127.0.0.1", "0.0.0.0", "169.254.169.254"}  # loopback, any, cloud metadata (used in SSRF tests)
IP_DOC_PREFIXES = ("192.0.2.", "198.51.100.", "203.0.113.")  # RFC 5737 documentation ranges
FORBIDDEN_PATHS = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "id_rsa*", "id_ed25519*",
    ".claude/*", "*/.claude/*", "CLAUDE.local.md", ".codex/*", ".cursor/*", ".aider*",
    ".DS_Store", "*/.DS_Store", "target/*", "node_modules/*", "*/node_modules/*", "*__pycache__/*",
    "*.safetensors", "*.gguf", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.onnx",
    "*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.7z", "*.pdf", "*.mp4", "*.mov",
)
# These files must spell out the banned AI words to enforce them; all other checks still apply.
GUARD_FILES = ("tools/git-guard.py", "AGENTS.md", "CLAUDE.md", ".githooks/*")


def git(*args, check=True):
    out = subprocess.run(["git", *args], capture_output=True, check=check)
    return out.stdout.decode("utf-8", "replace")


def root():
    return Path(git("rev-parse", "--show-toplevel").strip())


def load_list(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def denylist():
    path = Path(os.environ.get("TQ_GUARD_DENYLIST", "~/.config/tensorquay/guard-denylist")).expanduser()
    words = [w.lower() for w in load_list(path)]
    if not words:
        msg = f"guard: personal denylist {path} missing or empty"
        if os.environ.get("TQ_GUARD_REQUIRE_DENYLIST") == "1":
            sys.exit(msg)
        print(f"warning: {msg}; personal-name checks limited", file=sys.stderr)
    return words


def allowlist():
    """.guard-allow entries (reviewed in PRs): `<glob>` or `path:<glob>` allows a file type/size;
    `ip:<address>`; `personal-ok:<glob>` skips personal-data and IP checks (private internal docs only);
    `secret-ok:<glob>` skips the generic key=value heuristic (real token formats are always checked)."""
    allow = {"path": [], "ip": set(), "personal-ok": [], "secret-ok": []}
    for entry in load_list(root() / ".guard-allow"):
        kind, sep, value = entry.partition(":")
        if sep and kind in allow:
            allow[kind].add(value) if kind == "ip" else allow[kind].append(value)
        else:
            allow["path"].append(entry)
    return allow


def matches(path, patterns):
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def check_path(path, size, allow_globs, problems):
    if matches(path, allow_globs):
        return
    if matches(path, FORBIDDEN_PATHS) and path != ".env.example":
        problems.append(f"{path}: forbidden file type or location")
    if size is not None and size > MAX_BYTES:
        problems.append(f"{path}: {size / 1e6:.1f} MB exceeds {MAX_BYTES / 1e6:.0f} MB (use .guard-allow if intended)")


def check_line(path, lineno, text, deny, allow, problems):
    where = f"{path}:{lineno}"
    if not matches(path, GUARD_FILES) and FILE_ATTRIBUTION.search(text):
        problems.append(f"{where}: AI attribution marker")
    for rx in SECRETS:
        if rx.search(text):
            problems.append(f"{where}: possible secret ({rx.pattern[:24]}…)")
    if not matches(path, allow["secret-ok"]):
        for m in GENERIC_SECRET.finditer(text):
            if not FAKE_VALUE.search(m.group(3)):
                problems.append(f"{where}: possible secret (quoted value assigned to '{m.group(1)}')")
    if matches(path, allow["personal-ok"]):
        return
    for rx in PERSONAL:
        if rx.search(text):
            problems.append(f"{where}: personal path or e-mail")
    low = text.lower()
    for word in deny:
        if word in low:
            problems.append(f"{where}: personal identifier from the denylist")
    for ip in IPV4.findall(text):
        if ip in IP_OK or ip in allow["ip"] or ip.startswith(IP_DOC_PREFIXES):
            continue
        if all(int(p) <= 255 for p in ip.split(".")):
            problems.append(f"{where}: IP address {ip}")


def scan_patch(patch, deny, allow, problems):
    """Check the added lines of a unified diff (-U0)."""
    path, lineno = None, 0
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
        elif line.startswith("+") and path:
            check_line(path, lineno, line[1:], deny, allow, problems)
            lineno += 1


def check_identity(name, email, what, problems):
    if (name, email) != IDENTITY:
        problems.append(f"{what} is '{name} <{email}>', must be '{IDENTITY[0]} <{IDENTITY[1]}>'")


def check_message(msg, what, problems, require_format=True):
    body = "\n".join(ln for ln in msg.splitlines() if not ln.startswith("#")).strip()
    if MSG_BANNED.search(body):
        problems.append(f"{what}: message names an AI tool/vendor or carries an attribution trailer")
    first = body.splitlines()[0] if body else ""
    if require_format and not (CONVENTIONAL.match(first) or first.startswith(("Merge ", "Revert ", "fixup! ", "squash! "))):
        problems.append(f"{what}: first line must follow Conventional Commits ('type(scope): subject', ≤ 100 chars)")


def report(problems, stage):
    if problems:
        print(f"\n✖ git guard ({stage}) blocked this:", file=sys.stderr)
        for p in dict.fromkeys(problems):
            print(f"  - {p}", file=sys.stderr)
        print("Rules: AGENTS.md → Git guardrails. Fix the cause; never bypass with --no-verify.\n", file=sys.stderr)
        sys.exit(1)


def pre_commit():
    problems = []
    for var in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        m = re.match(r"(.*) <(.*)>", git("var", var).strip())
        check_identity(m.group(1), m.group(2), var.split("_")[1].lower(), problems)
    allow = allowlist()
    for entry in filter(None, git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split("\0")):
        size = int(git("cat-file", "-s", f":{entry}").strip())
        check_path(entry, size, allow["path"], problems)
    scan_patch(git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff"), denylist(), allow, problems)
    report(problems, "pre-commit")


def commit_msg(path):
    problems = []
    check_message(Path(path).read_text(encoding="utf-8"), "commit message", problems)
    report(problems, "commit-msg")


def check_commit(sha, deny, allow, problems):
    short = sha[:8]
    an, ae, cn, ce = git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce", sha).strip().split("\0")
    check_identity(an, ae, f"{short} author", problems)
    check_identity(cn, ce, f"{short} committer", problems)
    check_message(git("show", "-s", "--format=%B", sha), short, problems)
    for row in git("diff-tree", "-r", "--root", "--no-commit-id", "--diff-filter=ACMR", sha).splitlines():
        meta, path = row.split("\t", 1)
        blob = meta.split()[3]
        check_path(path, int(git("cat-file", "-s", blob).strip()), allow["path"], problems)
    scan_patch(git("show", "--format=", "-U0", "--no-color", "--no-ext-diff", sha), deny, allow, problems)


def pre_push(remote, url):
    problems = []
    if not REMOTE_OK.search(url):
        problems.append(f"remote '{remote}' ({url}) is not a TensorQuay GitHub repository")
    deny, allow = denylist(), allowlist()
    for line in sys.stdin.read().splitlines():
        local_ref, local_sha, remote_ref, remote_sha = line.split()
        if local_sha == ZERO:
            if remote_ref in PROTECTED_BRANCHES:
                problems.append(f"deleting {remote_ref} is not allowed")
            continue
        if remote_sha == ZERO:
            shas = git("rev-list", local_sha, "--not", f"--remotes={remote}").split()
        else:
            ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", remote_sha, local_sha]).returncode == 0
            if not ancestor and os.environ.get("TQ_GUARD_ALLOW_FORCE") != "1":
                problems.append(f"{remote_ref}: force-push (history rewrite) needs PM approval; see AGENTS.md")
            shas = git("rev-list", f"{remote_sha}..{local_sha}", check=False).split()
        for sha in shas:
            check_commit(sha, deny, allow, problems)
    report(problems, "pre-push")


def audit():
    problems = []
    deny, allow = denylist(), allowlist()
    for row in git("log", "--all", "--format=%h%x00%an%x00%ae%x00%cn%x00%ce").splitlines():
        sha, an, ae, cn, ce = row.split("\0")
        check_identity(an, ae, f"{sha} author", problems)
        check_identity(cn, ce, f"{sha} committer", problems)
        check_message(git("show", "-s", "--format=%B", sha), sha, problems, require_format=False)
    for path in filter(None, git("ls-files", "-z").split("\0")):
        full = root() / path
        size = full.stat().st_size if full.exists() else None
        check_path(path, size, allow["path"], problems)
        if size and size <= MAX_BYTES and b"\0" not in full.read_bytes()[:8192]:
            for n, text in enumerate(full.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                check_line(path, n, text, deny, allow, problems)
    report(problems, "audit")
    print("✔ git guard audit: clean")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "pre-commit":
        pre_commit()
    elif cmd == "commit-msg" and len(argv) == 3:
        commit_msg(argv[2])
    elif cmd == "pre-push" and len(argv) == 4:
        pre_push(argv[2], argv[3])
    elif cmd == "audit":
        audit()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
