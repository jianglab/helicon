#!/usr/bin/env python
"""Refuse a commit that carries what identifies private data.

What identifies a private or unpublished data set (its owner's home or cluster folders,
their user name, RELION job folders, CryoSPARC project and job numbers) must not get into
the history of a public repository: once pushed, it stays. This checks the paths, the added
lines and the commit messages against rules about the *shape* of such things, which need no
list and no set-up from anybody:

* a home folder (``/home/<name>/``, ``/Users/<name>/``) or a cluster folder (``/net``,
  ``/scratch``, ``/mnt``, ``/lustre``, ``/gpfs``, ``/work``, ``/nfs``) that is not a placeholder;
* a RELION job folder (``Class2D/job010``) and a CryoSPARC project and job (``P12/J34``);  # private-ok
* the user name, the home folder and the host name of the person who runs the check.

A line that mentions EMPIAR (a public data set) or carries ``private-ok`` is not checked: use
the second for a test or a text that is *about* such a thing. Optionally a person can add
names of their own, which no rule can know, in a file kept outside the repository
(``$HELICON_PRIVATE_NAMES`` or ``~/.config/helicon/private_names.txt``): one name a line,
matched without regard to case; ``re:`` starts a regular expression; ``#`` a comment.

    python tools/check_private_names.py --staged           # the added lines and the paths
    python tools/check_private_names.py --message MSGFILE  # a commit message (commit-msg hook)
    python tools/check_private_names.py --range main..HEAD # commits not pushed yet (CI)
    python tools/check_private_names.py --all              # every line of every tracked file

Use it as the hooks in ``.pre-commit-config.yaml`` (``pre-commit install --hook-type
pre-commit --hook-type commit-msg``).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_LIST = Path.home() / ".config" / "helicon" / "private_names.txt"
SKIP_MARK = "private-ok"

# (what it is, a regular expression), by the shape of what identifies private data. A RELION
# job of three digits (job670) is a real job; job001 to job099 are what examples use
_PLACEHOLDERS = r"user|username|runner|you|me|name|example\w*|someone|\w{1,2}"
SHAPE_RULES = [
    (
        "a home folder",
        rf"/(?:home|Users)/(?!(?:{_PLACEHOLDERS})/)(?![$<{{*])[\w.-]+/",
    ),
    (
        "a cluster folder",
        r"(?<![\w.])/(?:net|scratch|mnt|lustre|gpfs|work|nfs)/(?!data/)[\w.-]+/[\w.-]+",
    ),
    (
        "a RELION job folder",
        r"\b(?:Class2D|Class3D|Refine3D|InitialModel|Extract|AutoPick|CtfFind|MotionCorr"
        r"|Select|Import|Polish|PostProcess|MaskCreate|LocalRes|MultiBody|JoinStar"
        r"|ManualPick|External|Subtract)/job(?!123\b)[1-9]\d{2,}",
    ),
    ("a CryoSPARC project and job", r"(?<![\w])P\d{2,}[/ _-]J\d{2,}\b"),
]
_NOT_A_PERSON = {"root", "user", "runner", "runneradmin", "ubuntu", "vscode", "admin"}
_NOT_A_HOST = {"localhost", "runner", "ubuntu", "debian", "docker", "github"}


class Rule:
    """One thing to refuse: what it is called and the pattern that finds it."""

    def __init__(self, label: str, pattern: re.Pattern):
        self.label = label
        self.pattern = pattern

    def search(self, line: str):
        return self.pattern.search(line)


def own_identity() -> list[tuple[str, re.Pattern]]:
    """The user name, the home folder and the host name of the person running the check."""
    import getpass
    import socket

    found = []
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    if len(user) >= 4 and user.lower() not in _NOT_A_PERSON:
        found.append(("your user name", re.compile(rf"\b{re.escape(user)}\b")))
    home = str(Path.home())
    if len(Path(home).name) >= 4 and Path(home).name.lower() not in _NOT_A_PERSON:
        found.append(("your home folder", re.compile(re.escape(home))))
    host = socket.gethostname().split(".")[0]
    if len(host) >= 4 and host.lower() not in _NOT_A_HOST:
        found.append(("your host name", re.compile(rf"\b{re.escape(host)}\b")))
    return found


def shape_rules() -> list[Rule]:
    """The rules that need no list: by shape, and the identity of the person running it."""
    rules = [Rule(label, re.compile(rx)) for label, rx in SHAPE_RULES]
    return rules + [Rule(label, rx) for label, rx in own_identity()]


def load_patterns(path: str | Path | None = None) -> list[Rule]:
    """The rules of the person's own list of names; none if there is no list."""
    path = Path(path or os.environ.get("HELICON_PRIVATE_NAMES") or DEFAULT_LIST)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    patterns = []
    for line in lines:
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        body = line[3:] if line.startswith("re:") else re.escape(line)
        patterns.append(Rule("a name on your list", re.compile(body, re.IGNORECASE)))
    return patterns


def _skipped(line: str) -> bool:
    return SKIP_MARK in line or "empiar" in line.lower()


def find(text: str, patterns: list[Rule]) -> list[str]:
    """The lines of ``text`` that match a rule: ``line number: the line   <- what it is``."""
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        if _skipped(line):
            continue
        for rule in patterns:
            if rule.search(line):
                hits.append(f"{number}: {line.strip()[:140]}   <- {rule.label}")
                break
    return hits


def _why(hit: str) -> str:
    return hit.split("<- ")[-1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def check_staged(patterns: list[Rule]) -> list[str]:
    """The added lines and the new paths of what is staged that name something private."""
    problems = []
    for path in _git(
        "diff", "--cached", "--name-only", "--diff-filter=AMR"
    ).splitlines():
        problems += [f"path: {path}   <- {_why(h)}" for h in find(path, patterns)]
    current = ""
    for line in _git("diff", "--cached", "--unified=0", "--no-color").splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+"):
            problems += [f"{current}: {h}" for h in find(line[1:], patterns)]
    return problems


def check_range(patterns: list[Rule], revision_range: str) -> list[str]:
    """The commit messages and the added lines of commits in a range."""
    problems = []
    for sha in _git("rev-list", revision_range).split():
        message = _git("show", "-s", "--format=%B", sha)
        problems += [f"{sha[:8]} message {h}" for h in find(message, patterns)]
        diff = _git("show", "--unified=0", "--no-color", "--format=", sha)
        current = ""
        for line in diff.splitlines():
            if line.startswith("+++ "):
                current = line[6:] if line.startswith("+++ b/") else line[4:]
                problems += [
                    f"{sha[:8]} path: {current}   <- {_why(h)}"
                    for h in find(current, patterns)
                ]
            elif line.startswith("+"):
                problems += [
                    f"{sha[:8]} {current}: {h}" for h in find(line[1:], patterns)
                ]
    return problems


def check_all(patterns: list[Rule]) -> list[str]:
    """Every line and path of every tracked text file (an audit; old lines count too)."""
    problems = []
    for path in _git("ls-files").splitlines():
        problems += [f"path: {path}   <- {_why(h)}" for h in find(path, patterns)]
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        problems += [f"{path}:{h}" for h in find(text, patterns)]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--staged", action="store_true")
    what.add_argument("--message", metavar="FILE")
    what.add_argument("--range", metavar="A..B")
    what.add_argument("--all", action="store_true")
    parser.add_argument("--names", help="the list of names (default: see --help text)")
    args = parser.parse_args(argv)
    patterns = shape_rules() + load_patterns(args.names)
    if args.staged:
        problems = check_staged(patterns)
    elif args.message:
        problems = [
            f"message {h}"
            for h in find(Path(args.message).read_text(encoding="utf-8"), patterns)
        ]
    elif args.all:
        problems = check_all(patterns)
    else:
        problems = check_range(patterns, args.range)
    if problems:
        print(
            "What identifies private data is in what you are committing:",
            file=sys.stderr,
        )
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        print(
            "Use a neutral name or a placeholder (/home/user/...); a line that is about "
            f"such a thing can carry '{SKIP_MARK}'. See the top of tools/check_private_names.py.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
