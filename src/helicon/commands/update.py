"""Update helicon to the newest release (a version tag such as v2026.10)

Only releases count, not the commits between them. The newest release is looked
up on PyPI when helicon is published there, otherwise among the version tags of
the public GitHub repository. How it is then installed depends on how helicon
was installed: from PyPI, from a git URL, or as a git checkout (pip install -e).
"""

import json
import logging
import subprocess
import sys
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)

GITHUB_URL = "https://github.com/jianglab/helicon.git"
PYPI_URL = "https://pypi.org/pypi/helicon/json"
TIMEOUT = 20  # seconds, for each network request


def _version(text: str):
    """A comparable version from ``text`` such as ``v2026.10`` or ``2026.9.dev3``."""
    from packaging.version import InvalidVersion, Version

    try:
        return Version(text.removeprefix("v"))
    except InvalidVersion:
        return None


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command and return its output; a failure raises HeliconError."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        raise HeliconError(f"cannot run {' '.join(cmd[:2])}: {e}") from e
    if out.returncode != 0:
        raise HeliconError(f"{' '.join(cmd[:3])} failed: {out.stderr.strip()}")
    return out.stdout


def latest_on_pypi() -> str | None:
    """The newest version helicon has on PyPI, or None if it is not there."""
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=TIMEOUT) as r:
            info = json.load(r)["info"]
    except (OSError, ValueError, KeyError):  # 404, no network, odd reply
        return None
    # a project of the same name that is not ours is not an update
    ours = [info.get("home_page") or ""] + list(
        (info.get("project_urls") or {}).values()
    )
    if not any("jianglab" in (u or "") for u in ours):
        return None
    return info.get("version")


def tags_on_github() -> list[str]:
    """The release tags (``v...``) of the public repository, oldest first."""
    out = _run(["git", "ls-remote", "--tags", "--refs", GITHUB_URL, "v[0-9]*"])
    tags = [line.split("refs/tags/")[-1] for line in out.splitlines() if line.strip()]
    # release candidates (v2026.10rc1) are tags too, but not releases
    return sorted(
        (t for t in tags if _version(t) and not _version(t).is_prerelease),
        key=_version,
    )


def latest_release() -> tuple[str, str]:
    """The newest release as (version, where it was found: "pypi" or "github")."""
    version = latest_on_pypi()
    if version:
        return version, "pypi"
    tags = tags_on_github()
    if not tags:
        raise HeliconError(f"no release tags found at {GITHUB_URL}")
    return tags[-1].removeprefix("v"), "github"


def install_kind() -> str:
    """How this helicon is installed: "checkout", "git" or "pip"."""
    if (Path(__file__).resolve().parents[3] / ".git").exists():
        return "checkout"
    try:
        direct = metadata.distribution("helicon").read_text("direct_url.json")
        if direct and "vcs_info" in json.loads(direct):
            return "git"
    except (metadata.PackageNotFoundError, ValueError):
        pass
    return "pip"


def update_checkout(tag: str) -> str:
    """Bring the git checkout to the release ``tag``; returns what was done."""
    root = Path(__file__).resolve().parents[3]
    if _run(["git", "status", "--porcelain", "--untracked-files=no"], root).strip():
        raise HeliconError(
            f"{root} has uncommitted changes; commit or stash them, then update"
        )
    # fetched by name from the public repository, whatever "origin" is
    _run(
        ["git", "fetch", "--no-tags", GITHUB_URL, f"refs/tags/{tag}:refs/tags/{tag}"],
        root,
    )
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", tag, "HEAD"], cwd=root
        ).returncode
        == 0
    ):
        return f"{root} already contains {tag}"
    before = _run(["git", "rev-parse", "HEAD"], root).strip()
    on_branch = (
        subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=root, capture_output=True
        ).returncode
        == 0
    )
    # a branch moves forward to the tag, never merges or rewrites; a checkout
    # that is not on a branch (it sits on an older tag) moves to the new tag
    if on_branch:
        _run(["git", "merge", "--ff-only", tag], root)
    else:
        _run(["git", "checkout", "--quiet", tag], root)
    note = f"{root} is now at {tag}"
    changed = _run(
        ["git", "diff", "--name-only", before, "HEAD", "--", "pyproject.toml"], root
    )
    if changed.strip():
        note += (
            "\npyproject.toml changed: run 'pip install -e .' in that folder "
            "to bring the dependencies up to date"
        )
    return note


def update_installed(version: str, source: str, kind: str) -> str:
    """Install release ``version`` with pip; returns what was done."""
    pip = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if source == "pypi" and kind == "pip":
        _run(pip + [f"helicon=={version}"])
    else:
        _run(pip + [f"git+{GITHUB_URL}@v{version.removeprefix('v')}"])
    return f"installed helicon {version}"


def add_args(parser):
    """Add CLI arguments for the update command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to attach arguments to.
    """
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report whether a newer release exists, change nothing",
    )
    parser.add_argument(
        "--to",
        metavar="<release>",
        type=str,
        help="update to this release (for example v2026.10) instead of the newest",
        default="",
    )
    parser.add_argument(
        "--yes", action="store_true", help="do not ask before updating", default=False
    )


def main(args):
    """Check for a newer release and install it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Raises
    ------
    HeliconError
        If the release cannot be found or installed.
    """
    from helicon._versioninfo import describe, version_info

    current = version_info()
    kind = install_kind()
    if args.to:
        if not _version(args.to):
            raise HeliconError(f"{args.to!r} is not a release such as v2026.10")
        version, source = args.to.removeprefix("v"), None
        if kind == "pip":
            source = "pypi" if latest_on_pypi() else "github"
        source = source or "github"
    else:
        version, source = latest_release()
    print(f"installed: {describe(current)}")
    print(f"release:   {version} (from {source})")

    have = _version(current["version"])
    want = _version(version)
    if not args.to and have is not None:
        released = _version(have.base_version)  # 2026.10.dev3 follows 2026.10
        if want <= released:
            print(
                "this helicon is past the newest release; nothing to update"
                if have.is_devrelease
                else "helicon is up to date"
            )
            return
    if args.check:
        print(f"a newer release is available: run 'helicon update' to get {version}")
        return
    if not args.yes:
        if not sys.stdin.isatty():
            raise HeliconError("not a terminal: add --yes to update without asking")
        if input(f"update to {version}? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not updated")
            return
    if kind == "checkout":
        print(update_checkout(f"v{version}"))
    else:
        print(update_installed(version, source, kind))
