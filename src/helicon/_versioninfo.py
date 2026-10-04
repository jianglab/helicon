"""Where the helicon version number and commit come from.

The version is set by the git tag of a release (``v2026.10``). In a git
checkout, including an editable install (``pip install -e``), it is asked of
git each time, so it follows ``git pull`` and new commits without a
reinstall. Otherwise (an installed wheel) it is the value that
setuptools-scm wrote to ``_version.py`` when the package was built.
"""

import re
import subprocess
from pathlib import Path

UNKNOWN = "0+unknown"


def _from_git(root: Path) -> dict | None:
    """The version, commit and modified flag of the checkout at ``root``."""
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--match", "v[0-9]*", "--long", "--dirty"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    m = re.fullmatch(
        r"v(?P<tag>.+)-(?P<distance>\d+)-g(?P<commit>[0-9a-f]+)(?P<dirty>-dirty)?",
        out.stdout.strip(),
    )
    if m is None:
        return None
    distance = int(m["distance"])
    version = m["tag"] if distance == 0 else f"{m['tag']}.dev{distance}"
    return dict(version=version, commit=m["commit"], modified=bool(m["dirty"]))


def _from_build() -> dict | None:
    """The values written into the package when it was built."""
    try:
        from . import _version
    except ImportError:
        return None
    commit = getattr(_version, "commit_id", None)  # setuptools-scm: "gdf9d7ee2e"
    if commit:
        commit = commit.removeprefix("g")[:7]
    return dict(
        version=getattr(_version, "version", UNKNOWN),
        commit=commit,
        modified=False,
    )


def version_info() -> dict:
    """The version of this helicon.

    Returns
    -------
    dict
        ``version`` (str), ``commit`` (short git hash, or None if unknown) and
        ``modified`` (True if the checkout has uncommitted changes).
    """
    root = Path(__file__).resolve().parents[2]
    info = _from_git(root) if (root / ".git").exists() else None
    return info or _from_build() or dict(version=UNKNOWN, commit=None, modified=False)


def describe(info: dict | None = None) -> str:
    """The version for people, e.g. ``2026.10.dev3 (43d8e7d, modified)``."""
    info = info or version_info()
    notes = [x for x in (info["commit"], "modified" if info["modified"] else "") if x]
    return info["version"] + (f" ({', '.join(notes)})" if notes else "")
