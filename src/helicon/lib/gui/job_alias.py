"""The aliases of the jobs of a RELION project.

RELION gives a job an alias by two things: a symbolic link named by the alias in the folder of the
job type that points to the job (``Class2D/my_alias -> ../Class2D/job010/``), and the alias in the
table of the processes of ``default_pipeline.star``. :func:`set_alias` does both, as RELION does,
:func:`aliases` reads them from the links.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = ["aliases", "check_alias", "set_alias"]

_JOB = re.compile(r"^job\d+$")


def aliases(type_folder: str | Path) -> dict[str, list[str]]:
    """The aliases of the jobs in the folder of a job type: ``{job name: [alias, ...]}``.

    An alias is a link in the folder that points to a job (``job010``) of the same folder.

    Parameters
    ----------
    type_folder : str or pathlib.Path
        The folder of a job type (``Class2D``).

    Returns
    -------
    dict
        The aliases of each job that has one, sorted.
    """
    found: dict[str, list[str]] = {}
    try:
        with os.scandir(type_folder) as entries:
            for entry in entries:
                if not entry.is_symlink():
                    continue
                job = alias_target(entry.path, Path(type_folder).name)
                if job is not None:
                    found.setdefault(job, []).append(entry.name)
    except OSError:
        return {}
    return {job: sorted(names) for job, names in found.items()}


def alias_target(link: str | Path, type_name: str) -> str | None:
    """The job that a link in a job-type folder points to (``job010``), or None if it does not
    point to a job of that folder (the link ``../Class2D/job010/`` or ``job010``)."""
    try:
        target = Path(os.readlink(link))
    except OSError:
        return None
    parts = [p for p in target.parts if p not in (".", "/")]
    if parts and _JOB.match(parts[-1]):
        if (
            len(parts) == 1
            or parts[-2:-1] == [type_name]
            and parts[0] in ("..", type_name)
        ):
            return parts[-1]
    return None


def check_alias(alias: str) -> str | None:
    """Why a name cannot be an alias, or None if it can."""
    if not alias:
        return "The alias is empty."
    if re.search(r"[\s/\\]", alias):
        return "An alias cannot have spaces or slashes."
    if alias.startswith("."):
        return "An alias cannot start with a dot."
    if _JOB.match(alias):
        return "An alias cannot look like the name of a job."
    return None


def set_alias(job_folder: str | Path, alias: str | None) -> None:
    """Give a job an alias, change it, or (with None) take it away.

    The link of an earlier alias is removed, the new one made, and the alias is written in the table
    of the processes of ``default_pipeline.star``, which is written whole and replaced, with the
    folder ``.relion_lock`` held as RELION does while it writes the pipeline.

    Parameters
    ----------
    job_folder : str or pathlib.Path
        The folder of the job (``Class2D/job010``).
    alias : str or None
        The new alias; None takes the alias away.

    Raises
    ------
    ValueError
        The folder is not a job, or the alias cannot be used (a name that is taken, say).
    OSError
        RELION holds the lock of the project, or a file cannot be written.
    """
    job_folder = Path(job_folder)
    job = job_folder.name
    type_folder = job_folder.parent
    root = type_folder.parent
    if not _JOB.match(job):
        raise ValueError(f"{job_folder} is not a job folder")
    if alias is not None:
        problem = check_alias(alias)
        if problem:
            raise ValueError(problem)
        taken = type_folder / alias
        if (taken.exists() or taken.is_symlink()) and (
            alias_target(taken, type_folder.name) != job
        ):
            raise ValueError(f"{alias} is already in use in {type_folder.name}")
    lock = root / ".relion_lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise OSError(
            "RELION is writing the pipeline of the project; try again in a moment"
        )
    try:
        for old in aliases(type_folder).get(job, []):
            (type_folder / old).unlink()
        if alias is not None:
            os.symlink(f"../{type_folder.name}/{job}/", type_folder / alias)
        _write_pipeline_alias(
            root, f"{type_folder.name}/{job}/", alias, type_folder.name
        )
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def _write_pipeline_alias(
    root: Path, process: str, alias: str | None, type_name: str
) -> None:
    """Put the alias in the row of the process in ``default_pipeline.star`` (no file, no change)."""
    path = root / "default_pipeline.star"
    try:
        lines = path.read_text().splitlines(keepends=True)
    except OSError:
        return
    columns: list[str] = []
    in_table = False
    out, changed = [], False
    for line in lines:
        text = line.strip()
        if text.startswith("data_"):
            in_table = text == "data_pipeline_processes"
            columns = []
        elif in_table and text.startswith("_rln"):
            columns.append(text.split()[0].lstrip("_"))
        elif (
            in_table
            and columns
            and text
            and not text.startswith(("#", "loop_"))
            and "rlnPipeLineProcessName" in columns
            and "rlnPipeLineProcessAlias" in columns
        ):
            parts = text.split()
            name_at = columns.index("rlnPipeLineProcessName")
            alias_at = columns.index("rlnPipeLineProcessAlias")
            if len(parts) > max(name_at, alias_at) and parts[name_at] == process:
                parts[alias_at] = f"{type_name}/{alias}/" if alias else "None"
                line = " ".join(parts) + "\n"
                changed = True
        out.append(line)
    if not changed:
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text("".join(out))
    try:
        os.chmod(temporary, path.stat().st_mode & 0o7777)
    except OSError:
        pass
    temporary.replace(path)
