"""The two files of a 2D classification: its parameters and its class averages.

HelicalPitch and AbInitio3D each ask for both, and the file browser opens them
with both. Given either one, :func:`companion` finds the other beside it:

- RELION: ``run_it025_data.star`` and ``run_it025_classes.mrcs`` (also
  ``run_data.star`` and ``run_classes.mrcs``);
- cryoSPARC: ``J63_020_particles.cs`` and ``J63_020_class_averages.mrc``.

Kept free of heavy imports, so the file browser can use it too.
"""

from __future__ import annotations

import re
from pathlib import Path

_RELION = re.compile(r"^(?P<stem>.*?)_(?P<kind>data\.star|classes\.mrcs)$")
_CRYOSPARC = re.compile(
    r"^(?P<stem>J\d+(?:_\d+)?)_(?P<kind>particles\.cs|class_averages\.mrc)$"
)
_PARTNER = {
    "data.star": "classes.mrcs",
    "classes.mrcs": "data.star",
    "particles.cs": "class_averages.mrc",
    "class_averages.mrc": "particles.cs",
}


def _local_file(path) -> Path | None:
    """``path`` as an existing local file, or None (URLs, missing files)."""
    if not path or "://" in str(path):
        return None
    try:
        p = Path(str(path).strip()).expanduser()
        return p if p.is_file() else None
    except (OSError, ValueError):
        return None


def is_params_file(path) -> bool:
    """Whether ``path`` names a Class2D parameter file (.star or .cs)."""
    return Path(str(path)).suffix.lower() in (".star", ".cs")


def companion(path) -> str | None:
    """The other file of the 2D classification that ``path`` belongs to.

    Parameters
    ----------
    path : str or Path
        A local Class2D parameter file (``.star``, ``.cs``) or class-average
        stack (``.mrcs``, ``.mrc``).

    Returns
    -------
    str or None
        The absolute path of the other file when it exists beside ``path``;
        None for a URL, a missing file, or a name that follows neither
        RELION's nor cryoSPARC's.
    """
    p = _local_file(path)
    if p is None:
        return None
    p = p.absolute()
    for pattern in (_RELION, _CRYOSPARC):
        m = pattern.match(p.name)
        if m:
            other = p.parent / f"{m['stem']}_{_PARTNER[m['kind']]}"
            if other.is_file():
                return str(other)
    if p.suffix.lower() == ".mrc":
        # a cryoSPARC job's averages whose particles are named otherwise
        from helicon.lib import cryosparc_project

        try:
            found = cryosparc_project.particles_dataset(p)
        except Exception:
            found = None
        if found is not None and Path(found).is_file():
            return str(Path(found).absolute())
    return None


def needs_filling(current, path) -> bool:
    """Whether the field holding ``current`` should take ``path``'s companion.

    Not when it already names an existing local file in the same folder as
    ``path``: that was chosen, not left over.
    """
    here = _local_file(current)
    there = _local_file(path)
    if here is None or there is None:
        return True
    return here.absolute().parent != there.absolute().parent


def local_path(value) -> Path | None:
    """A field's value as an existing local file: a path or a ``file://`` URL.

    Parameters
    ----------
    value : str or Path
        What a "server" field or a "url" field holds.

    Returns
    -------
    Path or None
        The absolute file, or None for a web URL or a missing file.
    """
    text = str(value or "").strip()
    if text.lower().startswith("file://"):
        from urllib.parse import unquote, urlparse

        text = unquote(urlparse(text).path)
    p = _local_file(text)
    return p.absolute() if p is not None else None


def project_folder(params_path, image=None) -> str:
    """The RELION or cryoSPARC project a Class2D parameter file belongs to.

    That is the folder the segments' image paths start from, which
    relion_reconstruct needs. Looked for, in order: the folder (the project's,
    or one above the parameter file) from which ``image`` exists; the project
    root by its marker -- ``default_pipeline.star`` for RELION, ``project.json``
    for cryoSPARC; and, without either, the usual depth of a job folder below
    its project (RELION ``Class2D/job010/``, cryoSPARC ``J63/``).

    Parameters
    ----------
    params_path : str or Path
        The parameter file (``.star`` or ``.cs``), as a path or ``file://`` URL.
    image : str, optional
        One segment's image path, as the parameter file gives it (relative).

    Returns
    -------
    str
        The project folder, or "" when the file is not local.
    """
    # absolute, but symlinks left alone: cryoSPARC links imported data into its
    # jobs, and resolving the links would leave the project
    star = local_path(params_path)
    if star is None:
        return ""
    root = None
    for folder in star.parents:  # RELION: the project root holds the pipeline
        if (folder / "default_pipeline.star").is_file():
            root = folder
            break
    if root is None:
        from helicon.lib import cryosparc_project

        try:
            root = cryosparc_project.find_project_root(star)
        except OSError:
            root = None
    if image and not Path(image).is_absolute():
        for folder in ([root] if root is not None else []) + list(star.parents):
            if (folder / image).exists():
                return str(folder)
    if root is not None:
        return str(root)
    depth = 1 if star.suffix.lower() == ".cs" else 2
    return str(star.parents[depth]) if len(star.parents) > depth else ""
