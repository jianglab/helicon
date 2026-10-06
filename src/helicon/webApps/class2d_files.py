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
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

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


def _describe(name):
    """What the name of a Class2D file says about its job, or None.

    ``name`` is a path, a URL or just a file name (an upload). Returns a dict with
    ``program`` ("RELION" or "cryoSPARC"), ``stem`` (``run_it025``, ``J63_020``),
    ``file`` (the file name), ``folder`` (its last two folders, "" when the name has
    no folder) and ``iteration`` (an int, or None).
    """
    text = str(name or "").strip()
    if not text:
        return None
    path = unquote(urlparse(text).path) if "://" in text else text
    p = PurePosixPath(path.replace("\\", "/"))
    for program, pattern in (("RELION", _RELION), ("cryoSPARC", _CRYOSPARC)):
        m = pattern.match(p.name)
        if m:
            stem = m["stem"]
            it = re.search(r"it(\d+)$", stem) or re.search(r"_(\d+)$", stem)
            return dict(
                program=program,
                stem=stem,
                file=p.name,
                folder="/".join(p.parent.parts[-2:]),
                iteration=int(it.group(1)) if it else None,
            )
    return None


def pair_problem(params_name, images_name) -> str | None:
    """Why two files do not look like the two files of one Class2D job, or None.

    The parameter file and the class averages of a job are named alike: RELION's
    ``run_it025_data.star`` and ``run_it025_classes.mrcs`` in one job folder,
    cryoSPARC's ``J63_020_particles.cs`` and ``J63_020_class_averages.mrc``. Files
    whose names follow these patterns but not the same pair are named here. A name
    that follows neither (a renamed copy) cannot be judged, and gives None.

    Parameters
    ----------
    params_name, images_name : str
        The parameter file and the class-average file: a path, a URL, or a bare
        file name (an upload).
    """
    p, i = _describe(params_name), _describe(images_name)
    if p is None or i is None:
        return None
    if p["program"] != i["program"]:
        return (
            f"{p['file']} is named as {p['program']} writes its parameter file and "
            f"{i['file']} as {i['program']} writes its class averages, so they are "
            "not from the same job."
        )
    found = []
    if p["stem"] != i["stem"]:
        if p["iteration"] is not None and i["iteration"] is not None:
            same_job = re.sub(r"\d+$", "", p["stem"]) == re.sub(r"\d+$", "", i["stem"])
            found.append(
                f"the iterations differ ({p['stem']} and {i['stem']})"
                if same_job
                else f"the names differ ({p['stem']} and {i['stem']})"
            )
        else:
            found.append(f"the names differ ({p['stem']} and {i['stem']})")
    if p["folder"] and i["folder"] and p["folder"] != i["folder"]:
        found.append(f"the folders differ ({p['folder']} and {i['folder']})")
    if not found:
        return None
    example = (
        "J63_020_particles.cs with J63_020_class_averages.mrc"
        if p["program"] == "cryoSPARC"
        else "run_it025_data.star with run_it025_classes.mrcs"
    )
    return (
        "The two files do not look like a pair: "
        + "; ".join(found)
        + f". The files of one job are named alike, for example {example}."
    )


def abundance_error_message(
    params_name,
    images_name,
    n_param_classes=None,
    n_image_classes=None,
    highest_class=None,
):
    """The text for a Class2D parameter file and class averages that do not go together.

    Parameters
    ----------
    params_name, images_name : str
        As for :func:`pair_problem`.
    n_param_classes, highest_class : int, optional
        The number of classes the parameter file uses, and the highest class number.
    n_image_classes : int, optional
        The number of class averages in the image file.
    """
    parts = [
        "Failed to get class abundance from the provided Class2D parameter and "
        "image files."
    ]
    if n_image_classes is not None and highest_class is not None:
        used = (
            f"{n_param_classes} classes (numbered up to {highest_class})"
            if n_param_classes is not None
            else f"classes numbered up to {highest_class}"
        )
        parts.append(
            f"The parameter file has {used}; the image file has {n_image_classes}."
        )
    why = pair_problem(params_name, images_name)
    if why:
        parts.append(why)
    parts.append("Make sure that the two files are for the same Class2D job.")
    return " ".join(parts)


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


def relion_project_root(star) -> str:
    """The RELION project root of a file in one of its job folders, or "".

    See :func:`helicon.lib.io.get_relion_project_folder`: the folder with
    ``.gui_projectdir``, else the one with ``default_pipeline.star`` that is not a
    job folder.

    Parameters
    ----------
    star : str or Path
        A file in a job folder, such as the Class2D ``run_it020_data.star``.
    """
    from helicon.lib.io import get_relion_project_folder

    return get_relion_project_folder(str(star)) or ""


def project_folder(params_path, image=None) -> str:
    """The RELION or cryoSPARC project a Class2D parameter file belongs to.

    That is the folder the segments' image paths start from, which
    relion_reconstruct needs. Looked for, in order: the folder (the project's,
    or one above the parameter file) from which ``image`` exists; the project
    root by its marker -- ``.gui_projectdir`` for RELION, ``project.json``
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
    # RELION: the project root, by its .gui_projectdir (not a job folder, which in
    # RELION 5 also holds a copy of default_pipeline.star)
    root = Path(relion_project_root(star)) if relion_project_root(star) else None
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
