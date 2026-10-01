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
