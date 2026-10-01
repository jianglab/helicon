"""Reconstruct the segments of an AbInitio3D selection with relion_reconstruct."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def find_relion_reconstruct():
    """Path of ``relion_reconstruct`` on the PATH, or None."""
    return shutil.which("relion_reconstruct")


def _absolute_image_names(names, project_dir):
    """``rlnImageName`` values with the stack paths made absolute."""
    root = Path(project_dir).expanduser()
    out = []
    for name in names:
        index, _, path = str(name).rpartition("@")
        stack = Path(path)
        if not stack.is_absolute():
            stack = root / stack
        out.append(f"{index}@{stack}" if index else str(stack))
    return out


def reconstruct(segments, project_dir, cpu=1, csym=1, extra_args=(), work_dir=None):
    """A map, without helical symmetry, from segments with their angles and origins.

    Parameters
    ----------
    segments : pandas.DataFrame
        RELION particle rows with ``rlnImageName``, the angles and the origins,
        and the optics table in ``attrs["optics"]`` -- what
        ``helical_pitch_phase.select_segments`` returns.
    project_dir : str
        The directory the image paths in ``rlnImageName`` start from.
    cpu : int, optional
        Threads for ``relion_reconstruct``.
    csym : int, optional
        Cyclic symmetry to impose (``--sym c<csym>``). Defaults to 1.
    extra_args : sequence of str, optional
        More ``relion_reconstruct`` arguments.
    work_dir : str, optional
        Where the star file and the map are written. Defaults to a new
        temporary directory, which is left in place for the map to be read.

    Returns
    -------
    dict
        ``volume`` (z, y, x), ``apix``, ``path`` of the map, ``n_segments`` and
        the ``command`` that was run.

    Raises
    ------
    FileNotFoundError
        When relion_reconstruct is not on the PATH, or the first image stack is
        not under ``project_dir``.
    RuntimeError
        When relion_reconstruct fails; the message ends with its output.
    """
    import mrcfile
    import starfile

    exe = find_relion_reconstruct()
    if exe is None:
        raise FileNotFoundError("relion_reconstruct is not on the PATH")
    parts = segments.copy()
    parts["rlnImageName"] = _absolute_image_names(parts["rlnImageName"], project_dir)
    first = Path(parts["rlnImageName"].iloc[0].rpartition("@")[2])
    if not first.exists():
        raise FileNotFoundError(
            f"{first} does not exist: set the RELION project directory to the one "
            "the image paths in the star file start from"
        )
    # only RELION's own columns; the helicon ones would only draw warnings
    parts = parts[[c for c in parts.columns if c.startswith("rln")]]
    optics = segments.attrs.get("optics")
    folder = Path(work_dir or tempfile.mkdtemp(prefix="helicon_relion_"))
    folder.mkdir(parents=True, exist_ok=True)
    star = folder / "segments.star"
    blocks = dict(optics=optics, particles=parts) if optics is not None else parts
    starfile.write(blocks, str(star), overwrite=True)
    out = folder / "map.mrc"
    command = [
        exe,
        "--i",
        str(star),
        "--o",
        str(out),
        "--sym",
        "c1",
        "--j",
        str(int(cpu)),
    ]
    if "rlnDefocusU" in parts:
        command.append("--ctf")
    command += list(extra_args)
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode or not out.exists():
        raise RuntimeError(
            "relion_reconstruct failed:\n"
            + " ".join(command)
            + "\n"
            + (run.stdout + run.stderr)[-2500:]
        )
    with mrcfile.open(str(out)) as mrc:
        volume = np.asarray(mrc.data, dtype=np.float32).copy()
        apix = float(mrc.voxel_size.x)
    return dict(
        volume=volume,
        apix=apix,
        path=str(out),
        n_segments=len(parts),
        csym=int(max(csym, 1)),
        command=" ".join(command),
    )
