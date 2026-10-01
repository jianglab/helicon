"""Reconstruct the segments of an AbInitio3D selection with relion_reconstruct.

The MPI version, ``relion_reconstruct_mpi``, is used when it is installed:
measured on 20,000 segments of 256 pixels with 8 CPUs, 945 s for one thread,
419 s for eight threads (``--j 8``) and 39-46 s for eight MPI processes. It is
started with the ``mpirun`` of the MPI library it was built against -- another
one (the system's Open MPI 4, against a build on Open MPI 5) crashes it -- and
falls back to the threaded program if it fails.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Peak memory of one relion_reconstruct_mpi process, per voxel of the box
# (measured: 3.8 GB for a 256-pixel box, 5.9 GB for the first process), with
# room to spare.
_BYTES_PER_VOXEL = 300


def find_relion_reconstruct():
    """Path of ``relion_reconstruct`` on the PATH, or None."""
    return shutil.which("relion_reconstruct")


def _mpirun_for(exe):
    """The ``mpirun`` of the MPI library ``exe`` is linked against, or None.

    ``HELICON_MPIRUN`` overrides it; otherwise the library is found with
    ``ldd`` and its installation's ``bin/mpirun`` used, and failing that the
    ``mpirun`` on the PATH.
    """
    override = os.environ.get("HELICON_MPIRUN")
    if override:
        return override if shutil.which(override) else None
    try:
        out = subprocess.run(
            ["ldd", exe], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    m = re.search(r"libmpi\.so[.\d]*\s+=>\s+(\S+)", out)
    if m:
        mpirun = Path(m.group(1)).resolve().parent.parent / "bin" / "mpirun"
        if mpirun.exists():
            return str(mpirun)
    return shutil.which("mpirun")


def find_relion_reconstruct_mpi():
    """``(relion_reconstruct_mpi, mpirun)`` when both are found, or None."""
    exe = shutil.which("relion_reconstruct_mpi")
    if exe is None:
        return None
    mpirun = _mpirun_for(exe)
    return (exe, mpirun) if mpirun else None


def _mpi_layout(cpu, box, n_segments):
    """``(processes, threads per process)`` for ``cpu`` CPUs.

    Each process holds its own copy of the 3D arrays, so their number is also
    limited by the memory available; threads use the CPUs left over. At least
    a few hundred segments go to each process.
    """
    import helicon

    cpu = max(1, int(cpu))
    rank_gb = _BYTES_PER_VOXEL * float(box) ** 3 / 1024**3
    by_memory = helicon.available_cpu(mem_gb_per_cpu=max(rank_gb, 0.1))
    ranks = max(1, min(cpu, int(by_memory), int(n_segments) // 200))
    return ranks, max(1, cpu // ranks)


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
        CPUs to use: MPI processes (and threads in each) for
        ``relion_reconstruct_mpi``, threads for ``relion_reconstruct``.
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
        ``volume`` (z, y, x), ``apix``, ``path`` of the map, ``n_segments``,
        ``csym``, the ``command`` that was run and ``mpi`` (the number of MPI
        processes, 0 for the threaded program).

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
    args = ["--i", str(star), "--o", str(out), "--sym", f"c{int(max(csym, 1))}"]
    if "rlnDefocusU" in parts:
        args.append("--ctf")
    args += list(extra_args)

    def threaded():
        return [exe] + args + ["--j", str(max(1, int(cpu)))]

    command, mpi = threaded(), 0
    found = find_relion_reconstruct_mpi()
    if found is not None and int(cpu) > 1:
        box = _box_size(optics, segments)
        ranks, threads = _mpi_layout(cpu, box, len(parts))
        if ranks > 1:
            mpi_exe, mpirun = found
            command = (
                [
                    mpirun,
                    "--oversubscribe",
                    "--bind-to",
                    "none",
                    "-n",
                    str(ranks),
                    mpi_exe,
                ]
                + args
                + ["--j", str(threads)]
            )
            mpi = ranks
    run = subprocess.run(command, capture_output=True, text=True)
    if mpi and (run.returncode or not out.exists()):
        logger.warning(
            "relion_reconstruct_mpi failed, using relion_reconstruct: %s",
            (run.stdout + run.stderr)[-1000:],
        )
        command, mpi = threaded(), 0
        run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode or not out.exists():
        raise RuntimeError(
            f"{Path(command[0]).name} failed:\n"
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
        mpi=mpi,
    )


def _box_size(optics, segments):
    """The image size of the segments, in pixels."""
    for table in (optics, segments):
        if table is not None and "rlnImageSize" in table:
            return int(table["rlnImageSize"].iloc[0])
    return 256
