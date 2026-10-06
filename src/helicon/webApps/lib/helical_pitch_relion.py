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


_FUSED_CHECKED = {}


def fused_extract_available(exe=None):
    """Whether this ``relion_reconstruct`` can extract the segments itself.

    A modified RELION has ``--fused_extract``: it cuts each segment, in a
    rectangular box of any pixel size, from its micrograph and reconstructs a
    cuboid map, without the stored particle images. Found from its own help, once
    for each program.
    """
    exe = exe or find_relion_reconstruct()
    if exe is None:
        return False
    if exe not in _FUSED_CHECKED:
        try:
            run = subprocess.run(
                [exe, "--help"], capture_output=True, text=True, timeout=60
            )
            _FUSED_CHECKED[exe] = "--fused_extract" in run.stdout + run.stderr
        except (OSError, subprocess.SubprocessError):
            _FUSED_CHECKED[exe] = False
    return _FUSED_CHECKED[exe]


def micrographs_found(segments, project_dir):
    """Whether the segments' micrographs can be cut from, under ``project_dir``.

    Needs the micrograph name, the coordinates and the psi prior in the segments
    (what the fused extraction reads), and the first micrograph to exist: a path
    relative to ``project_dir``, or an absolute one.
    """
    if segments is None or not len(segments) or not str(project_dir or "").strip():
        return False
    needed = (
        "rlnMicrographName",
        "rlnCoordinateX",
        "rlnCoordinateY",
        "rlnAnglePsiPrior",
    )
    if any(c not in segments for c in needed):
        return False
    mic = Path(str(segments["rlnMicrographName"].iloc[0]))
    if not mic.is_absolute():
        mic = Path(str(project_dir).strip()).expanduser() / mic
    return mic.is_file()


def extraction_settings(project_dir, image_name):
    """How the segments were extracted, from the Extract job that made them.

    The fused extraction has to cut the segments as that job did, or the map is
    reconstructed from differently scaled and signed images. The job is found from
    the stack path of ``image_name`` (``Extract/job009/...``).

    Returns
    -------
    dict
        ``invert`` and ``norm`` (bool), ``bg_diameter`` (A, -1: not set) and
        ``helical_diameter`` (A, the outer diameter of the tube, or None if the
        segments were not extracted as helical tubes). RELION's defaults (invert,
        normalise, no tube) when the job is not found.
    """
    settings = dict(invert=True, norm=True, bg_diameter=-1.0, helical_diameter=None)
    stack = str(image_name).rpartition("@")[2]
    parts = Path(stack).parts
    if "Extract" not in parts:
        return settings
    job = Path(project_dir).expanduser().joinpath(*parts[: parts.index("Extract") + 2])
    star = job / "job.star"
    if not star.exists():
        return settings
    import starfile

    try:
        data = starfile.read(str(star))
        # one block is read as a table, several as a dict of them
        table = data["joboptions_values"] if isinstance(data, dict) else data
        table["rlnJobOptionVariable"]
    except (KeyError, ValueError, OSError):
        return settings
    values = dict(
        zip(
            table["rlnJobOptionVariable"].astype(str),
            table["rlnJobOptionValue"].astype(str),
        )
    )

    def yes(name, default):
        v = values.get(name)
        return default if v is None else v.strip().lower() in ("yes", "true", "1")

    def number(name):
        try:
            return float(values.get(name, "nan"))
        except ValueError:
            return float("nan")

    settings["invert"] = yes("do_invert", True)
    settings["norm"] = yes("do_norm", True)
    bg = number("bg_diameter")
    settings["bg_diameter"] = bg if np.isfinite(bg) else -1.0
    tube = number("helical_tube_outer_diameter")
    if yes("do_extract_helical_tubes", False) and np.isfinite(tube) and tube > 0:
        settings["helical_diameter"] = tube
    return settings


def fused_args(angpix, length, diameter, depth=None, settings=None):
    """The ``relion_reconstruct`` arguments for a fused extraction.

    Parameters
    ----------
    angpix : float
        Pixel size of the reconstruction, in A.
    length, diameter : int
        Size, in pixels at ``angpix``, of the segment boxes along the filament and
        across it. The cuboid map is ``diameter`` x ``diameter`` x ``depth``.
    depth : int, optional
        Size of the map along the filament axis. Defaults to ``length``.
    settings : dict, optional
        From :func:`extraction_settings`.
    """
    settings = settings or extraction_settings("", "")
    length, diameter = _even(length), _even(diameter)
    args = [
        "--fused_extract",
        "--fused_rotate_to_horizontal",
        "--angpix",
        f"{float(angpix):g}",
        "--box_x",
        str(length),
        "--box_y",
        str(diameter),
        "--box_z",
        str(_even(depth or length)),
    ]
    tube = settings.get("helical_diameter")
    if tube:
        args += ["--fused_helical_diameter", f"{float(tube):g}"]
    if settings.get("norm", True):
        bg = settings.get("bg_diameter", -1.0)
        outside = bg if bg and bg > 0 else (tube or 0.75 * diameter * float(angpix))
        args += [
            "--fused_norm",
            "--fused_bg_radius",
            f"{outside / 2 / float(angpix):g}",
        ]
    if settings.get("invert", True):
        args.append("--fused_invert_contrast")
    return args


def _even(n):
    """``n`` as an even integer of at least 2 (RELION's boxes are even)."""
    n = int(round(float(n)))
    return max(2, n + (n % 2))


def turn_to_horizontal(segments):
    """The orientations of segments whose boxes are turned to horizontal.

    ``--fused_rotate_to_horizontal`` cuts each box turned by minus the segment's
    psi prior, so that its tube lies along x. The orientations in the star file
    have to belong to those turned images: psi minus the psi prior, the origin
    shifts turned by the prior, and the defocus angle (the astigmatism turns with the
    image) plus it. Checked on EMPIAR-10940: the maps of 1,000 and of 7,433 segments
    correlate with the map of the stored images as well as that map does with itself,
    for this convention and not for the three others tried. Segments that already
    come from a turned extraction (``rlnParticleExtractionAngle``) are left as they
    are.

    Returns
    -------
    pandas.DataFrame
        A copy of ``segments``.
    """
    out = segments.copy()
    if "rlnParticleExtractionAngle" in out or "rlnAnglePsiPrior" not in out:
        return out
    prior = out["rlnAnglePsiPrior"].to_numpy(float)
    out["rlnAnglePsi"] = (out["rlnAnglePsi"].to_numpy(float) - prior + 180) % 360 - 180
    if "rlnOriginXAngst" in out and "rlnOriginYAngst" in out:
        a = np.deg2rad(prior)
        x = out["rlnOriginXAngst"].to_numpy(float)
        y = out["rlnOriginYAngst"].to_numpy(float)
        out["rlnOriginXAngst"] = x * np.cos(a) - y * np.sin(a)
        out["rlnOriginYAngst"] = x * np.sin(a) + y * np.cos(a)
    if "rlnDefocusAngle" in out:
        out["rlnDefocusAngle"] = (
            out["rlnDefocusAngle"].to_numpy(float) + prior + 90
        ) % 180 - 90
    return out


def reconstruct(
    segments, project_dir, cpu=1, csym=1, extra_args=(), work_dir=None, fused=None
):
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
        Where the star file and the map are written, and left. Defaults to a
        new temporary directory, which is removed once the map has been read
        (also when the run fails).
    fused : dict, optional
        Extract the segments from the micrographs, at another pixel size and in
        another box (:func:`fused_extract_available`): ``angpix`` (A), ``length``
        and ``diameter`` (pixels at ``angpix``), and optionally ``depth`` (the map's
        size along the filament, default ``length``). The segments need
        ``rlnMicrographName`` (a path under ``project_dir``), the coordinates and
        ``rlnAnglePsiPrior``. The map is then a cuboid.

    Returns
    -------
    dict
        ``volume`` (z, y, x), ``apix``, ``path`` of the map (None when it was
        written to a temporary directory, now removed), ``n_segments``,
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
    exe = find_relion_reconstruct()
    if exe is None:
        raise FileNotFoundError("relion_reconstruct is not on the PATH")
    parts = segments.copy()
    parts["rlnImageName"] = _absolute_image_names(parts["rlnImageName"], project_dir)
    first = Path(parts["rlnImageName"].iloc[0].rpartition("@")[2])
    if fused:
        if not fused_extract_available(exe):
            raise RuntimeError("this relion_reconstruct has no --fused_extract")
        need = [
            c
            for c in (
                "rlnMicrographName",
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnAnglePsiPrior",
            )
            if c not in parts
        ]
        if need:
            raise ValueError(
                "extracting from the micrographs needs "
                + ", ".join(need)
                + " in the star file"
            )
        root = Path(project_dir).expanduser()
        parts["rlnMicrographName"] = [
            str(m if Path(m).is_absolute() else root / m)
            for m in parts["rlnMicrographName"]
        ]
        mic = Path(parts["rlnMicrographName"].iloc[0])
        if not mic.exists():
            raise FileNotFoundError(
                f"{mic} does not exist: set the RELION project directory to the one "
                "the micrograph paths in the star file start from"
            )
        settings = extraction_settings(project_dir, segments["rlnImageName"].iloc[0])
        extra_args = list(extra_args) + fused_args(
            fused["angpix"],
            fused["length"],
            fused["diameter"],
            fused.get("depth"),
            settings,
        )
        parts = turn_to_horizontal(parts)
    elif not first.exists():
        raise FileNotFoundError(
            f"{first} does not exist: set the RELION project directory to the one "
            "the image paths in the star file start from"
        )
    # only RELION's own columns; the helicon ones would only draw warnings
    parts = parts[[c for c in parts.columns if c.startswith("rln")]]
    optics = segments.attrs.get("optics")
    own_folder = not work_dir
    folder = Path(work_dir or tempfile.mkdtemp(prefix="helicon_relion_"))
    try:
        return _reconstruct_in(
            folder,
            parts,
            segments,
            optics,
            exe,
            cpu,
            csym,
            extra_args,
            own_folder,
            fused,
        )
    finally:
        if own_folder:  # a temporary folder of ours: nothing is read from it later
            shutil.rmtree(folder, ignore_errors=True)


def _reconstruct_in(
    folder, parts, segments, optics, exe, cpu, csym, extra_args, own_folder, fused=None
):
    """The body of :func:`reconstruct`, writing into ``folder``."""
    import mrcfile
    import starfile

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
        if fused:  # a cuboid: the cube of the same number of voxels
            box = (
                _even(fused["diameter"]) ** 2
                * _even(fused.get("depth") or fused["length"])
            ) ** (1 / 3)
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
        path=None if own_folder else str(out),
        n_segments=len(parts),
        csym=int(max(csym, 1)),
        command=" ".join(command),
        mpi=mpi,
        fused=bool(fused),
    )


def _box_size(optics, segments):
    """The image size of the segments, in pixels."""
    for table in (optics, segments):
        if table is not None and "rlnImageSize" in table:
            return int(table["rlnImageSize"].iloc[0])
    return 256
