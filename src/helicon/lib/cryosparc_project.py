"""Reading the context a CryoSPARC project carries about its own files.

RELION says what a file is in its name: a job folder called ``Class2D`` holds
class averages, ``Refine3D`` holds maps, and a 2D stack is ``.mrcs`` while a
volume is ``.mrc``. CryoSPARC says none of that. Its job folders are ``J1``,
``J2``, ``J38``, and **everything is ``.mrc``** -- class averages, picking
templates, extracted particles, motion-corrected micrographs and 3D maps
alike.

So the shape of the data cannot be read off the suffix, and it cannot be read
off the header either. Measured on two real projects:

===============================  ===============  ==================
file                             dimensions       what it is
===============================  ===============  ==================
``J45_040_class_averages.mrc``   200 x 200 x 100  100 class averages
``templates.mrc``                180 x 180 x 10   10 picking templates
``..._particles.mrc``            200 x 200 x 328  328 particles
``J101_006_volume_map.mrc``      400 x 400 x 400  a map
``J100_mask.mrc``                400 x 400 x 400  a mask
``..._denoised.mrc``             2560 x 1819 x 1  a micrograph
===============================  ===============  ==================

A stack of 100 class averages has ``nz = 100``, so anything deciding by "more
than one Z slice" calls it a volume and offers to open it in ChimeraX.

What does distinguish them is the job that wrote them: its type, recorded in
``job.json``, and the naming convention CryoSPARC follows within a job. Both
are used here, the file name first because it is the more specific of the two.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .cache import cache as _cache, setup_cache_dir as _setup_cache_dir

# A project directory holds these; a job directory holds ``job.json``. The
# project marker is a file rather than a name pattern on purpose -- projects
# are named both ``P319`` and ``CS-phageg-slac`` in the wild.
PROJECT_MARKER = "project.json"
JOB_MARKER = "job.json"

# Metadata a project and its jobs keep beside the data, which is text to read
# rather than images to display.
METADATA_FILES = frozenset(
    {
        "project.json",
        "job.json",
        "job_manifest.json",
        "workspaces.json",
        "cs.lock",
    }
)

# Job types whose MRC output is a stack of 2D images, measured across two
# projects covering 38 distinct job types.
STACK_JOB_TYPES = frozenset(
    {
        "class_2D",
        "class_2D_new",
        "select_2D",
        "blob_picker_gpu",
        "template_picker_gpu",
        "manual_picker_v2",
        "denoise_train",
        "ctf_estimation",
        "patch_ctf_estimation_multi",
        "patch_motion_correction_multi",
        "reference_motion_correction",
        "extract_micrographs_multi",
        "extract_micrographs_cpu_parallel",
        "import_movies",
        "import_micrographs",
        "topaz_train",
        "topaz_extract",
    }
)

# Job types whose MRC output is a 3D volume: a map, a half map or a mask.
VOLUME_JOB_TYPES = frozenset(
    {
        "homo_abinit",
        "homo_refine",
        "homo_refine_new",
        "hetero_refine",
        "nonuniform_refine",
        "nonuniform_refine_new",
        "local_refine",
        "new_local_refine",
        "class_3D",
        "var_3D",
        "helix_refine",
        "import_volumes",
        "volume_tools",
        "local_resolution",
        "sharpen",
    }
)

# Names are the more specific signal and are checked first: a job type covers
# everything a job wrote, while these say what one file is.
_STACK_NAME_MARKERS = (
    "_class_averages",
    "_particles",
    "_denoised",
    "_patch_aligned",
    "_background",
    "_ctffind",
    "_ctf_diag_2d",
    "templates",
)

_VOLUME_NAME_MARKERS = (
    "_volume",
    "_map",
    "_mask",
)

# job.json is tens of kilobytes and a folder's files are all asked about in
# turn, so each job's type is read once and kept, keyed by path and mtime.
_JOB_TYPE_CACHE: dict[tuple[str, float], str | None] = {}


def _containing_dir(path: str | Path) -> Path | None:
    """The directory holding *path*, absolute but with symlinks left alone.

    Deliberately not ``Path.resolve()``. CryoSPARC imports data by symlinking
    it into the job's ``imported/`` directory, and resolving those links walks
    straight out of the project to wherever the movies actually live -- there
    is no ``job.json`` above that, so every imported file lost its job.
    """
    try:
        absolute = Path(os.path.abspath(str(path)))
    except (OSError, ValueError):
        return None
    try:
        if absolute.is_file():
            return absolute.parent
    except OSError:
        return absolute.parent
    return absolute


def is_cryosparc_project(folder: str | Path) -> bool:
    """Whether *folder* is the root of a CryoSPARC project.

    Parameters
    ----------
    folder : str or Path

    Returns
    -------
    bool
    """
    try:
        return (Path(folder) / PROJECT_MARKER).is_file()
    except OSError:
        return False


def find_project_root(path: str | Path) -> Path | None:
    """The CryoSPARC project *path* belongs to, if any.

    Parameters
    ----------
    path : str or Path
        A file or directory anywhere inside a project.

    Returns
    -------
    Path or None
    """
    current = _containing_dir(path)
    if current is None:
        return None
    for candidate in (current, *current.parents):
        if is_cryosparc_project(candidate):
            return candidate
    return None


def find_job_dir(path: str | Path) -> Path | None:
    """The job directory *path* belongs to, if any.

    Looks upward for ``job.json``, so a file in a job's ``extract/`` or
    ``motioncorrected/`` subdirectory is attributed to that job.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Path or None
    """
    current = _containing_dir(path)
    if current is None:
        return None
    for candidate in (current, *current.parents):
        try:
            if (candidate / JOB_MARKER).is_file():
                return candidate
        except OSError:
            continue
        if is_cryosparc_project(candidate):
            break
    return None


def job_type(path: str | Path) -> str | None:
    """The CryoSPARC job type that produced *path*.

    Parameters
    ----------
    path : str or Path
        A file or directory inside a job.

    Returns
    -------
    str or None
        The ``job_type`` recorded in ``job.json``, e.g. ``"class_2D_new"`` or
        ``"homo_refine_new"``, or None outside a CryoSPARC job.
    """
    job_dir = find_job_dir(path)
    if job_dir is None:
        return None
    marker = job_dir / JOB_MARKER
    try:
        key = (str(marker), marker.stat().st_mtime)
    except OSError:
        return None
    if key in _JOB_TYPE_CACHE:
        return _JOB_TYPE_CACHE[key]
    value = None
    try:
        with open(marker, "rb") as f:
            record = json.load(f)
        found = record.get("job_type") or record.get("type")
        if isinstance(found, str) and found:
            value = found
    except (OSError, ValueError):
        value = None
    _JOB_TYPE_CACHE[key] = value
    return value


def mrc_content(path: str | Path) -> str | None:
    """Whether an MRC file inside a CryoSPARC job holds a stack or a volume.

    Parameters
    ----------
    path : str or Path
        The MRC file.

    Returns
    -------
    str or None
        ``"stack"`` for 2D images, ``"volume"`` for a 3D map, or None when the
        file is not in a CryoSPARC job or its job type is not one we know. The
        caller should fall back to its own heuristics on None -- never assume
        a volume, since a 100-class average stack looks exactly like one.
    """
    name = Path(path).name.lower()
    stem = name.rsplit(".", 1)[0]

    for marker in _VOLUME_NAME_MARKERS:
        if marker in stem:
            return "volume"
    for marker in _STACK_NAME_MARKERS:
        if marker in stem:
            return "stack"

    kind = job_type(path)
    if kind is None:
        return None
    if kind in VOLUME_JOB_TYPES:
        return "volume"
    if kind in STACK_JOB_TYPES:
        return "stack"
    return None


def is_class_averages(path: str | Path) -> bool:
    """Whether *path* is a stack of 2D class averages.

    Those are the ones worth sorting by abundance and feeding to the helical
    apps, as ``Class2D`` job output is for RELION.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    bool
    """
    name = Path(path).name.lower()
    if "_class_averages" in name:
        return True
    return (
        job_type(path) in {"class_2D", "class_2D_new"} and mrc_content(path) == "stack"
    )


def is_metadata_file(path: str | Path) -> bool:
    """Whether *path* is one of a project's bookkeeping files."""
    return Path(path).name in METADATA_FILES


# Jobs that only exist for helical work.
HELICAL_JOB_TYPES = frozenset(
    {
        "helix_refine",
        "helix_refine_new",
        "filament_tracer",
        "filament_tracer_gpu",
    }
)

# The prefix CryoSPARC gives the per-particle filament columns
# (``filament/filament_uid``, ``filament/arc_length_A``, ...). This is the
# equivalent of RELION's rlnHelicalTubeID: it is written by filament tracing
# and carried along by everything downstream.
FILAMENT_FIELD_PREFIX = "filament/"

# How far back through a job's inputs to look for helical provenance.
_LINEAGE_DEPTH = 8

_HELICAL_CACHE: dict[tuple[str, float], bool] = {}


def _cs_has_filament_fields(cs_path: Path) -> bool:
    """Whether a ``.cs`` dataset carries the per-particle filament columns."""
    try:
        import numpy as np

        record = np.load(str(cs_path), mmap_mode="r", allow_pickle=False)
        names = record.dtype.names or ()
    except Exception:
        return False
    return any(str(n).startswith(FILAMENT_FIELD_PREFIX) for n in names)


def _parent_job_uids(job_dir: Path) -> list[str]:
    """The jobs feeding this one, read from its recorded input connections."""
    try:
        with open(job_dir / JOB_MARKER, "rb") as f:
            record = json.load(f)
    except (OSError, ValueError):
        return []
    uids = set()
    for group in record.get("input_slot_groups") or []:
        for connection in group.get("connections") or []:
            uid = connection.get("job_uid")
            if isinstance(uid, str) and uid:
                uids.add(uid)
    return sorted(uids)


def is_helical(path: str | Path) -> bool:
    """Whether *path* belongs to helical work, the way rlnIsHelix says so.

    CryoSPARC has no single flag for it, so three signals are used, in
    increasing cost:

    1. the job type -- ``helix_refine`` and ``filament_tracer`` exist for
       nothing else;
    2. the per-particle ``filament/`` columns in the job's ``.cs`` datasets,
       which filament tracing writes and everything downstream carries. This
       is the true counterpart of RELION's ``rlnHelicalTubeID``;
    3. the job's ancestry. A job that wrote no particle output of its own has
       nothing to test, and one that did may still be two steps removed from
       the tracing -- a 2D classification of particles extracted from traced
       filaments.

    Measured over three projects: every 2D classification in the helical
    project is recognised, seven of them by their own filament columns and two
    only through their ancestry, while all thirteen in the two non-helical
    projects are correctly rejected.

    Parameters
    ----------
    path : str or Path
        A file or directory inside a CryoSPARC job.

    Returns
    -------
    bool
    """
    job_dir = find_job_dir(path)
    if job_dir is None:
        return False
    marker = job_dir / JOB_MARKER
    try:
        key = (str(marker), marker.stat().st_mtime)
    except OSError:
        return False
    if key in _HELICAL_CACHE:
        return _HELICAL_CACHE[key]

    project = find_project_root(job_dir)
    seen: set[str] = set()
    frontier = [job_dir]
    result = False
    for _ in range(_LINEAGE_DEPTH):
        if result or not frontier:
            break
        next_frontier: list[Path] = []
        for candidate in frontier:
            name = candidate.name
            if name in seen:
                continue
            seen.add(name)
            if job_type(candidate) in HELICAL_JOB_TYPES:
                result = True
                break
            if _job_has_filament_data(candidate):
                result = True
                break
            if project is not None:
                for uid in _parent_job_uids(candidate):
                    parent = project / uid
                    if uid not in seen and (parent / JOB_MARKER).is_file():
                        next_frontier.append(parent)
        frontier = next_frontier

    _HELICAL_CACHE[key] = result
    return result


def _dataset_scan_order(name: str) -> tuple:
    """Sort key putting the datasets most likely to carry inherited columns first.

    Passthrough files come first. CryoSPARC keeps the parameters a job
    inherited -- as opposed to the ones it computed this iteration -- in
    ``*_passthrough_*.cs``, and for a ``select_2D`` job that is the *only*
    place the filament columns appear; its own iteration files have none.

    This ordering also has to survive jobs that hold a great many datasets: a
    helix_refine here carries up to 89 ``.cs`` files, and alphabetically
    ``J20_passthrough_particles.cs`` sorts after three dozen
    ``J20_0NN_particles.cs``, so any cap on the scan would have missed it.
    """
    return (
        "passthrough" not in name,
        "particle" not in name,
        name,
    )


def _job_has_filament_data(job_dir: Path) -> bool:
    """Whether any of a job's datasets carry filament columns."""
    try:
        names = sorted(
            (f.name for f in job_dir.iterdir() if f.suffix == ".cs"),
            key=_dataset_scan_order,
        )
    except OSError:
        return False
    for name in names[:16]:
        if _cs_has_filament_fields(job_dir / name):
            return True
    return False


# Where a 2D classification records what went into each class. The averages
# dataset gives one row per class, carrying the slice it occupies in the MRC;
# the particles dataset gives every particle's class assignment, and counting
# those is the abundance. RELION states the same thing directly as
# ``_rlnClassDistribution`` in model.star.
CLASS_ASSIGNMENT_FIELD = "alignments2D/class"
BLOB_INDEX_FIELD = "blob/idx"


def companion_dataset(mrc_path: str | Path) -> Path | None:
    """The ``.cs`` dataset describing an MRC written by CryoSPARC.

    Parameters
    ----------
    mrc_path : str or Path

    Returns
    -------
    Path or None
        ``J7_006_class_averages.cs`` beside ``J7_006_class_averages.mrc``.
    """
    candidate = Path(mrc_path).with_suffix(".cs")
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


def particles_dataset(mrc_path: str | Path) -> Path | None:
    """The particles ``.cs`` belonging to the same job output.

    Prefers the file sharing the averages' prefix -- ``J7_006_particles.cs``
    for ``J7_006_class_averages.mrc`` -- and otherwise takes the job's own
    particles, ignoring the passthrough and rejected sets, which carry no
    class assignment.

    Parameters
    ----------
    mrc_path : str or Path

    Returns
    -------
    Path or None
    """
    path = Path(mrc_path)
    direct = path.parent / (path.stem.replace("_class_averages", "_particles") + ".cs")
    try:
        if direct.is_file() and direct != path.with_suffix(".cs"):
            return direct
    except OSError:
        return None

    job_dir = find_job_dir(path)
    if job_dir is None:
        return None
    try:
        candidates = [
            f
            for f in sorted(job_dir.iterdir())
            if f.suffix == ".cs"
            and "particles" in f.name
            and "passthrough" not in f.name
            and "rejected" not in f.name
        ]
    except OSError:
        return None
    # the last, not the first: an iterative job writes
    # ``J20_000_particles.cs`` through ``J20_032_particles.cs``, and the
    # classes on display are the ones from the final iteration
    return candidates[-1] if candidates else None


def has_class_abundance(mrc_path: str | Path) -> bool:
    """Whether class abundances can be worked out for this MRC.

    A cheap check -- existence only. Counting the assignments means reading a
    particles dataset that can run to hundreds of megabytes, which is fine
    when the user asks to see the classes and not while browsing a folder.

    Parameters
    ----------
    mrc_path : str or Path

    Returns
    -------
    bool
    """
    if not is_class_averages(mrc_path):
        return False
    return (
        companion_dataset(mrc_path) is not None
        and particles_dataset(mrc_path) is not None
    )


def class_abundance(mrc_path: str | Path):
    """Each class average's share of the particles.

    Cached on disk, because the answer costs a full pass over the particles
    dataset -- 4.4 s for a 728 MB file over the network, against 0.1 s for a
    15 MB one -- and never changes for a job that has finished. The key
    includes both datasets' modification times, so a re-run job is recomputed
    rather than remembered wrongly.

    Parameters
    ----------
    mrc_path : str or Path
        The class averages MRC.

    Returns
    -------
    tuple or None
        ``(frame_indices, fractions)``: the slice each class occupies in the
        MRC, and the fraction of particles assigned to it. None when the
        datasets are missing or unreadable.
    """
    averages = companion_dataset(mrc_path)
    particles = particles_dataset(mrc_path)
    if averages is None or particles is None:
        return None
    try:
        stamps = (averages.stat().st_mtime, particles.stat().st_mtime)
    except OSError:
        return None
    return _class_abundance_cached(str(averages), str(particles), *stamps)


@_cache(
    expires_after=None,
    cache_dir=str(_setup_cache_dir() / "cryosparc"),
    verbose=0,
)
def _class_abundance_cached(
    averages_path: str,
    particles_path: str,
    averages_mtime: float,
    particles_mtime: float,
):
    """Count the class assignments. The mtimes are part of the cache key only."""
    import numpy as np

    averages = Path(averages_path)
    particles = Path(particles_path)

    try:
        rows = np.load(str(averages), mmap_mode="r", allow_pickle=False)
        names = rows.dtype.names or ()
        if BLOB_INDEX_FIELD in names:
            frames = np.asarray(rows[BLOB_INDEX_FIELD]).astype(int).ravel()
        else:
            frames = np.arange(len(rows))
    except Exception:
        return None
    if len(frames) == 0:
        return None

    try:
        assignments = np.load(str(particles), mmap_mode="r", allow_pickle=False)
        if CLASS_ASSIGNMENT_FIELD not in (assignments.dtype.names or ()):
            return None
        classes = np.asarray(assignments[CLASS_ASSIGNMENT_FIELD]).astype(int).ravel()
    except Exception:
        return None

    if classes.size == 0:
        return None
    counts = np.bincount(classes[classes >= 0], minlength=len(frames))
    total = counts.sum()
    if total <= 0:
        return None
    # row i of the averages dataset is class i, which is also the slice it
    # occupies; the frame index is read rather than assumed all the same
    fractions = counts[: len(frames)] / float(total)
    return frames, fractions


# A dataset that points at images records where they are and which slice each
# one occupies -- the same thing a RELION data.star says with
# ``_rlnImageName`` as ``idx@stack.mrcs``.
BLOB_PATH_FIELD = "blob/path"
BLOB_SHAPE_FIELD = "blob/shape"
BLOB_PSIZE_FIELD = "blob/psize_A"


def has_image_refs(cs_path: str | Path) -> bool:
    """Whether a ``.cs`` dataset points at 2D images.

    Cheap: only the dtype is read, never the rows. Particle and class-average
    datasets qualify; passthrough datasets carry inherited parameters and no
    images, and a volume's dataset points at a map rather than a stack.

    Parameters
    ----------
    cs_path : str or Path

    Returns
    -------
    bool
    """
    try:
        import numpy as np

        names = (
            np.load(str(cs_path), mmap_mode="r", allow_pickle=False).dtype.names or ()
        )
    except Exception:
        return False
    return BLOB_PATH_FIELD in names and BLOB_INDEX_FIELD in names


def image_refs(cs_path: str | Path):
    """The images a ``.cs`` dataset points at, ready for a lazy stack.

    ``blob/path`` is written relative to the project directory, so it is
    resolved against the project root rather than the job folder.

    Parameters
    ----------
    cs_path : str or Path

    Returns
    -------
    tuple or None
        ``(entries, shape, apix, n_skipped)`` where ``entries`` is a list of
        ``(frame_index, mrc_path, 0.0)`` in the order the dataset lists them,
        ``shape`` is ``(nx, ny)`` and ``apix`` the pixel size. None when the
        dataset points at no images.
    """
    path = Path(cs_path)
    if not has_image_refs(path):
        return None
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    root = find_project_root(path)
    return _image_refs_cached(str(path), str(root) if root else "", stamp)


@_cache(
    expires_after=None,
    cache_dir=str(_setup_cache_dir() / "cryosparc"),
    verbose=0,
)
def _image_refs_cached(cs_path: str, project_root: str, cs_mtime: float):
    """Resolve every row's image. The mtime is part of the cache key only.

    Cached because a particles dataset can hold over a million rows and
    reading the path column is a full pass over a file of several hundred
    megabytes -- the same reason the class abundances are cached.
    """
    import numpy as np

    try:
        rows = np.load(cs_path, mmap_mode="r", allow_pickle=False)
        names = rows.dtype.names or ()
        raw_paths = np.asarray(rows[BLOB_PATH_FIELD])
        indices = np.asarray(rows[BLOB_INDEX_FIELD]).astype(int).ravel()
    except Exception:
        return None
    if len(indices) == 0:
        return None

    shape = (0, 0)
    apix = 1.0
    try:
        if BLOB_SHAPE_FIELD in names:
            first = np.asarray(rows[BLOB_SHAPE_FIELD][0]).ravel()
            if first.size >= 2:
                # the dataset gives (ny, nx); a lazy stack wants (nx, ny)
                shape = (int(first[1]), int(first[0]))
        if BLOB_PSIZE_FIELD in names:
            value = float(np.asarray(rows[BLOB_PSIZE_FIELD]).ravel()[0])
            if value > 0:
                apix = value
    except Exception:
        pass

    base = Path(project_root) if project_root else Path(cs_path).parent
    resolved: dict[bytes, str | None] = {}
    entries = []
    skipped = 0
    for raw, index in zip(raw_paths, indices):
        key = raw if isinstance(raw, bytes) else str(raw).encode()
        if key not in resolved:
            text = key.decode("utf-8", "replace")
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = base / text
            resolved[key] = str(candidate) if candidate.is_file() else None
        target = resolved[key]
        if target is None:
            skipped += 1
            continue
        entries.append((int(index), target, 0.0))

    if not entries:
        return None
    return entries, shape, apix, skipped
