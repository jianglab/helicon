"""Parts of the file browser: a breadcrumb path bar, a sidebar of places and a preview pane.

Kept apart from ``file_browser`` so that module stays about the list of files. The
pure functions (the breadcrumb segments, the RELION project and job helpers, the
thumbnail and the preview of a file) do not need Qt and are tested without it; the
widgets use PySide6, as the browser does. Heavy packages (numpy, mrcfile, PIL) are
imported where they are used, so opening the browser stays quick.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "FlowLayout",
    "PathBar",
    "PlacesSidebar",
    "PreviewPane",
    "breadcrumb_parts",
    "build_preview",
    "Project",
    "find_project",
    "find_project_root",
    "cryosparc_job_info",
    "job_status",
    "list_job_types",
    "list_jobs",
    "project_jobs",
    "project_types",
    "read_cryosparc_job",
    "thumbnail_array",
]

# ── pure helpers ──────────────────────────────────────────────────────────────


def breadcrumb_parts(path: str | Path, home: str | Path | None = None):
    """The clickable segments of a path: ``[(label, full path), ...]``.

    The home folder is one segment, ``~``, when the path is inside it.

    Parameters
    ----------
    path : str or pathlib.Path
    home : str or pathlib.Path, optional
        Defaults to the user's home folder.

    Returns
    -------
    list of tuple of str
        From the first segment to the last.
    """
    path = Path(path)
    home = Path(home) if home is not None else Path.home()
    parts = path.parts
    out = []
    if (
        home.parts
        and parts[: len(home.parts)] == home.parts
        and len(parts) >= len(home.parts)
    ):
        out.append(("~", str(home)))
        start = len(home.parts)
    else:
        start = 0
    current = Path(*parts[:start]) if start else None
    for i in range(start, len(parts)):
        current = Path(parts[i]) if current is None else current / parts[i]
        out.append((parts[i], str(current)))
    return out


_JOB_FOLDER = re.compile(r"^job(\d+)$")
# the files RELION leaves in a job folder
_JOB_SUCCESS = "RELION_JOB_EXIT_SUCCESS"
_JOB_FAILURE = "RELION_JOB_EXIT_FAILURE"
_JOB_ABORTED = "RELION_JOB_EXIT_ABORTED"


def _is_job_folder_of(folder: Path) -> bool:
    """Whether a folder is a job of a project above it (``<project>/Class2D/job010``),
    though it may hold a ``default_pipeline.star`` of its own: RELION leaves a copy in
    its job folders, which must not make the job a project."""
    try:
        if (
            _JOB_FOLDER.match(folder.name)
            and (folder.parent.parent / "default_pipeline.star").is_file()
        ):
            return True
        if (
            _CRYOSPARC_JOB.match(folder.name)
            and (folder.parent / "project.json").is_file()
        ):
            return True
    except OSError:
        pass
    return False


def find_project_root(path: str | Path) -> Path | None:
    """The RELION project that holds a folder: the nearest folder above it with
    ``default_pipeline.star`` that is not a job folder of a project above; None if there
    is none."""
    path = Path(path)
    for folder in (path, *path.parents):
        try:
            if (folder / ".gui_projectdir").is_file() or (
                (folder / "default_pipeline.star").is_file()
                and not _is_job_folder_of(folder)
            ):
                return folder
        except OSError:
            return None
    return None


def job_status(folder: str | Path) -> str:
    """The state of a RELION job folder: ``success``, ``failure``, ``aborted``,
    ``running`` (it has been started and has not ended) or ``unknown``."""
    folder = Path(folder)
    try:
        if (folder / _JOB_SUCCESS).exists():
            return "success"
        if (folder / _JOB_FAILURE).exists():
            return "failure"
        if (folder / _JOB_ABORTED).exists():
            return "aborted"
        if (folder / "note.txt").exists() or (folder / "job.star").exists():
            return "running"
    except OSError:
        pass
    return "unknown"


def list_job_types(project: str | Path) -> list[str]:
    """The job types of a project that have jobs (``Class2D``, ``Extract``, ...), sorted."""
    out = []
    try:
        for entry in sorted(Path(project).iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                if any(_JOB_FOLDER.match(c.name) for c in entry.iterdir()):
                    out.append(entry.name)
            except OSError:
                continue
    except OSError:
        pass
    return out


def list_jobs(project: str | Path, job_type: str) -> list[tuple[str, str]]:
    """The jobs of a type: ``[(name, status), ...]`` by number, newest first."""
    folder = Path(project) / job_type
    jobs = []
    try:
        for entry in folder.iterdir():
            m = _JOB_FOLDER.match(entry.name)
            if m and entry.is_dir():
                jobs.append((int(m.group(1)), entry.name, job_status(entry)))
    except OSError:
        return []
    return [(name, status) for _, name, status in sorted(jobs, reverse=True)]


class Project(NamedTuple):
    """A RELION or CryoSPARC project: its ``kind`` (``relion`` or ``cryosparc``) and root."""

    kind: str
    root: Path

    @property
    def name(self) -> str:
        if self.kind == "cryosparc":
            title = _read_json(self.root / "project.json").get("title")
            if isinstance(title, str) and title:
                return title
        return self.root.name


_CRYOSPARC_JOB = re.compile(r"^J(\d+)$")
# the state of a CryoSPARC job, as job.json says it, in the terms of job_status
_CRYOSPARC_STATUS = {
    "completed": "success",
    "failed": "failure",
    "killed": "aborted",
    "running": "running",
    "launched": "running",
    "started": "running",
    "queued": "queued",
    "waiting": "queued",
    # built (its inputs and parameters set) but not queued to run
    "building": "pending",
}
# the names the CryoSPARC interface gives to the job types of helical and common work
_CRYOSPARC_NAMES = {
    "import_movies": "Import Movies",
    "import_micrographs": "Import Micrographs",
    "import_particles": "Import Particles",
    "import_volumes": "Import Volumes",
    "patch_motion_correction_multi": "Patch Motion Correction",
    "patch_ctf_estimation_multi": "Patch CTF Estimation",
    "curate_exposures_v2": "Curate Exposures",
    "manual_picker_v2": "Manual Picker",
    "blob_picker_gpu": "Blob Picker",
    "template_picker_gpu": "Template Picker",
    "filament_tracer_gpu": "Filament Tracer",
    "inspect_picks_v2": "Inspect Picks",
    "extract_micrographs_multi": "Extract From Micrographs",
    "class_2D_new": "2D Classification",
    "select_2D": "Select 2D Classes",
    "helix_refine": "Helical Refinement",
    "homo_refine_new": "Homogeneous Refinement",
    "hetero_refine": "Heterogeneous Refinement",
    "nonuniform_refine_new": "Non-Uniform Refinement",
    "new_local_refine": "Local Refinement",
    "homo_abinit": "Ab-Initio Reconstruction",
    "class_3D": "3D Classification",
    "sharpen": "Sharpening Tools",
    "volume_tools": "Volume Tools",
    "symmetry_expansion": "Symmetry Expansion",
    "particle_sets": "Particle Sets Tool",
    "exposure_sets": "Exposure Sets Tool",
    "ctf_refine_global": "Global CTF Refinement",
    "ctf_refine_local": "Local CTF Refinement",
    "reference_motion_correction": "Reference Based Motion Correction",
    "downsample_particles": "Downsample Particles",
    "restack_particles": "Restack Particles",
    "validation": "Validation (FSC)",
    "denoise_train": "Micrograph Denoiser (Train)",
    "denoise": "Micrograph Denoiser",
}


def _read_json(path: Path) -> dict:
    """A JSON file as a dict, or ``{}`` if it is missing or damaged."""
    try:
        with open(path, "rb") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def cryosparc_type_name(job_type: str) -> str:
    """The name of a CryoSPARC job type for people: the interface's, or the key made readable."""
    if job_type in _CRYOSPARC_NAMES:
        return _CRYOSPARC_NAMES[job_type]
    words = re.sub(r"(_new|_multi|_v\d+|_gpu)+$", "", job_type).split("_")
    return (
        " ".join(
            w if w.isupper() or w[:1].isdigit() else w.capitalize() for w in words if w
        )
        or job_type
    )


def read_cryosparc_job(folder: str | Path) -> dict | None:
    """What ``job.json`` says of a CryoSPARC job: ``uid``, ``type`` (the key), ``type_name``,
    ``title``, ``status`` (as :func:`job_status` gives it), ``created``, ``completed``
    and ``parents`` (the jobs feeding it); None if the folder holds no ``job.json``."""
    folder = Path(folder)
    record = _read_json(folder / "job.json")
    if not record:
        return None
    job_type = record.get("job_type") or record.get("type") or ""
    # the jobs feeding this one: listed in "parents", or else in the connections
    parents = [str(p) for p in record.get("parents") or []]
    if not parents:
        for group in record.get("input_slot_groups") or []:
            for connection in (group or {}).get("connections") or []:
                uid = (connection or {}).get("job_uid")
                if uid and uid not in parents:
                    parents.append(uid)
    return dict(
        uid=str(record.get("uid") or folder.name),
        type=str(job_type),
        type_name=cryosparc_type_name(str(job_type)) if job_type else "Job",
        title=str(record.get("title") or ""),
        status=_CRYOSPARC_STATUS.get(str(record.get("status", "")).lower(), "unknown"),
        raw_status=str(record.get("status") or ""),
        created=_cryosparc_time(record.get("started_at") or record.get("created_at")),
        completed=_cryosparc_time(record.get("completed_at")),
        parents=parents,
    )


def _cryosparc_time(value) -> str:
    """A time of ``job.json`` (``{"$date": "2026-10-05T22:52:30.382Z"}``, or a plain
    string) as ``2026-10-05 22:52``; empty if there is none."""
    if isinstance(value, dict):
        value = value.get("$date")
    if not isinstance(value, str):
        return ""
    return value[:16].replace("T", " ")


def find_project(path: str | Path) -> Project | None:
    """The RELION or CryoSPARC project a folder is in: the nearest one above it.

    A job folder is never the project, even where it holds a ``default_pipeline.star``
    of its own (RELION leaves one in each job folder).
    """
    path = Path(path)
    for folder in (path, *path.parents):
        try:
            # RELION's own mark of a project root, which its job folders do not have
            if (folder / ".gui_projectdir").is_file():
                return Project("relion", folder)
            if _is_job_folder_of(folder):
                continue
            if (folder / "default_pipeline.star").is_file():
                return Project("relion", folder)
            if (folder / "project.json").is_file():
                return Project("cryosparc", folder)
        except OSError:
            return None
    return None


# the jobs of a CryoSPARC project by type, kept until the project folder changes
_CRYOSPARC_SCAN: dict[str, tuple[float, dict]] = {}


def invalidate_project_cache() -> None:
    """Forget what was read of the projects (the refresh button does this)."""
    _CRYOSPARC_SCAN.clear()


def _cryosparc_jobs(root: Path) -> dict[str, list[tuple[int, str, dict]]]:
    """The jobs of a CryoSPARC project grouped by job type: ``{type: [(number, folder name, job)]}``."""
    try:
        stamp = root.stat().st_mtime
    except OSError:
        return {}
    cached = _CRYOSPARC_SCAN.get(str(root))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    groups: dict[str, list] = {}
    try:
        for entry in root.iterdir():
            m = _CRYOSPARC_JOB.match(entry.name)
            if not m or not entry.is_dir():
                continue
            job = read_cryosparc_job(entry)
            if job is not None:
                groups.setdefault(job["type"], []).append(
                    (int(m.group(1)), entry.name, job)
                )
    except OSError:
        return {}
    _CRYOSPARC_SCAN[str(root)] = (stamp, groups)
    return groups


def cryosparc_job_info(path: str | Path) -> dict | None:
    """The job of a folder directly under a CryoSPARC project root (see
    :func:`read_cryosparc_job`), or None. One read of the project serves every folder of it.
    """
    path = Path(path)
    try:
        if not (path.parent / "project.json").is_file():
            return None
    except OSError:
        return None
    if not _CRYOSPARC_JOB.match(path.name):
        return None
    for jobs in _cryosparc_jobs(path.parent).values():
        for _number, name, job in jobs:
            if name == path.name:
                return job
    return None


def project_types(project: Project) -> list[tuple[str, str, str]]:
    """The job types of a project that have jobs: ``[(key, label, folder), ...]``.

    A RELION type is a folder (``Class2D``); a CryoSPARC type is a job type
    (``class_2D_new``, labelled ``2D Classification``) with no folder of its own.
    """
    if project.kind == "relion":
        return [(t, t, str(project.root / t)) for t in list_job_types(project.root)]
    groups = _cryosparc_jobs(project.root)
    return sorted(
        ((t, cryosparc_type_name(t) if t else "Other", "") for t in groups),
        key=lambda x: x[1].lower(),
    )


def project_jobs(project: Project, key: str) -> list[tuple[str, str, str, str]]:
    """The jobs of a type: ``[(folder name, status, label, tooltip), ...]``, newest first.

    The label of a CryoSPARC job is its number and title (``J45  my classes``).
    """
    if project.kind == "relion":
        return [
            (name, status, name, f"{project.root / key / name}  ({status})")
            for name, status in list_jobs(project.root, key)
        ]
    out = []
    for number, name, job in sorted(
        _cryosparc_jobs(project.root).get(key, []), reverse=True
    ):
        label = f"{name}  {job['title']}" if job["title"] else name
        raw = f" ({job['raw_status']})" if job["raw_status"] else ""
        out.append((name, job["status"], label, f"{project.root / name}{raw}"))
    return out


def thumbnail_array(data, max_size: int = 256):
    """A grey-level picture of a 2D array: stretched between its 1st and 99th
    percentiles and cut down by whole steps to at most ``max_size`` pixels a side.

    Parameters
    ----------
    data : array_like
        2D.
    max_size : int, optional
        Defaults to 256.

    Returns
    -------
    numpy.ndarray
        2D ``uint8``.
    """
    import numpy as np

    a = np.asarray(data)
    step = max(1, int(np.ceil(max(a.shape) / max_size)))
    if step > 1:
        a = a[::step, ::step]
    a = a.astype(np.float32, copy=False)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _montage(frames, tile: int = 96, columns: int = 4):
    """Tiles of frames, each stretched on its own, in a grid with a thin gap."""
    import numpy as np

    pics = [thumbnail_array(f, tile) for f in frames]
    rows = (len(pics) + columns - 1) // columns
    gap = 2
    out = np.zeros((rows * (tile + gap) - gap, columns * (tile + gap) - gap), np.uint8)
    for i, pic in enumerate(pics):
        r, c = divmod(i, columns)
        y, x = r * (tile + gap), c * (tile + gap)
        h, w = pic.shape
        out[y + (tile - h) // 2 : y + (tile - h) // 2 + h, x + (tile - w) // 2 : x + (tile - w) // 2 + w] = pic  # fmt: skip
    return out


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


_MRC_SUFFIXES = {".mrc", ".mrcs", ".map", ".ccp4", ".rec", ".st", ".ali", ".ctf"}
_PIL_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
_MAX_PIL_PIXELS = 150_000_000
_STACK_PREVIEW_FRAMES = 12
_STACK_PREVIEW_BYTES = 96 * 2**20


def _folder_preview(path: Path, now: datetime | None = None) -> dict:
    out = dict(
        kind="folder", title=path.name or str(path), facts=[], text="", image=None
    )
    try:
        # os.scandir: a DirEntry knows if it is a folder without a stat, and no Path is
        # made for each entry. Listing 20,000 files with Path.iterdir and is_dir took 18 s of
        # a thread that held the main thread's GIL, so the window froze.
        with os.scandir(path) as listing:
            entries = [(e.name, e.is_dir()) for e in listing if e.name[:1] != "."]
    except OSError as e:
        out["note"] = f"Cannot read this folder: {e.strerror or e}"
        return out
    entries.sort(key=lambda e: (not e[1], e[0].lower()))
    n_dirs = sum(1 for _, is_dir in entries if is_dir)
    out["facts"].append(
        ("Contains", f"{len(entries) - n_dirs} files, {n_dirs} folders")
    )
    cs_job = read_cryosparc_job(path) if (path / "job.json").is_file() else None
    if cs_job is not None:
        out["facts"].append(
            ("CryoSPARC job", f"{cs_job['uid']}, {cs_job['type_name']}")
        )
        if cs_job["title"]:
            out["facts"].append(("Title", cs_job["title"]))
        out["facts"].append(("State", cs_job["raw_status"] or "unknown"))
        if cs_job["parents"]:
            out["facts"].append(("Inputs from", ", ".join(cs_job["parents"])))
        for label, key in (("Started", "created"), ("Finished", "completed")):
            if cs_job[key]:
                out["facts"].append((label, cs_job[key]))
        out["status"] = cs_job["status"]
    elif (path / "project.json").is_file():
        title = _read_json(path / "project.json").get("title") or ""
        n_jobs = sum(
            1
            for name, _ in entries
            if _CRYOSPARC_JOB.match(name) and (path / name / "job.json").is_file()
        )
        out["facts"].append(
            ("CryoSPARC project", f"{title}, {n_jobs} jobs".lstrip(", "))
        )
    elif _JOB_FOLDER.match(path.name):
        status = job_status(path)
        out["facts"].append(("RELION job", f"{path.parent.name}/{path.name}, {status}"))
        out["status"] = status
    elif (path / "default_pipeline.star").is_file() and not _is_job_folder_of(path):
        out["facts"].append(
            ("RELION project", f"{len(list_job_types(path))} job types")
        )
    try:
        out["facts"].append(
            (
                "Modified",
                datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            )
        )
    except OSError:
        pass
    shown = [name + ("/" if is_dir else "") for name, is_dir in entries[:14]]
    if len(entries) > 14:
        shown.append(f"… {len(entries) - 14} more")
    out["text"] = "\n".join(shown)
    return out


def _star_header(path: Path, limit: int = 400) -> str:
    """The data blocks of a STAR file and the first column names of each."""
    lines, block, cols = [], None, []

    def flush():
        if block is not None:
            more = f", … {len(cols) - 10} more" if len(cols) > 10 else ""
            lines.append(
                f"{block}: " + ", ".join(c.lstrip("_") for c in cols[:10]) + more
            )

    try:
        with open(path, errors="replace") as f:
            for i, line in enumerate(f):
                if i > limit and block is not None and cols:
                    break
                s = line.strip()
                if s.startswith("data_"):
                    flush()
                    block, cols = s, []
                elif s.startswith("_") and block is not None:
                    cols.append(s.split()[0])
        flush()
    except OSError:
        pass
    return "\n".join(lines)


def _array_facts(a) -> list[tuple[str, str]]:
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    return [
        ("Mean ± SD", f"{a.mean():.4g} ± {a.std():.4g}"),
        ("Range", f"{a.min():.4g} to {a.max():.4g}"),
    ]


def build_preview(path: str | Path) -> dict:
    """What to show of a file or folder: facts, a picture, a few lines of text.

    Reads only what it needs: the header of a map, one slice of it, the first frames
    of a stack, the first lines of a text file.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    dict
        ``kind`` (``folder``, ``image``, ``stack``, ``volume``, ``table``, ``text`` or
        ``file``), ``title``, ``facts`` (a list of ``(label, value)``), ``text``,
        ``image`` (a 2D ``uint8`` array or None) and, if something could not be read,
        ``note``.
    """
    path = Path(path)
    if path.is_dir():
        return _folder_preview(path)
    out = dict(kind="file", title=path.name, facts=[], text="", image=None)
    suffix = path.suffix.lower()
    try:
        st = path.stat()
        out["facts"].append(("Size", _human_size(st.st_size)))
        out["facts"].append(
            ("Modified", datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"))
        )
    except OSError as e:
        out["note"] = f"Cannot read this file: {e.strerror or e}"
        return out

    try:
        if suffix in _MRC_SUFFIXES:
            _preview_mrc(path, suffix, out)
        elif suffix in _PIL_SUFFIXES:
            _preview_pil(path, out)
        elif suffix == ".star":
            _preview_star(path, out)
        elif suffix == ".cs":
            _preview_cs(path, out)
        else:
            from helicon.lib.gui.file_browser import _is_text_file

            if _is_text_file(str(path)):
                out["kind"] = "text"
                with open(path, errors="replace") as f:
                    out["text"] = f.read(3000)
    except Exception as e:  # a damaged or unusual file must not break the browser
        out["note"] = f"Cannot preview this file: {e}"
    return out


def _preview_mrc(path: Path, suffix: str, out: dict) -> None:
    import mrcfile
    import numpy as np

    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        nx, ny, nz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
        apix = float(mrc.voxel_size.x)
        data = mrc.data
        if data is None:
            out["note"] = "No data in this file"
            return
        stack = suffix in (".mrcs", ".st", ".ali")
        if not stack and data.ndim == 3:
            # in a CryoSPARC job everything is .mrc: the job and the name say if it is
            # a stack of class averages or particles, not a map
            from helicon.lib import cryosparc_project

            stack = cryosparc_project.mrc_content(path) == "stack"
        if data.ndim == 2:
            out["kind"] = "image"
            out["facts"].insert(0, ("Dimensions", f"{nx} × {ny}"))
            image = thumbnail_array(data)
            sample = np.asarray(data[:: max(1, ny // 256), :: max(1, nx // 256)])
        elif stack:
            out["kind"] = "stack"
            out["facts"].insert(0, ("Images", f"{nz} of {nx} × {ny}"))
            # read no more than _STACK_PREVIEW_BYTES: the first frames of a stack of
            # large movie frames would otherwise be gigabytes to read for a thumbnail
            frame_bytes = max(int(nx) * int(ny) * data.dtype.itemsize, 1)
            n = max(
                1, min(nz, _STACK_PREVIEW_FRAMES, _STACK_PREVIEW_BYTES // frame_bytes)
            )
            frames = [np.asarray(data[i]) for i in range(n)]
            image = _montage(frames, columns=4 if n > 6 else min(n, 3) or 1)
            sample = frames[0]
        else:
            out["kind"] = "volume"
            out["facts"].insert(0, ("Dimensions", f"{nx} × {ny} × {nz}"))
            mid = np.asarray(data[nz // 2])
            image = thumbnail_array(mid)
            sample = mid
            out["facts"].append(("Shown", f"central slice, z = {nz // 2}"))
        if apix > 0:
            out["facts"].insert(1, ("Pixel size", f"{apix:.3f} Å"))
        out["facts"].extend(_array_facts(sample))
        out["image"] = image


def _preview_pil(path: Path, out: dict) -> None:
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as img:
        frames = getattr(img, "n_frames", 1)
        out["kind"] = "stack" if frames > 1 else "image"
        label = f"{img.width} × {img.height}"
        out["facts"].insert(
            0, ("Dimensions", label if frames == 1 else f"{frames} of {label}")
        )
        if img.width * img.height > _MAX_PIL_PIXELS:
            out["note"] = "Too large to show a picture quickly"
            return
        img.seek(0)
        arr = np.asarray(
            img.convert("L") if img.mode in ("P", "RGB", "RGBA", "LA", "1") else img
        )
    out["image"] = thumbnail_array(arr)
    out["facts"].extend(
        _array_facts(
            arr[:: max(1, arr.shape[0] // 256), :: max(1, arr.shape[1] // 256)]
        )
    )


def _preview_cs(path: Path, out: dict) -> None:
    """A CryoSPARC dataset (a structured NumPy array): its rows, and its fields by group."""
    import numpy as np

    data = np.load(path, mmap_mode="r", allow_pickle=False)
    out["kind"] = "table"
    out["facts"].insert(0, ("Rows", f"{len(data):,}"))
    names = data.dtype.names or ()
    out["facts"].insert(1, ("Fields", str(len(names))))
    groups: dict[str, int] = {}
    for name in names:
        group = name.split("/", 1)[0] if "/" in name else "(other)"
        groups[group] = groups.get(group, 0) + 1
    out["text"] = "\n".join(f"{g}: {n}" for g, n in groups.items())


def _preview_star(path: Path, out: dict) -> None:
    from helicon.lib.gui.file_browser import _get_file_info

    out["kind"] = "table"
    info, n_images, pixel = _get_file_info(str(path))
    if n_images:
        out["facts"].insert(0, ("Rows", n_images))
    if info:
        out["facts"].insert(1, ("Image size", info))
    if pixel:
        out["facts"].append(("Pixel size", pixel))
    out["text"] = _star_header(path)


# ── widgets ───────────────────────────────────────────────────────────────────


def themed_icon(
    name: str, color: str, selected: str | None = "#ffffff"
) -> QIcon | None:
    """A qtawesome icon in a colour, or None where qtawesome is not installed."""
    try:
        import qtawesome

        kwargs = {"color_selected": selected} if selected else {}
        return qtawesome.icon(name, color=color, **kwargs)
    except Exception:
        return None


class PathBar(QWidget):
    """The current folder as clickable segments (``~ › project › Class2D``); a click
    on the empty part, or Ctrl+L, turns it into a text box to type a path in.

    It has the text and ``returnPressed`` of a ``QLineEdit``, so it can stand for one.

    Signals
    -------
    pathClicked(str)
        A segment was clicked; the value is its full path.
    returnPressed()
        A path was typed and Return pressed (read it with :meth:`text`).
    """

    pathClicked = Signal(str)
    returnPressed = Signal()

    def __init__(self, path: str = "", parent=None):
        super().__init__(parent)
        self._path = str(path)
        self._home = None
        self._stack = QStackedWidget()
        self._crumbs = QFrame()
        self._crumbs.setObjectName("pathCrumbs")
        self._crumbs.setCursor(Qt.CursorShape.IBeamCursor)
        self._layout = QHBoxLayout(self._crumbs)
        self._layout.setContentsMargins(4, 0, 4, 0)
        self._layout.setSpacing(0)
        self._edit = QLineEdit(self._path)
        self._edit.returnPressed.connect(self._on_return)
        self._edit.installEventFilter(self)
        self._stack.addWidget(self._crumbs)
        self._stack.addWidget(self._edit)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._stack)
        self._build()

    # the text, as a line edit has it
    def text(self) -> str:
        return self._edit.text()

    def setText(self, path: str) -> None:
        self._path = str(path)
        self._edit.setText(self._path)
        self._stack.setCurrentWidget(self._crumbs)
        self._build()

    def edit(self) -> None:
        """Show the text box, with the path selected."""
        self._stack.setCurrentWidget(self._edit)
        self._edit.setFocus()
        self._edit.selectAll()

    def is_editing(self) -> bool:
        return self._stack.currentWidget() is self._edit

    def segments(self) -> list[tuple[str, str]]:
        """The ``(label, path)`` of the segments now shown, ``…`` for those folded away."""
        return list(self._shown)

    def _on_return(self) -> None:
        self.returnPressed.emit()
        self._stack.setCurrentWidget(self._crumbs)

    def eventFilter(self, obj, event):
        if obj is self._edit and event.type() == QEvent.Type.FocusOut:
            QTimer.singleShot(0, self._leave_edit)
        elif obj is self._edit and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self._edit.setText(self._path)
                self._leave_edit()
                return True
        return super().eventFilter(obj, event)

    def _leave_edit(self) -> None:
        if not self._edit.hasFocus():
            self._stack.setCurrentWidget(self._crumbs)

    def mousePressEvent(self, event):
        # the empty part of the bar: type a path
        if event.button() == Qt.MouseButton.LeftButton:
            self.edit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if (
            event.key() == Qt.Key.Key_L
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.edit()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._build()

    def _build(self) -> None:
        """Lay the segments out; the first ones fold into a ``…`` menu when they do not fit."""
        parts = breadcrumb_parts(self._path, self._home)
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        fm = self.fontMetrics()
        sep = fm.horizontalAdvance(" › ")
        room = max(self.width() - 16, 120)
        widths = [fm.horizontalAdvance(label) + 16 + sep for label, _ in parts]
        first = 0
        # keep the last segment always; fold from the front until the rest fit
        while (
            first < len(parts) - 1
            and sum(widths[first:]) + (fm.horizontalAdvance("…") + 24 if first else 0)
            > room
        ):
            first += 1
        self._shown = []
        if first:
            more = QToolButton()
            more.setText("…")
            more.setAutoRaise(True)
            more.setToolTip("Folders above")
            more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = QMenu(more)
            for label, full in reversed(parts[:first]):
                menu.addAction(label).triggered.connect(
                    lambda _=False, p=full: self.pathClicked.emit(p)
                )
            more.setMenu(menu)
            self._layout.addWidget(more)
            self._layout.addWidget(self._separator())
            self._shown.append(("…", ""))
        for i in range(first, len(parts)):
            label, full = parts[i]
            button = QToolButton()
            button.setText(label)
            button.setAutoRaise(True)
            button.setToolTip(full)
            if i == len(parts) - 1:
                font = button.font()
                font.setBold(True)
                button.setFont(font)
            button.clicked.connect(lambda _=False, p=full: self.pathClicked.emit(p))
            self._layout.addWidget(button)
            self._shown.append((label, full))
            if i < len(parts) - 1:
                self._layout.addWidget(self._separator())
        self._layout.addStretch(1)

    @staticmethod
    def _separator() -> QLabel:
        label = QLabel("›")
        label.setObjectName("pathSeparator")
        return label


def _is_project(project: Project) -> bool:
    """Whether the project's folder is still there and still marked as a project."""
    markers = (
        (".gui_projectdir", "default_pipeline.star")
        if project.kind == "relion"
        else ("project.json",)
    )
    try:
        return any((project.root / m).is_file() for m in markers)
    except OSError:
        return False


def _inside(path: str | Path, root: Path) -> bool:
    """Whether a path is the root or below it."""
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def _job_subpath(project: Project, type_item, name: str) -> str:
    """The folder of a job under the project root: ``Class2D/job010`` or ``J45``."""
    if project.kind == "relion":
        return f"{type_item.data(0, PlacesSidebar._ROLE_TYPE)}/{name}"
    return name


class PlacesSidebar(QTreeWidget):
    """Places on the left: home, the bookmarks, the recent folders and, inside a RELION
    project, its jobs by type with their state.

    Signals
    -------
    navigate(str)
        A place was clicked; the value is its folder.
    bookmark_removed(str)
        Remove was chosen from the menu of a bookmark.
    """

    navigate = Signal(str)
    bookmark_removed = Signal(str)

    _ROLE_PATH = Qt.ItemDataRole.UserRole
    _ROLE_KIND = Qt.ItemDataRole.UserRole + 1
    _ROLE_TYPE = Qt.ItemDataRole.UserRole + 2
    _STATUS_COLORS = {
        "success": "#4caf50",
        "failure": "#e5534b",
        "aborted": "#e5534b",
        "running": "#e0a93b",
        "queued": "#5aa0d8",
        "pending": "#8f8f8f",
        "unknown": "#8f8f8f",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(14)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setIconSize(QSize(14, 14))
        self.setExpandsOnDoubleClick(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.itemClicked.connect(self._clicked)
        self.itemExpanded.connect(self._expanded)
        self._colors = {
            "text": "#cccccc",
            "muted": "#8f8f8f",
            "folder": "#e0a93b",
            "file_icon": "#8fb0dc",
        }
        self._signature = None
        self._project: Project | None = None
        self._last_project: Project | None = None
        self._open_types: set[str] = set()
        self._current = ""

    def set_theme(self, colors: dict[str, str]) -> None:
        self._colors = colors
        self._signature = None  # icons are drawn again at the next refresh

    def invalidate(self) -> None:
        """Read the projects again at the next refresh."""
        invalidate_project_cache()
        self._signature = None

    def refresh(
        self, current: str, bookmarks=(), recent=(), home: str | None = None
    ) -> None:
        """Fill the sidebar for the folder ``current``, and mark it if it is a place."""
        home = home or str(Path.home())
        project = find_project(current)
        if project is not None:
            self._last_project = project
        elif self._last_project is not None and _is_project(self._last_project):
            # outside any project: the last one stays, so "Project folder" leads back
            project = self._last_project
        if project is not None and _inside(current, project.root):
            # the type of the job we are in is opened, so the job is seen
            key = self._type_of_job(project, current)
            if key is not None and key not in self._open_types:
                self._open_types.add(key)
                self._signature = None
        signature = (
            str(project.root) if project else None,
            project.kind if project else None,
            tuple(bookmarks),
            tuple(recent),
            home,
            self._colors.get("folder"),
        )
        if signature != self._signature:
            self._signature = signature
            self._project = project
            self._rebuild(project, bookmarks, recent, home)
        self._current = str(current)
        self._select_current()

    @staticmethod
    def _type_of_job(project: Project, current: str) -> str | None:
        """The type key of the job that holds a folder, or None if it is not in a job."""
        parts = Path(current).relative_to(project.root).parts
        if project.kind == "relion":
            if len(parts) >= 2 and _JOB_FOLDER.match(parts[1]):
                return parts[0]
        elif parts and _CRYOSPARC_JOB.match(parts[0]):
            job = read_cryosparc_job(project.root / parts[0])
            return job["type"] if job else None
        return None

    # building
    def _section(self, text: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([text.upper()])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        font = item.font(0)
        font.setPointSizeF(max(font.pointSizeF() - 1.5, 7))
        font.setBold(True)
        item.setFont(0, font)
        item.setForeground(0, self._brush(self._colors["muted"]))
        self.addTopLevelItem(item)
        item.setExpanded(True)
        return item

    @staticmethod
    def _brush(color: str):
        from PySide6.QtGui import QBrush, QColor

        return QBrush(QColor(color))

    def _place(
        self, parent, label, path, icon_name, color, tooltip=None
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [label])
        item.setData(0, self._ROLE_PATH, str(path))
        item.setToolTip(0, tooltip or str(path))
        icon = themed_icon(icon_name, color)
        if icon is not None:
            item.setIcon(0, icon)
        return item

    def _rebuild(self, project, bookmarks, recent, home) -> None:
        self.blockSignals(True)
        self.clear()
        c = self._colors
        places = self._section("Places")
        self._place(places, "Home", home, "fa5s.home", c["file_icon"]).setData(
            0, self._ROLE_KIND, "place"
        )
        self._place(places, "Computer", "/", "fa5s.hdd", c["file_icon"]).setData(
            0, self._ROLE_KIND, "place"
        )
        if project is not None:
            kind = "CryoSPARC" if project.kind == "cryosparc" else "RELION"
            node = self._section(f"{kind}  {project.name}")
            node.setToolTip(0, str(project.root))
            self._place(
                node,
                "Project folder",
                project.root,
                "fa5s.project-diagram",
                c["file_icon"],
            ).setData(0, self._ROLE_KIND, "project")
            for key, label, folder in project_types(project):
                type_item = QTreeWidgetItem(node, [label])
                type_item.setToolTip(0, label)
                type_item.setData(0, self._ROLE_PATH, folder)
                type_item.setData(0, self._ROLE_TYPE, key)
                type_item.setData(0, self._ROLE_KIND, "type")
                icon = themed_icon("fa5s.folder", c["folder"])
                if icon is not None:
                    type_item.setIcon(0, icon)
                # the jobs are listed when it is opened
                QTreeWidgetItem(type_item, ["…"]).setData(
                    0, self._ROLE_KIND, "placeholder"
                )
                if key in self._open_types:
                    self._fill_jobs(type_item)
                    type_item.setExpanded(True)
        marked = [path for path in bookmarks if Path(path).is_dir()]
        if marked:
            marks = self._section("Bookmarks")
            for path in marked:
                item = self._place(
                    marks, Path(path).name or path, path, "fa5s.star", "#e0a93b"
                )
                item.setData(0, self._ROLE_KIND, "bookmark")
        if recent:
            hist = self._section("Recent")
            for path in recent:
                self._place(
                    hist, Path(path).name or path, path, "fa5s.history", c["muted"]
                ).setData(0, self._ROLE_KIND, "recent")
        self.blockSignals(False)

    def _expanded(self, item: QTreeWidgetItem) -> None:
        if item.data(0, self._ROLE_KIND) != "type":
            return
        self._open_types.add(item.data(0, self._ROLE_TYPE))
        if self._fill_jobs(item):
            self._select_current()

    def _fill_jobs(self, type_item: QTreeWidgetItem) -> bool:
        """List the jobs of a type under it, once; whether it did."""
        if not (
            type_item.childCount() == 1
            and type_item.child(0).data(0, self._ROLE_KIND) == "placeholder"
        ):
            return False
        type_item.takeChild(0)
        project = self._project
        if project is None:
            return True
        for name, status, label, tooltip in project_jobs(
            project, type_item.data(0, self._ROLE_TYPE)
        ):
            job = QTreeWidgetItem(type_item, [label])
            job.setData(
                0,
                self._ROLE_PATH,
                str(project.root / _job_subpath(project, type_item, name)),
            )
            job.setData(0, self._ROLE_KIND, "job")
            job.setToolTip(0, tooltip)
            icon = themed_icon("fa5s.circle", self._STATUS_COLORS[status], None)
            if icon is not None:
                job.setIcon(0, icon)
        return True

    # which place to mark when a folder is more than one: the most specific
    _RANK = {"job": 0, "type": 1, "project": 2, "place": 3, "bookmark": 4, "recent": 5}

    def _select_current(self) -> None:
        self.blockSignals(True)
        self.clearSelection()
        best, best_rank = None, 99
        pending = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while pending:
            item = pending.pop()
            for i in range(item.childCount()):
                pending.append(item.child(i))
            if item.data(0, self._ROLE_PATH) == self._current:
                rank = self._RANK.get(item.data(0, self._ROLE_KIND), 9)
                if rank < best_rank:
                    best, best_rank = item, rank
        if best is not None:
            best.setSelected(True)
            self.scrollToItem(best)
        self.blockSignals(False)

    # events
    def _clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        path = item.data(0, self._ROLE_PATH)
        if item.data(0, self._ROLE_KIND) == "type":
            item.setExpanded(not item.isExpanded())
        if path:
            self.navigate.emit(str(path))

    def _menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None or item.data(0, self._ROLE_KIND) != "bookmark":
            return
        menu = QMenu(self)
        action = menu.addAction("Remove bookmark")
        if menu.exec(self.viewport().mapToGlobal(pos)) is action:
            self.bookmark_removed.emit(str(item.data(0, self._ROLE_PATH)))


class _PreviewWorker(QThread):
    """Reads a preview off the main thread; ``ready(serial, preview)`` at the end."""

    ready = Signal(int, dict)

    def __init__(self, serial: int, path: str, parent=None):
        super().__init__(parent)
        self._serial = serial
        self._path = path

    def run(self) -> None:
        self.ready.emit(self._serial, build_preview(self._path))


class PreviewPane(QWidget):
    """A picture, the facts and the first lines of the selected file or folder."""

    _EMPTY = "Select a file to see a preview"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self._title = QLabel("")
        self._title.setWordWrap(True)
        self._title.setObjectName("previewTitle")
        font = self._title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 1)
        self._title.setFont(font)
        self._picture = QLabel()
        self._picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._picture.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._facts = QLabel("")
        self._facts.setTextFormat(Qt.TextFormat.RichText)
        self._facts.setWordWrap(True)
        self._facts.setObjectName("previewFacts")
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFrameShape(QFrame.Shape.NoFrame)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setObjectName("previewNote")
        for widget in (self._title, self._picture, self._facts, self._note):
            layout.addWidget(widget)
        layout.addWidget(self._text, 1)
        # the room left when there is no text goes to the bottom, not between the parts
        layout.addStretch(0)
        self._muted = "#8f8f8f"
        self._serial = 0
        self._pixmap: QPixmap | None = None
        self._worker: _PreviewWorker | None = None
        self._pending: str | None = None
        self.clear()

    def set_theme(self, colors: dict[str, str]) -> None:
        self._muted = colors["muted"]
        self._note.setStyleSheet(f"color: {colors['muted']};")
        self._picture.setStyleSheet(f"background: {colors['alt']}; border-radius: 4px;")
        self._text.setStyleSheet(
            f"background: {colors['alt']}; color: {colors['text']}; border-radius: 4px; font-family: monospace;"
        )

    def clear(self) -> None:
        self._serial += 1
        self._title.setText("")
        self._picture.hide()
        self._facts.setText("")
        self._text.hide()
        self._note.setText(self._EMPTY)
        self._note.show()

    def show_path(self, path: str | None) -> None:
        """Preview a path; what is being read for an earlier one is dropped when it ends."""
        if not path:
            self.clear()
            return
        self._serial += 1
        self._title.setText(Path(path).name or str(path))
        self._note.setText("Reading…")
        self._note.show()
        # one read at a time: a click through many files starts only the last one
        self._pending = str(path)
        if self._worker is None or not self._worker.isRunning():
            self._start()

    def _start(self) -> None:
        path, self._pending = self._pending, None
        if path is None:
            return
        worker = _PreviewWorker(self._serial, path, self)
        worker.ready.connect(self._on_ready)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def _on_finished(self) -> None:
        if self._pending is not None:
            self._start()

    def _on_ready(self, serial: int, info: dict) -> None:
        if serial != self._serial:
            return  # another path was chosen meanwhile
        self.show_info(info)

    def show_info(self, info: dict) -> None:
        """Show a preview made by :func:`build_preview`."""
        self._title.setText(info.get("title", ""))
        rows = "".join(
            f"<tr><td style='color:{self._muted}; padding-right:10px'>{label}</td><td>{value}</td></tr>"
            for label, value in info.get("facts", [])
        )
        self._facts.setText(f"<table>{rows}</table>" if rows else "")
        image = info.get("image")
        if image is not None:
            h, w = image.shape
            qimage = QImage(image.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
            self._pixmap = QPixmap.fromImage(qimage)
            self._fit_picture()
            self._picture.show()
        else:
            self._pixmap = None
            self._picture.hide()
        text = info.get("text", "")
        self._text.setPlainText(text)
        self._text.setVisible(bool(text))
        note = info.get("note", "")
        self._note.setText(note)
        self._note.setVisible(bool(note))

    def _fit_picture(self) -> None:
        if self._pixmap is None:
            return
        width = max(self.width() - 24, 100)
        shown = self._pixmap.scaledToWidth(
            min(width, 360), Qt.TransformationMode.SmoothTransformation
        )
        self._picture.setPixmap(shown)
        self._picture.setFixedHeight(shown.height() + 8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_picture()

    def wait(self, ms: int = 5000) -> None:
        """Wait for a read in progress (for tests and when closing)."""
        if self._worker is not None:
            self._worker.wait(ms)


class FlowLayout(QLayout):
    """A layout that puts its widgets in a row and wraps to the next row when the
    row is full (the buttons of the action bar, which are many for some files).

    It has the ``addWidget``, ``insertWidget`` and ``removeWidget`` of a box layout.
    Hidden widgets take no room.

    Parameters
    ----------
    parent : QWidget, optional
    margin : int, optional
        Space around the rows, pixels. Defaults to 0.
    h_spacing, v_spacing : int, optional
        Space between widgets in a row and between rows. Default 6.
    """

    def __init__(
        self, parent=None, margin: int = 0, h_spacing: int = 6, v_spacing: int = 6
    ):
        super().__init__(parent)
        self._items: list = []
        self._h = h_spacing
        self._v = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    # the layout interface
    def addItem(self, item) -> None:
        self._items.append(item)
        self.invalidate()

    def insertWidget(self, index: int, widget) -> None:
        """Insert a widget at a position (as a box layout does)."""
        self.addChildWidget(widget)
        from PySide6.QtWidgets import QWidgetItem

        self._items.insert(index, QWidgetItem(widget))
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            item = self._items.pop(index)
            self.invalidate()
            return item
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect, apply: bool) -> int:
        """Place the widgets in rows, each centred vertically in its row; the height used."""
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        rows: list[list] = [[]]
        x = area.x()
        for item in self._items:
            if item.isEmpty():
                continue
            width = item.sizeHint().width()
            if x + width > area.right() + 1 and rows[-1]:
                rows.append([])
                x = area.x()
            rows[-1].append(item)
            x += width + self._h
        y = area.y()
        for number, row in enumerate(rows):
            height = max((item.sizeHint().height() for item in row), default=0)
            if apply:
                x = area.x()
                for item in row:
                    hint = item.sizeHint()
                    top = y + (height - hint.height()) // 2
                    item.setGeometry(QRect(QPoint(x, top), hint))
                    x += hint.width() + self._h
            y += height + (self._v if number < len(rows) - 1 else 0)
        return y - rect.y() + m.bottom()
