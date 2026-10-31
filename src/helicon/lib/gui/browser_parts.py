"""Parts of the file browser: a breadcrumb path bar, a sidebar of places and a preview pane.

Kept apart from ``file_browser`` so that module stays about the list of files. The
pure functions (the breadcrumb segments, the RELION project and job helpers, the
thumbnail and the preview of a file) do not need Qt and are tested without it; the
widgets use PySide6, as the browser does. Heavy packages (numpy, mrcfile, PIL) are
imported where they are used, so opening the browser stays quick.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap, QTextOption
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
    "contrast_limits",
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
    ``title``, ``status`` (as :func:`job_status` gives it), ``created``, ``completed``,
    ``parents`` (the jobs feeding it) and ``children`` (the jobs it feeds, as the file lists
    them); None if the folder holds no ``job.json``."""
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
        children=[str(c) for c in record.get("children") or []],
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
    """The jobs of a type: ``[(folder name, status, label, tooltip, alias), ...]``, newest first.

    The label of a CryoSPARC job is its number and title (``J45  my classes``); that of a RELION job
    that has an alias is the job and the alias (``job010(my_alias)``).
    """
    if project.kind == "relion":
        from .job_alias import aliases

        by_job = aliases(project.root / key)
        out = []
        for name, status in list_jobs(project.root, key):
            alias = by_job.get(name, [""])[0]
            label = f"{name}({alias})" if alias else name
            tip = f"{project.root / key / name}  ({status})"
            out.append(
                (
                    name,
                    status,
                    label,
                    tip + (f"\nalias {alias}" if alias else ""),
                    alias,
                )
            )
        return out
    out = []
    for number, name, job in sorted(
        _cryosparc_jobs(project.root).get(key, []), reverse=True
    ):
        label = f"{name}  {job['title']}" if job["title"] else name
        raw = f" ({job['raw_status']})" if job["raw_status"] else ""
        out.append((name, job["status"], label, f"{project.root / name}{raw}", ""))
    return out


def _cryosparc_links(root: Path, job: dict) -> dict[str, list[tuple[str, str | None]]]:
    """The jobs before and after a CryoSPARC job, as ``(label, folder or None)``.

    The children are those the file lists and those that name this job as a parent, so that a
    file that lacks the list still gives them. A job whose folder is gone has no folder.
    """
    by_name = {
        name: j for jobs in _cryosparc_jobs(root).values() for _n, name, j in jobs
    }
    children = list(job["children"])
    for name, other in by_name.items():
        if job["uid"] in other["parents"] and name not in children:
            children.append(name)

    def entry(uid: str) -> tuple[str, str | None]:
        other = by_name.get(uid)
        if other is None:
            return uid, None
        label = f"{uid}  {other['type_name']}"
        if other["title"]:
            label += f"  {other['title']}"
        return label, str(root / uid)

    def number(uid: str) -> int:
        m = _CRYOSPARC_JOB.match(uid)
        return int(m.group(1)) if m else 0

    return dict(
        parents=[entry(u) for u in sorted(job["parents"], key=number)],
        children=[entry(u) for u in sorted(children, key=number)],
    )


# the pipeline of a RELION project by file, kept until the file changes
_PIPELINES: dict[str, tuple[tuple[int, int], dict]] = {}


def _read_pipeline(root: Path) -> dict:
    """The jobs of ``default_pipeline.star`` and how they are connected.

    Returns ``{"processes": {name: alias}, "inputs": {process: [node, ...]}, "outputs":
    {process: [node, ...]}}``; names are as RELION writes them (``Class2D/job010/``).
    """
    path = root / "default_pipeline.star"
    try:
        info = path.stat()
        text = path.read_text(errors="replace")
    except OSError:
        return dict(processes={}, inputs={}, outputs={}, details={})
    stamp = (info.st_mtime_ns, info.st_size)
    cached = _PIPELINES.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    rows: dict[str, list[list[str]]] = {}
    headers: dict[str, list[str]] = {}
    block, in_loop, seen_rows = "", False, False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data_"):
            block, in_loop = line[5:], False
        elif line == "loop_":
            headers[block], in_loop, seen_rows = [], True, False
        elif in_loop and line.startswith("_"):
            headers[block].append(line.split()[0])
        elif in_loop and line and not line.startswith("#"):
            rows.setdefault(block, []).append(line.split())
            seen_rows = True
        elif not line and seen_rows:
            in_loop = False  # an empty line ends the rows of a loop

    def table(block_name: str, first: str, second: str) -> list[tuple[str, str]]:
        """Two columns of a loop, found by their names in its header."""
        header = headers.get(block_name, [])
        if first not in header or second not in header:
            return []
        i, j = header.index(first), header.index(second)
        return [(r[i], r[j]) for r in rows.get(block_name, []) if len(r) > max(i, j)]

    processes = dict(
        table(
            "pipeline_processes", "_rlnPipeLineProcessName", "_rlnPipeLineProcessAlias"
        )
    )
    # (type label, state label) of each process, for the graph of the project
    details = {
        name: (kind, state)
        for name, kind, state in _table3(
            headers,
            rows,
            "pipeline_processes",
            (
                "_rlnPipeLineProcessName",
                "_rlnPipeLineProcessTypeLabel",
                "_rlnPipeLineProcessStatusLabel",
            ),
        )
    }
    inputs: dict[str, list[str]] = {}
    for node, process in table(
        "pipeline_input_edges", "_rlnPipeLineEdgeFromNode", "_rlnPipeLineEdgeProcess"
    ):
        inputs.setdefault(process, []).append(node)
    outputs: dict[str, list[str]] = {}
    for process, node in table(
        "pipeline_output_edges", "_rlnPipeLineEdgeProcess", "_rlnPipeLineEdgeToNode"
    ):
        outputs.setdefault(process, []).append(node)
    result = dict(processes=processes, inputs=inputs, outputs=outputs, details=details)
    _PIPELINES[str(path)] = (stamp, result)
    return result


def _table3(headers, rows, block, wanted):
    """Three named columns of a loop of the pipeline."""
    header = headers.get(block, [])
    if any(w not in header for w in wanted):
        return []
    idx = [header.index(w) for w in wanted]
    return [tuple(r[i] for i in idx) for r in rows.get(block, []) if len(r) > max(idx)]


def _relion_links(root: Path, job: Path) -> dict[str, list[tuple[str, str | None]]]:
    """The jobs before and after a RELION job, as ``(label, folder or None)``.

    A parent is a job that wrote a file this job reads; a child is a job that reads a file this
    job wrote. A file that no job wrote (an import, a file of your own) has no parent.
    """
    pipeline = _read_pipeline(root)
    me = f"{job.parent.name}/{job.name}/"
    writer = {
        node: process
        for process, nodes in pipeline["outputs"].items()
        for node in nodes
    }
    parents, children = [], []
    for node in pipeline["inputs"].get(me, []):
        other = writer.get(node)
        if other and other != me and other not in parents:
            parents.append(other)
    mine = set(pipeline["outputs"].get(me, []))
    for process, nodes in pipeline["inputs"].items():
        if process != me and mine.intersection(nodes) and process not in children:
            children.append(process)

    def entry(process: str) -> tuple[str, str | None]:
        name = process.rstrip("/")
        alias = pipeline["processes"].get(process, "None").rstrip("/")
        label = name
        if alias and alias != "None":
            label += f"  {alias.rsplit('/', 1)[-1]}"
        folder = root / name
        return label, str(folder) if folder.is_dir() else None

    def number(process: str) -> int:
        m = re.search(r"job(\d+)", process)
        return int(m.group(1)) if m else 0

    return dict(
        parents=[entry(p) for p in sorted(parents, key=number)],
        children=[entry(p) for p in sorted(children, key=number)],
    )


# a picture of more bytes than this is cut down by taking every n-th pixel, not by binning: binning
# reads all of it
_BIN_BYTES = 512 * 2**20
# a value shared exactly by this share of the pixels is flat background (the zeros outside a mask)
_BACKGROUND_SHARE = 0.1


def _binned(data, step: int):
    """The mean of each ``step`` × ``step`` block of a 2D array, as float32.

    Read a few rows at a time, so that a memory-mapped micrograph is not read into memory whole;
    the rows and columns left over at the end are dropped.
    """
    import numpy as np

    ny, nx = data.shape
    h, w = ny // step, nx // step
    out = np.empty((h, w), dtype=np.float32)
    chunk = max(1, 256 // step)  # rows of the binned picture read at a time
    for y in range(0, h, chunk):
        stop = min(y + chunk, h)
        block = np.asarray(data[y * step : stop * step, : w * step], np.float32)
        out[y:stop] = block.reshape(stop - y, step, w, step).mean(axis=(1, 3))
    return out


def contrast_limits(data) -> tuple[float, float] | None:
    """The black and white points of a picture: the 1st and 99th percentiles of its pixels, leaving
    out the flat background.

    A masked map or a class average is mostly a single value outside its mask (0); with it the
    percentiles fall inside the density, which is then clipped white and black (a sixth of a masked
    filament, say). The value shared by a tenth of the pixels or more is left out when the picture
    has more than 256 levels: in an 8-bit picture many pixels share each level.

    Parameters
    ----------
    data : array_like

    Returns
    -------
    tuple[float, float] or None
        ``(black, white)`` with white > black, or None if the picture is flat or has no finite pixel.
    """
    import numpy as np

    a = np.asarray(data)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    values, counts = np.unique(finite, return_counts=True)
    signal = finite
    if len(values) > 256:
        mode = int(counts.argmax())
        if counts[mode] >= _BACKGROUND_SHARE * finite.size:
            signal = finite[finite != values[mode]]
    lo, hi = (float(v) for v in np.percentile(signal, (1, 99)))
    if hi <= lo:
        lo, hi = float(values[0]), float(values[-1])
    if hi <= lo:
        return None
    return lo, hi


def thumbnail_array(data, max_size: int = 256):
    """A grey-level picture of a 2D array: binned by whole steps to at most ``max_size`` pixels a
    side and stretched between the black and white points of :func:`contrast_limits`.

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
        # the mean of each block, not one pixel of it: the noise of a class average or a
        # micrograph falls with the binning, one pixel of each block keeps all of it
        if min(a.shape) >= step and a.nbytes <= _BIN_BYTES:
            a = _binned(a, step)
        else:
            a = a[::step, ::step]
    a = a.astype(np.float32, copy=False)
    limits = contrast_limits(a)
    if limits is None:
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = limits
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
    links = None
    cs_job = read_cryosparc_job(path) if (path / "job.json").is_file() else None
    if cs_job is not None:
        out["facts"].append(
            ("CryoSPARC job", f"{cs_job['uid']}, {cs_job['type_name']}")
        )
        if cs_job["title"]:
            out["facts"].append(("Title", cs_job["title"]))
        out["facts"].append(("State", cs_job["raw_status"] or "unknown"))
        for label, key in (("Started", "created"), ("Finished", "completed")):
            if cs_job[key]:
                out["facts"].append((label, cs_job[key]))
        out["status"] = cs_job["status"]
        links = _cryosparc_links(path.parent, cs_job)
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
        out["graph"] = dict(kind="cryosparc", root=str(path))
    elif _JOB_FOLDER.match(path.name):
        status = job_status(path)
        out["facts"].append(("RELION job", f"{path.parent.name}/{path.name}, {status}"))
        out["status"] = status
        root = find_project_root(path)
        if root is not None:
            links = _relion_links(root, path)
    elif (path / "default_pipeline.star").is_file() and not _is_job_folder_of(path):
        out["facts"].append(
            ("RELION project", f"{len(list_job_types(path))} job types")
        )
        out["graph"] = dict(kind="relion", root=str(path))
    try:
        out["facts"].append(
            (
                "Modified",
                datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            )
        )
    except OSError:
        pass
    if links is not None:
        # the jobs before and after this one, to go to with a click
        out["links"] = [
            (title, links[key])
            for title, key in (("Inputs from", "parents"), ("Used by", "children"))
            if links[key]
        ]
    if "graph" in out:
        return out  # the jobs as a graph to click, in place of the list of folders
    if cs_job is not None or _JOB_FOLDER.match(path.name):
        # what the job made, as pictures, in place of the list of its files
        from .job_outcome import build_outcome

        outcome = build_outcome(
            path,
            "cryosparc" if cs_job is not None else "relion",
            cs_job["type"] if cs_job is not None else None,
        )
        if outcome is not None:
            out["outcome"] = outcome
            out["facts"].extend(outcome["facts"])
        if cs_job is None:
            for name in _MICROGRAPH_STARS:
                if (path / name).is_file():
                    _add_micrographs(path / name, out)
                    if "micrographs" in out:
                        break
        # the files of a job are in the list of the window: no list of them here either
        out["text"] = ""
        return out
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
            from helicon.lib.gui.browser_files import _is_text_file

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


def _add_micrographs(path: Path, out: dict) -> None:
    """Mark a star file whose positions can be shown on the micrographs: a table of particles,
    the listing of a picking job, or the coordinate file of one micrograph of a picking job (the
    window then opens on the listing, at that micrograph). A button opens them."""
    from .job_outcome import (
        has_particle_coordinates,
        is_pick_listing,
        listing_of_coordinates,
    )

    if has_particle_coordinates(path) or is_pick_listing(path):
        out["micrographs"] = str(path)
        return
    found = listing_of_coordinates(path)
    if found is not None:
        out["micrographs"], out["micrograph"] = str(found[0]), found[1]


# the star files of a job folder whose positions can be shown on the micrographs, the first found
_MICROGRAPH_STARS = ("manualpick.star", "autopick.star", "particles.star")


def _preview_star(path: Path, out: dict) -> None:
    from helicon.lib.gui.browser_files import _get_file_info

    out["kind"] = "table"
    info, n_images, pixel = _get_file_info(str(path))
    if n_images:
        out["facts"].insert(0, ("Rows", n_images))
    if info:
        out["facts"].insert(1, ("Image size", info))
    if pixel:
        out["facts"].append(("Pixel size", pixel))
    out["text"] = _star_header(path)
    _add_micrographs(path, out)
    from .job_outcome import example_micrograph, is_micrograph_table

    if is_micrograph_table(path):
        # reading the micrograph takes a while: the table is shown first
        out["outcome"] = dict(
            title="Micrographs",
            panels=[],
            facts=[],
            refine=lambda: example_micrograph(path),
        )
    elif path.name.endswith("_data.star"):
        from .job_outcome import busiest_micrograph, is_particle_table

        if is_particle_table(path):
            # the micrograph with the most filaments comes later: it needs all of the table
            out["outcome"] = dict(
                title="Filaments",
                panels=[],
                facts=[],
                refine=lambda: busiest_micrograph(path),
            )


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


# the threads that read a project for the sidebar: a sidebar that is closed meanwhile must not take
# them down
_SCANNING: set = set()


class _ScanWorker(QThread):
    """Reads the job types of a project and the jobs of some of them off the main thread (a project
    on a network file system has thousands of folders to look at); ``done(serial, (types, jobs))``
    at the end, ``types`` None when they were not asked for."""

    done = Signal(int, object)

    def __init__(
        self,
        serial: int,
        project: "Project | None",
        keys,
        with_types: bool = True,
        bookmarks=(),
    ):
        super().__init__()
        self._serial = serial
        self._project = project
        self._keys = list(keys)
        self._with_types = with_types
        self._bookmarks = list(bookmarks)

    def run(self) -> None:
        types, jobs = None, {}
        try:
            if self._project is not None:
                types = project_types(self._project) if self._with_types else None
                jobs = {key: project_jobs(self._project, key) for key in self._keys}
        except Exception:
            types, jobs = None, {}
        # a bookmark of a folder that is gone is not shown: looking is slow on a network file system
        missing = [path for path in self._bookmarks if not Path(path).is_dir()]
        self.done.emit(self._serial, (types, jobs, missing))


class PlacesSidebar(QTreeWidget):
    """Places on the left: home, the bookmarks, the recent folders and, inside a RELION
    project, its jobs by type with their state.

    Signals
    -------
    navigate(str)
        A place was clicked; the value is its folder.
    bookmark_removed(str)
        Remove was chosen from the menu of a bookmark.
    bookmark_renamed(str, str)
        Rename was chosen from the menu of a bookmark: its folder and the new name ("" for the
        name of the folder).
    alias_requested(str, str)
        An alias was set, edited or removed from the menu of a RELION job: its folder and the new
        alias ("" to remove it).
    """

    navigate = Signal(str)
    bookmark_removed = Signal(str)
    bookmark_renamed = Signal(str, str)
    alias_requested = Signal(str, str)

    _ROLE_PATH = Qt.ItemDataRole.UserRole
    _ROLE_KIND = Qt.ItemDataRole.UserRole + 1
    _ROLE_TYPE = Qt.ItemDataRole.UserRole + 2
    _ROLE_ALIAS = Qt.ItemDataRole.UserRole + 3
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
        self.itemCollapsed.connect(self._collapsed)
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
        # what was read of each project, to show at once when it is looked at again, and the
        # numbers of the readings in progress (an answer for an earlier one is dropped)
        self._seen: dict[tuple[str, str], tuple[list, dict]] = {}
        self._scan_serial = 0
        self._project_node: QTreeWidgetItem | None = None
        self._bookmark_names: dict[str, str] = {}

    def set_theme(self, colors: dict[str, str]) -> None:
        self._colors = colors
        self._signature = None  # icons are drawn again at the next refresh

    def invalidate(self) -> None:
        """Read the projects again at the next refresh."""
        invalidate_project_cache()
        self._seen.clear()
        self._signature = None

    def refresh(
        self,
        current: str,
        bookmarks=(),
        recent=(),
        home: str | None = None,
        bookmark_names: dict | None = None,
    ) -> None:
        """Fill the sidebar for the folder ``current``, and mark it if it is a place."""
        home = home or str(Path.home())
        project = find_project(current)
        if project is not None:
            self._last_project = project
        elif self._last_project is not None and _is_project(self._last_project):
            # outside any project: the last one stays, so "Project folder" leads back
            project = self._last_project
        newly = None
        if (
            project is not None
            and str(current) != self._current
            and _inside(current, project.root)
        ):
            # going to a job opens its type (once, not at each refresh), so the job is seen; the
            # types that were opened and closed by hand are left as they are
            key = self._type_of_job(project, current)
            if key is not None and key not in self._open_types:
                self._open_types.add(key)
                newly = key
        signature = (
            str(project.root) if project else None,
            project.kind if project else None,
            tuple(bookmarks),
            tuple(recent),
            home,
            self._colors.get("folder"),
            tuple(sorted((bookmark_names or {}).items())),
        )
        if signature != self._signature:
            self._signature = signature
            self._project = project
            self._bookmark_names = dict(bookmark_names or {})
            self._rebuild(project, bookmarks, recent, home)
        elif newly is not None:
            self._expand_type(newly)  # in place: the rest of the tree stays as it is
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
        self._scan_serial += (
            1  # what is being read for the sidebar of before is of no use
        )
        self._project_node = None
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
            self._project_node = node
            seen = self._seen.get((str(project.root), project.kind))
            if seen is not None:
                self._populate(
                    node, project, *seen
                )  # until the reading that follows is there
            else:
                QTreeWidgetItem(node, ["Reading the jobs…"]).setData(
                    0, self._ROLE_KIND, "placeholder"
                )
        marked = list(
            bookmarks
        )  # the ones that are gone are taken away when they are looked at
        if marked:
            marks = self._section("Bookmarks")
            for path in marked:
                label = self._bookmark_names.get(path) or Path(path).name or path
                item = self._place(marks, label, path, "fa5s.star", "#e0a93b")
                item.setToolTip(
                    0,
                    path if label == (Path(path).name or path) else f"{label}\n{path}",
                )
                item.setData(0, self._ROLE_KIND, "bookmark")
        if recent:
            hist = self._section("Recent")
            for path in recent:
                self._place(
                    hist, Path(path).name or path, path, "fa5s.history", c["muted"]
                ).setData(0, self._ROLE_KIND, "recent")
        self.blockSignals(False)
        self._scan(project, marked)

    def _populate(self, node, project: "Project", types: list, jobs: dict) -> None:
        """Put the job types of a project, and the jobs of the open ones, under its node."""
        c = self._colors
        while node.childCount() > 1:  # the first is "Project folder"
            node.takeChild(1)
        for key, label, folder in types:
            type_item = QTreeWidgetItem(node, [label])
            type_item.setToolTip(0, label)
            type_item.setData(0, self._ROLE_PATH, folder)
            type_item.setData(0, self._ROLE_TYPE, key)
            type_item.setData(0, self._ROLE_KIND, "type")
            icon = themed_icon("fa5s.folder", c["folder"])
            if icon is not None:
                type_item.setIcon(0, icon)
            # the jobs are listed when it is opened
            QTreeWidgetItem(type_item, ["…"]).setData(0, self._ROLE_KIND, "placeholder")
            if key in self._open_types:
                if key in jobs:
                    self._add_jobs(type_item, project, jobs[key])
                type_item.setExpanded(True)

    def _scan(self, project: "Project | None", bookmarks=()) -> None:
        """Read the types of the project and the jobs of the open ones, and look for the folders of
        the bookmarks, in a thread."""
        if project is None and not bookmarks:
            return
        worker = _ScanWorker(
            self._scan_serial,
            project,
            sorted(self._open_types),
            bookmarks=bookmarks,
        )
        _SCANNING.add(worker)
        worker.done.connect(self._on_scanned)
        worker.finished.connect(lambda w=worker: _SCANNING.discard(w))
        worker.start(QThread.Priority.LowPriority)

    def _on_scanned(self, serial: int, result) -> None:
        types, jobs, missing = result
        if serial != self._scan_serial:
            return  # the sidebar was built again meanwhile
        project, node = self._project, self._project_node
        self.blockSignals(True)
        if project is not None and node is not None:
            self._seen[(str(project.root), project.kind)] = (types or [], jobs)
            self._populate(node, project, types or [], jobs)
        for i in range(self.topLevelItemCount()):
            section = self.topLevelItem(i)
            if section.text(0) != "BOOKMARKS":
                continue
            for j in reversed(range(section.childCount())):
                if section.child(j).data(0, self._ROLE_PATH) in missing:
                    section.takeChild(j)
            if not section.childCount():
                self.takeTopLevelItem(i)
            break
        self.blockSignals(False)
        if project is not None and node is not None:
            for i in range(node.childCount()):
                item = node.child(i)
                if item.data(0, self._ROLE_KIND) == "type" and item.isExpanded():
                    self._load_jobs(item)
        self._select_current()

    def _expand_type(self, key: str) -> None:
        """Open the type ``key`` in the tree, if it is there (its jobs are read when it opens)."""
        node = self._project_node
        for i in range(node.childCount() if node is not None else 0):
            item = node.child(i)
            if item.data(0, self._ROLE_TYPE) == key:
                item.setExpanded(True)
                return

    def _collapsed(self, item: QTreeWidgetItem) -> None:
        if item.data(0, self._ROLE_KIND) == "type":
            self._open_types.discard(item.data(0, self._ROLE_TYPE))

    def _expanded(self, item: QTreeWidgetItem) -> None:
        if item.data(0, self._ROLE_KIND) != "type":
            return
        self._open_types.add(item.data(0, self._ROLE_TYPE))
        self._load_jobs(item)

    def _load_jobs(self, item: QTreeWidgetItem) -> None:
        """List the jobs of an open type: at once if they were read before, else in a thread."""
        key = item.data(0, self._ROLE_TYPE)
        project = self._project
        if project is None or not self._has_placeholder(item):
            return
        seen = self._seen.get((str(project.root), project.kind))
        if seen is not None and key in seen[1]:
            self._add_jobs(item, project, seen[1][key])
            self._select_current()
            return
        worker = _ScanWorker(self._scan_serial, project, [key], with_types=False)
        _SCANNING.add(worker)
        worker.done.connect(self._on_jobs)
        worker.finished.connect(lambda w=worker: _SCANNING.discard(w))
        worker.start(QThread.Priority.LowPriority)

    def _on_jobs(self, serial: int, result) -> None:
        _types, jobs, _missing = result
        project, node = self._project, self._project_node
        if serial != self._scan_serial or project is None or node is None:
            return  # the sidebar was built again meanwhile
        seen = self._seen.setdefault((str(project.root), project.kind), ([], {}))
        seen[1].update(jobs)
        for i in range(node.childCount()):
            item = node.child(i)
            key = item.data(0, self._ROLE_TYPE)
            if key in jobs and self._has_placeholder(item):
                self._add_jobs(item, project, jobs[key])
        self._select_current()

    def _has_placeholder(self, type_item: QTreeWidgetItem) -> bool:
        return (
            type_item.childCount() == 1
            and type_item.child(0).data(0, self._ROLE_KIND) == "placeholder"
        )

    def _add_jobs(self, type_item: QTreeWidgetItem, project: "Project", jobs) -> None:
        """List the jobs ``[(folder name, status, label, tooltip), ...]`` under a type."""
        while type_item.childCount():
            type_item.takeChild(0)
        for name, status, label, tooltip, alias in jobs:
            job = QTreeWidgetItem(type_item, [label])
            job.setData(0, self._ROLE_ALIAS, alias)
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

    def wait(self, ms: int = 10000) -> None:
        """Wait for what is being read for the sidebar, and show it (for tests and when closing)."""
        import time

        from PySide6.QtWidgets import QApplication

        end = time.monotonic() + ms / 1000
        while time.monotonic() < end:
            QApplication.processEvents()  # an answer that has come may start another reading
            if not any(w.isRunning() for w in list(_SCANNING)):
                QApplication.processEvents()
                if not any(w.isRunning() for w in list(_SCANNING)):
                    break
            time.sleep(0.005)

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
            # scrolling to an item opens the folders above it: not when one was closed by hand
            parent, shown = best.parent(), True
            while parent is not None:
                shown = shown and parent.isExpanded()
                parent = parent.parent()
            if shown:
                self.scrollToItem(best)
        self.blockSignals(False)

    # events
    def _clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        path = item.data(0, self._ROLE_PATH)
        if item.data(0, self._ROLE_KIND) == "type":
            item.setExpanded(not item.isExpanded())
        if path:
            self.navigate.emit(str(path))

    def _run_menu(self, menu: QMenu, pos):
        """Show a menu at a place of the sidebar; the action that was chosen, or None."""
        return menu.exec(self.viewport().mapToGlobal(pos))

    def _menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        kind = item.data(0, self._ROLE_KIND)
        path = str(item.data(0, self._ROLE_PATH))
        menu = QMenu(self)
        if kind == "bookmark":
            rename = menu.addAction("Rename bookmark…")
            remove = menu.addAction("Remove bookmark")
            chosen = self._run_menu(menu, pos)
            if chosen is remove:
                self.bookmark_removed.emit(path)
            elif chosen is rename:
                self._rename_bookmark(item, path)
        elif (
            kind == "job"
            and self._project is not None
            and self._project.kind == "relion"
        ):
            alias = item.data(0, self._ROLE_ALIAS) or ""
            edit = menu.addAction("Edit alias…" if alias else "Add alias…")
            remove = menu.addAction("Remove alias") if alias else None
            chosen = self._run_menu(menu, pos)
            if chosen is edit:
                from PySide6.QtWidgets import QInputDialog

                text, ok = QInputDialog.getText(
                    self, "Alias of the job", f"Alias of {Path(path).name}:", text=alias
                )
                if ok and text.strip() != alias:
                    self.alias_requested.emit(path, text.strip())
            elif remove is not None and chosen is remove:
                self.alias_requested.emit(path, "")

    def _rename_bookmark(self, item: QTreeWidgetItem, path: str) -> None:
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(
            self,
            "Rename bookmark",
            f"Name of the bookmark of {path} (empty for the name of the folder):",
            text=item.text(0),
        )
        if ok:
            self.bookmark_renamed.emit(path, text.strip())


def _qimage_of(array) -> QImage:
    """A picture (``(h, w)`` grey or ``(h, w, 3)`` colour, uint8) as an image of its own pixels."""
    import numpy as np

    array = np.ascontiguousarray(array, dtype=np.uint8)
    h, w = array.shape[:2]
    if array.ndim == 2:
        image = QImage(array.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    else:
        image = QImage(array.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return image.copy()  # the bytes above are gone when this returns


def _pixmap_of(array, width: int) -> QPixmap:
    """A picture (``(h, w)`` grey or ``(h, w, 3)`` colour, uint8) as a pixmap ``width`` wide."""
    return QPixmap.fromImage(_qimage_of(array)).scaledToWidth(
        width, Qt.TransformationMode.SmoothTransformation
    )


class _PreviewWorker(QThread):
    """Reads a preview off the main thread; ``ready(serial, preview)`` at the end."""

    ready = Signal(int, dict)

    def __init__(self, serial: int, path: str, parent=None):
        super().__init__(parent)
        self._serial = serial
        self._path = path

    def run(self) -> None:
        self.ready.emit(self._serial, build_preview(self._path))


# the threads that count in the background: a pane that is closed meanwhile must not take them down
_REFINING: set = set()


class _RefineWorker(QThread):
    """Makes the exact outcome of a job whose first outcome had an estimate in it (slow: it reads
    all of a large table); ``done(serial, outcome)`` at the end, ``{}`` if it failed."""

    done = Signal(int, dict)

    def __init__(self, serial: int, refine):
        super().__init__()
        self._serial = serial
        self._refine = refine

    def run(self) -> None:
        try:
            outcome = self._refine() or {}
        except Exception:
            outcome = {}
        self.done.emit(self._serial, outcome)


class _MaskedMapWorker(QThread):
    """Writes the map of a mask job multiplied by the mask (:func:`job_outcome.masked_map_file`);
    ``done(path)`` at the end, ``""`` if it failed."""

    done = Signal(str)

    def __init__(self, map_path: str, mask_path: str):
        super().__init__()
        self._map_path = map_path
        self._mask_path = mask_path

    def run(self) -> None:
        from .job_outcome import masked_map_file

        try:
            path = str(masked_map_file(self._map_path, self._mask_path))
        except Exception:
            logging.getLogger(__name__).warning(
                "cannot write the masked map of %s", self._mask_path, exc_info=True
            )
            path = ""
        self.done.emit(path)


class PreviewPane(QWidget):
    """A picture, the facts, the links (the jobs before and after a job) and the first lines of
    the selected file or folder.

    Signals
    -------
    path_requested(str)
        A link to another job was clicked; the value is the folder of the job.
    file_requested(str)
        A file to open in its viewer, as with a double click in the list of files (the masked map
        of a mask job); the value is the path of the file.
    """

    path_requested = Signal(str)
    file_requested = Signal(str)

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
        # a right click saves the picture as an image file, as do those of a job below
        self._picture.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._picture.customContextMenuRequested.connect(self._picture_menu)
        self._facts = QLabel("")
        self._facts.setTextFormat(Qt.TextFormat.RichText)
        self._facts.setWordWrap(True)
        self._facts.setObjectName("previewFacts")
        # the buttons of the selected file (of the browser), below the facts: set_actions
        self._actions: QWidget | None = None
        # the jobs before and after a job: links to click, between the facts and the file list
        self._links = QLabel("")
        self._links.setTextFormat(Qt.TextFormat.RichText)
        self._links.setWordWrap(True)
        self._links.setObjectName("previewLinks")
        self._links.setOpenExternalLinks(False)
        self._links.linkActivated.connect(self._on_link)
        # a table of particles: the micrographs, each with its particles on it, in a window
        self._micrographs = QToolButton()
        self._micrographs.setText("View particles on micrographs…")
        self._micrographs.setToolTip(
            "List the micrographs of this table and show the particles on the one selected"
        )
        self._micrographs.clicked.connect(self.open_micrographs_window)
        self._micrographs_star: str | None = None
        self._micrographs_first: str | None = None
        self._micrograph_windows: dict = {}
        # the graph of the jobs of a project, made when a project folder is shown
        self._graph_box: QWidget | None = None
        self._graph_view = None
        self._graph_window = None
        self._graph_info: dict | None = None
        # what a job made: pictures with captions, in a scroll area
        self._outcome_scroll = None
        self._outcome_panels: list[tuple[QLabel, "np.ndarray"]] = []  # noqa: F821
        # the button of a mask job that opens its map multiplied by the mask
        self._masked_button: QToolButton | None = None
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFrameShape(QFrame.Shape.NoFrame)
        # long lines wrap at the width of the pane, a path without spaces anywhere
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._text.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setObjectName("previewNote")
        for widget in (
            self._title,
            self._picture,
            self._facts,
            self._micrographs,
            self._links,
            self._note,
        ):
            layout.addWidget(widget)
        layout.setAlignment(self._micrographs, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._text, 1)
        # the room left when there is no text goes to the bottom, not between the parts
        layout.addStretch(0)
        self._muted = "#8f8f8f"
        self._accent = "#4a8fd6"
        self._theme_colors: dict[str, str] = {}
        self._serial = 0
        self._base_facts: list = []
        self._pixmap: QPixmap | None = None
        self._worker: _PreviewWorker | None = None
        self._pending: str | None = None
        self.clear()

    def set_theme(self, colors: dict[str, str]) -> None:
        self._muted = colors["muted"]
        self._accent = colors.get("accent", self._accent)
        self._theme_colors = dict(colors)
        self._apply_graph_theme()
        for window in self._micrograph_windows.values():
            window.set_theme(colors)
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
        self._links.setText("")
        self._links.hide()
        self._micrographs_star = None
        self._micrographs.hide()
        if self._graph_box is not None:
            self._graph_box.hide()
        if self._outcome_scroll is not None:
            self._outcome_scroll.hide()
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
        worker.start(
            QThread.Priority.LowPriority
        )  # the window comes before a slow read

    def _on_finished(self) -> None:
        if self._pending is not None:
            self._start()

    def _on_ready(self, serial: int, info: dict) -> None:
        if serial != self._serial:
            return  # another path was chosen meanwhile
        self.show_info(info)
        outcome = info.get("outcome") or {}
        if outcome.get("refine") is not None:
            # the estimate is shown; the exact number comes when it has been counted
            self._base_facts = list(info.get("facts", []))[
                : len(info.get("facts", [])) - len(outcome.get("facts", []))
            ]
            worker = _RefineWorker(serial, outcome["refine"])
            _REFINING.add(worker)
            worker.done.connect(self._on_refined)
            worker.finished.connect(lambda w=worker: _REFINING.discard(w))
            worker.start()

    def _on_refined(self, serial: int, outcome: dict) -> None:
        if serial != self._serial or not outcome:
            return  # another path was chosen meanwhile, or it could not be counted
        self._set_facts(self._base_facts + list(outcome.get("facts", [])))
        if outcome.get("panels") and (
            self._outcome_scroll is None or self._outcome_scroll.isHidden()
        ):
            # pictures that only the second, slow reading makes (a table of particles)
            self._show_outcome(outcome)

    def show_info(self, info: dict) -> None:
        """Show a preview made by :func:`build_preview`."""
        self._title.setText(info.get("title", ""))
        self._set_facts(info.get("facts", []))
        self._show_links(info.get("links", []))
        self._micrographs_star = info.get("micrographs")
        self._micrographs_first = info.get("micrograph")
        self._micrographs.setVisible(self._micrographs_star is not None)
        self._show_graph(info.get("graph"))
        self._show_outcome(info.get("outcome"))
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

    def set_actions(self, bar: QWidget | None) -> None:
        """Show a bar of buttons below the facts, or take the one shown out (``None``).

        Parameters
        ----------
        bar : QWidget or None
            The buttons of the selected file. Whether it is shown is up to the caller.
        """
        layout = self.layout()
        if self._actions is not None:
            layout.removeWidget(self._actions)
        self._actions = bar
        if bar is not None:
            layout.insertWidget(layout.indexOf(self._facts) + 1, bar)

    def _set_facts(self, facts: list) -> None:
        rows = "".join(
            f"<tr><td style='color:{self._muted}; padding-right:10px'>{label}</td><td>{value}</td></tr>"
            for label, value in facts
        )
        self._facts.setText(f"<table>{rows}</table>" if rows else "")

    # more than this many jobs of one kind are cut, so that a hub job does not push the rest away
    _MAX_LINKS = 12

    def _show_links(self, groups: list) -> None:
        """Show the jobs before and after a job, each a link to its folder.

        ``groups`` is ``[(title, [(label, folder or None), ...]), ...]``: a job whose folder is
        gone (``None``) is shown as text.
        """
        from html import escape
        from urllib.parse import quote

        rows = []
        for title, items in groups:
            lines = []
            for label, folder in items[: self._MAX_LINKS]:
                if folder is None:
                    lines.append(
                        f"<span style='color:{self._muted}'>{escape(label)}</span>"
                    )
                else:
                    lines.append(
                        f"<a href='{quote(str(folder))}' style='color:{self._accent}; "
                        f"text-decoration:none'>{escape(label)}</a>"
                    )
            if len(items) > self._MAX_LINKS:
                more = len(items) - self._MAX_LINKS
                lines.append(f"<span style='color:{self._muted}'>… {more} more</span>")
            rows.append(
                f"<tr><td style='color:{self._muted}; padding-right:10px' valign='top'>"
                f"{escape(title)}</td><td>{'<br>'.join(lines)}</td></tr>"
            )
        self._links.setText(f"<table>{''.join(rows)}</table>" if rows else "")
        self._links.setVisible(bool(rows))

    def _ensure_outcome(self) -> None:
        """Make the area for what a job made when the first job is shown."""
        if self._outcome_scroll is not None:
            return
        from PySide6.QtWidgets import QScrollArea

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        column = QVBoxLayout(inner)
        column.setContentsMargins(0, 0, 4, 0)
        column.setSpacing(6)
        column.addStretch(1)
        scroll.setWidget(inner)
        self._outcome_scroll, self._outcome_column = scroll, column
        self.layout().insertWidget(self.layout().indexOf(self._note), scroll, 1)

    def _show_outcome(self, outcome: dict | None) -> None:
        """Show what a job made: each picture with its title and a line of caption."""
        if not outcome or not outcome["panels"]:  # counts only: the facts above say it
            if self._outcome_scroll is not None:
                self._outcome_scroll.hide()
            return
        self._ensure_outcome()
        column = self._outcome_column
        while column.count() > 1:  # all but the stretch at the end
            item = column.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._outcome_panels = []
        self._masked_button = None
        for panel in outcome["panels"]:
            heading = QLabel(panel["title"])
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            column.insertWidget(column.count() - 1, heading)
            picture = QLabel()
            picture.setAlignment(Qt.AlignmentFlag.AlignLeft)
            picture.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            picture.customContextMenuRequested.connect(
                lambda pos, p=picture, a=panel["image"], t=panel["title"]: (
                    self._save_menu(p, pos, _qimage_of(a), self._file_name(t))
                )
            )
            column.insertWidget(column.count() - 1, picture)
            self._outcome_panels.append((picture, panel["image"]))
            if panel.get("caption"):
                caption = QLabel(panel["caption"])
                caption.setWordWrap(True)
                caption.setStyleSheet(f"color: {self._muted};")
                column.insertWidget(column.count() - 1, caption)
        if outcome.get("masked_map"):
            button = QToolButton()
            button.setText("Open masked map")
            button.setToolTip(
                "Write the map multiplied by the mask to the cache of Helicon and open it"
            )
            button.clicked.connect(
                lambda _=False, m=outcome["masked_map"]: self.open_masked_map(m)
            )
            column.insertWidget(
                column.count() - 1, button, 0, Qt.AlignmentFlag.AlignLeft
            )
            self._masked_button = button
        self._fit_outcome()
        self._outcome_scroll.show()
        self._outcome_scroll.verticalScrollBar().setValue(0)

    def _fit_outcome(self) -> None:
        """Scale the pictures of the job to the width of the pane."""
        width = max(self.width() - 40, 120)
        for label, array in self._outcome_panels:
            label.setPixmap(_pixmap_of(array, min(width, 420)))

    def _ensure_graph(self) -> None:
        """Make the graph of the jobs when the first project is shown (it needs the graph module)."""
        if self._graph_box is not None:
            return
        from .job_graph import JobGraphView

        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        header = QHBoxLayout()
        title = QLabel("Jobs")
        title.setObjectName("previewGraphTitle")
        header.addWidget(title)
        header.addStretch(1)
        detach = QToolButton()
        detach.setText("Open in a window")
        detach.setToolTip("Show the jobs in a window of its own, to zoom and pan")
        detach.clicked.connect(self.open_graph_window)
        header.addWidget(detach)
        column.addLayout(header)
        view = JobGraphView()
        view.setMinimumHeight(180)
        view.job_clicked.connect(self.path_requested)
        view.graph_ready.connect(self._on_graph_ready)
        column.addWidget(view)
        self._graph_box, self._graph_view = box, view
        # after the facts and the links, before the note and the text
        self.layout().insertWidget(self.layout().indexOf(self._links) + 1, box)
        # a project folder has no list of folders: the graph takes the height that is left
        self.layout().setStretchFactor(box, 1)
        self._graph_title = title
        self._apply_graph_theme()

    def _apply_graph_theme(self) -> None:
        if self._graph_view is None or not self._theme_colors:
            return  # the colours come with the theme of the browser
        colors = dict(self._theme_colors)
        self._graph_view.set_theme(colors)
        self._graph_title.setStyleSheet(f"color: {self._muted};")
        if self._graph_window is not None:
            self._graph_window.set_theme(colors)

    def _show_graph(self, graph: dict | None) -> None:
        """Show the jobs of a project as a graph, read in the background."""
        self._graph_info = graph
        if graph is None:
            if self._graph_box is not None:
                self._graph_box.hide()
            return
        self._ensure_graph()
        self._graph_box.show()
        self._graph_view.load(graph["kind"], graph["root"])

    def _on_graph_ready(self) -> None:
        window = self._graph_window
        if (
            window is not None
            and window.isVisible()
            and self._graph_view.graph is not None
        ):
            window.view.set_graph(self._graph_view.graph)

    def open_graph_window(self) -> None:
        """Show the jobs in a window of its own, with the browser still where it is; a click
        on a job there goes to it, as in the preview."""
        if self._graph_info is None or self._graph_view is None:
            return
        from .job_graph import JobGraphWindow

        if self._graph_window is None:
            window = JobGraphWindow(self.window())
            window.view.job_clicked.connect(self.path_requested)
            self._graph_window = window
            self._apply_graph_theme()
        window = self._graph_window
        graph = self._graph_view.graph
        if graph is not None:
            window.view.set_graph(graph)
        else:
            window.view.load(self._graph_info["kind"], self._graph_info["root"])
        window.show()
        window.raise_()
        window.activateWindow()

    def open_micrographs_window(self) -> None:
        """Show the micrographs of the table of particles in the preview, in a window of its own:
        one window for each table, brought to the front when it is open."""
        star = self._micrographs_star
        if star is None:
            return
        from .micrograph_viewer import MicrographViewer

        window = self._micrograph_windows.get(star)
        if window is not None and self._micrographs_first is not None:
            window.select(self._micrographs_first)
        if window is None:
            window = MicrographViewer(star, self.window(), self._micrographs_first)
            window.destroyed.connect(
                lambda _=None, s=star: self._micrograph_windows.pop(s, None)
            )
            if self._theme_colors:
                window.set_theme(self._theme_colors)
            self._micrograph_windows[star] = window
        window.show()
        window.raise_()
        window.activateWindow()

    def open_masked_map(self, files: dict) -> None:
        """Write the map of a mask job multiplied by the mask, in a thread (a map can be large),
        and ask for it to be opened (``file_requested``) when it is written.

        Parameters
        ----------
        files : dict
            ``{"map": path, "mask": path}``, the ``"masked_map"`` of the outcome of the job.
        """
        worker = _MaskedMapWorker(files["map"], files["mask"])
        _REFINING.add(worker)
        button = self._masked_button
        if button is not None:
            button.setEnabled(False)
            button.setText("Writing the masked map…")
        worker.done.connect(lambda path, b=button: self._on_masked_map(path, b))
        worker.finished.connect(lambda w=worker: _REFINING.discard(w))
        worker.start()

    def _on_masked_map(self, path: str, button) -> None:
        if button is not None and button is self._masked_button:
            button.setEnabled(True)
            button.setText("Open masked map" if path else "Cannot write the masked map")
        if path:
            self.file_requested.emit(path)

    def set_current(self, folder: str | None) -> None:
        """Mark the job of the folder the browser is in on the graphs, in the preview and in the window."""
        if self._graph_view is not None:
            self._graph_view.set_current(folder)
        if self._graph_window is not None:
            self._graph_window.view.set_current(folder)

    def _on_link(self, href: str) -> None:
        from urllib.parse import unquote

        self.path_requested.emit(unquote(href))

    def _file_name(self, panel: str = "") -> str:
        """The name offered to save a picture of the preview: the shown file, then the panel."""
        name = Path(self._title.text()).stem or "preview"
        if panel:
            name += "_" + (re.sub(r"[^\w-]+", "_", panel).strip("_") or "picture")
        return name + ".png"

    def _picture_menu(self, pos) -> None:
        if self._pixmap is not None:
            self._save_menu(
                self._picture, pos, self._pixmap.toImage(), self._file_name()
            )

    def _run_menu(self, menu: QMenu, global_pos):
        """Show a menu where the pointer is; the action that was chosen, or None."""
        return menu.exec(global_pos)

    def _save_menu(self, label: QLabel, pos, image: QImage, name: str) -> None:
        """The menu of a picture of the preview: save it, at its own size, as an image file."""
        menu = QMenu(self)
        save = menu.addAction("Save image as…")
        if self._run_menu(menu, label.mapToGlobal(pos)) is save:
            from .viewer import _save_qimage

            _save_qimage(image, self, default_name=name)

    # the picture fits in a box this many pixels a side, or the width of the pane if narrower
    _PICTURE_BOX = 360

    def _fit_picture(self) -> None:
        """Scale the picture into a box, so that a tall one (a slice of a long map) does not
        push the facts and the buttons out of the pane; one enlarged much keeps sharp pixels.
        """
        if self._pixmap is None:
            return
        width = min(max(self.width() - 24, 100), self._PICTURE_BOX)
        height = self._PICTURE_BOX
        scale = min(width / self._pixmap.width(), height / self._pixmap.height())
        shown = self._pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            (
                Qt.TransformationMode.FastTransformation
                if scale >= 2
                else Qt.TransformationMode.SmoothTransformation
            ),
        )
        self._picture.setPixmap(shown)
        self._picture.setFixedHeight(shown.height() + 8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_picture()
        if self._outcome_panels:
            self._fit_outcome()

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
