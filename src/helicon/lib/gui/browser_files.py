"""What the file browser knows of a file or a folder, without Qt: its type, its size, its images,
its number of entries, how to sort and to filter names."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from helicon.lib.cache import cache, setup_cache_dir


_IMAGE_EXTENSIONS = {
    ".mrc",
    ".mrcs",
    ".star",
    ".cs",
    ".lst",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".sqlite",
    ".html",
    ".htm",
}

_DIGIT_RUN = re.compile(r"(\d+)")


def _natural_sort_key(text: str) -> list:
    """Sort key that compares runs of digits as numbers.

    Plain text order puts ``J10`` before ``J2``, which scatters a CryoSPARC
    project's jobs: they are named ``J1`` to ``J121`` with no zero padding, so
    the numbering only reads continuously if the digits are compared as
    numbers. RELION's zero-padded ``job001`` is unaffected either way, and
    micrograph names with frame numbers benefit the same way jobs do.

    Digit runs sort before text at the same position, so ``J2`` precedes
    ``J2a``, and the comparison is case-insensitive.

    Parameters
    ----------
    text : str

    Returns
    -------
    list
        A list of ``(kind, number, text)`` triples, comparable between names.
    """
    key = []
    for index, part in enumerate(_DIGIT_RUN.split(text or "")):
        if index % 2:
            key.append((0, int(part), ""))
        elif part:
            key.append((1, 0, part.lower()))
    return key


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _format_item_count(n: int) -> str:
    """Format a directory's entry count with proper singular/plural phrasing.

    Parameters
    ----------
    n : int
        Number of visible entries.

    Returns
    -------
    str
        ``"1 item"`` for a single entry, ``"N items"`` otherwise.
    """
    return "1 item" if n == 1 else f"{n} items"


def _name_matches_filter(name: str, pattern: str, use_regex: bool) -> bool:
    """Return whether ``name`` matches the browser's active filter.

    An empty pattern matches everything. Regex mode uses ``re.search`` (any
    match), glob mode uses ``fnmatch.fnmatch``. Invalid regexes match
    everything so a bad pattern never hides the whole listing.

    Parameters
    ----------
    name : str
        Entry name to test.
    pattern : str
        Active filename filter (``"*"`` when no filter is set).
    use_regex : bool
        Whether ``pattern`` is a regular expression.

    Returns
    -------
    bool
        Whether the name should be shown in the listing.
    """
    if not pattern:
        return True
    if use_regex:
        import re

        try:
            return bool(re.search(pattern, name))
        except re.error:
            return True
    else:
        import fnmatch

        return fnmatch.fnmatch(name, pattern)


def _count_folder_entries(path: str | Path) -> int:
    """Count the visible (non-hidden) entries of a directory.

    Mirrors the browser's visibility rule of skipping entries whose name
    starts with a dot. Missing or unreadable directories count as zero so
    callers never have to special-case them.

    Parameters
    ----------
    path : str or Path
        Directory to count.

    Returns
    -------
    int
        Number of visible entries (``0`` when the directory cannot be read).
    """
    try:
        return sum(1 for e in Path(path).iterdir() if not e.name.startswith("."))
    except OSError:
        return 0


def _count_dir_items(path: Path, filter_pattern: str, use_regex: bool) -> int | None:
    """Count the entries of a directory the way the browser lists them.

    Mirrors ``FileBrowserModel._load_directory`` visibility rules: hidden
    entries (leading dot) are skipped and non-directory entries must match
    the active filter. Directories are always visible.

    Parameters
    ----------
    path : Path
        Directory to count.
    filter_pattern : str
        Active filename filter (``"*"`` when no filter is set).
    use_regex : bool
        Whether ``filter_pattern`` is a regular expression.

    Returns
    -------
    int or None
        Number of visible entries, or ``None`` when the directory cannot be
        read (e.g. ``PermissionError``).
    """
    try:
        count = 0
        for entry in path.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                count += 1
            elif _name_matches_filter(entry.name, filter_pattern, use_regex):
                count += 1
        return count
    except (OSError, PermissionError):
        return None


def _is_text_file(path: str) -> bool:
    """Check if a file is a text file by attempting UTF-8 decode."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


# MRC-format image extensions. RELION/CTFFIND write CTF power spectra as MRC
# maps with a ``.ctf`` suffix (e.g. ``*_PS.ctf``), so those are treated as
# MRC images too.
_MRC_EXTENSIONS = {".mrc", ".mrcs", ".map", ".ctf"}

# MRC extensions that may hold a 3D volume; ``.mrcs`` is always a 2D stack.
_MRC_VOLUME_EXTENSIONS = {".mrc", ".map", ".ctf"}

# Common raster image formats opened as 2D images (napari's built-in
# readers handle them, e.g. PNG, JPEG, TIFF).
_RASTER_IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
    }
)

# Extensions with dedicated handlers/labels.  Everything else falls back
# to known text extension checks or uppercased extensions.
_KNOWN_EXTENSIONS = frozenset(
    {
        ".mrc",
        ".mrcs",
        ".map",
        ".ctf",
        ".star",
        ".bild",
        ".pdf",
        ".eps",
        ".cs",
        ".png",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".webp",
        ".html",
        ".htm",
    }
)

_KNOWN_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".py",
        ".pyi",
        ".md",
        ".rst",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".log",
        ".out",
        ".sh",
        ".zsh",
        ".bash",
        ".csv",
        ".tsv",
        ".ini",
        ".cfg",
        ".conf",
        ".env",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".js",
        ".ts",
        ".css",
        ".mod",
        ".inp",
        ".dat",
    }
)


def _get_file_type_label(filepath: str) -> str:
    """Return a human-readable type label for the file browser.

    Detection order:
    1. Known file types by suffix (.mrc, .mrcs, .star, .bild, etc.)
    2. Known text extensions -> "Text"
    3. Anything else -> uppercased extension (or "File" if no extension)
    """
    name = os.path.basename(filepath).lower()
    ext = os.path.splitext(name)[1]

    if ext == ".star":
        if any(name.endswith(s) for s in _METADATA_STAR_SUFFIXES):
            return "Metadata"
        return "STAR"

    if ext == ".bild":
        return "3D Plot"

    if ext == ".cs":
        return "CryoSPARC"

    from helicon.lib import cryosparc_project

    if cryosparc_project.is_metadata_file(filepath):
        return "Metadata"

    if ext in (".html", ".htm"):
        return "Browser"

    if ext == ".pdf":
        return "PDF"

    if ext == ".eps":
        return "EPS"

    if ext == ".ctf":
        # RELION CTF power spectra are MRC maps with a .ctf suffix.
        return "MRC"

    if ext in _KNOWN_TEXT_EXTENSIONS:
        return "Text"

    if ext in _KNOWN_EXTENSIONS:
        return ext.lstrip(".").upper()

    return ext.lstrip(".").upper() if ext else "File"


def _get_file_info(filepath: str) -> tuple[str, str, str]:
    """Returns (info, n_images, pixel_size) tuple."""
    name = Path(filepath).name.lower()
    if name.endswith(".star") and any(
        name.endswith(s) for s in _METADATA_STAR_SUFFIXES
    ):
        # Pipeline/optimiser/model/sampling/job star files describe metadata,
        # not image data; show them as plain entries (opened as text).
        return "", "", ""
    ext = Path(filepath).suffix.lower()
    info = ""
    n_images = ""
    pixel_size = ""

    if ext in _MRC_EXTENSIONS:
        try:
            import mrcfile

            with mrcfile.open(filepath, header_only=True) as mrc:
                nx = int(mrc.header.nx)
                ny = int(mrc.header.ny)
                nz = int(mrc.header.nz)
                apix = float(mrc.voxel_size.x)
                if apix > 0:
                    pixel_size = f"{apix:.3f} Å"
                stack = ext == ".mrcs"
                if not stack and nz > 1:
                    # everything is .mrc in a CryoSPARC job: its type says whether
                    # this is a stack of class averages or particles
                    from helicon.lib import cryosparc_project

                    stack = cryosparc_project.mrc_content(filepath) == "stack"
                if stack:
                    n_images = str(nz) if nz > 1 else "1"
                    info = f"{nx}×{ny}"
                else:
                    n_images = "1"
                    info = f"{nx}×{ny}" if nz == 1 else f"{nx}×{ny}×{nz}"
        except Exception:
            pass

    elif ext in (".tif", ".tiff"):
        try:
            from PIL import Image

            with Image.open(filepath) as img:
                if hasattr(img, "n_frames") and img.n_frames > 1:
                    n_images = str(img.n_frames)
                    info = f"{img.n_frames}×{img.width}×{img.height}"
                else:
                    n_images = "1"
                    info = f"{img.width}×{img.height}"
        except Exception:
            pass

    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
        try:
            from PIL import Image

            with Image.open(filepath) as img:
                n_images = "1"
                info = f"{img.width}×{img.height}"
        except Exception:
            pass

    elif ext == ".star":
        try:
            star_dir = Path(filepath).parent

            # State for line-by-line star parser (avoids loading the entire
            # file into a DataFrame — critical for large *data.star files).
            col_names: list[str] = []
            image_col_idx = -1
            size_col_idx = -1
            apix_col_idx = -1
            in_loop = False
            in_data = False
            current_data_block = ""
            n_particles = 0
            first_image_ref: str | None = None
            optics_size: int | None = None
            optics_pixel_size: float | None = None

            with open(filepath) as f:
                for line in f:
                    raw = line.rstrip("\n\r")
                    s = raw.strip()
                    if not s or s.startswith("#"):
                        continue

                    if s.startswith("data_"):
                        current_data_block = s[5:].strip().lower()
                        in_loop = False
                        in_data = False
                        col_names = []
                        image_col_idx = -1
                        size_col_idx = -1
                        apix_col_idx = -1
                        continue

                    if s == "loop_":
                        in_loop = True
                        col_names = []
                        image_col_idx = -1
                        size_col_idx = -1
                        apix_col_idx = -1
                        continue

                    if in_loop and s.startswith("_"):
                        col_names.append(s.split()[0])
                        idx = len(col_names) - 1
                        cl = s.lower()
                        if "imagename" in cl or "micrographname" in cl:
                            if image_col_idx < 0:
                                image_col_idx = idx
                        if current_data_block == "optics":
                            if "imagesize" in cl and size_col_idx < 0:
                                size_col_idx = idx
                            if "imagepixelsize" in cl and apix_col_idx < 0:
                                apix_col_idx = idx
                        continue

                    if not in_data:
                        in_data = True

                    parts = raw.split()

                    if current_data_block == "optics":
                        if (
                            optics_size is None
                            and size_col_idx >= 0
                            and size_col_idx < len(parts)
                        ):
                            try:
                                optics_size = int(parts[size_col_idx])
                            except ValueError:
                                pass
                        if (
                            optics_pixel_size is None
                            and apix_col_idx >= 0
                            and apix_col_idx < len(parts)
                        ):
                            try:
                                optics_pixel_size = float(parts[apix_col_idx])
                            except ValueError:
                                pass
                    else:
                        n_particles += 1
                        if (
                            first_image_ref is None
                            and image_col_idx >= 0
                            and image_col_idx < len(parts)
                        ):
                            first_image_ref = parts[image_col_idx]

            if n_particles > 0:
                n_images = str(n_particles)

                if first_image_ref:
                    img_rel = (
                        first_image_ref.split("@")[-1]
                        if "@" in first_image_ref
                        else first_image_ref
                    )

                    img_path_resolved = None
                    for ancestor in [star_dir] + list(star_dir.parents):
                        candidate = ancestor / img_rel
                        if candidate.is_file():
                            img_path_resolved = candidate
                            break

                    if img_path_resolved is not None:
                        img_ext = img_path_resolved.suffix.lower()
                        if img_ext in (".mrc", ".mrcs", ".map"):
                            import mrcfile

                            with mrcfile.open(
                                str(img_path_resolved), header_only=True
                            ) as mrc:
                                nx = int(mrc.header.nx)
                                ny = int(mrc.header.ny)
                                nz = int(mrc.header.nz)
                                apix = float(mrc.voxel_size.x)
                                if apix > 0:
                                    pixel_size = f"{apix:.3f} Å"
                                if img_ext == ".mrcs" or nz == 1:
                                    info = f"{nx}×{ny}"
                                else:
                                    info = f"{nx}×{ny}×{nz}"
                        else:
                            from PIL import Image

                            with Image.open(str(img_path_resolved)) as img:
                                info = f"{img.width}×{img.height}"

                # Fallback to data_optics values when image file not found.
                if not info and optics_size is not None:
                    info = f"{optics_size}×{optics_size}"
                if (
                    not pixel_size
                    and optics_pixel_size is not None
                    and optics_pixel_size > 0
                ):
                    pixel_size = f"{optics_pixel_size:.3f} Å"
        except Exception:
            pass

    elif ext == ".pdf":
        try:
            from PySide6.QtPdf import QPdfDocument

            doc = QPdfDocument()
            doc.load(filepath)
            n = doc.pageCount()
            if n > 0:
                n_images = str(n)
        except Exception:
            pass

    elif ext in (".html", ".htm"):
        pass

    return info, n_images, pixel_size


def _format_modified(mtime: datetime, now: datetime | None = None) -> str:
    """The modification time for the list: "Today 21:07", "Yesterday 09:30", or the date."""
    now = now or datetime.now()
    days = (now.date() - mtime.date()).days
    if days == 0:
        return "Today " + mtime.strftime("%H:%M")
    if days == 1:
        return "Yesterday " + mtime.strftime("%H:%M")
    return mtime.strftime("%Y-%m-%d %H:%M")


# Star files that describe pipelines/optimisation rather than image data.
# Must stay in sync with _METADATA_STAR_SUFFIXES in commands/display.py.
_METADATA_STAR_SUFFIXES = (
    "pipeline.star",
    "optimiser.star",
    "model.star",
    "sampling.star",
    "job.star",
    "extractpick.star",
    "frameimage.star",
    "autopick.star",
)


def _folder_is_helical(folder: str) -> bool:
    """Check if any ``model.star`` in *folder* has ``rlnIsHelix = 1``.

    The answer is cached on disk so repeated file selections within the same
    RELION job folder do not re-scan the STAR header. The cache key includes
    the name, size and modification time of every ``*model.star`` file, so a
    running job that writes a new (helical) ``model.star`` is re-checked
    instead of returning a stale answer.
    """
    try:
        signature = tuple(
            sorted(
                (f.name, f.stat().st_size, f.stat().st_mtime_ns)
                for f in Path(folder).glob("*model.star")
            )
        )
    except Exception:
        signature = ()
    if not signature:
        return False
    return _folder_is_helical_cached(folder, signature)


@cache(
    cache_dir=str(setup_cache_dir() / "file_browser"), expires_after=timedelta(days=7)
)
def _folder_is_helical_cached(folder: str, signature: tuple) -> bool:
    """Scan the ``*model.star`` files of *folder* for ``rlnIsHelix = 1``.

    Parameters
    ----------
    folder : str
        Folder to scan.
    signature : tuple
        ``(name, size, mtime_ns)`` of each ``*model.star`` file; only used
        as part of the cache key.

    Returns
    -------
    bool
        True if any of the files marks the job as helical.
    """
    try:
        folder_path = Path(folder)
        for f in folder_path.glob("*model.star"):
            try:
                if _star_has_helix_col(f):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _star_has_helix_col(star_path: Path) -> bool:
    """Check whether a RELION STAR file has ``rlnIsHelix = 1``.

    Handles two header styles:

    * **Column-definition** (``data.star``) — the header lists column names
      with a ``#N`` index (e.g. ``_rlnIsHelix  #12``).  The first data row
      is then inspected at the indicated column position.
    * **Key-value** (``model.star``) — the header contains
      ``_rlnIsHelix  <value>`` directly, so the value is checked in-place.
    """
    col_index = None
    in_header = True
    with open(star_path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if in_header:
                if stripped.startswith("_rlnIsHelix"):
                    # Key-value format: "_rlnIsHelix  1"
                    tail = stripped.split("_rlnIsHelix", 1)[1].strip()
                    if "#" in tail:
                        # Column-definition format: "_rlnIsHelix  #12"
                        col_index = int(tail.rsplit("#", 1)[-1]) - 1
                    else:
                        return tail == "1"
                elif stripped.startswith("data_"):
                    # Start of a data block — header continues until the
                    # next blank line.
                    continue
                elif stripped == "":
                    # RELION STAR headers: blank line ends the current
                    # block.  Reset so we re-enter header mode when the
                    # next ``data_`` block starts.
                    col_index = None
            elif col_index is not None:
                parts = stripped.split()
                if len(parts) > col_index:
                    return parts[col_index] == "1"
                return False
    return False


def _find_chimerax() -> str | None:
    """Locate the ChimeraX executable.

    Checks common install locations on each platform, then falls back to the
    ``PATH``. Returns the executable path or ``None`` if not found.
    """

    candidates = [
        "ChimeraX",
        "chimerax",
        "/opt/UCSF/ChimeraX/bin/chimerax",
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
        str(Path.home() / "Applications/ChimeraX.app/Contents/MacOS/ChimeraX"),
        "/usr/bin/chimerax",
        "/usr/local/bin/chimerax",
        r"C:\Program Files\ChimeraX\bin\ChimeraX.exe",
        r"C:\Program Files\UCSF\ChimeraX\ChimeraX.exe",
    ]
    for cand in candidates:
        found = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if found:
            return found
    return None


def _open_url(url: str) -> None:
    """Open *url* in the system browser (WSL/Linux/macOS-safe)."""
    try:
        from helicon.lib.shiny import _open_browser

        _open_browser(url)
        return
    except Exception:
        pass
    import webbrowser

    webbrowser.open(url, new=2)
