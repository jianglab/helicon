"""Star-file parsers for the display command.

Lazy, line-by-line readers that extract image references from RELION
``optimiser.star``/``model.star``/``*_data.star`` files without loading
the whole file into memory (a DataFrame read blocks for minutes on
million-row particle tables).
"""

from __future__ import annotations

# Star files that describe pipelines/optimisation rather than image data;
# opened as text, not as image/volume stacks.
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


def _is_metadata_star(path: str) -> bool:
    """Star files that describe pipelines/optimisation rather than image data.

    These should be opened as text, not as an image/volume stack.
    """
    from pathlib import Path

    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in _METADATA_STAR_SUFFIXES)


def _is_optimiser_star(path: str) -> bool:
    """Return True for RELION optimiser.star files.

    These contain references to MRC files whose center slices can be
    displayed in a gallery view.
    """
    from pathlib import Path

    return Path(path).name.lower().endswith("optimiser.star")


def _parse_optimiser_star(optimiser_path: str) -> list[str] | None:
    """Parse a RELION optimiser.star file and extract referenced MRC file paths.

    The optimiser.star file references model.star files, which in turn
    contain the actual MRC file paths in the ``_rlnReferenceImage`` column.

    Parameters
    ----------
    optimiser_path : str
        Path to the optimiser.star file.

    Returns
    -------
    list of str or None
        List of resolved MRC file paths, or None if parsing fails.
    """
    from pathlib import Path

    star_dir = Path(optimiser_path).parent
    model_star_paths: list[str] = []

    try:
        with open(optimiser_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("_rlnModelStarFile"):
                    parts = s.split()
                    if len(parts) >= 2:
                        model_rel = parts[-1]
                        for ancestor in [star_dir] + list(star_dir.parents):
                            candidate = ancestor / model_rel
                            if candidate.is_file():
                                model_star_paths.append(str(candidate))
                                break
    except Exception:
        return None

    if not model_star_paths:
        return None

    mrc_paths: list[str] = []
    for model_path in model_star_paths:
        result = _parse_model_star(model_path)
        if result:
            for p in result:
                if p not in mrc_paths:
                    mrc_paths.append(p)

    return mrc_paths if mrc_paths else None


def _parse_model_star(model_path: str) -> list[str] | None:
    """Extract referenced MRC file paths from a RELION model.star.

    Reads the ``data_model_classes`` section and resolves the
    ``_rlnReferenceImage`` column to absolute MRC paths.
    """
    from pathlib import Path

    model_dir = Path(model_path).parent
    in_loop = False
    in_data_model_classes = False
    col_names: list[str] = []
    ref_image_col_idx = -1
    mrc_paths: list[str] = []

    try:
        with open(model_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("data_"):
                    in_data_model_classes = "model_classes" in s
                    in_loop = False
                    col_names = []
                    ref_image_col_idx = -1
                    continue

                if s == "loop_":
                    in_loop = True
                    col_names = []
                    ref_image_col_idx = -1
                    continue

                if in_loop and s.startswith("_"):
                    col_names.append(s.split()[0])
                    if "referenceimage" in s.lower():
                        ref_image_col_idx = len(col_names) - 1
                    continue

                if not in_data_model_classes or ref_image_col_idx < 0:
                    continue

                if not in_loop:
                    continue

                parts = s.split()
                if ref_image_col_idx >= len(parts):
                    continue

                mrc_rel = parts[ref_image_col_idx]

                resolved = None
                for ancestor in [model_dir] + list(model_dir.parents):
                    candidate = ancestor / mrc_rel
                    if candidate.is_file():
                        resolved = str(candidate)
                        break

                if resolved and resolved not in mrc_paths:
                    mrc_paths.append(resolved)
    except Exception:
        return None

    return mrc_paths if mrc_paths else None


def _parse_class2d_model_star(
    model_path: str,
) -> tuple[list[tuple[str, int]], list[float]] | None:
    """Extract MRC references and class distributions from a RELION model.star.

    Reads the ``data_model_classes`` section and resolves
    ``_rlnReferenceImage`` (which may be ``idx@path.mrcs`` for Class2D)
    and ``_rlnClassDistribution`` (abundance) columns.

    Returns
    -------
    tuple of (entries, distributions) or None
        * ``entries``: list of ``(mrc_path, frame_idx)`` tuples.
        * ``distributions``: list of class distribution values (0-1).
    """
    from pathlib import Path

    model_dir = Path(model_path).parent
    in_loop = False
    in_data_model_classes = False
    col_names: list[str] = []
    ref_image_col_idx = -1
    class_dist_col_idx = -1
    entries: list[tuple[str, int]] = []
    distributions: list[float] = []

    try:
        with open(model_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("data_"):
                    in_data_model_classes = "model_classes" in s
                    in_loop = False
                    col_names = []
                    ref_image_col_idx = -1
                    class_dist_col_idx = -1
                    continue

                if s == "loop_":
                    in_loop = True
                    col_names = []
                    ref_image_col_idx = -1
                    class_dist_col_idx = -1
                    continue

                if in_loop and s.startswith("_"):
                    col_name = s.split()[0]
                    col_names.append(col_name)
                    if "referenceimage" in col_name.lower():
                        ref_image_col_idx = len(col_names) - 1
                    elif "classdistribution" in col_name.lower():
                        class_dist_col_idx = len(col_names) - 1
                    continue

                if not in_data_model_classes:
                    continue

                if not in_loop:
                    continue

                parts = s.split()
                if ref_image_col_idx < 0 or ref_image_col_idx >= len(parts):
                    continue

                ref_raw = parts[ref_image_col_idx]
                frame_idx = 0
                if "@" in ref_raw:
                    idx_str, file_part = ref_raw.split("@", 1)
                    frame_idx = int(idx_str) - 1
                else:
                    file_part = ref_raw

                resolved = None
                for ancestor in [model_dir] + list(model_dir.parents):
                    candidate = ancestor / file_part
                    if candidate.is_file():
                        resolved = str(candidate)
                        break

                if not resolved:
                    continue

                dist = 0.0
                if class_dist_col_idx >= 0 and class_dist_col_idx < len(parts):
                    try:
                        dist = float(parts[class_dist_col_idx])
                    except ValueError:
                        dist = 0.0

                entries.append((resolved, frame_idx))
                distributions.append(dist)
    except Exception:
        return None

    if not entries:
        return None

    return entries, distributions


def _parse_star_image_refs(
    star_path: str,
) -> tuple[list[tuple[int, str, float]], tuple, float, int] | None:
    """Parse a .star file line-by-line and build lazy image-stack entries.

    Extracts only the ImageName/MicrographName column instead of loading the
    entire file into a pandas DataFrame (which blocks for minutes on large
    *data.star files with millions of particles). Resolved MRC paths are
    cached because most particles reference frames from the same file.

    Parameters
    ----------
    star_path : str
        Path to the .star file.

    Returns
    -------
    tuple of (entries, first_shape, first_apix, n_skipped) or None
        * ``entries``: list of ``(frame_idx_0based, mrc_path, 0.0)`` tuples.
        * ``first_shape``: ``(nx, ny)`` or ``(nx, ny, nz)`` of the first image.
        * ``first_apix``: pixel size in Angstroms (fallback 1.0).
        * ``n_skipped``: number of data lines whose binary images could not
          be found on disk.
        Returns None if no image references could be resolved.
    """
    from pathlib import Path

    import mrcfile

    star_dir = Path(star_path).parent

    col_names: list[str] = []
    image_col_idx = -1
    in_loop = False
    in_data = False
    entries: list[tuple[int, str, float]] = []
    first_shape: tuple | None = None
    first_apix = 1.0
    n_skipped = 0
    resolved_cache: dict[str, str | None] = {}

    def _resolve(img_rel: str) -> str | None:
        if img_rel in resolved_cache:
            return resolved_cache[img_rel]
        resolved = None
        for ancestor in [star_dir] + list(star_dir.parents):
            candidate = ancestor / img_rel
            if candidate.is_file():
                resolved = str(candidate)
                break
        resolved_cache[img_rel] = resolved
        return resolved

    try:
        with open(star_path) as f:
            for line in f:
                raw = line.rstrip("\n\r")
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("data_") or s == "loop_":
                    in_loop = s == "loop_"
                    in_data = False
                    if in_loop:
                        col_names = []
                        image_col_idx = -1
                    continue

                if in_loop and s.startswith("_"):
                    col_names.append(s.split()[0])
                    if image_col_idx < 0:
                        cl = s.lower()
                        if "imagename" in cl or "micrographname" in cl:
                            image_col_idx = len(col_names) - 1
                    continue

                if image_col_idx < 0:
                    continue

                if not in_data:
                    in_data = True

                parts = raw.split()
                if image_col_idx >= len(parts):
                    continue
                image_ref = parts[image_col_idx]

                if "@" in image_ref:
                    idx_str, img_rel = image_ref.split("@", 1)
                else:
                    idx_str, img_rel = "1", image_ref

                try:
                    frame_idx = int(idx_str) - 1
                except ValueError:
                    continue

                resolved_path = _resolve(img_rel)
                if resolved_path is None:
                    n_skipped += 1
                    continue

                entries.append((frame_idx, resolved_path, 0.0))

                if first_shape is None:
                    with mrcfile.open(resolved_path, header_only=True) as mrc:
                        nx = int(mrc.header.nx)
                        ny = int(mrc.header.ny)
                        nz = int(mrc.header.nz)
                        first_shape = (
                            (nx, ny)
                            if nz == 1 or Path(resolved_path).suffix.lower() == ".mrcs"
                            else (nx, ny, nz)
                        )
                        first_apix = float(mrc.voxel_size.x)
                        if first_apix <= 0:
                            first_apix = 1.0
    except Exception:
        return None

    if not entries or first_shape is None:
        if n_skipped:
            return [], first_shape or (0, 0), first_apix, n_skipped
        return None

    return entries, first_shape, first_apix, n_skipped


def _find_model_star_from_optimiser(optimiser_path: str) -> str | None:
    """Find the model.star referenced by an optimiser.star file."""
    from pathlib import Path

    star_dir = Path(optimiser_path).parent

    try:
        with open(optimiser_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("_rlnModelStarFile"):
                    parts = s.split()
                    if len(parts) >= 2:
                        model_rel = parts[-1]
                        for ancestor in [star_dir] + list(star_dir.parents):
                            candidate = ancestor / model_rel
                            if candidate.is_file():
                                return str(candidate)
    except Exception:
        pass
    return None
