"""Exposure/optics group computation functions shared across CLI commands."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
from .collections import assign_to_groups, all_matched_attrs
from .epu import (
    movie_filename_patterns,
    guess_data_collection_software,
    extract_data_collection_time,
)

__all__ = [
    "assign_time_groups",
    "combine_groups",
    "extract_timestamps",
    "per_micrograph_ids",
    "per_micrograph_mapping",
    "propagate_ctf_median",
    "sync_group_columns",
]

logger = logging.getLogger(__name__)


def combine_groups(existing: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Combine existing group IDs and new subgroup IDs into unique combined IDs.

    For each pair (existing_group, new_subgroup), produces a unique sequential ID.
    This splits each existing group by the new subgroups.

    Parameters
    ----------
    existing :
        1D array-like of existing group IDs.
    new :
        1D array-like of new subgroup IDs (same length).

    Returns
    -------
    np.ndarray
        1D array of combined group IDs (1-indexed).
    """
    pairs = np.column_stack([existing, new])
    _, combined = np.unique(pairs, axis=0, return_inverse=True)
    return combined + 1


def extract_timestamps(
    micrographs: list[str], software: str, use_mtime_fallback: bool = False
) -> dict[str, float]:
    """Extract timestamps from micrograph filenames.

    Uses software-specific filename patterns to extract timestamps.
    Falls back to serial_number extraction, then float('inf') unless
    use_mtime_fallback is True.

    Parameters
    ----------
    micrographs :
        Iterable of micrograph filenames.
    software :
        Software key from movie_filename_patterns().
    use_mtime_fallback :
        If True, fall back to file modification time
        when no timestamp can be extracted from the filename.

    Returns
    -------
    dict
        Mapping of micrograph filename to timestamp.
    """
    patterns = movie_filename_patterns()
    pattern = patterns.get(software)
    result = {}
    for m in micrographs:
        ts = extract_data_collection_time(m, software=software)
        if ts is None and pattern:
            match = re.search(pattern, Path(m).name)
            if match and "serial_number" in match.groupdict():
                ts = float(match.group("serial_number"))
        if ts is None and use_mtime_fallback:
            try:
                ts = Path(m).resolve().stat().st_mtime
            except OSError:
                ts = None
        result[m] = ts if ts is not None else float("inf")
    return result


def per_micrograph_mapping(micrographs: list[str], start_id: int = 1) -> dict[str, int]:
    """Map each micrograph to a unique sequential group ID.

    Parameters
    ----------
    micrographs :
        Iterable of unique micrograph identifiers.
    start_id :
        First group ID (default 1).

    Returns
    -------
    dict
        Mapping of micrograph to group ID.
    """
    return {m: i + start_id for i, m in enumerate(micrographs)}


def per_micrograph_ids(
    names: np.ndarray,
    start_id: int = 1,
) -> np.ndarray:
    """Assign sequential group IDs per unique micrograph name.

    Parameters
    ----------
    names :
        Array of micrograph/group names (one per particle).
    start_id :
        First group ID (default 1).

    Returns
    -------
    np.ndarray
        Array of group IDs with the same length as *names*.
    """
    _, inverse = np.unique(names, return_inverse=True)
    return inverse + start_id


def propagate_ctf_median(data, group_id_name: str) -> None:
    """Set CTF parameters to median within each exposure group.

    Parameters
    ----------
    data : cryosparc.tools.Dataset
        CryoSPARC dataset.
    group_id_name : str
        Name of the exposure group ID column.
    """
    group_ids = np.sort(np.unique(data[group_id_name]))
    ctf_cols = (
        "ctf/cs_mm ctf/phase_shift_rad ctf/shift_A ctf/tilt_A "
        "ctf/trefoil_A ctf/tetra_A ctf/anisomag"
    )
    for gi in group_ids:
        mask = np.where(data[group_id_name] == gi)
        for col in ctf_cols.split():
            if col in data:
                data[col][mask] = np.median(data[col][mask])


def sync_group_columns(
    data, group_id_name: str, query_str: str = "exp_group_id"
) -> None:
    """Sync all related group ID columns to match the primary one.

    For example, if the dataset has ``ctf/exp_group_id``,
    ``location/exp_group_id``, and ``mscope_params/exp_group_id``, assigns
    the value of *group_id_name* to the others.

    Parameters
    ----------
    data : pd.DataFrame or cryosparc.tools.Dataset
        Data object.
    group_id_name : str
        Primary group ID column name.
    query_str : str, optional
        Substring to search for related columns. Defaults to ``"exp_group_id"``.
    """
    group_id_names_all = all_matched_attrs(data, query_str=query_str)
    if len(group_id_names_all) > 1:
        for attr in group_id_names_all:
            if attr != group_id_name:
                data[attr] = data[group_id_name]


def assign_time_groups(
    micrographs: list[str] | np.ndarray,
    source_group_ids: np.ndarray,
    group_id_lookup: np.ndarray,
    time_group_size: int,
    verbose: int = 0,
    use_mtime_fallback: bool | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, str]]:
    """Assign micrographs to time-based groups and map particles to groups.

    Detects data collection software, extracts timestamps, and assigns
    micrographs to groups of a given size sorted by collection time.

    Parameters
    ----------
    micrographs : list[str] or np.ndarray
        All unique micrograph filenames.
    source_group_ids : np.ndarray
        1-D array of sorted, unique source group IDs to split.
    group_id_lookup : np.ndarray
        Per-particle source group ID (same length as particle array).
    time_group_size : int
        Number of micrographs per time group. If negative, existing groups
        are combined into one first, then split with ``abs(time_group_size)``.
    verbose : int, optional
        Verbosity level (default 0).
    use_mtime_fallback : bool or None, optional
        If True, fall back to file modification time when no timestamp can be
        extracted from the filename. If False, use ``float("inf")`` for
        unparseable filenames. If None (default), auto-detect based on software:
        ``True`` for non-EPU, ``False`` for EPU.

    Returns
    -------
    new_group_ids : np.ndarray
        Per-particle new group IDs (1-indexed).
    micrograph_to_time : dict
        Mapping of micrograph filename to Unix timestamp.
    micrograph_to_time_str : dict
        Mapping of micrograph filename to formatted time string.
    """
    from .exceptions import HeliconError

    micrographs = np.asarray(micrographs)

    sample = micrographs[0] if isinstance(micrographs[0], str) else str(micrographs[0])
    software = guess_data_collection_software(sample)
    if software is None:
        known = ", ".join(sorted(movie_filename_patterns().keys()))
        logger.warning(
            "cannot detect the data collection software: %s\n\t"
            "I only know the filenames by %s",
            sample,
            known,
        )
        raise HeliconError("cannot detect data collection software")

    if use_mtime_fallback is None:
        use_mtime = software not in ("EPU", "EPU_old")
    else:
        use_mtime = use_mtime_fallback
    # Deduplicate micrographs for timestamp extraction, but keep full array for indexing
    unique_micrographs = list(dict.fromkeys(micrographs))
    micrograph_to_time = extract_timestamps(
        unique_micrographs, software, use_mtime_fallback=use_mtime
    )

    micrograph_to_time_str = {
        m: (
            datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H-%M-%S")
            if t != float("inf")
            else "unknown"
        )
        for m, t in micrograph_to_time.items()
    }

    micrograph_array = micrographs if micrographs.ndim == 1 else micrographs

    last_group_id = 0
    new_group_ids = np.zeros(len(group_id_lookup), dtype=int)

    for gi in source_group_ids:
        mask = np.where(group_id_lookup == gi)[0]
        group_micrographs = np.unique(micrograph_array[mask])
        group_times = [micrograph_to_time[m] for m in group_micrographs]
        time_2_subgroup = assign_to_groups(group_times, time_group_size)
        particle_subgroups = np.array(
            [time_2_subgroup[micrograph_to_time[m]] for m in micrograph_array[mask]]
        )
        new_group_ids[mask] = particle_subgroups + last_group_id
        last_group_id = int(np.max(new_group_ids))

    if verbose > 1:
        n_new = len(np.unique(new_group_ids))
        logger.info(f"\t{len(source_group_ids)} -> {n_new} groups")

    return new_group_ids, micrograph_to_time, micrograph_to_time_str
