"""Exposure/optics group computation functions shared across CLI commands."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import numpy as np
from .epu import movie_filename_patterns, extract_data_collection_time

__all__ = [
    "combine_groups",
    "extract_timestamps",
    "per_micrograph_mapping",
]


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
