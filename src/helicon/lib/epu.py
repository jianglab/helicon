"""EPU and serialEM movie filename parsing utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from .exceptions import HeliconIOError

__all__ = [
    "EPU_micrograph_path_2_movie_xml_path",
    "EPU_xml_2_beamshift",
    "assign_beamshift_groups",
    "check_foilhole_xml_files",
    "extract_beamshift",
    "extract_data_collection_time",
    "guess_data_collection_software",
    "movie_filename_patterns",
    "verify_data_collection_software",
]


def movie_filename_patterns() -> dict[str, str]:
    """Return regex patterns for movie filenames from various data collection software.

    Returns
    -------
    dict
        Mapping of software keys to regex pattern strings.
    """
    d = dict(
        # e.g. FoilHole_1464933_Data_427288_427290_20250502_213110_Fractions.mrc
        EPU_old=r"FoilHole_\d{7,8}_Data_\d{6,8}_\d{6,8}_(?P<timestamp>\d{8}_\d{6})_",
        # e.g. FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff
        EPU=r"FoilHole_\d{7,8}_Data_\d{7,8}_(?P<beamshift>\d{1,3})_(?P<timestamp>\d{8}_\d{6})_",
        # e.g. SAVED4M-DNA3_39-103_001_X+0Y+0-1.tif
        serialEM_pncc=r"_(?P<serial_number>\d{3})_(?P<beamshift>[XY][\+-]\d[XY][\+-]\d-\d)",
        # e.g. 250123_SF0431_01129_1-7.eer
        serialEM_embl_heidelberg=r"\d{6}_.{6}_(?P<serial_number>\d{5})_\d-(?P<beamshift>\d{1,2})[_\.]",
        # e.g. k2_1219_cva6X_00087.tif
        serialEM_cuhksz=r"_(?P<serial_number>\d{5})[_\.]",
    )
    return d


def guess_data_collection_software(filename: str) -> str | None:
    """Guess which data collection software produced a given filename.

    Parameters
    ----------
    filename :
        Name of the movie file to classify.

    Returns
    -------
    str or None
        String key of the matching software pattern, or None if no pattern matches.
    """
    import re

    software = None
    patterns = movie_filename_patterns()
    for p in patterns:
        if re.search(patterns[p], filename) is not None:
            software = p
            break
    return software


def verify_data_collection_software(filename: str, software: str) -> Any | None:
    """Verify that a filename matches the pattern for a given software.

    Parameters
    ----------
    filename :
        Name of the movie file to check.
    software :
        Key of the software pattern to verify against.

    Returns
    -------
    re.Match or None
        A re.Match object if the pattern matches, or None otherwise.
    """
    import re

    match = re.search(movie_filename_patterns()[software], filename)
    return match


def extract_data_collection_time(
    filename: str,
    software: str | None = None,
    pattern_names: tuple[str, ...] = ("timestamp",),
) -> float | None:
    """Extract the Unix timestamp from a data collection filename.

    Parameters
    ----------
    filename :
        Movie filename to extract the timestamp from.
    software :
        Software key (auto-guessed if None).
    pattern_names :
        Named groups to try, in order. Defaults to ("timestamp",).

    Returns
    -------
    float or None
        Unix timestamp as a float, or None if no matching pattern or group found.
    """
    import re

    if software is None:
        software = guess_data_collection_software(filename)
    if software is None:
        return None

    pattern = movie_filename_patterns().get(software)
    if pattern is None:
        return None

    match = re.search(pattern, filename)
    if not match:
        return None

    for name in pattern_names:
        try:
            datetime_str = match.group(name)
        except IndexError:
            continue
        from datetime import datetime

        datetime_obj = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
        if software == "EPU_old":
            from datetime import timezone

            datetime_obj = datetime_obj.replace(tzinfo=timezone.utc)
        return datetime_obj.timestamp()

    return None


def extract_beamshift(
    filename: str,
    software: str | None = None,
    pattern_names: tuple[str, ...] = ("beamshift", "serial_number"),
) -> str | None:
    """Extract beam shift or serial number from a data collection filename.

    Parameters
    ----------
    filename :
        Movie filename to extract the beam shift from.
    software :
        Software key (auto-guessed if None).
    pattern_names :
        Named groups to try, in order.
        Defaults to ("beamshift", "serial_number").

    Returns
    -------
    str or None
        The matched group string, or None if no matching pattern or group found.
    """
    import re

    if software is None:
        software = guess_data_collection_software(filename)
    if software is None:
        return None

    pattern = movie_filename_patterns().get(software)
    if pattern is None:
        return None

    match = re.search(pattern, filename)
    if not match:
        return None

    for name in pattern_names:
        try:
            return match.group(name)
        except IndexError:
            continue

    return None


def assign_beamshift_groups(
    micrographs: list[str], software: str, start_id: int = 1, **kwargs: Any
) -> dict[str, int]:
    """Assign 1-indexed beam shift group IDs to micrographs.

    Extracts the beam shift (or serial number) from each micrograph filename
    and maps unique values to sequential group IDs.

    Parameters
    ----------
    micrographs :
        Iterable of micrograph filenames.
    software :
        Software key (must have beamshift or serial_number in pattern).
    start_id :
        First group ID (default 1).
    **kwargs :
        n_per_stage_shift: For serialEM_cuhksz, modulo divisor for grouping.

    Returns
    -------
    dict
        Mapping of micrograph filename to group ID.
    """
    if software in ("EPU", "serialEM_pncc", "serialEM_embl_heidelberg"):
        mapping = {m: extract_beamshift(m, software=software) for m in micrographs}
        unique_vals = sorted(set(mapping.values()))
        id_map = {v: i + start_id for i, v in enumerate(unique_vals)}
        return {m: id_map[mapping[m]] for m in micrographs}

    elif software == "serialEM_cuhksz":
        n_per_stage_shift = int(kwargs.get("n_per_stage_shift", 1))
        result = {}
        for m in micrographs:
            i = int(extract_beamshift(m, software=software))
            if i > 0:
                i = i % n_per_stage_shift
                if i == 0:
                    i = n_per_stage_shift
            else:
                i = 0
            result[m] = i
        return result

    else:
        raise ValueError(f"Software {software!r} not supported for beam shift grouping")


def check_foilhole_xml_files(
    micrograph_paths: list | np.ndarray, xml_folder: str = ""
) -> None:
    """Check that FoilHole XML files exist for the given micrographs.

    Parameters
    ----------
    micrograph_paths : list or np.ndarray
        List of micrograph paths to check.
    xml_folder : str, optional
        Optional explicit folder containing XML files.

    Raises
    ------
    HeliconIOError
        If no FoilHole XML files are found.
    """
    sample = micrograph_paths[0]
    if xml_folder:
        xfp = Path(xml_folder)
        if xfp.exists() and xfp.is_dir() and list(xfp.glob("FoilHole_*.xml")):
            return
    if Path(sample).exists() and list(Path(sample).parent.glob("FoilHole_*.xml")):
        return
    raise HeliconIOError(
        f"Cannot find FoilHole XML files for {sample}. "
        "Specify xml_folder=<path> in the parameter string."
    )


def EPU_micrograph_path_2_movie_xml_path(
    micrograph_path: str | Path, xml_folder: str = ""
) -> Path:
    """Find the matching EPU XML file for a given micrograph path.

    Parameters
    ----------
    micrograph_path :
        Path to the EPU micrograph file.
    xml_folder :
        Optional folder to search for XML files.

    Returns
    -------
    Path
        Path to the matching XML file.
    """
    if not hasattr(EPU_micrograph_path_2_movie_xml_path, "xml_files"):
        EPU_micrograph_path_2_movie_xml_path.xml_files = {}
    xml_files = EPU_micrograph_path_2_movie_xml_path.xml_files

    folder = Path(xml_folder) if xml_folder else Path(micrograph_path).resolve().parent
    if folder not in xml_files:
        xml_files[folder] = list(folder.rglob("*.xml"))

    import re

    pattern = r"\d{21}_(FoilHole_\d{7,8}_Data_\d{6,8}_\d{6,8}_\d{8}_\d{6})"
    match = re.search(pattern, str(micrograph_path))
    if match:
        mid = match.group(1)
        matched_xml_files = [f for f in xml_files[folder] if str(f).find(mid) != -1]
        if not len(matched_xml_files):
            pattern_xml = f"*{mid}*.xml"
            raise HeliconIOError(
                f"cannot find the xml file ({pattern_xml}) in {str(folder)} for {str(micrograph_path)}"
            )
        if len(matched_xml_files) != 1:
            raise HeliconIOError(
                f"find {len(matched_xml_files)} xml files instead of 1 xml file in {str(folder)} for {str(micrograph_path)}"
            )
        return matched_xml_files[0]
    else:
        raise HeliconIOError(
            f"{str(micrograph_path)} filename is inconsistent with EPU output image filename pattern '{pattern}'"
        )


def EPU_xml_2_beamshift(xml_file: str | Path) -> tuple[float, float]:
    """Extract beam shift (x, y) values from an EPU XML file.

    Parameters
    ----------
    xml_file :
        Path to the EPU XML file.

    Returns
    -------
    tuple of float
        Tuple of (beamshift_x, beamshift_y) as floats.
    """
    import xmltodict

    with open(xml_file, "rb") as fp:
        xml = xmltodict.parse(fp, dict_constructor=dict)
    beamshift = xml["MicroscopeImage"]["microscopeData"]["optics"]["BeamShift"]
    beamshift = (float(beamshift["a:_x"]), float(beamshift["a:_y"]))
    return beamshift
