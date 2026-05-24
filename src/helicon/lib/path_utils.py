from __future__ import annotations

import logging, sys, os, time
from pathlib import Path
from typing import Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "which",
    "find_relion_project_folders",
    "get_direct_url",
    "get_file_size",
    "download_file_from_url",
    "get_emdb_id",
    "is_file_readable",
    "is_file_writable",
    "file_ready",
    "convert_file_path",
    "convert_dataframe_file_path",
    "check_required_columns",
]


def which(program: str, use_current_dir: int = 0) -> str | None:
    """Locate an executable in the system PATH (``which`` equivalent).

    Parameters
    ----------
    program : str
        Name or path of the executable.
    use_current_dir : int, optional
        If non-zero, prepend ``.`` to the PATH. Defaults to 0.

    Returns
    -------
    str or None
        Absolute path to the executable, or None if not found.
    """
    location = None
    if program.find(os.sep) != -1:
        file = os.path.abspath(program)
        if os.path.exists(file) and os.access(file, os.X_OK):
            location = os.path.abspath(file)
    else:
        path = os.environ["PATH"]
        if use_current_dir:
            path = ".:%s" % (path)
        dirs = path.split(":")
        for d in dirs:
            file = os.path.join(d, program)
            if os.path.exists(file) and os.access(file, os.X_OK):
                location = os.path.abspath(file)
                break
    return location


def find_relion_project_folders(
    start_folder=None, target_filename: str = "default_pipeline.star", verbose: int = 0
) -> list[Path]:
    """Find all RELION project folders containing a target filename.

    Parameters
    ----------
    start_folder : str or Path, optional
        Root directory to start the search. Defaults to home directory.
    target_filename : str, optional
        Filename to look for. Defaults to ``"default_pipeline.star"``.
    verbose : int, optional
        If non-zero, print progress. Defaults to 0.

    Returns
    -------
    list of Path
        Matching project folders.
    """
    if not (
        start_folder is not None
        and Path(start_folder).exists()
        and Path(start_folder).is_dir()
    ):
        start_folder = Path.home()
    else:
        start_folder = Path(start_folder)
    if verbose:
        logger.info(f"Searching {str(start_folder)} ...")

    project_folders = []
    for root, dirs, files in os.walk(start_folder):
        if target_filename in files:
            project_folders.append(Path(root))
            dirs.clear()
            if verbose:
                logger.info(f"{len(project_folders)}: {str(project_folders[-1])}")

    return project_folders


def get_direct_url(url: str) -> str:
    """Convert cloud storage URLs to direct download URLs.

    Supports Google Drive, Box, Dropbox, SharePoint, and OneDrive.

    Parameters
    ----------
    url : str
        Original cloud storage URL.

    Returns
    -------
    str
        Direct download URL.
    """
    import re

    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1") != -1:
            return url
        elif url.find("dl=0") != -1:
            return url.replace("dl=0", "dl=1")
        else:
            return url + "?dl=1"
    elif url.find("sharepoint.com") != -1 and url.find("guestaccess.aspx") != -1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64

        data_bytes64 = base64.b64encode(bytes(url, "utf-8"))
        data_bytes64_String = (
            data_bytes64.decode("utf-8").replace("/", "_").replace("+", "-").rstrip("=")
        )
        return (
            f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
        )
    else:
        return url


def get_file_size(url: str) -> int | None:
    """Get the file size of a remote resource via a HEAD request.

    Parameters
    ----------
    url : str
        URL to check.

    Returns
    -------
    int or None
        File size in bytes, or None if unknown.
    """
    import requests

    response = requests.head(url)
    if "Content-Length" in response.headers:
        file_size = int(response.headers["Content-Length"])
        return file_size
    else:
        return None


def download_file_from_url(
    url: str, target_file_name: str | None = None, return_filename: bool = False
) -> object | str:
    """Download a file from a URL or return a local file object.

    Parameters
    ----------
    url : str
        Remote URL or local file path.
    target_file_name : str, optional
        If given, write to this file path.
    return_filename : bool, optional
        If True, return the filename instead
        of a file object.

    Returns
    -------
    file object or str
        Opened file (readable binary) or filename.

    Raises
    ------
    IOError
        If the download fails.
    """
    import tempfile
    import requests
    import os

    if Path(url).is_file():
        return open(url, "rb")
    try:
        if target_file_name:
            fileobj = open(target_file_name, mode="wb")
        else:
            local_filename = url.split("/")[-1]
            suffix = "." + local_filename
            fileobj = tempfile.NamedTemporaryFile(suffix=suffix)
        with requests.get(url) as r:
            r.raise_for_status()  # Check for request success
            fileobj.write(r.content)
        if return_filename:
            return fileobj.name
        else:
            return fileobj
    except requests.exceptions.RequestException:
        logger.error("Failed to download %s", url, exc_info=True)
        raise IOError(f"ERROR: failed to down {url}")


def get_emdb_id(label: str) -> str | None:
    """Extract an EMDB identifier (e.g. EMD-1234) from a string.

    Parameters
    ----------
    label : str
        Input string.

    Returns
    -------
    str or None
        The matched EMDB ID, or None if not found.
    """
    import re

    pattern = r"(?i)(EMD[-_]\d{4,5})"
    match = re.search(pattern, str(label))
    if match:
        return match.group(1)
    return None


def is_file_readable(filename: str) -> bool:
    """Check whether a file exists and is readable.

    Parameters
    ----------
    filename : str
        Path to the file.

    Returns
    -------
    bool
        True if the file exists and is readable, False otherwise.
    """
    import os

    if not os.path.exists(filename):
        return False
    if os.path.isfile(filename):
        return os.access(filename, os.R_OK)
    else:
        return False


def is_file_writable(filename: str) -> bool:
    """Check whether a file (or its parent directory) is writable.

    Parameters
    ----------
    filename : str
        Path to check.

    Returns
    -------
    bool
        True if the file or its parent directory is writable.
    """
    import os

    if os.path.exists(filename):
        if os.path.isfile(filename):
            return os.access(filename, os.W_OK)
        else:
            return False
    pdir = os.path.dirname(filename)
    if not pdir:
        pdir = "."
    return os.access(pdir, os.W_OK)


def file_ready(
    filenames, wait: int = 0, minSize: int = 0
) -> int:  # wait given seconds and check again
    """Check whether a file or list of files exists and has content.

    If ``wait > 0``, polls until the file is ready or the deadline passes.

    Parameters
    ----------
    filenames : str or list of str
        File path(s) to check.
    wait : int, optional
        Maximum seconds to wait. Defaults to 0 (no wait).
    minSize : int, optional
        Minimum file size (bytes) required.

    Returns
    -------
    int
        1 if ready, 0 otherwise.
    """
    if type(filenames) == type([]):
        ready = 1
        for f in filenames:
            if not (os.path.exists(f) and os.path.getsize(f)):
                ready = 0
                break
        return ready
    else:
        ready = os.path.exists(filenames) and os.path.getsize(filenames) > minSize
    if ready:
        return 1
    elif wait > 0:
        deadline = time.time() + wait
        delay = 1
        while time.time() <= deadline:
            time.sleep(delay)
            ready = file_ready(filenames, wait=0, minSize=minSize)
            if ready:
                return 1
            else:
                delay *= 2
                now = time.time()
                if now + delay > deadline:
                    delay = deadline - now
        return file_ready(filenames, wait=0)
    else:
        return 0


def convert_file_path(
    filenames, to: str = "current", relpath_start: str = "."
) -> pd.Series:
    """Convert pandas Series of file paths between absolute, relative, and shortest forms.

    Parameters
    ----------
    filenames : pd.Series
        Series of file paths.
    to : str, optional
        Target format: "current", "absolute"/"abs"/"real",
        "relative"/"rel", or "shortest". Defaults to "current" (no-op).
    relpath_start : str, optional
        Base path for relative conversion.

    Returns
    -------
    pd.Series
        Converted file paths.
    """
    import pandas as pd

    if to == "current":
        return filenames
    assert to in "current absolute abs real relative rel shortest".split()
    assert isinstance(filenames, pd.Series)
    names = filenames.unique()
    mapping = {}

    for name in names:
        p = Path(name)
        p_abs = p.resolve()
        if to in ["real", "absolute", "abs"]:
            name2 = p_abs.as_posix()
        else:
            name2_rel = os.path.relpath(p_abs, relpath_start)
            if to in ["relative", "rel"]:
                name2 = name2_rel
            elif to == "shortest":
                if len(p_abs.as_posix()) < len(name2_rel):
                    name2 = p_abs.as_posix()
                else:
                    name2 = name2_rel
        if not (Path(name2).exists() or (Path(relpath_start) / Path(name2)).exists()):
            name2 = name
        mapping[name] = name2
    ret = filenames.map(mapping)
    return ret


def convert_dataframe_file_path(
    df, attr: str, to: str = "current", relpath_start: str = "."
) -> pd.Series:
    """Convert file paths in a DataFrame column, preserving stack indices.

    Handles cryoSPARC-style ``"stack_index@filename"`` format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column.
    attr : str
        Column name with file paths.
    to : str, optional
        Target path format (see :func:`convert_file_path`).
    relpath_start : str, optional
        Base path for relative conversion.

    Returns
    -------
    pd.Series
        Converted file paths, with stack indices re-attached.
    """
    if to == "current":
        return df[attr]
    if df[attr].iloc[0].find("@") != -1:
        tmp = df[attr].str.split("@", expand=True)
        indices, filenames = tmp.loc[:, 0], tmp.loc[:, 1]
        filenames = convert_file_path(filenames, to, relpath_start)
        ret = indices + "@" + filenames
    else:
        ret = convert_file_path(df[attr], to, relpath_start)
    return ret


def check_required_columns(data, required_cols: list | None = None) -> None:
    """Check that required columns exist in a cryoSPARC Dataset or DataFrame.

    Parameters
    ----------
    data : Dataset or DataFrame
        Data object to check.
    required_cols : list of str
        Column names that must be present.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    from cryosparc.dataset import Dataset

    if isinstance(data, Dataset):
        cols = data.fields()
    else:
        cols = data.columns
    missing_cols = [c for c in required_cols if c not in cols]
    if missing_cols:
        msg = f"required columns {' '.join(missing_cols)} are unavailable. Available columns are {' '.join(cols)}"
        logger.error(msg)
        raise ValueError(msg)
