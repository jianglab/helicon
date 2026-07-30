"""Generic image-loading utilities.

Shared between the denovo3D and HILL tabs (and potentially others).
Provides URL/file/EMDB loading with caching.
"""

from __future__ import annotations

import helicon
from pathlib import Path


def get_images_from_url(url: str) -> tuple:
    """Load a 2D/3D image from a URL or local file path.

    Parameters
    ----------
    url : str
        Remote URL or local file path.  Cloud-drive indirect URLs are
        automatically converted to direct download links.

    Returns
    -------
    tuple
        ``(data_array, apix_in_angstroms)``
    """
    url_final = helicon.get_direct_url(url)
    fileobj = helicon.download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud "
            f"drive file, make sure the link is a direct download link instead of "
            f"a link for preview"
        )
    data, apix = get_images_from_file(fileobj.name)
    return data, apix


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "denovo3D"), expires_after=7, verbose=0
)
def get_images_from_emdb(emdb_id: str) -> tuple:
    """Download a map from EMDB by ID.

    Parameters
    ----------
    emdb_id : str
        EMDB identifier (e.g. "10499" or "EMD-10499").

    Returns
    -------
    tuple
        ``(data_array, apix_in_angstroms)``
    """
    emdb = helicon.dataset.EMDB()
    data, apix = emdb(emdb_id)
    if data is None:
        raise IOError(f"ERROR: failed to download {emdb_id} from EMDB")
    return data, round(apix, 4)


def get_images_from_file(image_file: str) -> tuple:
    """Read an MRC/MRCS file and return the image data and pixel size.

    Parameters
    ----------
    image_file : str
        Path to an .mrc, .mrcs, .map, or .map.gz file.

    Returns
    -------
    tuple
        ``(data_array, apix_in_angstroms)``
    """
    import mrcfile

    with mrcfile.open(image_file) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, round(apix, 4)
