"""MRC image file I/O operations."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import numpy as np
from .exceptions import HeliconIOError

__all__ = [
    "change_map_axes_order",
    "display_map_orthoslices",
    "get_image_number",
    "get_image_size",
    "read_image_2d",
]


def get_image_number(imageFile: str, as2D: bool = False) -> int:
    """Get the number of images in an MRC file.

    Parameters
    ----------
    imageFile :
        Path to the MRC file.
    as2D :
        If True, return the number of 2D slices; otherwise return 1.

    Returns
    -------
    int
        Number of images in the file.
    """
    if not Path(imageFile).exists():
        raise HeliconIOError(f"cannot find image file {imageFile}")
    import mrcfile

    with mrcfile.open(imageFile, header_only=True) as mrc:
        if as2D:
            n = mrc.header.nz
        else:
            n = 1
    return n


def get_image_size(imageFile: str) -> tuple[int, int, int]:
    """Get the dimensions of an MRC image file.

    Parameters
    ----------
    imageFile :
        Path to the MRC file.

    Returns
    -------
    tuple of int
        Tuple of (nx, ny, nz) dimensions.
    """
    if not Path(imageFile).exists():
        raise HeliconIOError(f"cannot find image file {imageFile}")
    import mrcfile

    with mrcfile.open(imageFile, header_only=True) as mrc:
        nz = mrc.header.nz
        ny = mrc.header.ny
        nx = mrc.header.nx
    return (int(nx), int(ny), int(nz))


def read_image_2d(imageFile: str, i: int) -> np.ndarray:
    """Read a single 2D slice from an MRC stack.

    Parameters
    ----------
    imageFile :
        Path to the MRC file.
    i :
        Index of the slice to read.

    Returns
    -------
    np.ndarray
        2D numpy array of the image slice.
    """
    if not Path(imageFile).exists():
        raise HeliconIOError(f"cannot find image file {imageFile}")
    i = int(i)
    import mrcfile

    with mrcfile.open(imageFile) as mrc:
        nz = mrc.header.nz
        if 0 <= i < nz:
            return mrc.data[i]
        else:
            raise HeliconIOError(
                f"the requested image {i} is out of the valid range [0, {nz}) for image file {imageFile}"
            )


def change_map_axes_order(
    data: np.ndarray, header: Any, new_axes: list[str] | None = None
) -> tuple[np.ndarray, Any]:
    """Reorder the axes of a 3D map according to the MRC header axis mapping.

    Parameters
    ----------
    data :
        3D numpy array of the map data.
    header :
        MRC file header object with mapc/mapr/maps attributes.
    new_axes :
        Desired axis order as a list of "x", "y", "z". Defaults to ["x", "y", "z"].

    Returns
    -------
    tuple of (np.ndarray, Any)
        Tuple of (reordered_data, updated_header) with axes permuted accordingly.
    """
    if new_axes is None:
        new_axes = ["x", "y", "z"]
    map_axes = {"x": 0, "y": 1, "z": 2}
    try:
        current_axes_int = [header.mapc - 1, header.mapr - 1, header.maps - 1]
    except AttributeError:
        current_axes_int = [0, 1, 2]
    new_axes_int = [map_axes[a] for a in new_axes]
    data2 = np.moveaxis(data, current_axes_int, new_axes_int)
    header2 = header.copy()
    header2.mapc = new_axes_int[0] + 1
    header2.mapr = new_axes_int[1] + 1
    header2.maps = new_axes_int[2] + 1
    return data2, header2


def display_map_orthoslices(data: np.ndarray, title: str, hold: bool = False) -> None:
    """Display orthogonal slices through the center of a 3D volume.

    Parameters
    ----------
    data :
        3D numpy array of the volume data.
    title :
        Title string for the figure.
    hold :
        If True, block until the figure window is closed. Defaults to False.
    """
    if not sys.__stdin__.isatty():
        return
    nz, ny, nx = data.shape
    sz = data[nz // 2, :, :]
    sy = data[:, ny // 2, :]
    sx = data[:, :, nx // 2]
    images = [sx, sy, sz]
    titles = ["X=%d" % (nx // 2), "Y=%d" % (ny // 2), "Z=%d" % (nz // 2)]

    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 3.5), facecolor="w", edgecolor="w")
    fig.suptitle(title, fontsize=16)
    for i in range(3):
        sub = fig.add_subplot(1, 3, i + 1)
        sub.imshow(images[i], cmap="gray", aspect="equal", interpolation="bicubic")
        sub.set_title(titles[i], fontsize=12)
    fig.tight_layout()
    if hold:
        plt.show()
    else:
        plt.draw()
    plt.pause(0.05)
