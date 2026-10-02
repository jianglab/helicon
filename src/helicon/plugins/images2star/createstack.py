"""Handler for the createStack option."""

from __future__ import annotations
import logging
import helicon
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mrcfile

logger = logging.getLogger(__name__)


option_name = "createStack"


def add_args(parser):
    parser.add_argument(
        "--createStack",
        dest="createStack",
        type=str,
        metavar="output.mrcs:rescale2size=<n>:float16=<0|1>:force=<0|1>",
        help="create a new mrcs file to store all particles",
        default=None,
    )


def handle(data, args, index_d, param):
    """Handle the createStack option.

    Parameters
    ----------
    data : pd.DataFrame
        The particle data DataFrame.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (data, index_d) after processing.
    """
    if param:
        # outputFile:rescale2size=<n>:float16=<0|1>
        outputFile, param_dict = helicon.parse_param_str(param)

        if Path(outputFile).suffix != ".mrcs":
            suffix = Path(outputFile).suffix
            logger.error(
                "a .mrcs file is expected while you have specified %s! I will not do anything",
                outputFile,
            )
            return data, index_d

        images = data["rlnImageName"].str.split("@", expand=True)
        images.columns = ["pid", "filename"]
        images["pid"] = images["pid"].astype(int)

        attr = helicon.unique_attr_name(data, attr_prefix="rlnImageNameOrig")
        data[attr] = data["rlnImageName"]

        nx, ny, _ = helicon.get_image_size(images["filename"].iloc[0])
        nImage = len(data)

        newsize = int(param_dict.get("rescale2size", nx))
        float16 = int(param_dict.get("float16", 1))

        import mrcfile

        force = int(param_dict.get("force", 0))
        if not force:
            if Path(outputFile).exists():
                with mrcfile.open(outputFile, header_only=True) as mrc:
                    if not (
                        mrc.header.nx == newsize
                        and mrc.header.ny == newsize
                        and mrc.header.nz == nImage
                    ):
                        force = 1
            else:
                force = 1
        if force:
            from tqdm import tqdm

            if float16:
                mrc_mode = 12
            else:
                mrc_mode = 2
            with mrcfile.new_mmap(
                outputFile,
                shape=(nImage, newsize, newsize),
                mrc_mode=mrc_mode,
                fill=None,
                overwrite=True,
            ) as mrc:
                with mrcfile.open(images["filename"].iloc[0], header_only=True) as m:
                    apix0 = float(m.voxel_size.x)
                for i in tqdm(
                    list(range(nImage)), unit=" particles", disable=args.verbose > 1
                ):
                    if args.verbose > 1:
                        logger.info(
                            "\t%d/%d: adding %s:%d"
                            % (
                                i + 1,
                                nImage,
                                images["filename"].iloc[i],
                                images["pid"].iloc[i],
                            )
                        )
                    d = helicon.read_image_2d(
                        images["filename"].iloc[i], int(images["pid"].iloc[i] - 1)
                    )
                    if newsize != nx:
                        d = resize_image(d, newsize)
                    mrc.data[i, :, :] = d
                mrc.voxel_size = apix0 * nx / newsize
        images["pid"] = np.arange(nImage) + 1
        data["rlnImageName"] = images["pid"].astype(str) + "@" + outputFile
        optics = data.attrs.get("optics")
        if optics is not None and newsize != nx:
            optics = optics.copy()
            optics.loc[:, "rlnImageSize"] = newsize
            if "rlnImagePixelSize" in optics:
                optics.loc[:, "rlnImagePixelSize"] = (
                    optics.loc[:, "rlnImagePixelSize"] * nx / newsize
                )
            data.attrs["optics"] = optics
        index_d[option_name] += 1
    return data, index_d


def resize_image(image: np.ndarray, newsize: int) -> np.ndarray:
    """Resize a square image by cropping or zero-padding its Fourier transform.

    Parameters
    ----------
    image : np.ndarray
        2D square image.
    newsize : int
        The new image size in pixels.

    Returns
    -------
    np.ndarray
        The ``newsize`` x ``newsize`` image (float32) with the same mean value.
    """
    ny, nx = image.shape
    fft = np.fft.fftshift(np.fft.fft2(image))
    out = np.zeros((newsize, newsize), dtype=complex)
    # copy the central (lowest-frequency) part that both boxes have
    m_y, m_x = min(ny, newsize), min(nx, newsize)
    src_y, src_x = ny // 2 - m_y // 2, nx // 2 - m_x // 2
    dst = newsize // 2 - m_y // 2, newsize // 2 - m_x // 2
    out[dst[0] : dst[0] + m_y, dst[1] : dst[1] + m_x] = fft[
        src_y : src_y + m_y, src_x : src_x + m_x
    ]
    ret = np.fft.ifft2(np.fft.ifftshift(out)).real * (newsize * newsize) / (nx * ny)
    return ret.astype(np.float32)
