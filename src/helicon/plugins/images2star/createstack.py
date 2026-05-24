"""Handler for the createStack option."""

from __future__ import annotations
import logging
import helicon
import numpy as np
from pathlib import Path
import os
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

        if os.path.splitext(outputFile)[1] != ".mrcs":
            suffix = Path(outputFile).suffix
            logger.error(
                "a .mrcs file is expected while you have specified %s! I will not do anything",
                outputFile,
            )
            return data, index_d

        images = data["rlnImageName"].str.split("@", expand=True)
        images.columns = ["pid", "filename"]
        images.loc[:, "pid"] = images.loc[:, "pid"].astype(int)

        attr = helicon.unique_attr_name(data, attr_prefix="rlnImageNameOrig")
        data[attr] = data["rlnImageName"]

        nx, ny, _ = helicon.get_image_size(images["filename"].iloc[0])
        nImage = len(data)

        newsize = int(param_dict.get("rescale2size", nx))
        float16 = int(param_dict.get("float16", 1))

        import mrcfile

        force = int(param_dict.get("force", 0))
        if not force:
            if os.path.exists(outputFile):
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
                apix0 = None
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
                    if apix0 is None:
                        apix0 = d["apix_x"]
                    if newsize < nx:
                        d = d.FourTruncate(
                            newsize, newsize, 1, 1
                        )  # crop Fourier transform
                    elif newsize > nx:
                        d = d.FourInterpol(
                            newsize, newsize, 1, 1
                        )  # pad Fourier transform
                    d_numpy = helicon.EMNumPy.em2numpy(d)
                    mrc.data[i, :, :] = d_numpy
                mrc.voxel_size = apix0 * nx / newsize
        images.loc[:, "pid"] = np.arange(nImage) + 1
        data.loc[:, "rlnImageName"] = images["pid"].astype(str) + "@" + outputFile
        if optics is not None and newsize != nx:
            optics.loc[:, "rlnImageSize"] = newsize
            if "rlnImagePixelSize" in optics:
                optics.loc[:, "rlnImagePixelSize"] = (
                    optics.loc[:, "rlnImagePixelSize"] * nx / newsize
                )
        index_d[option_name] += 1
    return data, index_d
