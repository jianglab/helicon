"""Handler for the maskGold option."""

from __future__ import annotations
import helicon
from helicon.lib.exceptions import HeliconError
import numpy as np
import pandas as pd
from pathlib import Path
import mrcfile
import logging

logger = logging.getLogger(__name__)


option_name = "maskGold"


def add_args(parser):
    parser.add_argument(
        "--maskGold",
        metavar="value_sigma=<n>:gradient_sigma=<Å>:min_area=<Å^2>:both_sides=<0|1>:outdir=<str>:force=<0|1>:cpu=<n>",
        type=str,
        action="append",
        help="mask out electron dense (gold, ferritin, ice) pixels in images. disabled by default",
        default=None,
    )


def handle(data, args, index_d, param):
    """Handle the maskGold option.

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
        attrs_required = "rlnImageName rlnMicrographName".split()
        attrSrc = helicon.first_matched_attr(data, attrs_required)
        if attrSrc is None:
            raise HeliconError(
                f"the input does not have any of the columns: {' '.join(attrs_required)}"
            )

        # value_sigma=<n>:gradient_sigma=<Å>:min_area=<Å^2>:both_sides=<0|1>:outdir=<str>:force=<0|1>:cpu=<n>
        _, param_dict = helicon.parse_param_str(param)
        value_sigma = param_dict.get(
            "value_sigma", 4.0
        )  # value_sigma fold of mad above median
        gradient_sigma = param_dict.get(
            "gradient_sigma", 0
        )  # Å. 0 -> auto-decide, <0 -> disable
        min_area = param_dict.get("min_area", 100)  # Å^2
        both_sides = param_dict.get(
            "both_sides", 1
        )  # 0-remove large value pixels, 1-remove both large and small value pixels
        outdir = Path(param_dict.get("outdir", Path(args.output_starFile).stem))
        outdir.mkdir(parents=True, exist_ok=True)
        force = param_dict.get("force", 1)
        cpu = param_dict.get("cpu", 1)

        attr = helicon.unique_attr_name(data, attr_prefix=f"{attrSrc}Orig")
        data.loc[:, attr] = data[attrSrc]

        tmp = data[attrSrc].str.split("@", expand=True)
        data.loc[:, "tmp_mgraph_name"] = tmp.iloc[:, -1]
        if tmp.shape[1] > 1:
            data.loc[:, "tmp_mgraph_pid"] = tmp.iloc[:, 0]
        else:
            data.loc[:, "tmp_mgraph_pid"] = 1
        mgraphs = data.groupby("tmp_mgraph_name", sort=False)

        if gradient_sigma == 0:
            import mrcfile

            with mrcfile.mmap(data["tmp_mgraph_name"].values[0]) as mrc:
                ny, nx = mrc.data.shape[-2:]
                apix = mrc.voxel_size.x
            if ny > 2048 and nx > 2048:
                gradient_sigma = np.sqrt(min_area) * 10
                if args.verbose > 1:
                    logger.info(
                        f"\tgradient_sigma is set to {gradient_sigma:.1f} Å to remove brightness gradient of the micrographs ({nx}x{ny} pixels)"
                    )

        tasks = []
        for mi, (mgraphName, mgraphParticles) in enumerate(mgraphs):
            pid = mgraphParticles["tmp_mgraph_pid"].astype(int) - 1
            outputFile = Path(outdir) / Path(mgraphName).name
            if outputFile.exists():
                if outputFile.samefile(mgraphName):
                    raise HeliconError(
                        f"output {outputFile.as_posix()} will overwrite original image"
                    )
                if not force:
                    import mrcfile

                    with mrcfile.mmap(outputFile.as_posix()) as mrc:
                        n = mrc.header.nz.item()
                        if n == len(mgraphParticles):
                            if n > 1 or attrSrc in ["rlnImageName"]:
                                data.loc[mgraphParticles.index, attrSrc] = (
                                    pd.Series(list(range(1, n + 1))).map(
                                        "{:06d}".format
                                    )
                                    + "@"
                                    + outputFile.as_posix()
                                ).tolist()
                            else:
                                data.loc[mgraphParticles.index, attrSrc] = (
                                    outputFile.as_posix()
                                )
                            if args.verbose > 1:
                                if attrSrc in ["rlnMicrographName"]:
                                    logger.info(
                                        f"\tMicrograph {mi+1}/{len(mgraphs)}: {mgraphName} -> {outputFile.as_posix()} already done. skipped"
                                    )
                                else:
                                    logger.info(
                                        f"\tMicrograph {mi+1}/{len(mgraphs)}: {n} particles from {mgraphName} -> {outputFile.as_posix()} already done. skipped"
                                    )
                            continue
            if args.verbose > 1:
                if attrSrc in ["rlnImageName"]:
                    msg = f"\tMicrograph {mi+1}/{len(mgraphs)}: {len(mgraphParticles)} particles from {mgraphName} -> {outputFile.as_posix()}"
                else:
                    msg = f"\tMicrograph {mi+1}/{len(mgraphs)}: {mgraphName} -> {outputFile.as_posix()}"
            else:
                msg = None
            tasks.append((mgraphParticles, outputFile, msg))

        if tasks:
            if args.verbose > 2:
                logger.info(f"\tStart maskGold task for {len(tasks)} micrographs")
            from joblib import Parallel, delayed

            results = Parallel(
                n_jobs=cpu, verbose=max(0, args.verbose - 2), prefer="threads"
            )(
                delayed(maskGold_process_one_micrograph)(
                    t[0],
                    t[1],
                    value_sigma,
                    gradient_sigma,
                    min_area,
                    both_sides,
                    t[2],
                    max(0, args.verbose - 2),
                )
                for t in tasks
            )
            for result in results:
                indices, newImageFile = result
                if len(indices) > 1 or attrSrc in ["rlnImageName"]:
                    data.loc[indices, attrSrc] = (
                        pd.Series(list(range(1, len(indices) + 1))).map("{:06d}".format)
                        + "@"
                        + newImageFile.as_posix()
                    ).tolist()
                else:
                    data.loc[indices, attrSrc] = newImageFile.as_posix()

        data.drop(["tmp_mgraph_name", "tmp_mgraph_pid"], inplace=True, axis=1)
        index_d[option_name] += 1
    return data, index_d


def find_gold_mask(
    data: np.ndarray, value_sigma: float = 4, min_area: float = 200, both_sides: int = 1
) -> np.ndarray:
    """Find the electron dense (gold, ferritin, ice) regions of an image.

    Parameters
    ----------
    data : np.ndarray
        2D image.
    value_sigma : float, optional
        Pixels more than this many MADs away from the median are seeds of the
        dense regions. Defaults to 4.
    min_area : float, optional
        Smallest region (pixels) to keep. Defaults to 200.
    both_sides : int, optional
        1: mask both large and small value pixels; 0: large value pixels only.
        Defaults to 1.

    Returns
    -------
    np.ndarray
        Integer label image; 0 for the pixels to keep.
    """
    from skimage.filters import gaussian
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    from skimage.measure import label
    from skimage.morphology import remove_small_objects
    from scipy.stats import median_abs_deviation

    data = gaussian(data, sigma=0.25 * np.sqrt(min_area), mode="reflect")
    data = rescale_intensity(data, out_range=(-1, 1))
    median = np.median(data)
    mad = median_abs_deviation(data, axis=None)
    markers = np.zeros(data.shape, dtype=np.uint8)
    if both_sides:  # pixels with large values or small values
        markers[np.abs(data - median) < (value_sigma - 1.0) * mad] = 1
        markers[data > median + value_sigma * mad] = 2
        markers[data < median - value_sigma * mad] = 3
    else:  # pixels with large values only
        markers[data < median + (value_sigma - 1.0) * mad] = 1
        markers[data > median + value_sigma * mad] = 2
    labels = random_walker(data, markers, beta=10, mode="bf").astype(np.uint8)
    labels[labels < 2] = 0
    labels[labels >= 2] = 1
    labels = label(labels)
    labels = remove_small_objects(labels, min_size=min_area)
    return labels


def maskGold_process_one_micrograph(
    mgraphParticles: pd.DataFrame,
    outputFile: Path,
    value_sigma: float = 4,
    gradient_sigma: float = 50,
    min_area: float = 200,
    both_sides: int = 1,
    msg: str | None = None,
    verbose: int = 0,
) -> tuple[pd.Index, Path]:
    """Mask the electron dense pixels of the images of one micrograph/stack file.

    Parameters
    ----------
    mgraphParticles : pd.DataFrame
        Rows of one image file, with ``tmp_mgraph_name`` (file) and
        ``tmp_mgraph_pid`` (1-based image index) columns.
    outputFile : Path
        Output image file.
    value_sigma : float, optional
        See :func:`find_gold_mask`. Defaults to 4.
    gradient_sigma : float, optional
        Gaussian sigma (Å) used to remove brightness gradients; <=0 disables it.
        Defaults to 50.
    min_area : float, optional
        Smallest dense region to mask (Å^2). Defaults to 200.
    both_sides : int, optional
        See :func:`find_gold_mask`. Defaults to 1.
    msg : str, optional
        Message to log before processing.
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    tuple of (pd.Index, Path)
        Row index of ``mgraphParticles`` and ``outputFile``; image k of the
        output file is the k-th row.
    """
    from tqdm import tqdm

    n = len(mgraphParticles)
    if msg:
        logger.info(msg)
    with mrcfile.open(mgraphParticles["tmp_mgraph_name"].values[0]) as mrc_input:
        apix = float(mrc_input.voxel_size.x)
        apix2 = apix * apix
        is_stack = mrc_input.data.ndim == 3
        ny, nx = mrc_input.data.shape[-2:]
        if is_stack:
            data_out = np.zeros((n, ny, nx), dtype=np.float32)
            unit = " particles"
        else:
            if n != 1:
                raise HeliconError(
                    f"{mgraphParticles['tmp_mgraph_name'].values[0]} has 1 image but {n} rows refer to it"
                )
            data_out = np.zeros((ny, nx), dtype=np.float32)
            unit = " micrograph"

        for i, row in enumerate(
            tqdm(mgraphParticles.itertuples(), unit=unit, disable=verbose != 1)
        ):
            pid = int(row.tmp_mgraph_pid) - 1
            if is_stack:
                data = mrc_input.data[pid].astype(np.float32)
            else:
                data = mrc_input.data.astype(np.float32)
            if gradient_sigma > 0:
                import skimage.filters

                data_blurred = skimage.filters.gaussian(
                    data, sigma=gradient_sigma / apix, mode="reflect"
                )
                data -= data_blurred
                data += np.median(data_blurred)
            goldmask = find_gold_mask(data, value_sigma, min_area / apix2, both_sides)
            if verbose > 1:
                from skimage.measure import regionprops

                areas = sorted([r.area * apix2 for r in regionprops(goldmask)])
                if areas:
                    logger.info(
                        f"\t{outputFile.as_posix()} {i+1}/{n}: {np.count_nonzero(goldmask)*apix2:.1f} Å^2 in {len(areas)} regions ({areas[0]:.1f} - {areas[-1]:.1f} Å^2) are masked"
                    )
                else:
                    logger.info(f"\t{outputFile.as_posix()} {i+1}/{n}: nothing to mask")

            nonzeros = goldmask > 0
            if np.count_nonzero(nonzeros) > 0:
                data[nonzeros] = np.mean(data[~nonzeros])  # mean of non-gold pixels
            if is_stack:
                data_out[i] = data
            else:
                data_out = data
    with mrcfile.new(outputFile.as_posix(), data=data_out, overwrite=True) as mrc:
        mrc.voxel_size = apix
    return (mgraphParticles.index, outputFile)
