"""Handler for the maskGold option."""

from __future__ import annotations
import helicon
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
                "ERROR: the input does not have any of the columns: {' '.john(attrs_required)}"
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
                        "ERROR: output {outputFile.as_posix()} will overwrite original image"
                    )
                if not force:
                    import mrcfile

                    with mrcfile.mmap(outputFile.as_posix()) as mrc:
                        n = mrc.header.nz.item()
                        if n == len(mgraphParticles):
                            if n > 1 or attrSrc in ["rlnImageName"]:
                                data.loc[mgraphParticles.index, attSrc] = (
                                    pd.Series(list(range(1, n + 1))).map(
                                        "{:06d}".format
                                    )
                                    + "@"
                                    + outputFile.as_posix()
                                ).tolist()
                            else:
                                data.loc[mgraphParticles.index, attSrc] = (
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
                data.loc[indices, "rlnImageName"] = (
                    pd.Series(list(range(1, len(indices) + 1))).map("{:06d}".format)
                    + "@"
                    + newImageFile.as_posix()
                ).tolist()

        data.drop(["tmp_mgraph_name", "tmp_mgraph_pid"], inplace=True, axis=1)
        index_d[option_name] += 1
    return data, index_d
