"""Handler for the selectCommonHelices option."""

from __future__ import annotations
import helicon
import os
from helicon.lib.exceptions import HeliconError
import logging

logger = logging.getLogger(__name__)


option_name = "selectCommonHelices"


def add_args(parser):
    parser.add_argument(
        "--selectCommonHelices",
        type=str,
        metavar="starFile",
        action="append",
        help="select helices in the specified file (example: x.star) based on rlnMicrographName and rlnHelicalTubeID. disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the selectCommonHelices option.

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
    if len(param) > 0:
        # starfile
        sf, _ = helicon.parse_param_str(param)
        assert "rlnMicrographName" in data
        assert "rlnHelicalTubeID" in data

        if sf is None or not os.path.exists(sf):
            raise HeliconError(
                "\tERROR: option --selectCommonHelices %s has specified a non-existent file %s"
                % (param, sf)
            )
        data_sf = helicon.images2dataframe(
            sf,
            alternative_folders=args.folder,
            ignore_bad_particle_path=args.ignoreBadParticlePath,
            ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
            warn_missing_ctf=0,
            target_convention="relion",
        )
        if args.verbose > 1:
            logger.info("\t%d images found in %s" % (len(data_sf), sf))
        assert "rlnMicrographName" in data_sf
        assert "rlnHelicalTubeID" in data_sf

        # import tqdm
        # idx=[row in set(zip(data_sf["rlnMicrographName"],data["rlnHelicalTubeID"])) for row in tqdm.tqdm(list(zip(data["rlnMicrographName"],data["rlnHelicalTubeID"])))]
        # data2=data[idx]
        common_cols = ["rlnMicrographName", "rlnHelicalTubeID"]
        data2 = data.merge(
            data_sf[common_cols], on=common_cols, how="inner", suffixes=["", "_dup"]
        )
        data2 = data2[data.columns].drop_duplicates()

        data2.reset_index(drop=True, inplace=True)
        data2.attrs["optics"] = data.attrs.get("optics")

        if len(data2):
            if args.verbose > 1:
                logger.info(f"\t{len(data2)}/{len(data)} images retained")
            data = data2
        else:
            inputFileStr = (
                args.input_imageFiles
                if len(args.input_imageFiles) > 1
                else args.input_imageFiles[0]
            )
            data_ci = data.columns.get_loc("rlnMicrographName")
            data_sf_ci = data_sf.columns.get_loc("rlnMicrographName")
            logger.info(
                (
                    "\t%s %s: %s"
                    % (inputFileStr, "rlnMicrographName", data.iat[0, data_ci])
                )
            )
            logger.info(
                ("\t%s %s: %s" % (sf, "rlnMicrographName", data_sf.iat[0, data_sf_ci]))
            )
            raise HeliconError(
                "no common image found. Check if the files %s and %s include particles in the same folder"
                % (inputFileStr, sf)
            )
        index_d[option_name] += 1
    return data, index_d
