"""Handler for the excludeFile option."""

from __future__ import annotations
import os, logging
import helicon
from helicon.lib.exceptions import HeliconError
from .selectFile import _select_by_file

logger = logging.getLogger(__name__)

option_name = "excludeFile"


def add_args(parser):
    parser.add_argument(
        "--excludeFile",
        type=str,
        metavar="starFile:col1=<name>:col2=<name>:pattern=<str>",
        action="append",
        help=(
            "exclude images whose <col1> is present in the specified star file's <col2>. "
            "Example: ref.star:col1=rlnImageName:col2=rlnImageName:pattern=(.+)"
        ),
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the excludeFile option.

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
        sf, param_dict = helicon.parse_param_str(param)
        col1 = param_dict.get("col1", "rlnImageName")
        col2 = param_dict.get("col2", "rlnImageName")
        if col1 not in data:
            raise HeliconError(
                "\tERROR: column '%s' not found in data. Available columns: %s"
                % (col1, list(data.columns))
            )
        pattern = param_dict.get("pattern", None)

        if not os.path.exists(sf):
            raise HeliconError(
                "\tERROR: option --excludeFile has specified a non-existent file %s"
                % sf
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
            logger.info("\t%d images found in %s", len(data_sf), sf)

        if col2 not in data_sf:
            raise HeliconError(
                "\tERROR: column '%s' not found in %s. Available columns: %s"
                % (col2, sf, list(data_sf.columns))
            )

        from helicon import convert_dataframe_file_path

        sids = convert_dataframe_file_path(data_sf, col2, to="abs")

        data2 = _select_by_file(data, col1, sids, pattern, invert=True)

        if len(data2):
            if args.verbose > 1:
                logger.info("\t%d/%d images retained", len(data2), len(data))
            data = data2
        else:
            if args.verbose:
                logger.info(
                    "\tnothing to do when there is no particle image left. I will quit"
                )
        index_d[option_name] += 1
    return data, index_d
