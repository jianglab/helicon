"""Handler for the selectFile option."""

from __future__ import annotations
import os, logging
import helicon
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)

option_name = "selectFile"


def add_args(parser):
    parser.add_argument(
        "--selectFile",
        type=str,
        metavar="starFile:col1=<name>:col2=<name>:pattern=<str>",
        action="append",
        help=(
            "select images whose <col1> is present in the specified star file's <col2>. "
            "Example: ref.star:col1=rlnImageName:col2=rlnImageName:pattern=(.+)"
        ),
        default=[],
    )


def _select_by_file(data, col1, sids, pattern, invert=False):
    """Filter data based on values present in sids.

    Parameters
    ----------
    data : pd.DataFrame
    col1 : str
        Column in data to match.
    sids : pd.Series
        Values to match against (already cleaned).
    pattern : str or None
        Regex pattern to extract from values before matching.
    invert : bool
        If True, exclude matching rows (excludeFile behavior).
        If False, keep only matching rows (selectFile behavior).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    from helicon import convert_dataframe_file_path

    dids = convert_dataframe_file_path(data, col1, to="abs")
    dids = dids.apply(lambda row: row.lstrip("0"))
    sids = sids.apply(lambda row: row.lstrip("0"))

    if pattern:
        dids = dids.str.extract(pattern, expand=False)
        sids = sids.str.extract(pattern, expand=False)

    if invert:
        dids = dids[~dids.isin(sids)]
    else:
        dids = dids[dids.isin(sids)]

    return data.loc[dids.index, :].reset_index(drop=True)


def handle(data, args, index_d, param):
    """Handle the selectFile option.

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
                "\tERROR: option --selectFile has specified a non-existent file %s" % sf
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

        data2 = _select_by_file(data, col1, sids, pattern, invert=False)

        if len(data2):
            if args.verbose > 1:
                logger.info("\t%d/%d images retained", len(data2), len(data))
            data = data2
        else:
            inputFileStr = (
                args.input_imageFiles[0]
                if len(args.input_imageFiles) > 1
                else args.input_imageFiles[0]
            )
            raise HeliconError(
                "\tERROR: no common image found. Check if the files %s and %s include particles in the same folder"
                % (inputFileStr, sf)
            )
        index_d[option_name] += 1
    return data, index_d
