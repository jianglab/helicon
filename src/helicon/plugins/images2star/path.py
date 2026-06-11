"""Handler for the path option."""

from __future__ import annotations
import helicon
from pathlib import Path


option_name = "path"


def add_args(parser):
    parser.add_argument(
        "--path",
        metavar="<absolute|relative|real|shortest|current>",
        type=str,
        choices=["absolute", "relative", "real", "shortest", "current"],
        help="which type of file path is used for the images. default to current",
        default="current",
    )


def handle(data, args, index_d, param):
    """Handle the path option.

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
    if param != "current":
        path = param
        from helicon import convert_dataframe_file_path

        for attr in "rlnImageName rlnMicrographName rlnMovieName".split():
            if attr in data:
                from helicon import get_relion_project_folder

                output_star = Path(args.output_starFile).resolve()
                relion_proj_folder = get_relion_project_folder(str(output_star))
                relpath_start = (
                    str(output_star.parent)
                    if relion_proj_folder is None
                    else relion_proj_folder
                )
                data[attr] = convert_dataframe_file_path(
                    data, attr, to=path, relpath_start=relpath_start
                )
        index_d[option_name] += 1
    return data, index_d


import logging

logger = logging.getLogger(__name__)
