#!/usr/bin/env python

"""A command line tool that anaylzes/transforms 3D maps"""

from __future__ import annotations
import argparse, logging, sys
from helicon.lib.exceptions import (
    HeliconError,
    HeliconValidationError,
    HeliconFileExistsError,
)
from pathlib import Path
import numpy as np
import mrcfile
import helicon

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Analyze and transform a 3D map.

    Applies operations (flip hand, clip, FFT resample, helical symmetry,
    z-moving average) to the input map and writes the result.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    helicon.log_command_line()
    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose > 2
            else logging.INFO if args.verbose > 0 else logging.ERROR
        ),
        format="%(message)s",
        stream=sys.stdout,
    )

    if args.cpu < 1:
        args.cpu = helicon.available_cpu()

    with mrcfile.open(args.inputMapFile, mode="r") as mrc:
        data = mrc.data
        nz, ny, nx = data.shape
        apix = round(float(mrc.voxel_size.x), 4)
        if args.verbose > 0:
            logger.info(
                "Input map: %s\n\tnx,ny,nz=%d,%d,%d pixels\tsampling=%g \u00c5/pixel",
                args.inputMapFile,
                nx,
                ny,
                nz,
                apix,
            )

    if args.verbose > 1:
        title = f"{args.inputMapFile}: {nx}x{ny}x{nz} pixels | apix={apix}\u00c5/pixel"
        helicon.display_map_orthoslices(data, title=title, hold=False)

    index_d = {o: 0 for o in args.all_options}

    for option_name in args.all_options:
        if option_name in args.append_options:
            param = args.__dict__[option_name][index_d[option_name]]
        else:
            param = args.__dict__[option_name]

        if args.verbose:
            logger.info("%s: %s", option_name, param)

        from helicon.plugins.proc3d import dispatch

        data, apix, nx, ny, nz = dispatch(
            option_name, data, args, index_d, param, apix, nx, ny, nz
        )

        index_d[option_name] += 1

    if args.verbose > 1:
        logger.info(
            "Output map: %s\n\tnx,ny,nz=%d,%d,%d pixels\tsampling=%g \u00c5/pixel",
            str(args.outputMapFile),
            nx,
            ny,
            nz,
            apix,
        )

    with mrcfile.new(
        args.outputMapFile, data=data.astype(np.float32), overwrite=args.force
    ) as mrc:
        mrc.voxel_size = apix

    if args.verbose > 1:
        title = f"{str(args.outputMapFile)}: {nx}x{ny}x{nz} pixels | apix={apix}\u00c5/pixel"
        helicon.display_map_orthoslices(data, title=title, hold=True)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for the proc3d command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to attach arguments to.
    """
    parser.add_argument(
        "inputMapFile",
        type=str,
        metavar="<inputMapFile>",
        help="input 3D map file in MRC format",
    )

    parser.add_argument(
        "outputMapFile",
        type=str,
        nargs="?",
        metavar="<outputMapFile>",
        default=None,
        help="output 3D map file (optional, auto-generated from input if omitted)",
    )

    parser.add_argument(
        "--outputMapFile",
        type=str,
        dest="outputMapFile_opt",
        metavar="<filename>",
        help=argparse.SUPPRESS,
        default="",
    )

    parser.add_argument(
        "--force",
        type=int,
        metavar="<0|1>",
        help="force overwrite the output file. default to %(default)s",
        default=0,
    )

    parser.add_argument(
        "--verbose",
        type=int,
        metavar="<0|1|2>",
        help="verbose mode. default to %(default)s",
        default=2,
    )

    parser.add_argument(
        "--cpu",
        type=int,
        metavar="<n>",
        help="number of cpus to use. default to %(default)s and use all idle cpus",
        default=-1,
    )

    from helicon.plugins.proc3d import add_plugin_args

    add_plugin_args(parser)

    return parser


def check_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Validate and prepare proc3d arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    parser : argparse.ArgumentParser
        The argument parser (used to discover append actions).

    Returns
    -------
    argparse.Namespace
        The validated arguments.
    """
    args.append_options = [
        a.dest for a in parser._actions if type(a) is argparse._AppendAction
    ]
    all_options = helicon.get_option_list(sys.argv[1:])
    args.all_options = [
        o
        for o in all_options
        if o not in "cpu force inputMapFile outputMapFile_opt verbose".split()
    ]

    if args.outputMapFile is not None:
        args.outputMapFile = Path(args.outputMapFile)
    elif args.outputMapFile_opt:
        args.outputMapFile = Path(args.outputMapFile_opt)
    else:
        args.outputMapFile = Path(args.inputMapFile).with_suffix(".proc3d.mrc")

    if args.outputMapFile.exists() and not args.force:
        raise HeliconFileExistsError(
            f"output file {str(args.outputMapFile)} already exists. Use --force to overwrite it"
        )

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    args = check_args(args, parser)
    main(args)
