#!/usr/bin/env python

"""A command line tool that analyzes/transforms dataset(s) and saves the dataset in a RELION star file"""

from __future__ import annotations
import argparse, logging, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from helicon.lib.exceptions import (
    HeliconError,
    HeliconValidationError,
    HeliconFileExistsError,
)

logger = logging.getLogger(__name__)

pd.options.mode.copy_on_write = True

import helicon
from helicon.lib.io import getPixelSize, setPixelSize, pixelSizeAttrForImageAttr
from helicon.lib.analysis import estimate_inter_segment_distance
from helicon.plugins.images2star import dispatch


def main(args: argparse.Namespace) -> None:
    """Analyze and transform image datasets, saving to a RELION STAR file.

    Reads images from STAR/mrcs/lst files, applies filters/transformations,
    and writes the result as a STAR file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    helicon.log_command_line()
    if args.verbose <= 0:
        level = logging.ERROR
    elif args.verbose == 1:
        level = logging.WARNING
    elif args.verbose == 2:
        level = logging.INFO
    else:
        level = logging.DEBUG
    from rich.logging import RichHandler

    _handler = RichHandler(show_time=False, show_path=False, rich_tracebacks=True)
    _handler.setLevel(level)
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if args.cpu < 1:
        args.cpu = helicon.available_cpu()

    data = helicon.images2dataframe(
        args.input_imageFiles,
        csparc_passthrough_files=args.csparcPassthroughFiles,
        alternative_folders=args.folder,
        ignore_bad_particle_path=args.ignoreBadParticlePath,
        ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
        warn_missing_ctf=1,
        target_convention="relion",
    )

    try:
        optics = data.attrs["optics"]
    except KeyError:
        optics = None

    if args.verbose:
        image_name = helicon.first_matched_attr(
            data, attrs="rlnImageName rlnMicrographName rlnMicrographMovieName".split()
        )
        tmpCol = helicon.unique_attr_name(data, attr_prefix=image_name)
        data[tmpCol] = data[image_name].str.split("@", expand=True).iloc[:, -1]
        nMicrographs = len(data[tmpCol].unique())
        apixAttr = pixelSizeAttrForImageAttr(image_name)
        apix = getPixelSize(data, attrs=[apixAttr])
        infoParts = []
        if apix is not None:
            infoParts.append(f"pixel size={apix:.3f} Å/pixel")
        try:
            import mrcfile

            sample_file = data[tmpCol].iloc[0]
            with mrcfile.open(sample_file, permissive=True) as mrc:
                if mrc.data.ndim == 2:
                    ny, nx = mrc.data.shape
                else:
                    ny, nx = mrc.data.shape[-2:]
            infoParts.append(f"image size={nx}x{ny}")
        except Exception:
            pass
        if infoParts:
            apixStr = " (" + ", ".join(infoParts) + ")"
        else:
            apixStr = ""
        if "rlnHelicalTubeID" in data:
            nHelices = len(data.groupby([tmpCol, "rlnHelicalTubeID"]))
            dist_seg_median, dist_seg_mean, dist_seg_sigma, n_all = (
                estimate_inter_segment_distance(data)
            )
            if dist_seg_median is None:
                logger.info(
                    "Read in %d segments in %d helices from %d micrographs in %d image files%s",
                    len(data),
                    nHelices,
                    nMicrographs,
                    len(args.input_imageFiles),
                    apixStr,
                )
            else:
                read_msg = (
                    "Read in %d segments (extracted with %.2f\u00c5 inter-segment shift) in %d helices from %d micrographs in %d image files%s. Segment distances: %.2f\u00b1%.2f\u00c5."
                    % (
                        len(data),
                        dist_seg_median,
                        nHelices,
                        nMicrographs,
                        len(args.input_imageFiles),
                        apixStr,
                        dist_seg_mean,
                        dist_seg_sigma,
                    )
                )
                estimate_msg = "Estimate: ~%.1f%% of all (~%d) segments" % (
                    len(data) / n_all * 100,
                    n_all,
                )
                logger.info(read_msg)
                if dist_seg_sigma > dist_seg_median:
                    logger.warning(estimate_msg)
                    logger.warning(
                        "It appears that the filaments are badly fragmented, probably from Select2D/Select3D jobs. You can avoid filament fragmentation by runing the following command:\nhelicon images2star <input.star> <output.star> --recoverFullFilaments minFraction=<0.5>[:forcePickJob=<0|1>][:fullStarFile=<filename>]\nafter each Select2D/Select3D job",
                    )
                else:
                    logger.info(estimate_msg)
        elif (
            "rlnMicrographMovieName" in data
            and "rlnMicrographName" not in data
            and "rlnImageName" not in data
        ):
            logger.info(
                "Read in %d movies from %d files%s",
                nMicrographs,
                len(args.input_imageFiles),
                apixStr,
            )
        elif "rlnMicrographName" in data and "rlnImageName" not in data:
            logger.info(
                "Read in %d micrographs from %d files%s",
                nMicrographs,
                len(args.input_imageFiles),
                apixStr,
            )
        else:
            logger.info(
                "Read in %d particles in %d micrographs from %d image files%s",
                len(data),
                nMicrographs,
                len(args.input_imageFiles),
                apixStr,
            )
        if tmpCol in data:
            data.drop(tmpCol, inplace=True, axis=1)

    if args.micrographStar is not None and "rlnMicrographName" in data:
        import starfile

        ref = starfile.read(args.micrographStar)
        if isinstance(ref, dict):
            ref = ref.get(
                "particles",
                ref.get("data_particles", ref.get("micrographs", ref)),
            )
        if "rlnMicrographName" not in ref:
            raise HeliconError(
                f"--micrographStar file {args.micrographStar} has no rlnMicrographName column"
            )
        # Build mapping: cleaned CS basename -> reference STAR micrograph path
        ref_paths = ref["rlnMicrographName"].unique()
        path_map = {}
        for p in ref_paths:
            key = Path(p.split("@")[-1]).name  # strip @ prefix, get basename
            path_map[key] = p

        # Clean CS micrograph names and map to reference paths
        def _map_path(cs_path: str) -> str:
            key = helicon.clean_cs_micrograph_path(cs_path)
            if key in path_map:
                return path_map[key]
            logger.warning(
                "No matching micrograph in reference STAR for %s (cleaned: %s)",
                cs_path,
                key,
            )
            return cs_path

        data["rlnMicrographName"] = data["rlnMicrographName"].apply(_map_path)

    if len(data) == 0:
        raise HeliconError("nothing to do with 0 particles. I am going to quit")

    if args.first or args.last:
        if 0 < args.first < len(data):
            first = args.first
        else:
            first = 0
        if first < args.last < len(data):
            last = args.last
        else:
            last = len(data)
        data = data.iloc[first:last]
        data = data.reset_index(drop=True)  # important to do this

    index_d = {}
    for o in args.all_options:
        index_d[o] = 0

    for option_name in args.all_options:
        if option_name in args.append_options:
            param = args.__dict__[option_name][index_d[option_name]]
        else:
            param = args.__dict__[option_name]

        if args.verbose:
            logger.info("%s: %s", option_name, param)

        data, index_d = dispatch(option_name, data, args, index_d, param)
    if args.path != "absolute":
        from helicon import get_relion_project_folder, convert_dataframe_file_path

        relion_proj_folder = get_relion_project_folder(
            str(Path(args.output_starFile).resolve())
        )
        if relion_proj_folder:
            for attr in ["rlnImageName", "rlnMicrographName"]:
                if attr not in data:
                    continue
                data[attr] = convert_dataframe_file_path(
                    data, attr, to="relative", relpath_start=relion_proj_folder
                )

    if args.splitNumSets > 1:
        subsets = [[] for i in range(args.splitNumSets)]
        if args.splitMode in ["micrograph", "helicaltube"]:
            mapping = {
                "micrograph": "rlnMicrographName",
                "helicaltube": "rlnHelicalTubeID",
            }
            var = mapping[args.splitMode]
            if var not in data:
                raise HeliconError(
                    '--splitMode=%s requires "%s" in the input star file'
                    % (args.splitMode, var)
                )
            if var == "rlnHelicalTubeID":
                var = ["rlnMicrographName", "rlnHelicalTubeID"]
            mgraphs = data.groupby(var, sort=False)
            mgraphs = sorted(
                mgraphs, key=lambda x: len(x[1]), reverse=True
            )  # largest group -> smallest group
            for mgraphName, mgraphParticles in mgraphs:
                smallest_subset = min(subsets, key=lambda x: len(x))
                smallest_subset += list(mgraphParticles.index)
        else:
            if args.splitMode == "random":
                data = data.sample(frac=1).reset_index(drop=True)
            for si in range(args.splitNumSets):
                subsets[si] = list(range(si, len(data), args.splitNumSets))

        prefix = Path(args.output_starFile).stem
        suffix = Path(args.output_starFile).suffix
        for si, subset in enumerate(subsets):
            if args.splitNumSets == 2 and args.splitMode == "evenodd":
                imageSubSetFileName = f"{prefix}.{['e', 'o'][si]}{suffix}"
            else:
                imageSubSetFileName = f"{prefix}.subset-{si}{suffix}"

            data_subset = data.iloc[subset, :]
            data_subset = data_subset.sort_values(["rlnImageName"], ascending=True)
            data_subset["rlnRandomSubset"] = si + 1
            data_subset.reset_index(drop=True, inplace=True)
            data_subset.attrs["optics"] = optics
            helicon.dataframe2file(data_subset, imageSubSetFileName)
            if args.verbose:
                logger.info(
                    "Subset %d/%d: %d images saved to %s",
                    si + 1,
                    args.splitNumSets,
                    len(data_subset),
                    imageSubSetFileName,
                )
    else:
        helicon.dataframe2file(data, args.output_starFile)

        if args.verbose:
            filename = ""
            for choice in "rlnImageName rlnMicrographName".split():
                if choice in data:
                    filename = choice
                    break
            if filename:
                filename = data[filename].iloc[0].split("@")[-1]
                if Path(filename).exists():
                    nx, ny, _ = helicon.get_image_size(filename)
                    x, unit = helicon.bytes2units(nx * ny * 4 * len(data))
                    logger.info(
                        "%d images (%dx%d) saved to %s. Storage needed: %g %s",
                        len(data),
                        nx,
                        ny,
                        args.output_starFile,
                        round(x, 1),
                        unit,
                    )
                else:
                    logger.info(
                        "%d images saved to %s", len(data), args.output_starFile
                    )
            else:
                logger.info("%d images saved to %s", len(data), args.output_starFile)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for the images2star command.

    Combines infrastructure arguments with auto-discovered plugin arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to attach arguments to.

    Returns
    -------
    argparse.ArgumentParser
        The parser with arguments added.
    """
    # Infrastructure arguments
    parser.add_argument("input_imageFiles", nargs="+", help="input image file(s)")
    parser.add_argument("output_starFile", help="output star file name")
    parser.add_argument(
        "--csparcPassthroughFiles",
        metavar="<filename>",
        type=str,
        nargs="+",
        help="input cryosparc v2 passthrough file(s)",
        default=[],
    )
    parser.add_argument(
        "--first",
        type=int,
        metavar="<n>",
        help="first image to process. default to 0",
        default=0,
    )
    parser.add_argument(
        "--last",
        type=int,
        metavar="<n>",
        help="last image to process. default to the last image in the file",
        default=-1,
    )
    parser.add_argument(
        "--subset",
        metavar="<n>",
        type=int,
        help="which subset to keep. default to 0",
        default=0,
    )
    parser.add_argument(
        "--sets",
        metavar="<n>",
        type=int,
        help="number of subsets to split into. default to 1",
        default=1,
    )
    parser.add_argument(
        "--splitNumSets",
        metavar="<n>",
        type=int,
        help="number of subsets to split into. default to 1",
        default=1,
    )
    splitMode = ["evenodd", "random", "micrograph", "helicaltube"]
    parser.add_argument(
        "--splitMode",
        metavar="<%s>" % ("|".join(splitMode)),
        type=str,
        choices=splitMode,
        help="how to split image set",
        default="evenodd",
    )

    parser.add_argument(
        "--ignoreBadParticlePath",
        metavar="<0|1|2|3>",
        type=int,
        help="ignore bad particle image file path: 1-check but ignore missing files, 2-skip file checking, 3-skip file checking and recursive tracing of lst files. default: 0",
        default=0,
    )
    parser.add_argument(
        "--ignoreBadMicrographPath",
        metavar="<0|1>",
        type=int,
        help="ignore bad micrograph image file path. default: 1",
        default=1,
    )
    parser.add_argument(
        "--tag",
        metavar="<str>",
        type=str,
        help="add this tag to new binary image files",
        default="",
    )
    parser.add_argument(
        "--folder",
        metavar="<dirname>",
        type=str,
        nargs="*",
        help='Search these folders if the images cannot be found. default: ""',
        default=[],
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
        help="number of cpus to use. default to 1",
        default=1,
    )
    parser.add_argument(
        "--micrographStar",
        metavar="<starfile>",
        type=str,
        help=(
            "Reference RELION STAR file with rlnMicrographName entries. "
            "CryoSPARC micrograph names are cleaned (hash + _patch_aligned_doseweighted "
            "stripped) and mapped to matching entries from this STAR file."
        ),
        default=None,
    )
    parser.add_argument(
        "--force",
        type=int,
        metavar="<0|1>",
        help="force overwrite output images. default to 0",
        default=0,
    )
    parser.add_argument(
        "--ppid",
        metavar="<n>",
        type=int,
        help="Set the PID of the parent process, used for cross platform PPID",
        default=-1,
    )

    # Plugin-discovered arguments
    from helicon.plugins.images2star import add_plugin_args

    add_plugin_args(parser)

    return parser


def check_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Validate images2star command arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    parser : argparse.ArgumentParser
        The argument parser.

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
        if o
        not in "cpu first force ignoreBadParticlePath ignoreBadMicrographPath last folder splitNumSets splitMode micrographStar tag verbose".split()
    ]

    if Path(args.output_starFile).suffix not in ".star .cs .csv".split():
        raise HeliconValidationError(
            "the output file (%s) must be a .star, .cs, or .csv file"
            % (args.output_starFile)
        )

    if Path(args.output_starFile).exists() and not (
        args.force == 1 or args.splitNumSets > 1
    ):
        raise HeliconFileExistsError(
            "the output file (%s) exists. Use --force=1 to overwrite it"
            % (args.output_starFile)
        )

    if args.setCTF and not Path(args.setCTF).exists():
        raise HeliconValidationError(
            'option "--setCTF %s" specifies of a nonexistent file' % (args.setCTF)
        )

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    args = check_args(args, parser)
    main(args)
