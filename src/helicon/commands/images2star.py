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
from helicon.lib.images2star_engine import apply_options

logger = logging.getLogger(__name__)

import helicon
from helicon.lib.io import getPixelSize, setPixelSize, pixelSizeAttrForImageAttr
from helicon.lib.analysis import estimate_inter_segment_distance


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

    if getattr(args, "summary2D", False):
        from helicon.plugins.images2star import dispatch

        dispatch("summary2D", None, args, {"summary2D": 0}, args.summary2D)
        return

    if args.cpu < 1:
        args.cpu = helicon.available_cpu()

    source_index_attr = None
    if len(args.input_imageFiles) > 1:
        source_index_attr = "images2star_source_file_index"
    data = helicon.images2dataframe(
        args.input_imageFiles,
        csparc_passthrough_files=args.csparcPassthroughFiles,
        alternative_folders=args.folder,
        ignore_bad_particle_path=args.ignoreBadParticlePath,
        ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
        warn_missing_ctf=1,
        target_convention="relion",
        source_index_attr=source_index_attr,
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

    if source_index_attr is not None:
        data = join_filaments_from_multiple_files(data, source_index_attr, args)
        data.drop(source_index_attr, inplace=True, axis=1)

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

    data = apply_options(data, args.all_options, args, args.append_options)
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

        subset_files = split_subset_filenames(
            args.output_starFile, args.splitNumSets, args.splitMode
        )
        existing = [f for f in subset_files if f.exists()]
        if existing and args.force != 1:
            raise HeliconFileExistsError(
                "the output file(s) (%s) exist. Use --force=1 to overwrite them"
                % (" ".join(str(f) for f in existing))
            )
        for si, subset in enumerate(subsets):
            imageSubSetFileName = str(subset_files[si])
            data_subset = data.iloc[subset, :]
            if "rlnImageName" in data_subset:
                data_subset = data_subset.sort_values(["rlnImageName"], ascending=True)
            else:
                data_subset = data_subset.copy()
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


def split_subset_filenames(
    output_starFile: str, n_sets: int, split_mode: str
) -> list[Path]:
    """Names of the subset files written by ``--splitNumSets``.

    Parameters
    ----------
    output_starFile : str
        The output file given on the command line.
    n_sets : int
        Number of subsets.
    split_mode : str
        The ``--splitMode``.

    Returns
    -------
    list of Path
        One file per subset, in the folder of ``output_starFile``:
        ``<stem>.e<suffix>``/``<stem>.o<suffix>`` for an even/odd split into
        2 subsets, otherwise ``<stem>.subset-<i><suffix>``.
    """
    output = Path(output_starFile)
    if n_sets == 2 and split_mode == "evenodd":
        tags = ["e", "o"]
    else:
        tags = [f"subset-{si}" for si in range(n_sets)]
    return [output.with_name(f"{output.stem}.{tag}{output.suffix}") for tag in tags]


def join_filaments_from_multiple_files(
    data: pd.DataFrame, source_index_attr: str, args: argparse.Namespace
) -> pd.DataFrame:
    """Join helical filaments split across multiple input files.

    Only applies if all input files have helical particles. Duplicate
    particles (same ``rlnImageName``) are dropped, and on each micrograph the
    filament pieces from all input files that lie on overlapping straight
    lines are joined into one helical tube. In a joined tube, particles
    closer than half of the inter-box distance to a particle of another piece are
    removed as duplicates.

    Parameters
    ----------
    data : pd.DataFrame
        Particles from all input files, with ``source_index_attr`` holding
        the index of the input file of each particle.
    source_index_attr : str
        Column with the input file index.
    args : argparse.Namespace
        CLI arguments.

    Returns
    -------
    pd.DataFrame
        The particles with joined filaments.
    """
    name, param_dict = helicon.parse_param_str(str(args.joinFilaments))
    if name is not None and name.strip() == "0":
        return data
    if "rlnHelicalTubeID" not in data:
        return data
    helical = data.groupby(source_index_attr)["rlnHelicalTubeID"].apply(
        lambda s: s.notna().all()
    )
    if len(helical) < len(args.input_imageFiles) or not helical.all():
        return data
    required_attrs = "rlnMicrographName rlnCoordinateX rlnCoordinateY".split()
    missing_attrs = [a for a in required_attrs if a not in data]
    if missing_attrs:
        logger.warning(
            "cannot join the helical filaments from the %d input files: %s not available",
            len(args.input_imageFiles),
            " ".join(missing_attrs),
        )
        return data
    epsilon = float(param_dict.get("epsilon", 1.0))

    from helicon import convert_dataframe_file_path

    attrs = data.attrs
    n0 = len(data)
    if "rlnImageName" in data:
        image_abs = convert_dataframe_file_path(data, "rlnImageName", to="abs")
        data = data[~image_abs.duplicated(keep="first")].reset_index(drop=True)
    n_duplicates = n0 - len(data)

    mgraph_attr = helicon.unique_attr_name(data, attr_prefix="rlnMicrographName_abs")
    data[mgraph_attr] = convert_dataframe_file_path(data, "rlnMicrographName", to="abs")
    piece_attrs = [mgraph_attr, source_index_attr, "rlnHelicalTubeID"]
    n_pieces = data.groupby(piece_attrs, sort=False).ngroups
    apix = getPixelSize(
        data, attrs=["rlnMicrographPixelSize", "rlnMicrographOriginalPixelSize"]
    )
    inter_box_distance = helicon.estimate_inter_box_distance(data, piece_attrs)
    if args.verbose and inter_box_distance is not None:
        if apix:
            logger.info(
                "Inter-box distance: %.1f pixels (%.2fÅ)",
                inter_box_distance,
                inter_box_distance * apix,
            )
        else:
            logger.info("Inter-box distance: %.1f pixels", inter_box_distance)
    n1 = len(data)
    data = helicon.join_collinear_filaments(
        data,
        piece_attrs=piece_attrs[1:],
        micrograph_attr=mgraph_attr,
        epsilon=epsilon,
        apix_micrograph=apix,
        inter_box_distance=inter_box_distance,
    )
    n_overlapping = n1 - len(data)
    n_filaments = data.groupby([mgraph_attr, "rlnHelicalTubeID"], sort=False).ngroups
    data.drop(mgraph_attr, inplace=True, axis=1)
    data.attrs = attrs
    if args.verbose:
        logger.info(
            "Joined %d helical filaments from %d input files into %d filaments. "
            "Removed %d duplicate particles (same rlnImageName) and %d overlapping particles "
            "(closer than half of the inter-box distance) in the joined filaments",
            n_pieces,
            len(args.input_imageFiles),
            n_filaments,
            n_duplicates,
            n_overlapping,
        )
    return data


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
    parser.add_argument(
        "input_imageFiles",
        nargs="+",
        help="input image file(s), or a Class2D folder with --summary2D",
    )
    parser.add_argument(
        "output_starFile", help="output star file name, or PDF with --summary2D"
    )
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
        "--joinFilaments",
        metavar="<0|1>[:epsilon=<1.0>]",
        type=str,
        help=(
            "if multiple input files all have helical particles, join the filaments "
            "on the same micrograph that lie on overlapping straight lines "
            "(RELION start-end picking) into one helical tube, removing the particles "
            "that are closer than half of the inter-box distance to particles kept from a "
            "larger piece of the same tube. epsilon (pixels) is the "
            "tolerance of the |ab|-|pa|-|pb|<epsilon test of point p on line segment ab. "
            "default to %(default)s"
        ),
        default="1",
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
    from helicon.plugins.images2star import summary2d

    summary2d.check_args(args, parser, all_options)
    if getattr(args, "summary2D", False):
        args.all_options = []
        return args

    from helicon.plugins import plugin_options_in_argv
    from helicon.plugins.images2star import _plugins

    args.all_options = plugin_options_in_argv(sys.argv[1:], parser, _plugins)

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
