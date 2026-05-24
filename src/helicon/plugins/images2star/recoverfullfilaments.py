"""Handler for the recoverFullFilaments option."""

from __future__ import annotations
import logging
import helicon
import numpy as np
from pathlib import Path
import os

logger = logging.getLogger(__name__)


option_name = "recoverFullFilaments"


def add_args(parser):
    parser.add_argument(
        "--recoverFullFilaments",
        metavar="minFraction=<0.5>[:forcePickJob=<0|1>][:fullStarFile=<filename>]",
        type=str,
        help="recover the whole filaments if current filament has >=minFraction of the segments in the whole filament",
        default="",
    )


def handle(data, args, index_d, param):
    """Handle the recoverFullFilaments option.

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
    if len(param):
        if param.find("=") != -1:
            # minFraction=<0.5>[:forcePickJob=<0|1>][:fullStarFile=<filename>]
            _, param_dict = helicon.parse_param_str(param)
        else:
            param_dict = {}

        required_attrs = "rlnImageName rlnHelicalTubeID".split()

        forcePickJob = param_dict.get("forcePickJob", 0)
        if forcePickJob:
            required_attrs += "rlnMicrographName rlnCoordinateX rlnCoordinateY rlnHelicalTrackLengthAngst".split()

        missing_attrs = [p for p in required_attrs if p not in data]
        assert (
            missing_attrs == []
        ), f"\tERROR: requried parameters {' '.join(missing_attrs)} are not available"

        fullStarFile = param_dict.get("fullStarFile", None)

        def get_input_star_file(starFile, arg="--i "):
            from pathlib import Path

            sf = Path(starFile).resolve()
            pipelineFile = sf.parent / "job_pipeline.star"
            if not pipelineFile.exists():  # not in a RELION job folder
                return None
            noteFile = Path(sf).parent / "note.txt"
            if not noteFile.exists():
                return None
            relionProjectFolder = Path(noteFile).parent.parent.parent
            with open(noteFile) as fp:
                new_inputStarFile = None
                for l in fp.readlines()[::-1]:
                    pos = l.find(arg)
                    if pos != -1:
                        l2 = l[pos:]
                        pos2 = l2.find(" --")
                        if pos2 != -1:
                            s = l2[:pos2]
                        else:
                            s = l2
                        new_inputStarFile = s[len(arg) :].strip('"').strip().split()[0]
                        new_inputStarFile = relionProjectFolder / new_inputStarFile
                        return str(new_inputStarFile)
            return None

        if fullStarFile is None:

            def trace_back_to_extract_job(inputStarFile, forcePickJob=0, history=[]):
                history.append(str(inputStarFile))
                new_inputStarFile = get_input_star_file(inputStarFile)
                if new_inputStarFile is None:
                    return None
                if (
                    new_inputStarFile.find("Polish") != -1
                    or new_inputStarFile.find("Extract") != -1
                ):
                    if not forcePickJob:
                        history.append(new_inputStarFile)
                        return new_inputStarFile
                    parent_job_pick = get_input_star_file(
                        new_inputStarFile, arg="--coord_list "
                    )
                    parent_job_reextract = get_input_star_file(
                        new_inputStarFile, arg="--reextract_data_star "
                    )
                    if parent_job_pick and parent_job_pick.find("Pick") != -1:
                        history.append(new_inputStarFile)
                        return new_inputStarFile
                    if parent_job_reextract:
                        history.append(new_inputStarFile)
                        return trace_back_to_extract_job(
                            parent_job_reextract, forcePickJob, history
                        )
                return trace_back_to_extract_job(
                    new_inputStarFile, forcePickJob, history
                )

            history = []
            fullStarFile = trace_back_to_extract_job(
                args.input_imageFiles[0], forcePickJob, history
            )
            if args.verbose > 2:
                tmp = "\t->\n\t".join(history)
                logger.info(f"\t{tmp}")
            if fullStarFile is None:
                fullStarFile = history[-1]
                if len(history) > 1:
                    logger.warning(
                        "auto-traced back to '%s' but it is not the starting Polish shiny.star or Extract particles.star file. Will use it for recovery but you can manually specify the starting star file with --recoverFullFilaments fullStarFile=<filename>",
                        fullStarFile,
                    )

                else:
                    raise HeliconError(
                        "WARNING: failed to auto-find the Polish shiny.star or Extract particles.star file. Please manually specify it with --recoverFullFilaments fullStarFile=<filename>"
                    )
            if args.verbose > 2:
                logger.info(
                    f"\tWill use {str(fullStarFile)} to provide the full filaments"
                )

        parent_job_pick = get_input_star_file(fullStarFile, arg="--coord_list ")
        parent_job_reextract = get_input_star_file(
            fullStarFile, arg="--reextract_data_star "
        )
        if parent_job_pick is None and parent_job_reextract:
            logger.warning(
                "the source of the 'full' filaments\n\t%s\n\tis not a ManualPick or AutoPick job. The output star file will probably still be fragmented",
                parent_job_reextract,
            )

        data.convention = "relion"
        data = data.drop_duplicates(
            subset=["rlnImageName"], keep="last", inplace=False
        ).reset_index(drop=True)

        data2 = helicon.images2dataframe(
            fullStarFile,
            alternative_folders=args.folder,
            ignore_bad_particle_path=args.ignoreBadParticlePath,
            ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
            warn_missing_ctf=0,
            target_convention="relion",
        )
        data2 = data2.drop_duplicates(
            subset=["rlnImageName"], keep="last", inplace=False
        ).reset_index(drop=True)
        if args.verbose > 1:
            logger.info(f"\tRead in {len(data2):,} particles from {fullStarFile}")

        missing_attrs = [p for p in required_attrs if p not in data2]
        assert (
            missing_attrs == []
        ), f"\tERROR: {fullStarFile} does not have the requried parameters {' '.join(missing_attrs)}"

        if len(data) > len(data2):
            raise HeliconError(
                "\\tERROR: --recoverFullFilament option requires that {fullStarFile} ({len(data2)}) has the same number or more particles (>={len(data)})"
            )

        folders = [
            p
            for p in Path(data["rlnImageName"].iloc[0].split("@")[-1]).parents
            if str(p).find("/job") != -1
        ]
        folder_current = folders[-1]
        folders = [
            p
            for p in Path(data2["rlnImageName"].iloc[0].split("@")[-1]).parents
            if str(p).find("/job") != -1
        ]
        folder_new = folders[-1]

        n0 = len(data)
        helices = []

        from helicon import convert_dataframe_file_path

        if forcePickJob:
            data.loc[:, "rlnMicrographName_abs"] = convert_dataframe_file_path(
                data, "rlnMicrographName", to="abs"
            )
            data2.loc[:, "rlnMicrographName_abs"] = convert_dataframe_file_path(
                data2, "rlnMicrographName", to="abs"
            )
            if not (
                set(data["rlnMicrographName_abs"]).issubset(
                    set(data2["rlnMicrographName_abs"])
                )
            ):
                missing_micrographs = "\n\t".join(
                    sorted(
                        [
                            m
                            for m in set(data["rlnMicrographName_abs"])
                            if m not in set(data2["rlnMicrographName_abs"])
                        ]
                    )
                )
                raise HeliconError(
                    "\\tERROR: --recoverFullFilament option requires that {fullStarFile} contains identical set or a superset of micrographs. These micrographs are not in {fullStarFile}:\\t{missing_micrographs}"
                )
            sortby = [
                "rlnMicrographName_abs",
                "rlnHelicalTubeID",
                "rlnHelicalTrackLengthAngst",
            ]
            data = data.sort_values(sortby, ascending=True)
            data2 = data2.sort_values(sortby, ascending=True)
            mgraphs = data.groupby("rlnMicrographName_abs", sort=False)
            mgraphs2 = data2.groupby("rlnMicrographName_abs", sort=False)
            mgraphs_dict = {
                mgraph_name: mgraph_particles
                for mgraph_name, mgraph_particles in mgraphs
            }
            mgraphs2_dict = {
                mgraph_name: mgraph_particles
                for mgraph_name, mgraph_particles in mgraphs2
            }

            def on_line_segment(
                px,
                py,
                line_start_x,
                line_start_y,
                line_end_x,
                line_end_y,
                epsilon=1.0,
            ):
                d1 = np.sqrt((px - line_start_x) ** 2 + (py - line_start_y) ** 2)
                d2 = np.sqrt((px - line_end_x) ** 2 + (py - line_end_y) ** 2)
                d = np.sqrt(
                    (line_end_x - line_start_x) ** 2 + (line_end_y - line_start_y) ** 2
                )
                return abs(d - d1 - d2) < epsilon

            for mgraph_name in mgraphs_dict:
                filaments = mgraphs_dict[mgraph_name].groupby(
                    "rlnHelicalTubeID", sort=False
                )
                if mgraph_name not in mgraphs2_dict:
                    logger.error(
                        "micrograph %s is not in %s", mgraph_name, fullStarFile
                    )
                filaments2 = mgraphs2_dict[mgraph_name].groupby(
                    "rlnHelicalTubeID", sort=False
                )
                for filament_name, filament_segments in filaments:
                    matched = False
                    n = len(filament_segments)
                    cx0, cx1 = (
                        filament_segments["rlnCoordinateX"]
                        .astype(np.float32)
                        .values[[0, -1]]
                    )
                    cy0, cy1 = (
                        filament_segments["rlnCoordinateY"]
                        .astype(np.float32)
                        .values[[0, -1]]
                    )
                    for filament_name2, filament_segments2 in filaments2:
                        cx0_2, cx1_2 = (
                            filament_segments2["rlnCoordinateX"]
                            .astype(np.float32)
                            .values[[0, -1]]
                        )
                        cy0_2, cy1_2 = (
                            filament_segments2["rlnCoordinateY"]
                            .astype(np.float32)
                            .values[[0, -1]]
                        )
                        if on_line_segment(
                            cx0, cy0, cx0_2, cy0_2, cx1_2, cy1_2
                        ) and on_line_segment(cx1, cy1, cx0_2, cy0_2, cx1_2, cy1_2):
                            matched = True
                            n2 = len(filament_segments2)
                            indices = list(filament_segments2.index)
                            helices.append((n, n2, indices))
                    if not matched:
                        logger.warning(
                            "%s:helicalTubeID=%s: cannot find a matching helix in %s",
                            mgraph_name,
                            filament_name,
                            fullStarFile,
                        )
        else:
            if not (set(data[["rlnImageName"]]).issubset(set(data2[["rlnImageName"]]))):
                raise HeliconError(
                    "\\tERROR: --recoverFullFilament option requires that {fullStarFile} contains identical set or a superset of particles"
                )

            data.loc[:, "rlnImageName_abs"] = (
                convert_dataframe_file_path(data, "rlnImageName", to="abs")
                .str.split("@", expand=True)
                .iloc[:, -1]
            )
            data2.loc[:, "rlnImageName_abs"] = (
                convert_dataframe_file_path(data2, "rlnImageName", to="abs")
                .str.split("@", expand=True)
                .iloc[:, -1]
            )
            groups = data.groupby(["rlnImageName_abs", "rlnHelicalTubeID"], sort=False)
            groups2 = data2.groupby(
                ["rlnImageName_abs", "rlnHelicalTubeID"], sort=False
            )
            groups_dict = {
                group_name: group_particles for group_name, group_particles in groups
            }
            groups2_dict = {
                group_name: group_particles for group_name, group_particles in groups2
            }
            missing_helices = [
                f"{k[0]}:rlnHelicalTubeID={k[1]}"
                for k in groups_dict
                if k not in groups2_dict
            ]
            if missing_helices:
                s = "\n\t".join(missing_helices)
                logger.error(
                    "%s helices not found in %s:\n\t%s",
                    len(missing_helices),
                    fullStarFile,
                    s,
                )
                raise HeliconError(
                    "\\tMake sure that the input star file {' '.join(args.input_imageFiles)} and the fullStarFile {fullStarFile} are from the same Extract job"
                )

            for group_name in groups_dict:
                assert group_name in groups2_dict
                n = len(groups_dict[group_name])
                n2 = len(groups2_dict[group_name])
                indices = list(groups2_dict[group_name].index)
                helices.append((n, n2, indices))

        minFraction = float(param_dict.get("minFraction", -1))
        if not (0 <= minFraction <= 1):
            n1 = sum([helix[0] for helix in helices])
            n2 = sum([helix[1] for helix in helices])
            ng = sum([helix[0] for helix in helices if n1 / n2 >= 0.5])
            minFraction = min(0.5, max(0, (n1 - ng) / (n2 - 2 * ng)))
            if args.verbose > 1:
                logger.info(f"\tminFraction set to {minFraction:.2f}")

        nsegments = []
        fractions = []
        indices = []
        for n1, n2, helix_indices in helices:
            nsegments.append(n2)
            fractions.append(n1 / n2)
            if n1 / n2 < minFraction:
                continue
            indices += helix_indices
        data = data2.iloc[indices]
        drop_attrs = [
            attr
            for attr in "rlnImageName_abs rlnMicrographName_abs".split()
            if attr in data
        ]
        if drop_attrs:
            data = data.drop(drop_attrs, inplace=False, axis=1)
        data = data.reset_index(drop=True)  # important to do this

        if args.verbose > 1:
            logger.info(f"\t{n0} -> {len(data)} helical segments")
            if folder_current != folder_new:
                logger.warning(
                    "the output star file now points to particles in folder\n\t%s\n\tinstead of\n\t%s",
                    folder_new,
                    folder_current,
                )
        if args.verbose > 2:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
            hbin = axes[0].hexbin(
                nsegments, fractions, bins="log", gridsize=50, cmap="jet"
            )
            fig.colorbar(hbin, ax=axes[0], label="#filaments")
            axes[1].hist(fractions, bins=50, edgecolor="white", linewidth=1)
            axes[0].set(xlabel="Filament Length (#segments)", ylabel="Fraction")
            axes[1].set(xlabel="Fraction", ylabel="# Filaments")
            plt.savefig(f"{os.path.splitext(args.output_starFile)[0]}.pdf")
            plt.show()

        index_d[option_name] += 1
    return data, index_d
