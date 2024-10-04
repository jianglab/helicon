#!/usr/bin/env python

"""A command line tool that analyzes/transforms dataset(s) and saves the dataset in RELION star file"""

import argparse, math, os, sys, types
from pathlib import Path
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

import helicon


def main(args):
    helicon.log_command_line()

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
    except:
        optics = None

    if args.verbose:
        image_name = helicon.first_matched_atrr(
            data, attrs="rlnMicrographMovieName rlnMicrographName rlnImageName".split()
        )
        tmpCol = helicon.unique_attr_name(data, attr_prefix=image_name)
        data[tmpCol] = data[image_name].str.split("@", expand=True).iloc[:, -1]
        nMicrographs = len(data[tmpCol].unique())
        apix = getPixelSize(data)
        if apix is not None:
            apixStr = f" (pixel size={apix:.3f} Å/pixel)"
        else:
            apixStr = ""
        if "rlnHelicalTubeID" in data:
            nHelices = len(data.groupby([tmpCol, "rlnHelicalTubeID"]))
            dist_seg_median, dist_seg_mean, dist_seg_sigma, n_all = (
                estimate_inter_segment_distance(data)
            )
            if dist_seg_median is None:
                print(
                    f"Read in {len(data)} segments in {nHelices} helices from {nMicrographs} micrographs in {len(args.input_imageFiles)} image files{apixStr}"
                )
            else:
                print(
                    f"Read in {len(data):,} segments (extracted with {dist_seg_median:.2f}Å inter-segment shift) in {nHelices:,} helices from {nMicrographs:,} micrographs in {len(args.input_imageFiles)} image files{apixStr}. Segment distances: {dist_seg_mean:.2f}±{dist_seg_sigma:.2f}Å. Estimate: ~{len(data)/n_all*100:.1f}% of all (~{n_all:,}) segments"
                )
                if dist_seg_sigma > dist_seg_median:
                    helicon.color_print(
                        f"It appears that the filaments are badly fragmented, probably from Select2D/Select3D jobs. You can avoid filament fragmentation by runing the following command:\nimages2star.py <input.star> <output.star> --recoverFullFilaments minFraction=<0.5>[:forcePickJob=<0|1>][:fullStarFile=<filename>]\nafter each Select2D/Select3D job"
                    )
        elif (
            "rlnMicrographMovieName" in data
            and "rlnMicrographName" not in data
            and "rlnImageName" not in data
        ):
            print(
                f"Read in {nMicrographs} movies from {len(args.input_imageFiles)} files{apixStr}"
            )
        elif "rlnMicrographName" in data and "rlnImageName" not in data:
            print(
                f"Read in {nMicrographs} micrographs from {len(args.input_imageFiles)} files{apixStr}"
            )
        else:
            print(
                f"Read in {len(data)} particles in {nMicrographs} micrographs from {len(args.input_imageFiles)} image files{apixStr}"
            )
        if tmpCol in data:
            data.drop(tmpCol, inplace=True, axis=1)

    if len(data) == 0:
        helicon.color_print(
            "WARNING: nothing to do with 0 particles. I am going to quit"
        )
        sys.exit(-1)

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
            print("%s: %s" % (option_name, param))

        if option_name == "removeDuplicates" and param:
            vars = [v for v in param if v not in data]
            if vars:
                helicon.color_print(f"\tWARNING: {vars} are not valid parameters")
            vars = [v for v in param if v in data]
            if len(vars) < 1:
                print(f"\tnothing to do when no valid parameter is provided")
            else:
                data2 = data.drop_duplicates(vars)
                if args.verbose:
                    print(
                        f"{len(data2)} image retained after removing {len(data) - len(data2)} images with duplicate {vars}"
                    )
                data = data2.reset_index(drop=True)  # important to do this

        if option_name == "minDuplicates" and param > 0:
            minN = param
            attr = None
            for a in "rlnImageName rlnMicrographName".split():
                if a in data:
                    attr = a
                    break
            if attr is None:
                helicon.color_print(
                    f"\tERROR: required parameter (rlnImageName or rlnMicrographName) is not available"
                )
                sys.exit(-1)

            from helicon import convert_dataframe_file_path

            tmp = convert_dataframe_file_path(data, attr, to="abs")
            retained = tmp.map(tmp.value_counts() >= minN)
            data2 = data[retained]
            if len(data2) < 1:
                helicon.color_print(f"\tWarning: no image is retained")
                sys.exit(-1)
            data2 = data2.drop_duplicates([attr])
            if args.verbose > 1:
                print(f"\t{len(data2)} images retained")
            data = data2.reset_index(drop=True)  # important to do this

        elif option_name == "randomSample" and 0 < param < len(data):
            data = data.sample(args.randomSample)
            data.reset_index(drop=True, inplace=True)
            index_d[option_name] += 1

        elif option_name == "recoverFullFilaments" and len(param):
            if param.find("=") != -1:
                # minFraction=<0.5>[:forcePickJob=<0|1>][:fullStarFile=<filename>]
                param_dict = helicon.parsemodopt2(param)
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
                            new_inputStarFile = (
                                s[len(arg) :].strip('"').strip().split()[0]
                            )
                            new_inputStarFile = relionProjectFolder / new_inputStarFile
                            return str(new_inputStarFile)
                return None

            if fullStarFile is None:

                def trace_back_to_extract_job(
                    inputStarFile, forcePickJob=0, history=[]
                ):
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
                    print(f"\t{tmp}")
                if fullStarFile is None:
                    fullStarFile = history[-1]
                    if len(history) > 1:
                        helicon.color_print(
                            f"WARNING: auto-traced back to '{fullStarFile}' but it is not the starting Polish shiny.star or Extract particles.star file. Will use it for recovery but you can manually specify the starting star file with --recoverFullFilaments fullStarFile=<filename>"
                        )

                    else:
                        helicon.color_print(
                            f"WARNING: failed to auto-find the Polish shiny.star or Extract particles.star file. Please manually specify it with --recoverFullFilaments fullStarFile=<filename>"
                        )
                        sys.exit(-1)
                if args.verbose > 2:
                    print(
                        f"\tWill use {str(fullStarFile)} to provide the full filaments"
                    )

            parent_job_pick = get_input_star_file(fullStarFile, arg="--coord_list ")
            parent_job_reextract = get_input_star_file(
                fullStarFile, arg="--reextract_data_star "
            )
            if parent_job_pick is None and parent_job_reextract:
                helicon.color_print(
                    f"\tWarning: the source of the 'full' filaments\n\t{parent_job_reextract}\n\tis not a ManualPick or AutoPick job. The output star file will probably still be fragmented"
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
                print(f"\tRead in {len(data2):,} particles from {fullStarFile}")

            missing_attrs = [p for p in required_attrs if p not in data2]
            assert (
                missing_attrs == []
            ), f"\tERROR: {fullStarFile} does not have the requried parameters {' '.join(missing_attrs)}"

            if len(data) > len(data2):
                helicon.color_print(
                    f"\tERROR: --recoverFullFilament option requires that {fullStarFile} ({len(data2)}) has the same number or more particles (>={len(data)})"
                )
                sys.exit(-1)

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
                    helicon.color_print(
                        f"\tERROR: --recoverFullFilament option requires that {fullStarFile} contains identical set or a superset of micrographs. These micrographs are not in {fullStarFile}:\t{missing_micrographs}"
                    )

                    sys.exit(-1)
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
                        (line_end_x - line_start_x) ** 2
                        + (line_end_y - line_start_y) ** 2
                    )
                    return abs(d - d1 - d2) < epsilon

                for mgraph_name in mgraphs_dict:
                    filaments = mgraphs_dict[mgraph_name].groupby(
                        "rlnHelicalTubeID", sort=False
                    )
                    if mgraph_name not in mgraphs2_dict:
                        helicon.color_print(
                            "\tERROR: micrograph {mgraph_name} is not in {fullStarFile}"
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
                            helicon.color_print(
                                "\tWarning: {mgraph_name}:helicalTubeID={helix_name}: cannot find a matching helix in {fullStarFile}"
                            )
            else:
                if not (
                    set(data[["rlnImageName"]]).issubset(set(data2[["rlnImageName"]]))
                ):
                    helicon.color_print(
                        f"\tERROR: --recoverFullFilament option requires that {fullStarFile} contains identical set or a superset of particles"
                    )
                    sys.exit(-1)

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
                groups = data.groupby(
                    ["rlnImageName_abs", "rlnHelicalTubeID"], sort=False
                )
                groups2 = data2.groupby(
                    ["rlnImageName_abs", "rlnHelicalTubeID"], sort=False
                )
                groups_dict = {
                    group_name: group_particles
                    for group_name, group_particles in groups
                }
                groups2_dict = {
                    group_name: group_particles
                    for group_name, group_particles in groups2
                }
                missing_helices = [
                    f"{k[0]}:rlnHelicalTubeID={k[1]}"
                    for k in groups_dict
                    if k not in groups2_dict
                ]
                if missing_helices:
                    s = "\n\t".join(missing_helices)
                    helicon.color_print(
                        f"\tERROR: {len(missing_helices)} helices not found in {fullStarFile}:\n\t{s}"
                    )
                    helicon.color_print(
                        f"\tMake sure that the input star file {' '.join(args.input_imageFiles)} and the fullStarFile {fullStarFile} are from the same Extract job"
                    )
                    sys.exit(-1)

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
                    print(f"\tminFraction set to {minFraction:.2f}")

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
                print(f"\t{n0} -> {len(data)} helical segments")
                if folder_current != folder_new:
                    helicon.color_print(
                        f"\tWarning: the output star file now points to particles in folder\n\t{folder_new}\n\tinstead of\n\t{folder_current}"
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

        elif option_name == "select" and len(param) == 2:
            var, val = param
            if var in data:
                vmin, vmax = data[var].min(), data[var].max()
                vals = val.split(",")
                if pd.api.types.is_integer_dtype(data[var]):
                    vals = list(map(int, vals))
                elif pd.api.types.is_float_dtype(data[var]):
                    vals = list(map(float, vals))
                data = data[data[var].isin(vals)]
                if len(data) < 1:
                    helicon.color_print(
                        f"WARNING: this selection has excluded all images. Hint: the actual data range is [{vmin}, {vmax}]"
                    )
                    sys.exit(-1)
            else:
                if args.verbose:
                    helicon.color_print(
                        '\tWARNING: the variable "%s" specified by option "--select %s %s" does NOT exist'
                        % (var, var, val)
                    )
            index_d[option_name] += 1

        elif option_name == "selectValueRange" and len(param) == 3:
            var, val1, val2 = param
            if var in data:
                vmin, vmax = data[var].min(), data[var].max()
                if pd.api.types.is_integer_dtype(data[var]):
                    val1 = int(val1)
                    val2 = int(val2)
                elif pd.api.types.is_float_dtype(data[var]):
                    val1 = float(val1)
                    val2 = float(val2)
                data = data.loc[(data[var] > val1) & (data[var] < val2)]
                if len(data) < 1:
                    helicon.color_print(
                        f"WARNING: this selection has excluded all images. Hint: the actual data range is [{vmin}, {vmax}]"
                    )
                    sys.exit(-1)
            else:
                if args.verbose:
                    helicon.color_print(
                        '\tWARNING: the variable "%s" specified by option "--selectValueRange %s %s %s" does NOT exist'
                        % (var, var, val1, val2)
                    )
            index_d[option_name] += 1

        elif option_name == "selectRatioRange" and len(param) == 3:
            var, val1, val2 = param
            if var in data:
                vmin, vmax = data[var].min(), data[var].max()
                if pd.api.types.is_integer_dtype(data[var]):
                    val1 = int(val1)
                    val2 = int(val2)
                elif pd.api.types.is_float_dtype(data[var]):
                    val1 = float(val1)
                    val2 = float(val2)
                else:
                    if args.verbose:
                        helicon.color_print(
                            'ERROR: the variable "%s" specified by option "--selectRatioRange %s %s %s" is NOT a number type'
                            % (var, var, val1, val2)
                        )
                        sys.exit(-1)
                data[var] = data[var].astype(float)
                val1 = float(val1)
                val2 = float(val2)
                if val1 == 0:
                    valmin = data[var].min()
                else:
                    valmin = data[var].nsmallest(int(len(data) * val1)).iloc[-1]
                if val2 == 1:
                    valmax = data[var].max() + 0.1
                else:
                    valmax = data[var].nsmallest(int(len(data) * val2) + 1).iloc[-1]
                data = data.loc[(data[var] >= valmin) & (data[var] < valmax)]
                if len(data) < 1:
                    helicon.color_print(
                        f"WARNING: this selection has excluded all images. Hint: the actual data range is [{vmin}, {vmax}]"
                    )
                    sys.exit(-1)
            elif var.lower() == "index":
                val1 = int(round(float(val1) * len(data)))
                val2 = int(round(float(val2) * len(data)))
                if val1 < 0:
                    val1 = 0
                if val2 < 0:
                    val2 = len(data)
                data = data.iloc[val1:val2]
            else:
                if args.verbose:
                    helicon.color_print(
                        '\tERROR: the variable "%s" specified by option "--selectRatioRange %s %s %s" does NOT exist'
                        % (var, var, val1, val2)
                    )
                    sys.exit(-1)
            if args.verbose > 1:
                print("\t%d images selected" % (len(data)))
            if not len(data):
                if args.verbose:
                    print(
                        "\tNothing to do when there is no particle image left. I will quit"
                    )
                sys.exit(-1)
            index_d[option_name] += 1

        elif option_name in ["selectFile", "excludeFile"] and len(param) > 0:
            # starfile:col1=<name>:col2=<name>:pattern=<str>
            sf, param_dict = helicon.parsemodopt(param)
            col1 = param_dict.get("col1", "rlnImageName")
            col2 = param_dict.get("col2", "rlnImageName")
            assert col1 in data
            pattern = param_dict.get("pattern", None)
            if not os.path.exists(sf):
                helicon.color_print(
                    "\tERROR: option --selectFile=%s has specified a non-existent file %s"
                    % (args.selectFile, sf)
                )
                sys.exit(-1)
            data_sf = helicon.images2dataframe(
                sf,
                alternative_folders=args.folder,
                ignore_bad_particle_path=args.ignoreBadParticlePath,
                ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
                warn_missing_ctf=0,
                target_convention="relion",
            )
            if args.verbose > 1:
                print("\t%d images found in %s" % (len(data_sf), sf))
            assert col2 in data_sf
            from helicon import convert_dataframe_file_path

            dids = convert_dataframe_file_path(data, col1, to="abs")
            sids = convert_dataframe_file_path(data_sf, col2, to="abs")
            if pattern:
                dids = dids.str.extract(pattern, expand=False)
                sids = sids.str.extract(pattern, expand=False)
            if option_name in ["selectFile"]:
                dids = dids[dids.isin(sids)]
            else:
                dids = dids[~dids.isin(sids)]
            data2 = data.loc[dids.index, :]
            data2.reset_index(drop=True, inplace=True)
            if len(data2):
                if args.verbose > 1:
                    print(f"\t{len(data2)}/{len(data)} images retained")
                data = data2
            else:
                inputFileStr = (
                    args.input_imageFiles
                    if len(args.input_imageFiles) > 1
                    else args.input_imageFiles[0]
                )
                if option_name in ["selectFile"]:
                    helicon.color_print(
                        (
                            "\tERROR: no common image found. Check if the files %s and %s include particles in the same folder"
                            % (inputFileStr, sf)
                        )
                    )
                    data_ci = data.columns.get_loc(col1)
                    data_sf_ci = data_sf.columns.get_loc(col2)
                    print(("\t%s %s: %s" % (inputFileStr, col1, data.iat[0, data_ci])))
                    print(("\t%s %s: %s" % (sf, col2, data_sf.iat[0, data_sf_ci])))
                    sys.exit(-1)
            index_d[option_name] += 1

        elif option_name == "selectCommonHelices" and len(param) > 0:
            # starfile:col1=<name>:col2=<name>:pattern=<str>
            sf, _ = helicon.parsemodopt(param)
            assert "rlnMicrographName" in data
            assert "rlnHelicalTubeID" in data
            if not os.path.exists(sf):
                helicon.color_print(
                    "\tERROR: option --selectCommonHelices %s has specified a non-existent file %s"
                    % (args.selectCommonHelices, sf)
                )
                sys.exit(-1)
            data_sf = helicon.images2dataframe(
                sf,
                alternative_folders=args.folder,
                ignore_bad_particle_path=args.ignoreBadParticlePath,
                ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
                warn_missing_ctf=0,
                target_convention="relion",
            )
            if args.verbose > 1:
                print("\t%d images found in %s" % (len(data_sf), sf))
            assert "rlnMicrographName" in data_sf
            assert "rlnHelicalTubeID" in data_sf
            
            #import tqdm
            #idx=[row in set(zip(data_sf["rlnMicrographName"],data["rlnHelicalTubeID"])) for row in tqdm.tqdm(list(zip(data["rlnMicrographName"],data["rlnHelicalTubeID"])))]
            #data2=data[idx]
            common_cols=["rlnMicrographName","rlnHelicalTubeID"]
            data2=data.merge(data_sf[common_cols],on=common_cols,how='inner',suffixes=['','_dup'])
            data2=data2[data.columns].drop_duplicates()
            
            data2.reset_index(drop=True, inplace=True)
            data2.attrs["optics"] = optics
            
            if len(data2):
                if args.verbose > 1:
                    print(f"\t{len(data2)}/{len(data)} images retained")
                data = data2
            else:
                inputFileStr = (
                    args.input_imageFiles
                    if len(args.input_imageFiles) > 1
                    else args.input_imageFiles[0]
                )
                helicon.color_print(
                    (
                        "\tERROR: no common image found. Check if the files %s and %s include particles in the same folder"
                        % (inputFileStr, sf)
                    )
                )
                data_ci = data.columns.get_loc("rlnMicrographName")
                data_sf_ci = data_sf.columns.get_loc("rlnMicrographName")
                print(("\t%s %s: %s" % (inputFileStr, "rlnMicrographName", data.iat[0, data_ci])))
                print(("\t%s %s: %s" % (sf, "rlnMicrographName", data_sf.iat[0, data_sf_ci])))
                sys.exit(-1)
            index_d[option_name] += 1

        elif option_name in ["selectByParticleLocation"] and len(param) > 0:
            # starfile:maxDist=<pixel>
            required_attrs = "rlnMicrographName rlnCoordinateX rlnCoordinateY".split()
            missing_attrs = [p for p in required_attrs if p not in data]
            assert (
                missing_attrs == []
            ), f"\tERROR: requried parameters {' '.join(missing_attrs)} are not available"

            sf, param_dict = helicon.parsemodopt(param)
            maxDist = param_dict.get("maxDist", 5)
            assert Path(sf).exists(), f"ERROR: {sf} does not exist"
            assert maxDist >= 0

            data_sf = helicon.images2dataframe(
                sf,
                alternative_folders=args.folder,
                ignore_bad_particle_path=args.ignoreBadParticlePath,
                ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
                warn_missing_ctf=0,
                target_convention="relion",
            )
            if args.verbose > 1:
                print(f"\t{len(data_sf)} images found in {sf}")

            missing_attrs = [p for p in required_attrs if p not in data_sf]
            assert (
                missing_attrs == []
            ), f"\tERROR: requried parameters {' '.join(missing_attrs)} are not available in {sf}"

            from helicon import convert_dataframe_file_path

            data["sbpl_rlnMicrographName"] = convert_dataframe_file_path(
                data, "rlnMicrographName", to="abs"
            )
            data_sf["sbpl_rlnMicrographName"] = convert_dataframe_file_path(
                data_sf, "rlnMicrographName", to="abs"
            )

            from scipy.spatial import distance

            group2 = {
                gname: g for gname, g in data_sf.groupby("sbpl_rlnMicrographName")
            }
            matched_indices = []
            for gname, g in data.groupby("sbpl_rlnMicrographName"):
                if gname not in group2:
                    continue
                cx = g["rlnCoordinateX"].values
                cy = g["rlnCoordinateY"].values
                cx2 = group2[gname]["rlnCoordinateX"].values
                cy2 = group2[gname]["rlnCoordinateY"].values

                loc = np.vstack((cx, cy)).T
                loc2 = np.vstack((cx2, cy2)).T
                dist_matrix = distance.cdist(loc, loc2, "euclidean")
                row_indices = np.where(np.min(dist_matrix, axis=1) <= maxDist)[0]
                matched_indices += list(g.index[row_indices])

            data2 = data.loc[matched_indices, :]
            data2.reset_index(drop=True, inplace=True)
            if args.verbose > 1:
                print(f"\t{len(data2)}/{len(data)} images retained")
            if len(data2) <= 0:
                helicon.color_print("\tWARNING: no particle left. I will quit")
                sys.exit(-1)
            data = data2
            data.drop(["sbpl_rlnMicrographName"], axis=1, inplace=True)
            index_d[option_name] += 1

        elif option_name == "sets" and param > 1:
            sets = param
            data = data[args.subset :: sets]
            if args.verbose > 1:
                print("\t%d/%d: %d images selected" % (args.subset, sets, len(data)))
            index_d[option_name] += 1

        elif option_name == "normEulerDist" and len(param) == 2:
            bin, nkeep = param
            nkeep = int(nkeep)

            def assignEulerBins(rottilt):
                import math

                rot, tilt = rottilt
                tilt = int(tilt / bin + 0.5) * bin
                if tilt == 0 or tilt == 180:
                    rot = 0
                else:
                    angBinSizeTmp = bin / math.sin(tilt * math.pi / 180)
                    rot = int(rot / angBinSizeTmp + 0.5) * angBinSizeTmp
                return (tilt, rot)

            binAngles = data[["rlnAngleRot", "rlnAngleTilt"]].apply(
                assignEulerBins, axis=1
            )
            binAssignments = binAngles.groupby(binAngles, sort=False)

            counts = binAssignments.size().sort_values(ascending=True)
            elbow = counts[helicon.findElbowPoint(counts)]
            if nkeep < 1:
                nkeep = elbow

            if args.verbose > 1:
                print(
                    "\tNumber of particles in %d Euler groups: Mean=%.1f\tSigma=%.1f\tMedian=%d\tElbow=%d"
                    % (
                        len(binAssignments),
                        counts.mean(),
                        counts.std(),
                        counts.median(),
                        elbow,
                    )
                )
                print(
                    "\tUpto %d best particles in each angular bin will be retained"
                    % (nkeep)
                )

            binAssignments = dict(list(binAssignments))
            indices = []
            binEulers = sorted(binAssignments.keys())
            for bi, be in enumerate(binEulers):
                bm = binAssignments[be]
                binPtcls = data.iloc[bm.index, :]
                if "rlnLogLikeliContribution" in binPtcls:
                    binPtcls = binPtcls.sort_values(
                        "rlnLogLikeliContribution", ascending=1
                    )  # the larger rlnLogLikeliContribution the better
                    binPtcls2 = binPtcls.tail(n=nkeep)
                else:
                    if len(binPtcls) > nkeep:
                        binPtcls2 = binPtcls.sample(n=nkeep)
                    else:
                        binPtcls2 = binPtcls
                if args.verbose > 2:
                    print(
                        "\t%3d/%3d:\talt=%.3g\taz=%.3g\t%d\t->\t%d"
                        % (
                            bi + 1,
                            len(binEulers),
                            be[0],
                            be[1],
                            len(binPtcls),
                            len(binPtcls2),
                        )
                    )
                indices.extend(binPtcls2.index)
            indices.sort()
            data = data.iloc[indices, :]
            index_d[option_name] += 1

        elif option_name == "apix" and param > 0:
            setPixelSize(data, apix_new=param)
            index_d[option_name] += 1

        elif option_name == "setParm" and param:
            if len(param) % 2:
                helicon.color_print(
                    "ERROR: you specified odd number of --setParm arguments. Only even number of arguments are allowed for var val pairs"
                )
                sys.exit(1)
            for i in range(len(param) // 2):
                var, val = param[2 * i : 2 * (i + 1)]
                if var in helicon.Relion_OpticsGroup_Parameters:
                    try:
                        data.attrs["optics"].loc[:, var] = helicon.guess_data_type(val)(
                            val
                        )
                    except:
                        data.loc[:, var] = helicon.guess_data_type(val)(val)
                else:
                    data.loc[:, var] = helicon.guess_data_type(val)(val)
            index_d[option_name] += 1

        elif (
            option_name == "setBeamTiltClass" and param > 0
        ):  # one group per micrograph
            micrographNames = (
                data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
            )
            mgraphs = micrographNames.groupby(micrographNames, sort=False)

            for mi, mgraph in enumerate(mgraphs):
                mgraphName, mgraphParticles = mgraph
                data.loc[mgraphParticles.index, "rlnBeamTiltClass"] = mi + 1
            index_d[option_name] += 1

        elif option_name == "psiPrior180" and param:
            var = "rlnAnglePsiPrior"
            if var not in data:
                helicon.color_print(
                    f"ERROR: parameter {var} does not exist. Cannot add a value to it"
                )
                sys.exist(-1)
            data1 = data
            data2 = data1.copy()
            data2.loc[:, var] += 180.0
            var = "rlnHelicalTubeID"
            if var in data2:
                idMax = data2[var].astype(int).max()
                idMax = helicon.ceil_power_of_10(idMax)
                data2.loc[:, var] += idMax
            data = pd.concat((data1, data2), axis=0)
            try:
                data.attrs = data1.attrs
            except:
                pass
            index_d[option_name] += 1

        elif option_name == "addParm" and len(param) == 2:
            var, val = param
            if var not in data:
                helicon.color_print(
                    f"ERROR: parameter {var} does not exist. Cannot add a value to it"
                )
            data.loc[:, var] += float(val)
            index_d[option_name] += 1

        elif option_name == "multParm" and len(param) == 2:
            var, val = param
            if var not in data:
                helicon.color_print(
                    f"ERROR: parameter {var} does not exist. Cannot multiply it by another value"
                )
            data[var] *= float(val)
            index_d[option_name] += 1

        elif option_name == "duplicateParm" and len(param):
            for var_from, var_to in param:
                if var_from not in data:
                    helicon.color_print(
                        (
                            "\tWARNING: %s does not exist. Cannot duplicate %s to %s"
                            % (var_from, var_from, var_to)
                        )
                    )
                    continue
                if var_to in data:
                    helicon.color_print(
                        (
                            "\tWARNING: %s already exists. Will not duplicating %s to %s"
                            % (var_to, var_from, var_to)
                        )
                    )
                    continue
                data[var_to] = data[var_from]
            index_d[option_name] += 1

        elif option_name == "renameParm" and len(param):
            cols = {}
            for parm in zip(*[iter(param)] * 2):
                var_old, var_new = parm
                if var_old not in data:
                    helicon.color_print(
                        (
                            "\tWARNING: %s does not exist. Cannot rename %s to %s"
                            % (var_old, var_old, var_new)
                        )
                    )
                    continue
                if var_new in data:
                    helicon.color_print(
                        (
                            "\tWARNING: %s already exists. Cannot duplicate %s to %s"
                            % (var_new, var_old, var_new)
                        )
                    )
                    continue
                cols[var_old] = var_new
            data.rename(columns=cols, inplace=True)
            index_d[option_name] += 1

        elif option_name == "keepParm" and len(param):
            dropParms = [c for c in data if c not in param]
            data = data.drop(dropParms, inplace=False, axis=1)
            index_d[option_name] += 1

        elif option_name == "delParm" and len(param):
            invalidParms = []
            dropParms = []
            for p in param:
                p = p.strip("_")
                if p in data:
                    dropParms.append(p)
                else:
                    invalidParms.append(p)
            if invalidParms:
                helicon.color_print(f"\tWARNING: {invalidParms} do not exist")
            if dropParms:
                data = data.drop(dropParms, inplace=False, axis=1)
            index_d[option_name] += 1

        # copy parameters in the other star file
        elif option_name == "copyParm" and len(param) >= 1:
            starFile = param[0]
            try:
                vars = param[1:]
            except:
                vars = []  # copy all parameters if a list is not specified

            data.convention = "relion"
            data = data.drop_duplicates(
                subset=["rlnImageName"], keep="last", inplace=False
            )

            data2 = helicon.images2dataframe(
                starFile,
                alternative_folders=args.folder,
                ignore_bad_particle_path=args.ignoreBadParticlePath,
                ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
                warn_missing_ctf=0,
                target_convention="relion",
            )
            data2 = data2.drop_duplicates(
                subset=["rlnImageName"], keep="last", inplace=False
            )
            if args.verbose > 1:
                print(("\tRead in %d particles from %s" % (len(data2), starFile)))

            if len(data) > len(data2):
                helicon.color_print(
                    (
                        "\tERROR: --copyParm option requires that %s (%d) has the same number or more particles (>=%d needed)"
                        % (starFile, len(data2), len(data))
                    )
                )
                sys.exit(-1)

            if not (set(data[["rlnImageName"]]).issubset(set(data2[["rlnImageName"]]))):
                helicon.color_print(
                    (
                        "\tERROR: --copyParm option requires that %s contains identical set or a superset of particles"
                        % (starFile)
                    )
                )
                sys.exit(-1)

            if len(vars):
                copyVars = [v for v in vars if v[0] != "~"]
                skipVars = [v[1:] for v in vars if v[0] == "~"]
                if skipVars and args.verbose > 1:
                    print(("\tSkipping parameters: %s" % (" ".join(skipVars))))
                if len(copyVars):
                    invalidParms = [v for v in copyVars if v not in data2]
                    if len(invalidParms):
                        helicon.color_print(
                            (
                                '\tWARNING: parameters "%s" not in the list of parameters %s from file %s. ignored'
                                % (
                                    " ".join(invalidParms),
                                    list(data2.columns),
                                    starFile,
                                )
                            )
                        )
                    validParms = [v for v in copyVars if v in data2]
                else:
                    validParms = [
                        v for v in data2 if v not in skipVars + ["rlnImageName"]
                    ]
            else:
                validParms = [v for v in data2 if v not in ["rlnImageName"]]

            if args.verbose > 1:
                print(("\tCopying parameters: %s" % (" ".join(validParms))))

            for v in validParms:
                if v not in data:
                    data.loc[:, v] = np.nan

            from helicon import convert_dataframe_file_path

            data.loc[:, "rlnImageName_abs"] = convert_dataframe_file_path(
                data, "rlnImageName", to="abs"
            )
            data2.loc[:, "rlnImageName_abs"] = convert_dataframe_file_path(
                data2, "rlnImageName", to="abs"
            )
            data.set_index(["rlnImageName_abs"], inplace=True)
            data2.set_index(["rlnImageName_abs"], inplace=True)
            data[validParms] = data2.loc[data.index, validParms]
            data.reset_index(drop=True, inplace=True)
            index_d[option_name] += 1

        elif option_name == "replaceStr" and len(param) == 3:
            var, oldStr, newStr = param
            if var in data:
                data[var] = data[var].str.replace(oldStr, newStr)
            else:
                helicon.color_print(
                    ("\tWARNING: variable %s does not exist. Skipped" % (var))
                )
            index_d[option_name] += 1

        elif option_name == "replaceImageName" and param:
            replaceImageName = param
            if not os.path.exists(replaceImageName):
                helicon.color_print(("\tERROR: %s does not exist" % (replaceImageName)))
                sys.exit(-1)

            nImage = helicon.EMUtil.get_image_count(replaceImageName)
            if nImage != len(data):
                helicon.color_print(
                    f"\tERROR: {replaceImageName} contains {len(nImage)} particles, different from the expected {len(data)} particles"
                )
                sys.exit(-1)

            data["rlnImageName"] = (
                pd.Series(list(range(1, nImage + 1))).map("{:06d}".format)
                + "@"
                + replaceImageName
            )
            index_d[option_name] += 1

        elif option_name == "setCTF" and param:
            setCTF = param
            data["rlnVoltage"] = 0
            data["rlnSphericalAberration"] = 0
            data["rlnAmplitudeContrast"] = 0
            if "rlnDetectorPixelSize" not in data:
                data["rlnDetectorPixelSize"] = 5
            data["rlnMagnification"] = 0
            data["rlnDefocusU"] = 0
            data["rlnDefocusV"] = 0
            data["rlnDefocusAngle"] = 0

            ctfparms = readCtfparmFile(setCTF)

            micrographNames = (
                data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
            )
            mgraphs = micrographNames.groupby(micrographNames, sort=False)

            def setMicrographCTF(mgraphName, mgraphParticles, data, ctfparms):
                mid = os.path.basename(mgraphName)
                mid = os.path.splitext(mid)[0]
                mid2 = mid.split(".")[0]

                d = None
                if mid in ctfparms:
                    d = ctfparms[mid]
                elif mid2 in ctfparms:
                    d = ctfparms[mid2]
                else:
                    helicon.color_print(
                        "\tERROR: cannot find ctf parmeters for micrograph %s"
                        % (mgraphName)
                    )
                    sys.exit(1)

                data.loc[mgraphParticles.index, "rlnVoltage"] = d["voltage"]
                data.loc[mgraphParticles.index, "rlnSphericalAberration"] = d["cs"]
                data.loc[mgraphParticles.index, "rlnAmplitudeContrast"] = (
                    d["ampcont"] / 100.0
                )  # [0, 100] -> [0, 1]
                data.loc[mgraphParticles.index, "rlnMagnification"] = (
                    data.loc[mgraphParticles.index, "rlnDetectorPixelSize"]
                    * 1e4
                    / d["apix"]
                )

                rlnDefocusU, rlnDefocusV, rlnDefocusAngle = (
                    helicon.eman_astigmatism_to_relion(
                        d["defocus"], d["dfdiff"], d["dfang"]
                    )
                )
                data.loc[mgraphParticles.index, "rlnDefocusU"] = rlnDefocusU
                data.loc[mgraphParticles.index, "rlnDefocusV"] = rlnDefocusV
                data.loc[mgraphParticles.index, "rlnDefocusAngle"] = rlnDefocusAngle

            for mgraphName, mgraphParticles in mgraphs:
                setMicrographCTF(mgraphName, mgraphParticles, data, ctfparms)
            index_d[option_name] += 1

        elif option_name == "copyCtf" and len(param) >= 1:
            targetStarFile = param
            print(f"Copying CTF parameters from {targetStarFile}")

            data.convention = "relion"
            data = data.drop_duplicates(
                subset=["rlnImageName"], keep="last", inplace=False
            )

            data2 = helicon.images2dataframe(
                targetStarFile,
                alternative_folders=args.folder,
                ignore_bad_particle_path=args.ignoreBadParticlePath,
                ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
                warn_missing_ctf=1,
                target_convention="relion",
            )
            data2 = data2.drop_duplicates(
                subset=["rlnImageName"], keep="last", inplace=False
            )

            if args.verbose > 1:
                print(("\tRead in %d particles from %s" % (len(data2), targetStarFile)))

            common_optics_groups = set(optics["rlnOpticsGroup"].values) & set(
                data2.attrs["optics"]["rlnOpticsGroup"].values
            )
            if common_optics_groups:
                # copy 'rlnBeamTiltX', 'rlnBeamTiltY', 'rlnOddZernike', 'rlnEvenZernike' from the same optics group
                ctf_parms_candidate = [
                    "rlnBeamTiltX",
                    "rlnBeamTiltY",
                    "rlnOddZernike",
                    "rlnEvenZernike",
                ]
                ctf_parms = []
                for key in ctf_parms_candidate:
                    if key in data2.attrs["optics"]:
                        ctf_parms.append(key)
                        if key not in optics:
                            optics.loc[:, key] = 0
                if ctf_parms:
                    # for optics_group in data2.attrs["optics"].loc[:,'rlnOpticsGroup']:
                    #    if optics_group in optics['rlnOpticsGroup'].values:
                    #        optics.loc[optics['rlnOpticsGroup']==optics_group,ctf_parms]=data2.attrs["optics"].loc[data.attrs["optics"]['rlnOpticsGroup']==optics_group,ctf_parms].values
                    for optics_group in common_optics_groups:
                        optics.loc[
                            optics["rlnOpticsGroup"] == optics_group, ctf_parms
                        ] = (
                            data2.attrs["optics"]
                            .loc[
                                data.attrs["optics"]["rlnOpticsGroup"] == optics_group,
                                ctf_parms,
                            ]
                            .values
                        )
                    data.attrs["optics"] = optics

            # copy from the same micrograph (average for particles in the same micrograph)
            ctf_parms = [
                "rlnDefocusU",
                "rlnDefocusV",
                "rlnDefocusAngle",
                "rlnCtfBfactor",
                "rlnCtfScalefactor",
                "rlnPhaseShift",
            ]
            for v in ctf_parms:
                if v not in data:
                    data.loc[:, v] = np.nan

            data2["mean_defocus"] = (data2["rlnDefocusU"] + data2["rlnDefocusV"]) / 2
            data2["delta_defocus"] = (data2["rlnDefocusU"] - data2["rlnDefocusV"]) / 2
            data2["astig_x"] = data2["delta_defocus"] * np.cos(
                np.deg2rad(data2["rlnDefocusAngle"])
            )
            data2["astig_y"] = data2["delta_defocus"] * np.sin(
                np.deg2rad(data2["rlnDefocusAngle"])
            )
            data2 = data2.groupby("rlnMicrographName").mean()
            data2["mean_astig"] = np.sqrt(data2["astig_x"] ** 2 + data2["astig_y"] ** 2)
            data2["mean_astig_angle"] = (
                np.arctan2(data2["astig_y"], data2["astig_x"]) * 180 / np.pi
            )
            for micrograph in data2.index:
                if micrograph in data["rlnMicrographName"].values:
                    micrograph_rows = data["rlnMicrographName"] == micrograph
                    data.loc[micrograph_rows, "rlnDefocusU"] = (
                        data2.loc[micrograph, "mean_defocus"]
                        + data2.loc[micrograph, "mean_astig"]
                    )
                    data.loc[micrograph_rows, "rlnDefocusV"] = (
                        data2.loc[micrograph, "mean_defocus"]
                        - data2.loc[micrograph, "mean_astig"]
                    )
                    data.loc[
                        micrograph_rows,
                        [
                            "rlnDefocusAngle",
                            "rlnCtfBfactor",
                            "rlnCtfScalefactor",
                            "rlnPhaseShift",
                        ],
                    ] = data2.loc[
                        micrograph,
                        [
                            "mean_astig_angle",
                            "rlnCtfBfactor",
                            "rlnCtfScalefactor",
                            "rlnPhaseShift",
                        ],
                    ].values

        elif option_name in ["sortby", "rsortby"] and param:
            sortby = param
            # check if the parameters are in the dataframe
            badParms = [v for v in sortby if v not in data.columns]
            if badParms:
                if len(badParms) > 1:
                    helicon.color_print(
                        "\tERROR: parameters %s do not exist" % (" ".join(badParms))
                    )
                else:
                    helicon.color_print(
                        "\tERROR: parameter %s does not exist" % (badParms[0])
                    )
                sys.exit(-1)

            tmpCol = "tmp_sort_rlnImageName"
            if "rlnImageName" in sortby:
                tmp = data["rlnImageName"].str.split("@", expand=True)
                data[tmpCol] = tmp.iloc[:, -1] + "@" + tmp.iloc[:, 0]
                sortby = [tmpCol if v == "rlnImageName" else v for v in sortby]

            ascending = False if option_name == "rsortby" else True
            data_sorted = data.sort_values(sortby, ascending=ascending)
            if tmpCol in data_sorted.columns:
                data = data_sorted.drop(tmpCol, axis=1)
            else:
                data = data_sorted
            data.reset_index(drop=True, inplace=True)
            index_d[option_name] += 1

        elif option_name == "path" and param != "current":
            path = param
            from helicon import convert_dataframe_file_path

            for attr in "rlnImageName rlnMicrographName".split():
                if attr in data:
                    from helicon import get_relion_project_folder

                    relion_proj_folder = get_relion_project_folder(
                        os.path.abspath(args.output_starFile)
                    )
                    relpath_start = (
                        os.path.dirname(os.path.abspath(args.output_starFile))
                        if relion_proj_folder is None
                        else relion_proj_folder
                    )
                    data[attr] = convert_dataframe_file_path(
                        data, attr, to=path, relpath_start=relpath_start
                    )
            index_d[option_name] += 1

        elif option_name == "minStack" and param:
            tmp = data["rlnImageName"].str.split("@", expand=True)
            indices, micrographNames = tmp.iloc[:, 0], tmp.iloc[:, -1]

            mgraphs = micrographNames.groupby(micrographNames, sort=False)

            count = 0
            subdir = os.path.splitext(args.output_starFile)[0]
            if not os.path.isdir(subdir):
                os.mkdir(subdir)

            for mgraphName, mgraphParticles in mgraphs:
                mgraphName2 = os.path.join(subdir, os.path.basename(mgraphName))
                n = len(mgraphParticles)
                if not (
                    os.path.exists(mgraphName2)
                    and helicon.EMUtil.get_image_count(mgraphName2) == n
                ):
                    particles_indices = sorted(
                        list(indices.iloc[mgraphParticles.index].astype(int))
                    )
                    for i in range(n):
                        i2 = particles_indices[i] - 1
                        d.read_image(mgraphName, i2)
                        d.write_image(mgraphName2, i)
                rlnImageName = (
                    pd.Series(list(range(1, n + 1))).map("{:06d}".format)
                    + "@"
                    + mgraphName2
                )
                data.loc[mgraphParticles.index, "rlnImageName"] = (
                    rlnImageName.values
                )  # critical to use values, otherwise the index-aligned assignment will mess up the values
                count += 1
                if args.verbose:
                    print(
                        "\t%d/%d: %d/%d images in %s saved to %s"
                        % (
                            count,
                            len(mgraphs),
                            len(mgraphParticles),
                            helicon.EMUtil.get_image_count(mgraphName),
                            mgraphName,
                            mgraphName2,
                        )
                    )
            index_d[option_name] += 1

        elif option_name == "fullStack" and param:
            valid_cols = set(
                "rlnVoltage rlnDefocusU rlnDefocusV rlnDefocusAngle rlnSphericalAberration rlnDetectorPixelSize rlnMagnification rlnAmplitudeContrast rlnMicrographName rlnGroupName rlnGroupNumber".split()
            )
            cols_to_keep = [c for c in data if c in valid_cols]

            micrographNames = (
                data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
            )

            mgraphs = micrographNames.groupby(micrographNames, sort=False)

            dataframes = []
            count = 0
            for mgraphName, mgraphParticles in mgraphs:
                n = helicon.EMUtil.get_image_count(mgraphName)
                rlnImageName = (
                    pd.Series(list(range(1, n + 1))).map("{:06d}".format)
                    + "@"
                    + mgraphName
                )
                df = pd.DataFrame()
                df["rlnImageName"] = rlnImageName
                tmpdf = data.loc[mgraphParticles.index]
                for ci, c in enumerate(cols_to_keep):
                    df[c] = tmpdf[c].values[0]
                dataframes.append(df)
                count += 1
                if args.verbose:
                    print(
                        "\t%d/%d: %s:\t%d -> %d images"
                        % (count, len(mgraphs), mgraphName, len(mgraphParticles), n)
                    )
            data = pd.concat(dataframes)
            data = data.reset_index(drop=True)  # important to do this
            index_d[option_name] += 1

        elif option_name == "createStack" and param:
            # outputFile:rescale2size=<n>:float16=<0|1>
            outputFile, param_dict = helicon.parsemodopt(param)
            if os.path.splitext(outputFile)[1] != ".mrcs":
                suffix = Path(outputFile).suffix
                helicon.color_print(
                    f"\tERROR: a .mrcs file is expected while you have specified {outputFile}! I will not do anything"
                )
                continue

            images = data["rlnImageName"].str.split("@", expand=True)
            images.columns = ["pid", "filename"]
            images.loc[:, "pid"] = images.loc[:, "pid"].astype(int)

            attr = helicon.unique_attr_name(data, attr_prefix="rlnImageNameOrig")
            data[attr] = data["rlnImageName"]

            nx, ny, _ = helicon.get_image_size(images["filename"].iloc[0])
            nImage = len(data)

            newsize = int(param_dict.get("rescale2size", nx))
            float16 = int(param_dict.get("float16", 1))

            import mrcfile

            force = int(param_dict.get("force", 0))
            if not force:
                if os.path.exists(outputFile):
                    with mrcfile.open(outputFile, header_only=True) as mrc:
                        if not (
                            mrc.header.nx == newsize
                            and mrc.header.ny == newsize
                            and mrc.header.nz == nImage
                        ):
                            force = 1
                else:
                    force = 1
            if force:
                from tqdm import tqdm

                if float16:
                    mrc_mode = 12
                else:
                    mrc_mode = 2
                with mrcfile.new_mmap(
                    outputFile,
                    shape=(nImage, newsize, newsize),
                    mrc_mode=mrc_mode,
                    fill=None,
                    overwrite=True,
                ) as mrc:
                    apix0 = None
                    for i in tqdm(
                        list(range(nImage)), unit=" particles", disable=args.verbose > 1
                    ):
                        if args.verbose > 1:
                            print(
                                "\t%d/%d: adding %s:%d"
                                % (
                                    i + 1,
                                    nImage,
                                    images["filename"].iloc[i],
                                    images["pid"].iloc[i],
                                )
                            )
                        d = helicon.read_image_2d(
                            images["filename"].iloc[i], int(images["pid"].iloc[i] - 1)
                        )
                        if apix0 is None:
                            apix0 = d["apix_x"]
                        if newsize < nx:
                            d = d.FourTruncate(
                                newsize, newsize, 1, 1
                            )  # crop Fourier transform
                        elif newsize > nx:
                            d = d.FourInterpol(
                                newsize, newsize, 1, 1
                            )  # pad Fourier transform
                        d_numpy = helicon.EMNumPy.em2numpy(d)
                        mrc.data[i, :, :] = d_numpy
                    mrc.voxel_size = apix0 * nx / newsize
            images.loc[:, "pid"] = np.arange(nImage) + 1
            data.loc[:, "rlnImageName"] = images["pid"].astype(str) + "@" + outputFile
            if optics is not None and newsize != nx:
                optics.loc[:, "rlnImageSize"] = newsize
                if "rlnImagePixelSize" in optics:
                    optics.loc[:, "rlnImagePixelSize"] = (
                        optics.loc[:, "rlnImagePixelSize"] * nx / newsize
                    )
            index_d[option_name] += 1

        elif option_name == "calibratePixelSize" and param:
            standard_sample = param
            # ice ring at 3.661Å: https://journals.iucr.org/d/issues/2021/04/00/tz5104/index.html
            supported_standards = dict(
                graphene=2.13, graphene_oxide=2.13, go=2.13, gold=2.355, ice=3.661
            )  # unit: Angstrom
            target_res = supported_standards[standard_sample.lower()]
            apix, pixelSize_source = getPixelSize(data, return_pixelSize_source=True)
            if apix is None:
                helicon.color_print(
                    '\tERROR: cannot find "rlnImagePixelSize" or "rlnMicrographPixelSize"'
                )
                sys.exit(-1)
            half_corner_res = 1.0 / (1 / (2 * apix) * (1 + np.sqrt(2)) / 2)
            if target_res <= half_corner_res:
                helicon.color_print(
                    f"\tERROR: target resolution {target_res} Å for {param} is beyond the limit ({half_corner_res:.2f} Å = (1+sqrt(2))/2 * Nyquist resolution)"
                )
                sys.exit(-1)

            search_range = 0.05  #  # default range: +/- 5%
            corner_res = 2 * apix / np.sqrt(2)
            res_low = target_res * (1 + search_range)
            res_high = max(corner_res, target_res * (1 - search_range))
            r_samples = 100  # 0.1% stepsize
            theta_samples = (
                int(
                    np.pi
                    / (
                        (1 / res_high - 1 / res_low)
                        / (r_samples - 1)
                        / (1 / target_res)
                    )
                )
                + 1
            )  # equal sampling in radial and angular directions
            if args.verbose > 1:
                if args.verbose > 2:
                    print(
                        f"\tCurrent {pixelSize_source}: {apix} Å (Nyquist={2*apix} Å)"
                    )
                print(
                    f"\tResolution range (±{search_range*100}%) to search for diffraction peak ({target_res} Å) of {param}: {res_low:.2f} -> {res_high:.2f} Å"
                )

            def fft_resolution_range(
                images,
                apix,
                res_low=0,
                res_high=0,
                r_samples=-1,
                theta_samples=180,
                return_R_only=False,
            ):
                import finufft
                import numpy as np

                R0 = 1 / res_low if res_low > 0 else 0
                R1 = 1 / res_high if res_high > 0 else 1 / (2 * apix)
                nr = r_samples if r_samples > 0 else min(images.shape[-2:]) // 2
                R = np.linspace(start=R0, stop=R1, num=nr, endpoint=True)
                if return_R_only:
                    return R

                Theta = np.linspace(
                    start=0, stop=np.pi, num=theta_samples, endpoint=False
                )
                Theta, R = np.meshgrid(Theta, R, indexing="ij")
                Y = (2 * np.pi * apix * R * np.sin(Theta)).flatten(order="C")
                X = (2 * np.pi * apix * R * np.cos(Theta)).flatten(order="C")
                from finufft import nufft2d2

                if len(images.shape) > 2:
                    if len(images) > 1:
                        fft = nufft2d2(
                            x=X, y=Y, f=images.astype(np.complex128), eps=1e-6
                        )
                    else:
                        fft = nufft2d2(
                            x=X, y=Y, f=images[0].astype(np.complex128), eps=1e-6
                        )
                else:
                    fft = nufft2d2(x=X, y=Y, f=images.astype(np.complex128), eps=1e-6)
                if len(images.shape) > 2:
                    new_shape = list(images.shape[:-2]) + list(R.shape)
                else:
                    new_shape = R.shape
                fft = fft.reshape(new_shape)
                return fft

            def calibrateMag_process_one_micrograph(
                imageFile, apix, res_low, res_high, r_samples, theta_samples
            ):
                import mrcfile

                with mrcfile.open(imageFile) as mrc:
                    images = mrc.data
                if len(images.shape) == 2:
                    images = np.expand_dims(images, axis=0)
                fft = fft_resolution_range(
                    images,
                    apix,
                    res_low,
                    res_high,
                    r_samples,
                    theta_samples,
                    return_R_only=False,
                )
                pwr = np.abs(fft)
                pwr_1d = pwr.max(axis=tuple(range(len(pwr.shape) - 1)))
                # pwr_1d = pwr.max(axis=1)
                # pwr_1d = pwr_1d.mean(axis=0)
                pwr_1d -= np.median(pwr_1d)
                from scipy.stats import median_abs_deviation

                pwr_curve = pwr_1d / median_abs_deviation(pwr_1d)
                n_ptcl = len(images)
                return (pwr_curve, n_ptcl)

            mapping = dict(
                rlnImagePixelSize="rlnImageName",
                rlnMicrographPixelSize="rlnMicrographName",
            )
            imageFiles = (
                data[mapping[pixelSize_source]]
                .str.split("@", expand=True)
                .iloc[:, -1]
                .unique()
            )

            from tqdm import tqdm
            from joblib import Parallel, delayed

            results = list(
                tqdm(
                    Parallel(
                        return_as="generator",
                        n_jobs=args.cpu if len(imageFiles) > 1 else 1,
                    )(
                        delayed(calibrateMag_process_one_micrograph)(
                            imageFile, apix, res_low, res_high, r_samples, theta_samples
                        )
                        for imageFile in imageFiles
                    ),
                    unit="micrograph",
                    total=len(imageFiles),
                    disable=len(imageFiles) < 2
                    or (args.cpu > 1 and args.verbose < 2)
                    or (args.cpu == 1 and args.verbose != 2),
                )
            )

            pwr_curves = []
            n_ptcls = []
            for result in results:
                pwr_curve, n_ptcl = result
                pwr_curves.append(pwr_curve)
                n_ptcls.append(n_ptcl)
            pwr_curves = np.vstack(pwr_curves)
            n_ptcls = np.array(n_ptcls)
            pwr_mean = (
                np.sum(pwr_curves * np.expand_dims(n_ptcls, axis=1), axis=0)
                / n_ptcls.sum()
            )
            from scipy.signal import detrend

            pwr_mean = detrend(pwr_mean)

            import mrcfile

            with mrcfile.open(imageFiles[0]) as mrc:
                images = mrc.data
            R = fft_resolution_range(
                images[0],
                apix,
                res_low,
                res_high,
                r_samples,
                theta_samples=180,
                return_R_only=True,
            )
            res_peak = 1 / R[np.argmax(pwr_mean)]
            apix_new = round(apix * target_res / res_peak, 3)  # precision: 0.1%
            if args.verbose > 1:
                outputFile = (
                    os.path.splitext(args.output_starFile)[0]
                    + f".calibrateMag.{pixelSize_source}={apix}.txt"
                )
                np.savetxt(
                    outputFile,
                    np.hstack((R.reshape((len(R), 1)), pwr_mean.reshape((len(R), 1)))),
                )
                print(
                    f"\tAverage power spectra saved to {outputFile} using the original {pixelSize_source} {apix}"
                )
                import matplotlib.pyplot as plt

                plt.plot(R, pwr_mean, label=f"original pixel size={apix}")
                plt.axvline(x=1 / target_res, color="r", linestyle="dashed")
                plt.xlabel(f"Spatial Frequency (1/Å)")
                plt.ylabel("Power Spectra")
                plt.title(f"{standard_sample}: expected peak at {target_res} Å")
            if apix_new != apix:
                if args.verbose > 1:
                    R2 = R * apix / apix_new
                    outputFile2 = (
                        os.path.splitext(args.output_starFile)[0]
                        + f".calibrateMag.{pixelSize_source}={apix_new}.txt"
                    )
                    np.savetxt(
                        outputFile2,
                        np.hstack(
                            (R2.reshape((len(R), 1)), pwr_mean.reshape((len(R), 1)))
                        ),
                    )
                    print(
                        f"\tAverage power spectra also saved to {outputFile2} using the new, calibrated {pixelSize_source} {apix_new}"
                    )
                    plt.plot(R2, pwr_mean, label=f"calibrated pixel size={apix_new}")
                setPixelSize(data, apix_new=apix_new, update_defocus=True)
                if args.verbose > 0:
                    print(
                        f"\tCalibrated {pixelSize_source}: {apix_new} ({100*(apix_new-apix)/apix:.1f}% from the original {pixelSize_source} {apix}). {pixelSize_source}, rlnDefocusU, and rlnDefocusV have been updated to use the calibrated {pixelSize_source} {apix_new}"
                    )
            else:
                if args.verbose > 0:
                    print(
                        f"\tCongratulations! Your original {pixelSize_source} {apix} is accurate without a need to adjust"
                    )
            if args.verbose > 1:
                plt.legend(loc="upper right")
                plt.show()

        elif option_name == "process" and param:
            process = param

            data_tmp = helicon.dataframe_convert(data, target="jspr")
            data_tmp = helicon.dataframe_jspr2dict(data_tmp)

            processors = []
            for p in process:
                processorname, param_dict = helicon.parsemodopt(p)
                if not param_dict:
                    param_dict = {}
                if processorname in helicon.outplaceprocs:
                    processors.append((processorname, param_dict, 1))
                else:
                    processors.append((processorname, param_dict, 0))

            tag = args.tag if args.tag else "-".join([p[0] for p in processors])
            if tag:
                tag = "." + tag.strip(".")

            micrographNames = (
                data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
            )
            mgraphs = micrographNames.groupby(micrographNames, sort=False)

            mcount = 0
            d = helicon.EMData()
            for mgraphName, mgraphParticles in mgraphs:
                tmpdata = data.loc[mgraphParticles.index]
                filename = tmpdata["rlnImageName"].iloc[0].split("@")[-1]
                newfilename = os.path.splitext(filename)[0] + tag + ".mrcs"
                if not os.access(os.path.dirname(newfilename), os.W_OK):
                    newfilename = os.path.basename(newfilename)
                pcount = 0
                for ri, row in tmpdata.iterrows():
                    pid, filename = row["rlnImageName"].split("@")
                    pid = int(pid) - 1
                    d.read_image(filename, pid)

                    attrs = d.get_attr_dict()
                    attrs.update(data_tmp[ri])
                    d.set_attr_dict(attrs)

                    for processorName, processorparams, outplace in processors:
                        if outplace:
                            d = d.process(processorName, processorparams)
                        else:
                            d.process_inplace(processorName, processorparams)
                    d.write_image(newfilename, pcount)
                    pcount += 1
                mcount += 1
                if args.verbose:
                    print(
                        (
                            "\tMicrograph %d/%d: %d particles from %s are processed and saved to %s"
                            % (mcount, len(mgraphs), pcount, filename, newfilename)
                        )
                    )
                data.loc[mgraphParticles.index, "rlnImageName"] = (
                    pd.Series(list(range(1, pcount + 1))).map("{:06d}".format)
                    + "@"
                    + newfilename
                ).tolist()
            index_d[option_name] += 1

        elif option_name == "maskGold" and param:
            attrs_required = "rlnImageName rlnMicrographName".split()
            attrSrc = helicon.first_matched_atrr(data, attrs_required)
            if attrSrc is None:
                helicon.color_print(
                    f"ERROR: the input does not have any of the columns: {' '.john(attr_required)}"
                )
                sys.exit(-1)

            # value_sigma=<n>:gradient_sigma=<Å>:min_area=<Å^2>:both_sides=<0|1>:outdir=<str>:force=<0|1>:cpu=<n>
            param_dict = helicon.parsemodopt2(param)
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
                        print(
                            f"\tgradient_sigma is set to {gradient_sigma:.1f} Å to remove brightness gradient of the micrographs ({nx}x{ny} pixels)"
                        )

            tasks = []
            for mi, (mgraphName, mgraphParticles) in enumerate(mgraphs):
                pid = mgraphParticles["tmp_mgraph_pid"].astype(int) - 1
                outputFile = Path(outdir) / Path(mgraphName).name
                if outputFile.exists():
                    if outputFile.samefile(mgraphName):
                        helicon.color_print(
                            f"ERROR: output {outputFile.as_posix()} will overwrite original image"
                        )
                        sys.exit(-1)
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
                                        print(
                                            f"\tMicrograph {mi+1}/{len(mgraphs)}: {mgraphName} -> {outputFile.as_posix()} already done. skipped"
                                        )
                                    else:
                                        print(
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
                    print(f"\tStart maskGold task for {len(tasks)} micrographs")
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

        elif option_name == "estimateHelicalAngleVariance" and param:
            missing_attrs = [
                p
                for p in "rlnImageName rlnHelicalTubeID rlnAngleTilt rlnAnglePsi rlnAngleRot".split()
                if p not in data
            ]
            assert (
                missing_attrs == []
            ), f"\tERROR: requried parameters {' '.join(missing_attrs)} are not available"

            from helicon import convert_dataframe_file_path

            data.loc[:, "rlnImageName_abs"] = (
                convert_dataframe_file_path(data, "rlnImageName", to="abs")
                .str.split("@", expand=True)
                .iloc[:, -1]
            )
            groups = data.groupby(["rlnImageName_abs", "rlnHelicalTubeID"], sort=False)
            groups = [group_particles for group_name, group_particles in groups]
            nsegments = []
            tilt_means = []
            tilt_sigmas = []
            psi_sigmas = []
            rot_sigmas = []
            from scipy.stats import circmean, circstd
            from tqdm import tqdm

            for group_particles in tqdm(
                groups, unit=" filaments", disable=args.verbose < 1
            ):
                nsegments.append(len(group_particles))
                tilt = group_particles["rlnAngleTilt"].astype(np.float32).values
                tilt_means.append(np.rad2deg(circmean(np.deg2rad(tilt))))
                tilt_sigma = np.rad2deg(circstd(np.deg2rad(tilt)))
                data.loc[group_particles.index, "rlnAngleTiltSigma"] = round(
                    tilt_sigma, 2
                )
                tilt_sigmas.append(tilt_sigma)
                psi = group_particles["rlnAnglePsi"].astype(np.float32).values
                psi = np.rad2deg(
                    np.arccos(np.cos(2 * np.deg2rad(psi)))
                )  # to make the psi angles independent of polarity
                psi_sigma = np.rad2deg(circstd(np.deg2rad(psi)))
                data.loc[group_particles.index, "rlnAnglePsiSigma"] = round(
                    psi_sigma, 2
                )
                psi_sigmas.append(psi_sigma)
                rot = group_particles["rlnAngleRot"].astype(np.float32).values
                rot_sigma = np.rad2deg(circstd(np.deg2rad(rot)))
                data.loc[group_particles.index, "rlnAngleRotSigma"] = round(
                    rot_sigma, 2
                )
                rot_sigmas.append(rot_sigma)
            data = data.drop(["rlnImageName_abs"], inplace=False, axis=1)
            data = data.reset_index(drop=True)  # important to do this
            index_d[option_name] += 1
            if args.verbose > 1:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 14))
                for ai, angle in enumerate("Tilt Psi Rot".split()):
                    axes[0, ai].hist(
                        data[f"rlnAngle{angle}"],
                        bins=50,
                        edgecolor="white",
                        linewidth=1,
                    )
                    axes[0, ai].set(xlabel=f"{angle} (°)", ylabel="# Filaments")
                axes[0, 3].scatter(tilt_means, tilt_sigmas)
                axes[0, 3].set(xlabel=f"Tilt (°)", ylabel="Tilt Sigma (°)")
                angles = [
                    ("Tilt", tilt_sigmas),
                    ("Psi", psi_sigmas),
                    ("Rot", rot_sigmas),
                ]
                for ai, angle in enumerate(angles):
                    angle_str, angle_sigma = angle
                    axes[1, ai].hist(
                        angle_sigma, bins=50, edgecolor="white", linewidth=1
                    )
                    axes[2, ai].plot(range(len(angle_sigma)), sorted(angle_sigma))
                    hbin = axes[3, ai].hexbin(
                        nsegments, angle_sigma, bins="log", gridsize=50, cmap="jet"
                    )
                    fig.colorbar(hbin, ax=axes[3, ai], label="# Filaments")
                    axes[1, ai].set(
                        xlabel=f"{angle_str} Sigma (°)", ylabel="# Filaments"
                    )
                    axes[2, ai].set(
                        xlabel="Rank (# Filaments)", ylabel=f"{angle_str} Sigma (°)"
                    )
                    axes[3, ai].set(
                        xlabel="Filament Length (# Segments)",
                        ylabel=f"{angle_str} Sigma (°)",
                    )
                from itertools import combinations

                for pi, pair in enumerate(combinations(angles, 2)):
                    (angle_str_1, angle_sigma_1), (angle_str_2, angle_sigma_2) = pair
                    hbin = axes[pi + 1, 3].hexbin(
                        angle_sigma_1,
                        angle_sigma_2,
                        bins="log",
                        gridsize=50,
                        cmap="jet",
                    )
                    fig.colorbar(hbin, ax=axes[pi + 1, 3], label="# Filaments")
                    axes[pi + 1, 3].set(
                        xlabel=f"{angle_str_1} Sigma (°)",
                        ylabel=f"{angle_str_2} Sigma (°)",
                    )
                plt.savefig(
                    f"{os.path.splitext(args.output_starFile)[0]}.tilt_psi_rot_sigma.pdf"
                )
                plt.tight_layout()
                plt.show()

        elif option_name == "estimateHelicalTubeLength" and param:
            badParms = [
                v
                for v in "rlnImageName rlnHelicalTubeID rlnCoordinateX rlnCoordinateY".split()
                if v not in data
            ]
            if badParms:
                s = "s" if len(badParms) > 1 else ""
                helicon.color_print(
                    f"\tERROR: parameter{s} {' '.join(badParms)} do not exist"
                )
                sys.exit(-1)
            data = estimate_helicalTube_length(data, verbose=args.verbose)

        elif option_name == "resetInterSegmentDistance" and param > 0:
            badParms = [
                v
                for v in "rlnImageName rlnHelicalTubeID rlnCoordinateX rlnCoordinateY".split()
                if v not in data
            ]
            if badParms:
                s = "s" if len(badParms) > 1 else ""
                helicon.color_print(
                    f"\tERROR: parameter{s} {' '.join(badParms)} do not exist"
                )
                sys.exit(-1)

            apix_micrograph = 0
            try:
                optics = data.attrs["optics"]
                for attr in [
                    "rlnMicrographPixelSize",
                    "rlnMicrographOriginalPixelSize",
                ]:
                    if attr in optics:
                        apix_micrograph = optics[attr].iloc[0]
                        break
            finally:
                if apix_micrograph <= 0:
                    helicon.color_print(
                        "\tERROR: neither rlnMicrographPixelSize nor rlnMicrographOriginalPixelSize is available"
                    )
                    sys.exit(-1)

            data = reset_inter_segment_distance(
                data,
                new_inter_segment_distance=param,
                apix_micrograph=apix_micrograph,
                verbose=args.verbose,
            )

        elif option_name == "keepOneParticlePerHelicalTube" and param:
            var = ""
            for v in "rlnMicrographName rlnImageName".split():
                if v in data:
                    var = v
                    break
            if not var:
                helicon.color_print(
                    f"\tERROR: rlnMicrographName or rlnImageName must be available"
                )
                sys.exit(-1)
            if "rlnHelicalTubeID" not in data:
                helicon.color_print(
                    f"\tERROR: parameter rlnHelicalTubeID is not available"
                )
                sys.exit(-1)

            if "@" in data[var].iloc[0]:
                tmp = data.loc[:, var].str.split("@", expand=True)
                var = "filename"
                data.loc[:, var] = tmp.iloc[:, 1]

            data = data.groupby(
                [var, "rlnHelicalTubeID"], as_index=False, sort=False
            ).first()
            if var == "filename":
                data.drop(["filename"], inplace=True, axis=1)
            if args.verbose > 1:
                print(f"\t{len(data)} helices found")

        elif option_name == "keepOneParticlePerMicrograph" and param:
            var = ""
            for v in "rlnMicrographName rlnImageName".split():
                if v in data:
                    var = v
                    break
            if not var:
                helicon.color_print(
                    f"\tERROR: rlnMicrographName or rlnImageName must be available"
                )
                sys.exit(-1)

            if "@" in data[var].iloc[0]:
                tmp = data.loc[:, var].str.split("@", expand=True)
                var = "filename"
                data.loc[:, var] = tmp.iloc[:, 1]

            data = data.groupby([var], as_index=False, sort=False).first()
            if var == "filename":
                data.drop(["filename"], inplace=True, axis=1)
            if args.verbose > 1:
                print("\t%d micrographs found" % (len(data)))

        elif option_name == "assignOpticGroupByBeamShift" and param != "no":
            # choices = "no auto EPU serialEM_pncc".split()
            try:
                optics_orig = data.attrs["optics"]
            except:
                optics_orig = None
            if optics_orig is None:
                helicon.color_print(f"\tERROR: data_optics block must be available")
                sys.exit(-1)

            image_name = helicon.first_matched_atrr(
                data,
                attrs="rlnMicrographMovieName rlnMicrographName rlnImageName".split(),
            )
            if image_name is None:
                helicon.color_print(
                    f"\tERROR: rlnMicrographMovieName, rlnMicrographName or rlnImageName must be available"
                )
                sys.exit(-1)

            required_cols = "rlnOpticsGroup".split()
            missing_cols = [c for c in required_cols if c not in data]
            if missing_cols:
                helicon.color_print(
                    f"\tERROR: required attrs {' '.join(missing_cols)} must be available"
                )
                sys.exit(-1)

            if param == "auto":
                format = helicon.guess_data_collection_software(
                    filename=data[image_name].iloc[0]
                )
                if format is None:
                    helicon.color_print(
                        f"\tERROR: cannot detect the format of filename {image_name}: {data[image_name].iloc[0]}"
                    )
                    sys.exit(-1)
                else:
                    if args.verbose > 1:
                        print(
                            f"\tAuto-detect the format as {format} based on {image_name}"
                        )
            else:
                format = param
                if (
                    helicon.verify_data_collection_software(
                        data[image_name].iloc[0], format
                    )
                    is None
                ):
                    helicon.color_print(
                        f"\tERROR: the specified format {format} is inconsistent with filename {image_name}: {data[image_name].iloc[0]}. If you are not sure, specify auto as the format and let me guess for you"
                    )
                    sys.exit(-1)

            optics = optics_orig.copy().iloc[0:0]

            tmp_col = "TEMP_beam_shift_pos"
            ogs = data.groupby("rlnOpticsGroup", sort=False)
            og_count = 0
            pattern = helicon.movie_filename_patterns()[format]
            for ogName, ogData in ogs:
                optics_row_index = optics_orig[
                    optics_orig["rlnOpticsGroup"].astype(str) == str(ogName)
                ].last_valid_index()
                ogData[tmp_col] = ogData.loc[:, image_name].str.extract(pattern)
                if format in ["EPU"]:
                    ogData[tmp_col] = ogData[tmp_col].astype(int)
                else:
                    ogData[tmp_col] = ogData[tmp_col].astype(str)
                unique_beam_shift_pos = sorted(ogData[tmp_col].unique())
                n = len(unique_beam_shift_pos)
                mapping = {
                    p: pi + 1 + og_count for pi, p in enumerate(unique_beam_shift_pos)
                }
                if args.verbose > 10:
                    print(f"{mapping=}")
                data.loc[ogData.index, "rlnOpticsGroup"] = ogData[tmp_col].map(mapping)
                new_rows = pd.concat(
                    [optics_orig.iloc[[optics_row_index]]] * n, ignore_index=True
                )
                new_rows["rlnOpticsGroup"] = np.arange(
                    og_count + 1, og_count + 1 + n, dtype=int
                )
                new_rows["rlnOpticsGroupName"] = "opticsGroup" + new_rows[
                    "rlnOpticsGroup"
                ].astype(str)
                optics = pd.concat([optics, new_rows], ignore_index=True)
                og_count += n
            data.attrs["optics"] = optics
            if args.verbose > 1:
                print(f"\t{len(ogs)} optics groups -> {len(optics)} optic groups")

        elif option_name == "assignOpticGroupByTime" and param > 0:
            try:
                optics_orig = data.attrs["optics"]
            except:
                optics_orig = None
            if optics_orig is None:
                helicon.color_print(f"\tERROR: data_optics block must be available")
                sys.exit(-1)

            image_name = helicon.first_matched_atrr(
                data,
                attrs="rlnMicrographMovieName rlnMicrographName rlnImageName".split(),
            )
            if image_name is None:
                helicon.color_print(
                    f"\tERROR: rlnMicrographMovieName, rlnMicrographName or rlnImageName must be available"
                )
                sys.exit(-1)

            software = helicon.guess_data_collection_software(
                filename=data[image_name].iloc[0]
            )
            if software in ["EPU"]:
                required_cols = "rlnOpticsGroup".split()
                if args.verbose > 2:
                    print(
                        f"\tIt appears that you used EPU to collect the movies. Data collection time will be extracted from the file names specified in the {image_name} column"
                    )
            else:
                required_cols = "rlnOpticsGroup rlnMicrographMovieName".split()
                image_name = "rlnMicrographMovieName"
                if args.verbose > 2:
                    helicon.color_print(
                        f"\tData collection time will use the file modification time of the movie files specified in the rlnMicrographMovieName column. Make sure that the file modification times are indeed the movie collection times"
                    )

            missing_cols = [c for c in required_cols if c not in data]
            if missing_cols:
                helicon.color_print(
                    f"\tERROR: required attrs {' '.join(missing_cols)} must be available"
                )
                sys.exit(-1)

            movies = data[image_name].unique()
            if software in ["EPU"]:
                moive2time = {
                    m: helicon.extract_EPU_data_collection_time(m) for m in movies
                }
            else:
                moive2time = {m: Path(m).resolve().stat().st_mtime for m in movies}
            from datetime import datetime

            moive2time_str = {
                m: datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H-%M-%S")
                for m, t in moive2time.items()
            }

            optics = optics_orig.copy().iloc[0:0]

            ogs = data.groupby("rlnOpticsGroup", sort=False)
            og_count = 0
            for ogName, ogData in ogs:
                optics_row_index = optics_orig[
                    optics_orig["rlnOpticsGroup"].astype(str) == str(ogName)
                ].last_valid_index()
                times = [moive2time[m] for m in ogData[image_name].unique()]
                time2group = helicon.assign_to_groups(times, n=param)
                movie2group = {
                    m: time2group[moive2time[m]] + og_count
                    for m in ogData[image_name].unique()
                }
                if args.verbose > 10:
                    print(
                        f"{movie2group=} {len(movie2group)=} {len(set(movie2group.values()))=}"
                    )
                data.loc[ogData.index, "rlnOpticsGroup"] = ogData[image_name].map(
                    movie2group
                )
                data.loc[ogData.index, "rlnMovieCollectionTime"] = ogData[
                    image_name
                ].map(moive2time_str)
                n = len(data.loc[ogData.index, "rlnOpticsGroup"].unique())
                new_rows = pd.concat(
                    [optics_orig.iloc[[optics_row_index]]] * n, ignore_index=True
                )
                new_rows["rlnOpticsGroup"] = np.arange(
                    og_count + 1, og_count + 1 + n, dtype=int
                )
                new_rows["rlnOpticsGroupName"] = "opticsGroup" + new_rows[
                    "rlnOpticsGroup"
                ].astype(str)
                optics = pd.concat([optics, new_rows], ignore_index=True)
                og_count += n
            data.attrs["optics"] = optics
            if args.verbose > 1:
                print(f"\t{len(ogs)} optics groups -> {len(optics)} optic groups")

        elif option_name == "assignOpticGroupPerMicrograph" and param:
            try:
                optics_orig = data.attrs["optics"]
            except:
                optics_orig = None
            if optics_orig is None:
                helicon.color_print(f"\tERROR: data_optics block must be available")
                sys.exit(-1)

            image_name = helicon.first_matched_atrr(
                data,
                attrs="rlnMicrographMovieName rlnMicrographName rlnImageName".split(),
            )
            if image_name is None:
                helicon.color_print(
                    f"\tERROR: rlnMicrographMovieName, rlnMicrographName or rlnImageName must be available"
                )
                sys.exit(-1)

            required_cols = "rlnOpticsGroup".split()
            missing_cols = [c for c in required_cols if c not in data]
            if missing_cols:
                helicon.color_print(
                    f"\tERROR: required attrs {' '.join(missing_cols)} must be available"
                )
                sys.exit(-1)

            tmp_col = "TEMP_image_name"
            data[tmp_col] = data[image_name].str.split("@", expand=True).iloc[:, -1]
            mgraphs = data.groupby(tmp_col, sort=False)

            optics = pd.concat(
                [optics_orig.iloc[[0]]] * len(mgraphs), ignore_index=True
            )
            for gi, (mgraphName, mgraphData) in enumerate(mgraphs):
                data.loc[mgraphData.index, "rlnOpticsGroup"] = gi + 1
                new_row = optics_orig.copy().iloc[0]
                optics.loc[gi, "rlnOpticsGroup"] = gi + 1
                optics.loc[gi, "rlnOpticsGroupName"] = f"opticsGroup{gi+1}"
            data.attrs["optics"] = optics
            data.drop(tmp_col, axis=1, inplace=True)
            if args.verbose > 1:
                print(
                    f"\t{len(mgraphs)} micrographs -> {len(data.attrs["optics"])} optic groups"
                )

        elif option_name == "splitByMicrograph" and param:
            if "rlnMicrographName" in data:
                micrographNames = data["rlnMicrographName"]
            else:
                micrographNames = (
                    data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
                )
            mgraphs = micrographNames.groupby(micrographNames, sort=False)

            count = 0
            prefix = os.path.splitext(args.output_starFile)[0]
            for mgraphName, mgraphParticles in mgraphs:
                tmpStarFile = "%s.%s.star" % (
                    prefix,
                    os.path.basename(os.path.splitext(mgraphName)[0]),
                )
                tmpdata = data.loc[mgraphParticles.index]
                helicon.dataframe2file(tmpdata, tmpStarFile)
                count += 1
                if args.verbose > 1:
                    print(
                        "\t%d/%d: %d images saved to %s"
                        % (count, len(mgraphs), len(mgraphParticles), tmpStarFile)
                    )
            sys.exit()

        elif option_name == "showTime" and param:
            if param in data:
                fileAttr = param
            else:
                fileAttr = helicon.first_matched_atrr(
                    data,
                    attrs="rlnMicrographMovieName rlnMicrographName rlnImageName".split(),
                )
            tmpCol = helicon.unique_attr_name(data, attr_prefix=fileAttr)
            data.loc[:, tmpCol] = data[fileAttr].str.split("@", expand=True).iloc[:, -1]
            timeCol = f"{fileAttr}CreateTime"
            files = data.groupby(tmpCol, sort=False)
            for fileName, fileParticles in files:
                data.loc[fileParticles.index, timeCol] = os.path.getctime(fileName)
            data.drop(tmpCol, inplace=True, axis=1)
            if args.verbose > 1:
                print(
                    f"\tThe create time of {len(files):,} {fileAttr} files added to a new column {timeCol}"
                )

    if args.path != "absolute":
        from helicon import get_relion_project_folder, convert_dataframe_file_path

        relion_proj_folder = get_relion_project_folder(
            os.path.abspath(args.output_starFile)
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
                helicon.color_print(
                    (
                        '\tERROR: --splitMode=%s requires "%s" in the input star file'
                        % (args.splitMode, var)
                    )
                )
                sys.exit(-1)
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

        prefix, suffix = os.path.splitext(args.output_starFile)
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
                print(
                    "\tSubset %d/%d: %d images saved to %s"
                    % (si + 1, args.splitNumSets, len(data_subset), imageSubSetFileName)
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
                if os.path.exists(filename):
                    nx, ny, _ = helicon.get_image_size(filename)
                    x, unit = helicon.bytes2units(nx * ny * 4 * len(data))
                    print(
                        f"{len(data):,} images ({nx}x{ny}) saved to {args.output_starFile}. Storage needed: {round(x,1):g} {unit}"
                    )
                else:
                    print(f"{len(data):,} images saved to {args.output_starFile}")
            else:
                print(f"{len(data):,} images saved to {args.output_starFile}")


def maskGold_process_one_micrograph(
    mgraphParticles,
    outputFile,
    value_sigma=4,
    gradient_sigma=50,
    min_area=200,
    both_sides=1,
    msg=None,
    verbose=0,
):
    def findGoldMask(data, value_sigma=4, min_area=200, both_sides=1):
        import numpy as np
        from skimage.filters import gaussian
        from skimage.exposure import rescale_intensity
        from skimage.restoration import denoise_tv_chambolle
        from skimage.segmentation import random_walker
        from skimage.measure import label, regionprops
        from skimage.morphology import remove_small_objects
        from scipy.stats import median_abs_deviation

        data = gaussian(data, sigma=0.25 * np.sqrt(min_area), mode="reflect")
        data = rescale_intensity(data, out_range=(-1, 1))
        # data = denoise_tv_chambolle(data, weight = 0.8, multichannel = False)
        median = np.median(data)
        mad = median_abs_deviation(data, axis=None)
        markers = np.zeros(data.shape, dtype=np.uint8)
        if (
            both_sides
        ):  # remove negative and positive sides (pixels with large values or small values)
            markers[np.abs(data - median) < (value_sigma - 1.0) * mad] = 1
            markers[data > median + value_sigma * mad] = 2
            markers[data < median - value_sigma * mad] = 3
        else:  # remove positive side only (pixels with large values)
            markers[data < median + (value_sigma - 1.0) * mad] = 1
            markers[data > median + value_sigma * mad] = 2
        labels = random_walker(data, markers, beta=10, mode="bf").astype(np.uint8)
        labels[labels < 2] = 0
        labels[labels >= 2] = 1
        labels = label(labels)
        labels = remove_small_objects(labels, min_size=min_area)
        return labels

    n = len(mgraphParticles)

    import mrcfile

    mrc_input = mrcfile.open(mgraphParticles["tmp_mgraph_name"].values[0])
    apix = mrc_input.voxel_size.x
    apix2 = apix * apix
    if len(mrc_input.data.shape) == 3:
        _, ny, nx = mrc_input.data.shape
        data_out = np.zeros((n, ny, nx), dtype=np.float32)
        unit = " particles"
    else:
        ny, nx = mrc_input.data.shape
        data_out = np.zeros((ny, nx), dtype=np.float32)
        unit = " micrograph"

    i = 0
    if msg:
        print(msg)
    from tqdm import tqdm

    for row in tqdm(mgraphParticles.itertuples(), unit=unit, disable=verbose != 1):
        pid = int(row.tmp_mgraph_pid) - 1
        if len(mrc_input.data.shape) == 3:
            data = mrc_input.data[pid] * 1.0
        else:
            assert i == 0, f"ERROR: accessing image {i+1} when there is only one image"
            data = mrc_input.data * 1.0
        data = data.astype(np.float32)
        if gradient_sigma > 0:
            import skimage.filters

            data_blurred = skimage.filters.gaussian(
                data, sigma=gradient_sigma / apix, mode="reflect"
            )
            data -= data_blurred
            data += np.median(data_blurred)
        goldmask = findGoldMask(data, value_sigma, min_area / apix2, both_sides)
        if verbose > 1:
            from skimage.measure import label, regionprops

            props = regionprops(goldmask)
            areas = sorted([p.area * apix2 for p in props])
            if len(areas) > 1:
                print(
                    f"\t{outputFile.as_posix()} {i+1}/{len(mgraphParticles)}: {np.count_nonzero(goldmask)*apix2:.1f} Å^2 in {len(areas)} regions ({areas[0]:.1f} - {areas[-1]:.1f} Å^2) are masked"
                )
            elif len(areas) == 1:
                print(
                    f"\t{outputFile.as_posix()} {i+1}/{len(mgraphParticles)}: {np.count_nonzero(goldmask)*apix2:.1f} Å^2 in 1 region are masked"
                )
            else:
                print(
                    f"\t{outputFile.as_posix()} {i+1}/{len(mgraphParticles)}: nothing to mask"
                )

        nonzeros = goldmask > 0
        if np.count_nonzero(nonzeros) > 0:
            mean = np.mean(data[~nonzeros])  # mean of non-gold pixels
            data[nonzeros] = mean
        if len(mrc_input.data.shape) == 3:
            data_out[i] = data
        else:
            data_out = data
        i += 1
    mrc_input.close()
    mrc_output = mrcfile.new(outputFile.as_posix(), data=data_out, overwrite=True)
    mrc_output.close()
    return (mgraphParticles.index, outputFile)


def estimate_inter_segment_distance(data):
    for attr in ["rlnImageName", "rlnHelicalTubeID", "rlnHelicalTrackLengthAngst"]:
        if attr not in data:
            return None, None, None, None

    data, data_orig = data.copy(), data
    temp = data["rlnImageName"].str.split("@", expand=True)
    data.loc[:, "pid"] = temp.iloc[:, 0].astype(int)
    filename = "micrograph"
    data.loc[:, filename] = temp.iloc[:, 1]
    data = data.sort_values([filename, "pid"], ascending=True)
    data.reset_index(drop=True, inplace=True)

    helices = data.groupby([filename, "rlnHelicalTubeID"], sort=False)

    import numpy as np

    dists_all = []
    lengths = []
    for _, particles in helices:
        lengths.append(particles["rlnHelicalTrackLengthAngst"].astype(np.float32).max())
        if len(particles) < 2:
            continue
        dists = np.sort(
            particles["rlnHelicalTrackLengthAngst"].astype(np.float32).values
        )
        dists = dists[1:] - dists[:-1]
        dists_all.append(dists)
    dists_all = np.hstack(dists_all)
    dist_seg_median = np.median(dists_all)  # Angstrom
    dist_seg_mean = np.mean(dists_all)  # Angstrom
    dist_seg_sigma = np.std(dists_all)  # Angstrom
    n_max = np.sum(np.round(np.array(lengths) / dist_seg_median) + 1).astype(int)
    return dist_seg_median, dist_seg_mean, dist_seg_sigma, n_max


def reset_inter_segment_distance(
    data,
    new_inter_segment_distance,
    apix_micrograph,
    current_inter_segment_distance=-1,
    verbose=0,
):
    if (
        current_inter_segment_distance > 0
        and new_inter_segment_distance == current_inter_segment_distance
    ):
        return data

    for attr in ["rlnHelicalTubeID", "rlnCoordinateX", "rlnCoordinateY"]:
        if attr not in data:
            return None
    if "rlnImageName" in data:
        tmp = data.loc[:, "rlnImageName"].str.split("@", expand=True)
        data.loc[:, "risd_pid"] = tmp.iloc[:, 0].astype(int)
        data.loc[:, "risd_filename"] = tmp.iloc[:, 1]
        filename = "risd_filename"
    else:
        return None

    if "rlnMicrographName" in data:
        filename = "rlnMicrographName"

    if current_inter_segment_distance <= 0:
        current_inter_segment_distance = estimate_inter_segment_distance(data)[0]

    if new_inter_segment_distance == current_inter_segment_distance:
        data.drop(["risd_filename", "risd_pid"], inplace=True, axis=1)
        return data

    cdist = current_inter_segment_distance / apix_micrograph  # Angstrom -> pixel
    ndist = new_inter_segment_distance / apix_micrograph

    import numpy as np
    from tqdm import tqdm

    data2 = []
    helices = data.groupby([filename, "rlnHelicalTubeID"], sort=False)
    for _, particles in tqdm(helices, unit=" helicaltubes", disable=verbose < 1):
        if len(particles) < 2:
            data2.append(particles.reset_index(drop=True))
            continue
        particles_sorted = particles.sort_values(
            ["risd_pid"], ascending=True
        ).reset_index(drop=True)
        x = particles_sorted.loc[:, "rlnCoordinateX"].astype(float).values  # pixel
        y = particles_sorted.loc[:, "rlnCoordinateY"].astype(float).values
        pos, xy_fit = helicon.line_fit_projection(
            x, y, w=None, ref_i=0, return_xy_fit=True
        )
        n0 = len(pos)
        unit_vec = (xy_fit[-1] - xy_fit[0]) / (pos[-1] - pos[0])
        right = np.arange(pos[0], pos[-1] + cdist / 2 + 0.1, ndist)
        left = np.arange(pos[0] - ndist, pos[0] - cdist / 2, -ndist)
        if len(left):
            left.sort()
            pos_new = np.hstack((left, right))
        else:
            pos_new = right
        n = len(pos_new)
        xy_new = xy_fit[0] + pos_new.reshape((n, 1)) * unit_vec
        if n <= n0:
            df_tmp = particles_sorted.iloc[:n].reset_index(drop=True)
        else:
            df_tmp = particles_sorted.iloc[:n0].reset_index(drop=True)
            index_repeat = [df_tmp.index[-1]] * (n - n0)  # replicate last segment
            df_tmp = df_tmp.append(df_tmp.iloc[index_repeat], ignore_index=True)
        df_tmp.loc[:, "rlnCoordinateX"] = xy_new[:, 0]
        df_tmp.loc[:, "rlnCoordinateY"] = xy_new[:, 1]
        if "rlnHelicalTrackLengthAngst" in df_tmp:
            df_tmp.loc[:, "rlnHelicalTrackLengthAngst"] = (
                pos_new - pos_new[0]
            ) * apix_micrograph
        data2.append(df_tmp)

    data2 = pd.concat(data2)
    data2.drop(["risd_filename", "risd_pid"], inplace=True, axis=1)

    try:
        data2.attrs = data.attrs
    except:
        pass

    return data2


# set inplace
def estimate_helicalTube_length(data, inter_segment_distance=-1, verbose=0):
    for attr in ["rlnHelicalTubeID", "rlnCoordinateX", "rlnCoordinateY"]:
        if attr not in data:
            return None
    if "rlnImageName" in data:
        tmp = data.loc[:, "rlnImageName"].str.split("@", expand=True)
        data.loc[:, "ehl_pid"] = tmp.iloc[:, 0].astype(int)
        filename = "ehl_filename"
        data.loc[:, filename] = tmp.iloc[:, 1]
    else:
        return None

    if "rlnMicrographName" in data:
        filename = "rlnMicrographName"

    if inter_segment_distance <= 0:
        inter_segment_distance = estimate_inter_segment_distance(data)[0]

    helices = data.groupby([filename, "rlnHelicalTubeID"], sort=False)

    from tqdm import tqdm
    import numpy as np

    for _, particles in tqdm(helices, unit=" helicaltubes", disable=verbose < 1):
        if "rlnHelicalTrackLengthAngst" in particles:
            data.loc[particles.index, "rlnHelicalTubeLength"] = round(
                particles["rlnHelicalTrackLengthAngst"].max(), 1
            )
        else:
            pids = particles["ehl_pid"].astype(int).values
            helical_len = (pids.max() - pids.min() + 1) * inter_segment_distance
            data.loc[particles.index, "rlnHelicalTubeLength"] = round(helical_len, 1)

    data.drop(["ehl_filename", "ehl_pid"], inplace=True, axis=1)
    return data


def getPixelSize(
    data,
    attrs=[
        "rlnMicrographOriginalPixelSize",
        "rlnMicrographPixelSize",
        "rlnImagePixelSize",
        "rlnImageName",
        "rlnMicrographName",
    ],
    return_pixelSize_source=False,
):
    try:
        sources = [data.attrs["optics"]]
    except:
        sources = []
    sources += [data]
    for source in sources:
        if source is None:
            continue
        for attr in attrs:
            if attr in source:
                if attr in ["rlnImageName", "rlnMicrographName"]:
                    import mrcfile, pathlib

                    folder = Path(data["starFile"].iloc[0])
                    if folder.is_symlink():
                        folder = folder.readlink()
                    folder = folder.resolve().parent
                    filename = source[attr].iloc[0].split("@")[-1]
                    filename = str((folder / "../.." / filename).resolve())
                    with mrcfile.open(filename, header_only=True) as mrc:
                        apix = float(mrc.voxel_size.x)
                else:
                    apix = float(source[attr].iloc[0])
                if return_pixelSize_source:
                    return apix, attr
                return apix
    return None


def setPixelSize(data, apix_new, update_defocus=False):
    apix_old, pixelSize_source = getPixelSize(data, return_pixelSize_source=True)
    if update_defocus:
        for attr in "rlnDefocusU rlnDefocusV".split():
            if attr in data:
                data.loc[:, attr] = data.loc[:, attr].astype(float) * (
                    (apix_new / apix_old) ** 2
                )
    try:
        data.attrs["optics"].loc[:, pixelSize_source] = apix_new
    except:
        pass
    if pixelSize_source in data:
        data.loc[:, pixelSize_source] = apix_new


def readCtfparmFile(filename):  # EMAN1 ctfparm.txt format file
    fp = open(filename, "rt")
    ret = {}
    for l in fp.readlines():
        mid, rest = l.split()
        mid2 = mid.split(".")[0]  # relax name matching
        params = rest.split(",")
        if len(params) == 14:
            (
                defocus,
                dfdiff,
                dfang,
                bfactor,
                amplitude,
                ampcont,
                noise1,
                noise2,
                noise3,
                noise4,
                voltage,
                cs,
                apix,
            ) = list(map(float, params[:13]))
        elif len(params) == 12:
            (
                defocus,
                bfactor,
                amplitude,
                ampcont,
                noise1,
                noise2,
                noise3,
                noise4,
                voltage,
                cs,
                apix,
            ) = list(map(float, params[:11]))
            dfdiff = 0
            dfang = 0
        else:
            helicon.color_print(
                "\tERROR: wrong format for line in %s\n%s\n" % (filename, l)
            )
            sys.exit(1)
        d = {}
        d["defocus"] = abs(defocus)
        d["dfdiff"] = dfdiff
        d["dfang"] = dfang
        d["ampcont"] = ampcont * 100
        d["voltage"] = voltage
        d["cs"] = cs
        d["apix"] = apix
        ret[mid] = d
        ret[mid2] = d
    fp.close()
    return ret


def add_args(parser):
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
        "--randomSample",
        metavar="<n>",
        type=int,
        help="take random n images subset. disabled by default",
        default=0,
    )
    parser.add_argument(
        "--recoverFullFilaments",
        metavar="minFraction=<0.5>[:forcePickJob=<0|1>][:fullStarFile=<filename>]",
        type=str,
        help="recover the whole filaments if current filament has >=minFraction of the segments in the whole filament",
        default="",
    )
    parser.add_argument(
        "--select",
        type=str,
        metavar=("<var>", "<val1<,val2>...>"),
        nargs=2,
        help="select images with exact matching of the specified variable value(s). disabled by default",
        default=[],
    )
    parser.add_argument(
        "--selectValueRange",
        type=str,
        metavar=("<var>", "<valmin>", "<valmax>"),
        nargs=3,
        help="select images with the variable value in the specified range. disabled by default",
        default=[],
    )
    parser.add_argument(
        "--selectRatioRange",
        type=str,
        metavar=("<var>", "<ratio min>", "<ratio max>"),
        nargs=3,
        help="select images with the variable value in the specified ratio range. disabled by default",
        default=[],
    )
    parser.add_argument(
        "--selectByParticleLocation",
        type=str,
        metavar="starFile:maxDist=<pixel>",
        action="append",
        help="select particles that are at the same locations in the micrograph (example: x.star:maxDist=10). disabled by default",
        default=[],
    )
    parser.add_argument(
        "--selectFile",
        type=str,
        metavar="starFile[:col1=<name>:col2=<name>:pattern=<str>]",
        action="append",
        help='select images in the specified file (example: x.star:col1=rlnImageName:col2=rlnImageOriginalName:pattern=".*(16jul.*-a).*"). disabled by default',
        default=[],
    )
    parser.add_argument(
        "--selectCommonHelices",
        type=str,
        metavar="starFile",
        action="append",
        help='select helices in the specified file (example: x.star) based on rlnMicrographName and rlnHelicalTubeID. disabled by default',
        default=[],
    )
    parser.add_argument(
        "--excludeFile",
        type=str,
        metavar="starFile[:col1=<name>:col2=<name>:pattern=<str>]",
        action="append",
        help='exclude images in the specified file (example: x.star:col1=rlnImageName:col2=rlnImageOriginalName:pattern=".*(16jul.*-a).*"). disabled by default',
        default=[],
    )
    parser.add_argument(
        "--sortby",
        type=str,
        action="append",
        metavar="<parameter>",
        nargs="+",
        help="sort (small to large) by the specified parameter(s). disabled by default",
        default=[],
    )
    parser.add_argument(
        "--rsortby",
        type=str,
        action="append",
        metavar="<parameter>",
        nargs="+",
        help="reverse sort (large to small) by the specified parameter(s). disabled by default",
        default=[],
    )
    parser.add_argument(
        "--minDuplicates",
        metavar="<n>",
        type=int,
        help="only keep images >=n duplicates. disabled by default",
        default=0,
    )
    parser.add_argument(
        "--removeDuplicates",
        metavar="<var>",
        nargs="+",
        type=str,
        help="remove images with duplicate parameters. disabled by default",
        default="",
    )
    parser.add_argument(
        "--normEulerDist",
        type=float,
        metavar=("<angle bin size>", "<nkeep>"),
        nargs=2,
        help="reduce Euler (view) preference by removing the worst particles in over-populated angular bins",
        default=[],
    )
    parser.add_argument(
        "--setCTF",
        metavar="<filename>",
        type=str,
        help="set ctf parameters stored in this file (EMAN1 ctfparm.txt) to the output star file",
        default="",
    )
    parser.add_argument(
        "--copyCtf",
        metavar="<starfile>",
        type=str,
        help="Star file to copy CTF parameters from. Should be a CTF refinement output file.",
    )
    parser.add_argument(
        "--setParm",
        metavar=("<var> <val>"),
        type=str,
        nargs="+",
        help="set parameter var val pair for each image",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--setBeamTiltClass",
        metavar="<0|1>",
        type=int,
        help="set rlnBeamTiltClass column, one group per micrograph",
        default=0,
    )
    parser.add_argument(
        "--keepParm",
        metavar="<var>",
        type=str,
        nargs="+",
        help="keep parameter var for each image, remove other parameters",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--delParm",
        metavar="<var>",
        type=str,
        nargs="+",
        action="append",
        help="remove parameter var for each image",
        default=[],
    )
    parser.add_argument(
        "--copyParm",
        metavar="<starfile< var ~var ...>>",
        type=str,
        nargs="+",
        help="copy the specified parameters or all parameters if no var is specified. ~var will skip copying var",
        default=[],
    )
    parser.add_argument(
        "--addParm",
        metavar="<var> <val>",
        type=str,
        nargs=2,
        help="modify parameter: var+=val",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--multParm",
        metavar="<var> <val>",
        type=str,
        nargs=2,
        help="modify parameter: var*=val",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--duplicateParm",
        metavar="<from> <to>",
        type=str,
        nargs=2,
        help="duplicate parameter",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--renameParm",
        metavar="<old> <new>",
        type=str,
        nargs=2,
        help="rename parameter",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--psiPrior180",
        metavar="<0|1>",
        type=int,
        help="duplicate data by adding 180 degrees to rlnAnglePsiPrior",
        default=0,
    )
    parser.add_argument(
        "--replaceStr",
        metavar=("<var>", "<original str>", "<new str>"),
        type=str,
        nargs=3,
        help="replace substr in the variable with new str",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--replaceImageName",
        metavar="<new mrcs file>",
        type=str,
        help="replace rlnImageName column by the provided mrcs file that has the same number of particles",
        default="",
    )
    parser.add_argument(
        "--apix",
        type=float,
        metavar="<A/pixel>",
        help="set mag to have this sampling",
        default=0,
    )
    parser.add_argument(
        "--path",
        metavar="<absolute|relative|real|shortest|current>",
        type=str,
        choices=["absolute", "relative", "real", "shortest", "current"],
        help="which type of file path is used for the images. default to current",
        default="current",
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
    choices = "no auto EPU serialEM_pncc".split()
    parser.add_argument(
        "--assignOpticGroupByBeamShift",
        choices=choices,
        metavar=f"<{'|'.join(choices)}>",
        help="assign images to optic groups according to the beam shifts, one group per beam shift position. default to no",
        default="no",
    )
    parser.add_argument(
        "--assignOpticGroupByTime",
        type=int,
        metavar="<n>",
        help="assign images to optic groups according to data collection time, n movies per group. disabled by default",
        default=-1,
    )
    parser.add_argument(
        "--assignOpticGroupPerMicrograph",
        type=bool,
        metavar="<0|1>",
        help="assign images to optic groups, one group per micrograph. default to 0",
        default=0,
    )
    parser.add_argument(
        "--splitByMicrograph",
        type=bool,
        metavar="<0|1>",
        help="split the output into separate star files, one per micrograph. default to 0",
        default=0,
    )
    parser.add_argument(
        "--minStack",
        type=int,
        metavar="<0|1>",
        help="generate a new set of mrcs files including only the subset of particles in this stack. default to 0",
        default=0,
    )
    parser.add_argument(
        "--fullStack",
        type=int,
        metavar="<0|1>",
        help="generate a star stack including all particles in the referenced image files. default to 0",
        default=0,
    )
    parser.add_argument(
        "--createStack",
        dest="createStack",
        type=str,
        metavar="output.mrcs:rescale2size=<n>:float16=<0|1>:force=<0|1>",
        help="create a new mrcs file to store all particles",
        default=None,
    )
    choices = "graphene graphen_oxide go gold ice".split()
    parser.add_argument(
        "--calibratePixelSize",
        metavar="<%s>" % ("|".join(choices)),
        type=str,
        choices=choices,
        help="calibrate mag based on known diffraction peak resolution of standards such as graphene/graphene oxide at 2.13Å or gold at 2.355Å",
        default=None,
    )
    parser.add_argument(
        "--process",
        metavar="processor_name:param1=value1:param2=value2",
        type=str,
        nargs="+",
        action="append",
        help="apply a processor named 'processorname' with all its parameters/values.",
    )
    parser.add_argument(
        "--maskGold",
        metavar="value_sigma=<n>:gradient_sigma=<Å>:min_area=<Å^2>:both_sides=<0|1>:outdir=<str>:force=<0|1>:cpu=<n>",
        type=str,
        action="append",
        help="mask out electron dense (gold, ferritin, ice) pixels in images. disabled by default",
        default=None,
    )
    parser.add_argument(
        "--estimateHelicalAngleVariance",
        metavar="<0|1>",
        type=int,
        help="estimate the variance of the tilt, psi, rot angles of segments in the same helical tube/filament",
        default=0,
    )
    parser.add_argument(
        "--estimateHelicalTubeLength",
        metavar="<0|1>",
        type=int,
        help="estimate the length of each helical filament/tube",
        default=0,
    )
    parser.add_argument(
        "--resetInterSegmentDistance",
        metavar="<Å>",
        type=float,
        help="reset inter-segment distance by adding/removing 'particles' with updated 'rlnCoordinateX rlnCoordinateY rlnHelicalTrackLengthAngst' parameters. Warning: the output star file is meaningful only for particle extraction and it should NOT be used for 2d/3d classification or 3d refinement",
        default=0,
    )
    parser.add_argument(
        "--keepOneParticlePerHelicalTube",
        metavar="<0|1>",
        type=int,
        help="keep only one segment of each helical filament/tube",
        default=0,
    )
    parser.add_argument(
        "--keepOneParticlePerMicrograph",
        metavar="<0|1>",
        type=int,
        help="keep only one segment of each micrograph",
        default=0,
    )
    parser.add_argument(
        "--showTime",
        metavar="<attr>",
        type=str,
        help="include file create time of the attr in the output star file. disabled by default",
        default=None,
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
        metavar="<0|1>",
        help="verbose mode. default to 2",
        default=3,
    )
    parser.add_argument(
        "--cpu",
        type=int,
        metavar="<n>",
        help="number of cpus to use. default to 1",
        default=1,
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

    return parser


def check_args(args, parser):
    args.append_options = [
        a.dest for a in parser._actions if type(a) is argparse._AppendAction
    ]
    all_options = helicon.get_option_list(sys.argv[1:])
    args.all_options = [
        o
        for o in all_options
        if o
        not in "cpu first force ignoreBadParticlePath ignoreBadMicrographPath last folder splitNumSets splitMode tag verbose".split()
    ]

    if Path(args.output_starFile).suffix not in ".star .cs .csv".split():
        helicon.color_print(
            "\tERROR: the output file (%s) must be a .star, .cs, or .csv file"
            % (args.output_starFile)
        )
        sys.exit(-1)

    if os.path.exists(args.output_starFile) and not (
        args.force == 1 or args.splitNumSets > 1
    ):
        helicon.color_print(
            "\tERROR: the output file (%s) exists. Use --force=1 to overwrite it"
            % (args.output_starFile)
        )
        sys.exit(-1)

    if args.setCTF and not os.path.exists(args.setCTF):
        helicon.color_print(
            'ERROR: option "--setCTF %s" specifies of a nonexistent file'
            % (args.setCTF)
        )
        sys.exit(1)

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
