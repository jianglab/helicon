#!/usr/bin/env python

"""A command line tool that interacts with a CryoSPARC server and performs image analysis tasks"""

import argparse, sys
from pathlib import Path
import numpy as np
import helicon


def main(args):
    helicon.log_command_line()

    if args.cpu < 1:
        args.cpu = helicon.available_cpu()

    if args.csFile:
        from cryosparc.dataset import Dataset

        data_orig = Dataset.load(args.csFile)
    else:
        cs = helicon.connect_cryosparc()
        job = cs.find_job(args.projectID, args.jobID)
        data_orig = job.load_output("particles")

    data = data_orig.copy()

    if args.verbose > 1:
        image_name = helicon.first_matched_atrr(
            data, attrs="location/micrograph_path blob/path".split()
        )
        if image_name is None:
            helicon.color_print(
                f"\tERROR: location/micrograph_path or blob/path must be available"
            )
            sys.exit(-1)
        micrographs = np.unique(data[image_name])
        if args.verbose > 1:
            if args.csFile:
                print(
                    f"{args.csFile}: {len(data)} particles from {len(micrographs)} micrographs"
                )
            else:
                print(
                    f"{args.projectID}/{args.workspaceID}/{args.jobID}: {len(data)} particles from {len(micrographs)} micrographs"
                )

    if args.verbose > 10:
        print(data)

    output_title = ""
    output_slots = set()

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

        if option_name == "assignExposureGroupByBeamShift" and param:
            group_ids_orig = np.sort(np.unique(data["ctf/exp_group_id"]))

            image_name = helicon.first_matched_atrr(
                data, attrs="location/micrograph_path blob/path".split()
            )
            if image_name is None:
                helicon.color_print(
                    f"\tERROR: location/micrograph_path or blob/path must be available"
                )
                sys.exit(-1)

            software = helicon.guess_data_collection_software(data[image_name][0])
            if software is None:
                helicon.color_print(
                    f"\tWARNING: cannot detect the data collection software using {image_name}: {data[image_name][0]}\n\tI only know the filenames by {', '.join(sorted(helicon.movie_filename_patterns().keys()))}"
                )
                sys.exit(-1)

            if software in ["EPU"]:
                extractBeamShift = helicon.extract_EPU_beamshift_pos
            elif software in ["serialEM_pncc"]:
                extractBeamShift = helicon.extract_serialEM_pncc_beamshift

            # split by beamshift groups
            def get_micrograph_path_2_beamshift_groups(micrographs):
                mapping = {m: extractBeamShift(m) for m in micrographs}
                mapping2 = {
                    s: si + 1 for si, s in enumerate(sorted(set(mapping.values())))
                }
                return {m: mapping2[mapping[m]] for m in micrographs}

            micrographs = np.unique(data[image_name])
            micrograph_path_2_beamshift_group = get_micrograph_path_2_beamshift_groups(
                micrographs
            )
            particle_group = [
                micrograph_path_2_beamshift_group[row[image_name]]
                for row in data.rows()
            ]
            data["ctf/exp_group_id"] = np.array(particle_group)

            group_ids = np.sort(np.unique(data["ctf/exp_group_id"]))
            for gi in group_ids:
                mask = np.where(data["ctf/exp_group_id"] == gi)
                for (
                    col
                ) in "ctf/cs_mm ctf/phase_shift_rad ctf/shift_A ctf/tilt_A ctf/trefoil_A ctf/tetra_A ctf/anisomag".split():
                    if col in data:
                        data[col][mask] = np.median(data[col][mask])

            output_slots.add("ctf")
            output_title += f"->{len(group_ids)} beamshift groups"

            if args.verbose > 1:
                print(f"\t{len(group_ids_orig)} -> {len(group_ids)} exposure groups")

        elif option_name == "assignExposureGroupByTime" and abs(param) > 0:
            time_group_size = param

            group_ids_orig = np.sort(np.unique(data["ctf/exp_group_id"]))

            if (
                time_group_size < 0 and len(group_ids_orig) > 1
            ):  # combine previous groups (if there are) into a single group first
                if args.verbose > 1:
                    print(
                        f"\tCombining {len(group_ids_orig)} exposure groups into 1 group"
                    )
                data["ctf/exp_group_id"] = 1
                group_ids_orig = np.sort(np.unique(data["ctf/exp_group_id"]))
                time_group_size = abs(time_group_size)

            image_name = helicon.first_matched_atrr(
                data, attrs="location/micrograph_path blob/path".split()
            )
            if image_name is None:
                helicon.color_print(
                    f"\tERROR: location/micrograph_path or blob/path must be available"
                )
                sys.exit(-1)

            software = helicon.guess_data_collection_software(data[image_name][0])
            if software is None:
                helicon.color_print(
                    f"\tWARNING: cannot detect the data collection software using {image_name}: {data[image_name][0]}\n\tI only know the filenames by {', '.join(sorted(helicon.movie_filename_patterns().keys()))}"
                )
                sys.exit(-1)
            elif software not in ["EPU", "EPU_old"]:
                helicon.color_print(
                    f"\tWARNING: I can only detect data collection time for EPU-collected data. It appears that you used {software} to collect the data"
                )
                sys.exit(-1)

            if software in ["EPU"]:
                extractDataCollectionTime = helicon.extract_EPU_data_collection_time
            elif software in ["EPU_old"]:
                extractDataCollectionTime = helicon.extract_EPU_old_data_collection_time

            micrographs = np.unique(data[image_name])
            micrograph_path_2_time = {
                m: extractDataCollectionTime(m) for m in micrographs
            }
            last_group_id = 0
            new_particle_group_ids = np.zeros(len(data))
            for gi in group_ids_orig:
                mask = np.where(data["ctf/exp_group_id"] == gi)
                group_micrographs = np.unique(data[image_name][mask])
                group_micrograph_time = [
                    micrograph_path_2_time[m] for m in group_micrographs
                ]
                group_time_2_subgroup = helicon.assign_to_groups(
                    group_micrograph_time, time_group_size
                )
                group_particle_2_subgroup = [
                    group_time_2_subgroup[micrograph_path_2_time[m]]
                    for m in data[image_name][mask]
                ]
                new_particle_group_ids[mask] = (
                    np.array(group_particle_2_subgroup) + last_group_id
                )
                last_group_id = np.max(new_particle_group_ids)
            data["ctf/exp_group_id"] = new_particle_group_ids

            group_ids = np.sort(np.unique(data["ctf/exp_group_id"]))
            for gi in group_ids:
                mask = np.where(data["ctf/exp_group_id"] == gi)
                for (
                    col
                ) in "ctf/cs_mm ctf/phase_shift_rad ctf/shift_A ctf/tilt_A ctf/trefoil_A ctf/tetra_A ctf/anisomag".split():
                    if col in data:
                        data[col][mask] = np.median(data[col][mask])

            output_slots.add("ctf")
            output_title += f"->{len(group_ids)} time groups"

            if args.verbose > 1:
                print(f"\t{len(group_ids_orig)} -> {len(group_ids)} exposure groups")

        elif option_name == "assignExposureGroupPerMicrograph" and param:
            group_ids_orig = np.sort(np.unique(data["ctf/exp_group_id"]))

            image_name = helicon.first_matched_atrr(
                data, attrs="location/micrograph_path blob/path".split()
            )
            if image_name is None:
                helicon.color_print(
                    f"\tERROR: location/micrograph_path or blob/path must be available"
                )
                sys.exit(-1)

            micrographs = np.unique(data[image_name])
            for mi, m in enumerate(micrographs):
                mask = np.where(data[image_name] == m)
                data["ctf/exp_group_id"][mask] = mi + 1

            group_ids = np.sort(np.unique(data["ctf/exp_group_id"]))

            output_slots.add("ctf")
            output_title += f"->{len(group_ids)} per-micrograph groups"

            if args.verbose > 1:
                print(f"\t{len(group_ids_orig)} -> {len(group_ids)} exposure groups")

        elif option_name == "copyExposureGroup" and param:
            group_ids_orig = np.sort(np.unique(data["ctf/exp_group_id"]))

            dataFrom = helicon.images2dataframe(
                inputFiles=param,
                ignore_bad_particle_path=True,
                ignore_bad_micrograph_path=True,
                warn_missing_ctf=0,
                target_convention="relion",
            )

            helicon.check_required_columns(
                dataFrom, required_cols=["rlnMicrographMovieName", "rlnOpticsGroup"]
            )
            dataFrom["rlnOpticsGroup"] = dataFrom["rlnOpticsGroup"].astype(int)
            dataFrom["rlnOpticsGroup"] = (
                dataFrom["rlnOpticsGroup"].astype(int)
                - np.min(dataFrom["rlnOpticsGroup"])
                + 1
            )
            mapping = {}
            for i, row in dataFrom.iterrows():
                mapping[Path(row["rlnMicrographMovieName"]).stem.split(".")[0]] = row[
                    "rlnOpticsGroup"
                ]

            image_name = helicon.first_matched_atrr(
                data, attrs="location/micrograph_path blob/path".split()
            )
            if image_name is None:
                helicon.color_print(
                    f"\tERROR: location/micrograph_path or blob/path must be available"
                )
                sys.exit(-1)

            micrographs = np.unique(data[image_name])
            from tqdm import tqdm

            for mi, m in tqdm(
                enumerate(micrographs),
                total=len(micrographs),
                desc="\tProcessing micrographs",
                unit="micrograph",
            ):
                group = 0
                for k, v in mapping.items():
                    if m.find(k) != -1:
                        group = v
                        break
                mask = np.where(data[image_name] == m)
                data["ctf/exp_group_id"][mask] = group
                if group == 0:
                    helicon.color_print(
                        f"\tWARNING: cannot find matching optics group info in {param} for {m}. Assign it to exposure group 0"
                    )

            group_ids = np.sort(np.unique(data["ctf/exp_group_id"]))

            output_slots.add("ctf")
            output_title += (
                f"->{len(group_ids)} exposure groups copied from {Path(param).name}"
            )

            if args.verbose > 1:
                print(f"\t{len(group_ids_orig)} -> {len(group_ids)} exposure groups")

        elif option_name == "splitByMicrograph" and param:
            col_mid = "location/micrograph_uid"
            mids = np.unique(data[col_mid])
            masks = [data[col_mid] == mid for mid in mids]
            counts = [np.sum(m) for m in masks]
            group1, group2 = helicon.split_array(counts)

            col_split = "alignments3D/split"
            if col_split not in data:
                data.add_fields([col_split], ["u4"])

            for gi, g in enumerate([group1, group2]):
                for mid_index in g:
                    data[col_split][masks[mid_index]] = gi

            output_slots.add("alignments3D")
            output_title += f"->per-micrograph split"

            if args.verbose > 1:
                print(
                    f"\twhole  dataset: {len(mids)} micrographs, {len(data)} particles"
                )
                print(
                    f"\thalf dataset 1: {len(group1)} micrographs, {np.sum(data[col_split]==0)} particles"
                )
                print(
                    f"\thalf dataset 2: {len(group2)} micrographs, {np.sum(data[col_split]==1)} particles"
                )

    if args.csFile or args.saveLocal:
        if args.csFile:
            output_file = (
                f"{Path(args.csFile).stem}"
                + (output_title if output_title else ".output")
                + ".cs"
            )
        else:
            output_file = (
                f"{args.projectID}_{args.workspaceID}_{args.jobID}"
                + output_title
                + ".cs"
            )
        output_file = "-".join(output_file.split())
        output_file = output_file.replace(" ", "-")
        output_file = output_file.replace("->", "_")
        output_file = output_file.replace("/", "_")
        data.save(output_file)
        if args.verbose > 1:
            print(f"The results are saved to {output_file}")
    else:
        project = cs.find_project(args.projectID)
        new_jobID = project.save_external_result(
            workspace_uid=args.workspaceID,
            dataset=data,
            type="particle",
            name="particles",
            slots=list(output_slots),
            passthrough=(job.uid, "particles"),
            title=f"{args.jobID}" + output_title,
        )
        if args.verbose > 1:
            print(
                f"The results are saved to a new CryoSPARC external job: {args.projectID}/{args.workspaceID}/{new_jobID}"
            )


def add_args(parser):
    parser.add_argument(
        "--csFile",
        type=str,
        metavar="<filename>",
        help="input cryosparc particles cs file",
        default=None,
    )

    parser.add_argument(
        "--projectID",
        type=str,
        metavar="<Pxx>",
        help="input cryosparc project id (short version: -p)",
        default=None,
    )
    parser.add_argument(
        "--workspaceID",
        type=str,
        metavar="<Wx>",
        help="input cryosparc workspace id",
        default=None,
    )
    parser.add_argument(
        "--jobID",
        type=str,
        metavar="<Jxx>",
        help="input cryosparc job id",
        default=None,
    )

    parser.add_argument(
        "--assignExposureGroupByBeamShift",
        type=bool,
        metavar="<0|1>",
        help="assign images to exposure groups according to the beam shifts, one group per beam shift position. default to %(default)s",
        default=0,
    )
    parser.add_argument(
        "--assignExposureGroupByTime",
        type=int,
        metavar="<n>",
        help="assign images to exposure groups according to data collection time, n movies per group. disabled by default",
        default=-1,
    )
    parser.add_argument(
        "--assignExposureGroupPerMicrograph",
        type=bool,
        metavar="<0|1>",
        help="assign images to exposure groups, one group per micrograph. default to %(default)s",
        default=0,
    )
    parser.add_argument(
        "--copyExposureGroup",
        type=str,
        metavar="<star file>",
        help="copy the optics group info from this star file. rlnMicrographMovieName and rlnOpticsGroup must be in this star file. disabled by default",
        default=0,
    )
    parser.add_argument(
        "--splitByMicrograph",
        type=bool,
        metavar="<0|1>",
        help="split the dataset by micrograph. default to %(default)s",
        default=0,
    )
    parser.add_argument(
        "--saveLocal",
        type=int,
        metavar="<0|1>",
        help="save results to a local cs file instead of creating a new external job on the CryoSPARC server. default to %(default)s",
        default=0,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        metavar="<0|1>",
        help="verbose mode. default to %(default)s",
        default=3,
    )
    parser.add_argument(
        "--cpu",
        type=int,
        metavar="<n>",
        help="number of cpus to use. default to %(default)s",
        default=1,
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
        if o not in "cpu jobID projectID saveLocal verbose workspaceID".split()
    ]

    if (args.projectID or args.workspaceID or args.jobID) and args.csFile:
        msg = f"You should only specify options for CryoSPARC server (--projectID --workspaceID --jobID) or local file (--csFile), but not both"
        helicon.color_print(msg)
        raise ValueError(msg)

    if not ((args.projectID and args.workspaceID and args.jobID) or args.csFile):
        msg = f"You should specify options for either CryoSPARC server (--projectID --workspaceID --jobID) or local file (--csFile)"
        helicon.color_print(msg)
        raise ValueError(msg)

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
