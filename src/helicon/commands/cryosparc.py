#!/usr/bin/env python

"""A command line tool that interacts with a CryoSPARC server and performs image analysis tasks"""

import argparse, sys
from pathlib import Path
import numpy as np
import helicon
from cryosparc.dataset import Dataset


def main(args):
    helicon.log_command_line()

    if args.cpu < 1:
        args.cpu = helicon.available_cpu()
    if args.csFile:
        input_project_folder = Path(args.csFile).resolve().parent.parent
        data_orig = Dataset.load(args.csFile)
        input_type = "particle" if "blob/path" in data_orig else "exposure"
    else:
        cs = helicon.connect_cryosparc()
        project = cs.find_project(args.projectID)
        input_project_folder = project.dir()
        input_job = cs.find_job(args.projectID, args.jobID)
        input_group = input_job.doc["output_result_groups"][args.groupIndex]
        input_group_name = input_group["name"]
        input_type = input_group["type"]
        data_orig = input_job.load_output(input_group_name)

    if data_orig is None or not len(data_orig):
        helicon.color_print(f"WARNING: no data in the input. Nothing to do.")
        sys.exit(-1)

    if args.saveLocal:
        output_project_folder = Path(".")
    else:
        output_project_folder = input_project_folder
        output_job = None

    data = data_orig.copy()

    attrs = "movie_blob/path micrograph_blob/path location/micrograph_path blob/path".split()
    micrograph_name = helicon.first_matched_attr(data, attrs=attrs)
    if micrograph_name is None:
        helicon.color_print(
            f"\tERROR: at least one of the {len(attrs)} parameters ({' '.join(attrs)}) must be available"
        )
        sys.exit(-1)

    if args.verbose > 1:
        micrographs = np.unique(data[micrograph_name])
        if input_type == "particle":
            if args.verbose > 1:
                if args.csFile:
                    print(
                        f"{args.csFile}: {len(data):,} particles from {len(micrographs):,} micrographs"
                    )
                else:
                    print(
                        f"{args.projectID}/{args.workspaceID}/{args.jobID}: {len(data):,} particles from {len(micrographs):,} micrographs"
                    )
        else:
            if args.verbose > 1:
                if args.csFile:
                    print(f"{args.csFile}: {len(micrographs):,} micrographs")
                else:
                    print(
                        f"{args.projectID}/{args.workspaceID}/{args.jobID}: {len(micrographs):,} micrographs"
                    )

    if args.verbose > 10:
        print(data)

    exp_group_id_names_all = helicon.all_matched_attrs(data, query_str="exp_group_id")
    exp_group_id_name = helicon.first_matched_attr(
        data,
        attrs="ctf/exp_group_id location/exp_group_id mscope_params/exp_group_id".split(),
    )

    output_title = ""
    output_slots = set()

    index_d = {option_name: 0 for option_name in args.all_options}

    for option_name in args.all_options:
        if option_name in args.append_options:
            param = args.__dict__[option_name][index_d[option_name]]
            index_d[option_name] += 1
        else:
            param = args.__dict__[option_name]

        if args.verbose:
            print("%s: %s" % (option_name, param))

        if option_name == "assignExposureGroupByBeamShift" and (
            param is not None and param != "0"
        ):
            source_group_ids = np.sort(np.unique(data[exp_group_id_name]))

            software = helicon.guess_data_collection_software(data[micrograph_name][0])
            if software is None:
                helicon.color_print(
                    f"\tWARNING: cannot detect the data collection software using {micrograph_name}: {data[micrograph_name][0]}\n\tI only know the filenames by {', '.join(sorted(helicon.movie_filename_patterns().keys()))}"
                )
                sys.exit(-1)

            micrographs = np.sort(np.unique(data[micrograph_name]))

            if software in ["EPU", "serialEM_pncc"]:
                if software in ["EPU"]:
                    extractBeamShift = helicon.extract_EPU_beamshift_pos
                elif software in ["serialEM_pncc"]:
                    extractBeamShift = helicon.extract_serialEM_pncc_beamshift

                # split by beamshift groups
                def get_micrograph_2_beamshift_groups(micrographs):
                    mapping = {m: extractBeamShift(m) for m in micrographs}
                    mapping2 = {
                        s: si + 1 for si, s in enumerate(sorted(set(mapping.values())))
                    }
                    return {m: mapping2[mapping[m]] for m in micrographs}

                micrograph_to_beamshift_clusters = get_micrograph_2_beamshift_groups(
                    micrographs
                )
            elif software in ["EPU_old"]:
                _, param_dict = helicon.parse_param_str(param)
                xml_folder = param_dict.get("xml_folder", "")
                min_cluster_size = int(param_dict.get("min_micrographs_per_group", 4))

                from tqdm import tqdm

                @helicon.cache(
                    cache_dir=str(helicon.cache_dir / "cryosparc"),
                    expires_after=7,
                    verbose=0,
                )  # 7 days
                def EPU_micrograph_path_2_movie_xml_path(micrograph_path, xml_folder):
                    return helicon.EPU_micrograph_path_2_movie_xml_path(
                        micrograph_path=micrograph_path, xml_folder=xml_folder
                    )

                xml_files_dict = {
                    m: EPU_micrograph_path_2_movie_xml_path(
                        micrograph_path=input_project_folder / m, xml_folder=xml_folder
                    )
                    for m in tqdm(
                        micrographs,
                        total=len(micrographs),
                        desc="Finding xml files",
                        unit="micrograph",
                    )
                }

                @helicon.cache(
                    cache_dir=str(helicon.cache_dir / "cryosparc"),
                    expires_after=7,
                    verbose=0,
                )  # 7 days
                def EPU_micrograph_path_2_beamshift(m):
                    xml_file = xml_files_dict[m]
                    beamshift = helicon.EPU_xml_2_beamshift(xml_file=xml_file)
                    return beamshift

                micrographs_to_beamshifts = {
                    m: EPU_micrograph_path_2_beamshift(m)
                    for m in tqdm(
                        micrographs,
                        total=len(micrographs),
                        desc="Parsing xml files",
                        unit="micrograph",
                    )
                }

                @helicon.cache(
                    cache_dir=str(helicon.cache_dir / "cryosparc"),
                    ignore=["cpu", "verbose"],
                    expires_after=7,
                    verbose=0,
                )  # 7 days
                def assign_beamshifts_to_cluster(
                    beamshifts, range_n_clusters, min_cluster_size, cpu, verbose
                ):
                    return helicon.assign_beamshifts_to_cluster(
                        beamshifts=beamshifts,
                        range_n_clusters=range_n_clusters,
                        min_cluster_size=min_cluster_size,
                        cpu=cpu,
                        verbose=verbose,
                    )

                beamshifts = np.array(list(micrographs_to_beamshifts.values()))
                beamshift_clusters = assign_beamshifts_to_cluster(
                    beamshifts=beamshifts,
                    range_n_clusters=range(2, 200),
                    min_cluster_size=min_cluster_size,
                    cpu=args.cpu,
                    verbose=args.verbose,
                )
                assert len(beamshifts) == len(beamshift_clusters)
                micrograph_to_beamshift_clusters = {
                    m: beamshift_clusters[mi]
                    for mi, m in enumerate(micrographs_to_beamshifts.keys())
                }

                if "mscope_params/beam_shift" in data:
                    data["mscope_params/beam_shift"] = np.array(
                        [
                            micrographs_to_beamshifts[row[micrograph_name]]
                            for row in data.rows()
                        ]
                    )

            exposure_groups = [
                micrograph_to_beamshift_clusters[row[micrograph_name]]
                for row in data.rows()
            ]
            data[exp_group_id_name] = np.array(exposure_groups)
            if len(exp_group_id_names_all) > 1:
                for attr in exp_group_id_names_all:
                    if attr != exp_group_id_name:
                        data[attr] = data[exp_group_id_name]

            group_ids = np.sort(np.unique(data[exp_group_id_name]))
            for gi in group_ids:
                mask = np.where(data[exp_group_id_name] == gi)
                for (
                    col
                ) in "ctf/cs_mm ctf/phase_shift_rad ctf/shift_A ctf/tilt_A ctf/trefoil_A ctf/tetra_A ctf/anisomag".split():
                    if col in data:
                        data[col][mask] = np.median(data[col][mask])

            for attr in exp_group_id_names_all:
                slot = attr.split("/")[0]
                output_slots.add(slot)
            output_title += f"->{len(group_ids)} beamshift groups"

            if args.verbose > 1:
                print(f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups")

            if (
                args.verbose > 1
                and "beamshifts" in locals()
                and "beamshift_clusters" in locals()
            ):
                if args.csFile:
                    output_file = (
                        f"{Path(args.csFile).stem}"
                        + (output_title if output_title else ".output")
                        + ".pdf"
                    )
                else:
                    output_file = (
                        f"{args.projectID}_{args.workspaceID}_{args.jobID}"
                        + output_title
                        + ".pdf"
                    )
                output_file = "-".join(output_file.split())
                output_file = output_file.replace(" ", "-")
                output_file = output_file.replace("->", "_")
                output_file = output_file.replace("/", "_")

                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 8))
                scatter = plt.scatter(
                    beamshifts[:, 0],
                    beamshifts[:, 1],
                    c=beamshift_clusters,
                    cmap="tab20",
                    s=2,
                )
                plt.colorbar(scatter, label="Exposure Group")
                plt.xlabel("Beam Shift X")
                plt.ylabel("Beam Shift Y")
                plt.title("Exposure groups by beam shifts")
                plt.savefig(output_file)
                print(
                    f"\tPlot of exposure group assignments based on beam shifts is saved to {output_file}"
                )
                plt.show()
                plt.close()

        elif option_name == "assignExposureGroupByTime" and abs(param) > 0:
            time_group_size = param

            source_group_ids = np.sort(np.unique(data[exp_group_id_name]))

            if (
                time_group_size < 0 and len(source_group_ids) > 1
            ):  # combine previous groups (if there are) into a single group first
                if args.verbose > 1:
                    print(
                        f"\tCombining {len(source_group_ids)} exposure groups into 1 group"
                    )
                data[exp_group_id_name] = 1
                source_group_ids = np.sort(np.unique(data[exp_group_id_name]))
                time_group_size = abs(time_group_size)

            software = helicon.guess_data_collection_software(data[micrograph_name][0])
            if software is None:
                helicon.color_print(
                    f"\tWARNING: cannot detect the data collection software using {micrograph_name}: {data[micrograph_name][0]}\n\tI only know the filenames by {', '.join(sorted(helicon.movie_filename_patterns().keys()))}"
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

            micrographs = np.unique(data[micrograph_name])
            micrograph_path_2_time = {
                m: extractDataCollectionTime(m) for m in micrographs
            }
            last_group_id = 0
            new_particle_group_ids = np.zeros(len(data))
            for gi in source_group_ids:
                mask = np.where(data[exp_group_id_name] == gi)
                group_micrographs = np.unique(data[micrograph_name][mask])
                group_micrograph_time = [
                    micrograph_path_2_time[m] for m in group_micrographs
                ]
                group_time_2_subgroup = helicon.assign_to_groups(
                    group_micrograph_time, time_group_size
                )
                group_particle_2_subgroup = [
                    group_time_2_subgroup[micrograph_path_2_time[m]]
                    for m in data[micrograph_name][mask]
                ]
                new_particle_group_ids[mask] = (
                    np.array(group_particle_2_subgroup) + last_group_id
                )
                last_group_id = np.max(new_particle_group_ids)
            data[exp_group_id_name] = new_particle_group_ids
            if len(exp_group_id_names_all) > 1:
                for attr in exp_group_id_names_all:
                    if attr != exp_group_id_name:
                        data[attr] = data[exp_group_id_name]

            group_ids = np.sort(np.unique(data[exp_group_id_name]))
            for gi in group_ids:
                mask = np.where(data[exp_group_id_name] == gi)
                for (
                    col
                ) in "ctf/cs_mm ctf/phase_shift_rad ctf/shift_A ctf/tilt_A ctf/trefoil_A ctf/tetra_A ctf/anisomag".split():
                    if col in data:
                        data[col][mask] = np.median(data[col][mask])

            for attr in exp_group_id_names_all:
                slot = attr.split("/")[0]
                output_slots.add(slot)
            output_title += f"->{len(group_ids)} time groups"

            if args.verbose > 1:
                print(f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups")

        elif option_name == "assignExposureGroupPerMicrograph" and param:
            source_group_ids = np.sort(np.unique(data[exp_group_id_name]))

            micrographs = np.unique(data[micrograph_name])
            for mi, m in enumerate(micrographs):
                mask = np.where(data[micrograph_name] == m)
                data[exp_group_id_name][mask] = mi + 1

            if len(exp_group_id_names_all) > 1:
                for attr in exp_group_id_names_all:
                    if attr != exp_group_id_name:
                        data[attr] = data[exp_group_id_name]

            group_ids = np.sort(np.unique(data[exp_group_id_name]))

            for attr in exp_group_id_names_all:
                slot = attr.split("/")[0]
                output_slots.add(slot)
            output_title += f"->{len(group_ids)} per-micrograph groups"

            if args.verbose > 1:
                print(f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups")

        elif option_name == "copyExposureGroupAssignments" and param:
            source_group_ids = np.sort(np.unique(data[exp_group_id_name]))

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

            micrographs = np.unique(data[micrograph_name])
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
                mask = np.where(data[micrograph_name] == m)
                data[exp_group_id_name][mask] = group
                if group == 0:
                    helicon.color_print(
                        f"\tWARNING: cannot find matching optics group info in {param} for {m}. Assign it to exposure group 0"
                    )

            if len(exp_group_id_names_all) > 1:
                for attr in exp_group_id_names_all:
                    if attr != exp_group_id_name:
                        data[attr] = data[exp_group_id_name]

            group_ids = np.sort(np.unique(data[exp_group_id_name]))

            for attr in exp_group_id_names_all:
                slot = attr.split("/")[0]
                output_slots.add(slot)
            output_title += (
                f"->{len(group_ids)} exposure groups copied from {Path(param).name}"
            )

            if args.verbose > 1:
                print(f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups")

        elif option_name == "copyExposureGroupParameters" and param:
            param_dict_default = dict(
                source_cs_file="",
                source_job_id="",
                beam_tilt=1,
                cs=1,
                trefoil=1,
                tetrafoil=1,
                anisomag=1,
            )
            _, param_dict = helicon.parse_param_str(param)
            param_dict, param_changed, param_unsuppported = helicon.validate_param_dict(
                param=param_dict, param_ref=param_dict_default
            )
            if len(param_unsuppported):
                helicon.color_print(
                    f"\tWARNING: ignoring unknown parameters: {param_unsuppported}"
                )
            if args.verbose > 2:
                print(f"\tCustom parameters: {param_changed}")

            if param_dict["source_cs_file"]:
                if param_dict["source_job_id"]:
                    helicon.color_print(
                        f"\tWARNING: both source_cs_file and source_job_id are specified. I will use source_cs_file"
                    )
                source_data_name = param_dict["source_cs_file"]
                source_data = Dataset.load(param_dict["source_cs_file"])
                source_exp_group_id_names_all = helicon.all_matched_attrs(
                    source_data, query_str="exp_group_id"
                )
                if len(source_exp_group_id_names_all) == 0:
                    helicon.color_print(
                        f"\tERROR: {param_dict['source_cs_file']} does not contain exp_group_id"
                    )
                    sys.exit(-1)
            elif param_dict["source_job_id"]:
                source_job = cs.find_job(args.projectID, param_dict["source_job_id"])
                input_particle_group_name = None
                for g in source_job.doc["output_result_groups"]:
                    if g["type"] == "particle":
                        input_particle_group_name = g["name"]
                        break
                if not input_particle_group_name:
                    helicon.color_print(
                        f"ERROR: {source_job} does not provide particles"
                    )
                    sys.exit(-1)
                source_data_name = source_job.doc["uid"]
                source_data = source_job.load_output(input_particle_group_name)
                source_exp_group_id_names_all = helicon.all_matched_attrs(
                    source_data, query_str="exp_group_id"
                )
                if len(source_exp_group_id_names_all) == 0:
                    helicon.color_print(
                        f"\tERROR: {param_dict['source_job_id']} does not contain exp_group_id"
                    )
                    sys.exit(-1)
            else:
                helicon.color_print(
                    f"\tERROR: either source_cs_file or source_job_id must be specified"
                )
                sys.exit(-1)

            source_exp_group_id_name = helicon.first_matched_attr(
                source_data,
                attrs="ctf/exp_group_id location/exp_group_id mscope_params/exp_group_id".split(),
            )

            ctf_params_to_copy = []
            if int(param_dict["beam_tilt"]):
                ctf_params_to_copy.append("ctf/tilt_A")
            if int(param_dict["cs"]):
                ctf_params_to_copy.append("ctf/cs_mm")
            if int(param_dict["trefoil"]):
                ctf_params_to_copy.append("ctf/trefoil_A")
            if int(param_dict["tetrafoil"]):
                ctf_params_to_copy.append("ctf/tetra_A")
            if int(param_dict["anisomag"]):
                ctf_params_to_copy.append("ctf/anisomag")

            ctf_params_to_copy = [p for p in ctf_params_to_copy if p in source_data]
            if not ctf_params_to_copy:
                helicon.color_print(
                    f"\tWARNING: No exposure group parameters found in the source dataset. Available parameters: {', '.join(source_data.keys())}"
                )
                sys.exit(-1)

            group_ids = np.sort(np.unique(data[exp_group_id_name]))
            source_group_ids = np.unique(source_data[source_exp_group_id_name])

            for group_id in group_ids:
                mask = np.where(data[exp_group_id_name] == group_id)
                if group_id in source_group_ids:
                    source_mask = np.where(source_data[exp_group_id_name] == group_id)
                    for p in ctf_params_to_copy:
                        data[p][mask] = np.median(source_data[p][source_mask])
                else:
                    for p in ctf_params_to_copy:
                        data[p][mask] = np.median(source_data[p])

            for attr in exp_group_id_names_all:
                slot = attr.split("/")[0]
                output_slots.add(slot)
            output_slots.add("ctf")
            output_title += f"->copied params {' '.join(ctf_params_to_copy)} of {len(group_ids)} exposure groups from {source_data_name}"

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

        elif option_name == "extractParticles" and param:
            if (
                "location/center_x_frac" not in data
                or "location/center_y_frac" not in data
            ):
                helicon.color_print(
                    f"ERROR: --extractParticles option requires location/center_x_frac, location/center_y_frac parameters in the input data"
                )
                sys.exit(-1)

            param_dict_default = dict(
                box_size=0,
                fft_crop_size=0,
                flip_y=0,
                recenter=1,
                replace_ctf=0,
                normalize=1,
                fill_mode="random",
                sign=-1,
                n_micrographs=-1,
                fp16=1,
                micrographs_cs_file="",
                micrographs_job_id="",
                reuse_job_id="",
                force=0,
                plot_pdf=0,
            )
            _, param_dict = helicon.parse_param_str(param)
            param_dict, param_changed, param_unsuppported = helicon.validate_param_dict(
                param=param_dict, param_ref=param_dict_default
            )
            if len(param_unsuppported):
                helicon.color_print(
                    f"\tWARNING: ignoring unknown parameters: {param_unsuppported}"
                )
            if args.verbose > 2:
                print(f"\tCustom parameters: {param_changed}")
            box_size = int(param_dict["box_size"])
            if box_size <= 0:
                helicon.color_print("\tERROR: box_size (>0) must be specified")
                sys.exit(-1)
            fft_crop_size = int(param_dict["fft_crop_size"])
            if fft_crop_size <= 0 or fft_crop_size > box_size:
                fft_crop_size = box_size
            recenter = int(param_dict["recenter"]) > 0
            replace_ctf = int(param_dict["replace_ctf"]) > 0
            sign = int(param_dict["sign"])
            flip_y = int(param_dict["flip_y"]) > 0
            fill_mode = param_dict["fill_mode"]
            normalize = int(param_dict["normalize"]) > 0
            fp16 = int(param_dict["fp16"]) > 0
            n_micrographs = int(param_dict["n_micrographs"])
            micrographs_job_id = param_dict["micrographs_job_id"]
            micrographs_cs_file = param_dict["micrographs_cs_file"]
            reuse_job_id = param_dict["reuse_job_id"]
            force = param_dict["force"]
            plot_pdf = int(param_dict["plot_pdf"]) > 0

            output_slots.add("blob")
            output_slots.add("location")

            col_mid = "location/micrograph_uid"
            micrograph_input = ""
            if "location/micrograph_path" not in data:
                if col_mid not in data:
                    helicon.color_print(
                        f"ERROR: {col_mid} must be in the input data when the input data does not have location/micrograph_path parameters"
                    )
                    sys.exit(-1)
                if not (micrographs_cs_file or micrographs_job_id):
                    helicon.color_print(
                        "ERROR: micrographs_cs_file or micrographs_job_id must be provided when the input data does not have location/micrograph_path parameters"
                    )
                    sys.exit(-1)
            if replace_ctf and not (micrographs_cs_file or micrographs_job_id):
                helicon.color_print(
                    f"\tERROR: micrographs_cs_file or micrographs_job_id must be provided when replace_ctf is specified"
                )
                sys.exit(-1)
            if micrographs_cs_file or micrographs_job_id:
                if micrographs_cs_file:
                    micrograph_input = micrographs_cs_file
                    data_micrographs = Dataset.load(micrographs_cs_file)
                    if (
                        "uid" not in data_micrographs
                        or "micrograph_blob/path" not in data_micrographs
                    ):
                        helicon.color_print(
                            f"ERROR: {micrographs_cs_file} does not contain uid and micrograph_blob/path"
                        )
                        sys.exit(-1)
                else:
                    micrograph_input = f"{args.projectID}/{micrographs_job_id}"
                    micrograph_input_job = cs.find_job(
                        args.projectID, micrographs_job_id
                    )
                    input_micrographs_group_name = None
                    for g in micrograph_input_job.doc["output_result_groups"]:
                        if g["type"] == "exposure":
                            input_micrographs_group_name = g["name"]
                            break
                    if not input_micrographs_group_name:
                        helicon.color_print(
                            f"ERROR: {micrographs_job_id} does not provide micrographs"
                        )
                        sys.exit(-1)
                    data_micrographs = micrograph_input_job.load_output(
                        input_micrographs_group_name
                    )
                    if (
                        "uid" not in data_micrographs
                        or "micrograph_blob/path" not in data_micrographs
                    ):
                        helicon.color_print(
                            f"ERROR: {micrographs_job_id} result {input_micrographs_group_name} does not contain uid and micrograph_blob/path. Available parameters are: {' '.join(data_micrographs.keys())}"
                        )
                        sys.exit(-1)

                # Check if all micrograph IDs in data exist in data_micrographs
                data_mids = set(data[col_mid])
                micrographs_mids = set(data_micrographs["uid"])
                missing_mids = data_mids - micrographs_mids
                if missing_mids:
                    helicon.color_print(
                        f"\tERROR: {len(missing_mids)} micrograph IDs in the input data are not found in the micrographs dataset"
                    )
                    sys.exit(-1)

                if "location/micrograph_path" not in data:
                    data.add_fields(["location/micrograph_path"], [str])

                cols_ctf = [
                    col for col in data_micrographs if col.split("/")[0] == "ctf"
                ]
                cols_ctf_missing_names = [
                    col[0]
                    for col in data_micrographs.descr()
                    if col[0] in cols_ctf and col[0] not in data
                ]
                cols_ctf_missing_types = [
                    col[1]
                    for col in data_micrographs.descr()
                    if col[0] in cols_ctf and col[0] not in data
                ]
                if len(cols_ctf_missing_names):
                    data.add_fields(cols_ctf_missing_names, cols_ctf_missing_types)

                if replace_ctf:
                    cols_ctf_to_copy = cols_ctf
                else:
                    cols_ctf_to_copy = cols_ctf_missing_names
                if len(cols_ctf_to_copy):
                    output_slots.add("ctf")

                for mid in data_mids:
                    particle_row_mask = np.where(data[col_mid] == mid)
                    micrograph_row_mask = np.where(data_micrographs["uid"] == mid)
                    data["location/micrograph_path"][particle_row_mask] = (
                        data_micrographs["micrograph_blob/path"][micrograph_row_mask][0]
                    )
                    for col in cols_ctf_to_copy:
                        data[col][particle_row_mask] = data_micrographs[col][
                            micrograph_row_mask
                        ][0]

            if flip_y:
                data["location/center_y_frac"] = 1 - data["location/center_y_frac"]

            if recenter and (
                "alignments3D/shift" in data or "alignments2D/shift" in data
            ):
                if "alignments3D/shift" in data:
                    alignment_psize = data["alignments3D/psize_A"]
                    alignment_shift = data["alignments3D/shift"]
                    output_slots.add("alignments3D")
                elif "alignments2D/shift" in data:
                    alignment_psize = data["alignments2D/psize_A"]
                    alignment_shift = data["alignments2D/shift"]
                    output_slots.add("alignments2D")

                micrograph_psize = data["location/micrograph_psize_A"]
                mic_shape_y, mic_shape_x = data["location/micrograph_shape"].T
                shift_x = alignment_psize * alignment_shift[:, 0] / micrograph_psize
                shift_y = alignment_psize * alignment_shift[:, 1] / micrograph_psize
                new_loc_x = data["location/center_x_frac"] * mic_shape_x - shift_x
                new_loc_y = data["location/center_y_frac"] * mic_shape_y - shift_y

                data["location/center_x_frac"] = new_loc_x / mic_shape_x
                data["location/center_y_frac"] = new_loc_y / mic_shape_y
                if "alignments3D/shift" in data:
                    data["alignments3D/shift"][:] = [0, 0]
                else:
                    data["alignments2D/shift"][:] = [0, 0]

            if "blob" not in data.prefixes():
                data.add_fields(
                    fields=[
                        "blob/path",
                        "blob/idx",
                        "blob/shape",
                        "blob/psize_A",
                        "blob/sign",
                        "blob/import_sig",
                    ],
                    dtypes=["|O", "<u4", ("<u4", (2,)), "<f4", "<f4", "<u8"],
                )

            reuse_result_folder = None
            if args.projectID and not args.saveLocal:
                output_job = project.create_external_job(
                    args.workspaceID,
                    title="Extract Particles",
                    desc=f"{' '.join(sys.argv)}",
                )
                output_job.connect(
                    target_input="particles",
                    source_job_uid=args.jobID,
                    source_output=input_group_name,
                    title="Particles",
                )
                output_job.add_output(
                    type="particle",
                    name="extracted_particles",
                    slots=sorted(list(output_slots)),
                    passthrough="particles",
                    title="Particles extracted",
                )
                if micrographs_job_id is not None:
                    output_job.connect(
                        target_input="micrographs",
                        source_job_uid=micrographs_job_id,
                        source_output=input_micrographs_group_name,
                        title="Micrographs",
                    )
                    output_job.add_output(
                        type="exposure",
                        name="micrographs",
                        slots=[],
                        passthrough="micrographs",
                        title="Passthrough micrographs",
                    )
                    # output_job.save_output("micrographs", data_micrographs)
                output_job.mkdir("extract")
                particle_dir = f"{output_job.uid}/extract"
                if reuse_job_id:
                    reuse_job = project.find_job(reuse_job_id)
                    source_path = Path(reuse_job.dir()) / "extract"
                    if source_path.exists() and source_path.is_dir():
                        reuse_result_folder = source_path
                output_job.start(status="running")
            else:
                output_job = None
                particle_dir = "extract"
                Path(particle_dir).mkdir(parents=True, exist_ok=True)

            tasks = []
            mids = np.unique(data[col_mid])
            n_micrographs = len(mids) if n_micrographs <= 0 else n_micrographs
            for mid in mids[:n_micrographs]:
                tasks.append(
                    (
                        data,
                        mid,
                        box_size,
                        fft_crop_size,
                        input_project_folder,
                        output_project_folder,
                        particle_dir,
                        sign,
                        fill_mode,
                        normalize,
                        fp16,
                        plot_pdf,
                    )
                )

            if args.verbose > 1:
                print(
                    f"\tStart extracting {len(data):,} particles from {n_micrographs:,} micrographs"
                )

            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=args.cpu) as executor:
                n_skipped = 0
                futures = []
                results = []
                for ti, task in enumerate(tasks):
                    (
                        data,
                        mid,
                        box_size,
                        fft_crop_size,
                        input_project_folder,
                        output_project_folder,
                        particle_dir,
                        sign,
                        fill_mode,
                        normalize,
                        fp16,
                        plot_pdf,
                    ) = task
                    subset = data.query({"location/micrograph_uid": mid})

                    micrograph_path = subset["location/micrograph_path"][0]
                    skip = False
                    if not force and reuse_result_folder:
                        source_file = Path(
                            reuse_result_folder / f"{Path(micrograph_path).stem}.mrcs"
                        )
                        if source_file.exists() and source_file.is_file():
                            import mrcfile

                            with mrcfile.open(source_file, header_only=True) as mrc:
                                nx = mrc.header.nx
                                ny = mrc.header.ny
                                nz = mrc.header.nz
                                if (
                                    nz == len(subset)
                                    and ny == nx
                                    and ny == fft_crop_size
                                ):
                                    target_file = Path(
                                        project.dir() / particle_dir / source_file.name
                                    )
                                    target_file.hardlink_to(source_file)
                                    skip = True
                                    n_skipped += 1

                    if skip:
                        apix = (
                            subset["location/micrograph_psize_A"][0]
                            * box_size
                            / fft_crop_size
                        )
                        result = subset.copy()
                        result["blob/path"] = f"{particle_dir}/{source_file.name}"
                        result["blob/idx"] = np.arange(len(result))
                        result["blob/shape"] = [(fft_crop_size, fft_crop_size)] * len(
                            result
                        )
                        result["blob/psize_A"] = apix
                        result["blob/sign"] = [sign] * len(result)
                        result["blob/import_sig"] = [1] * len(result)
                        results.append(result)
                        if args.verbose > 1:
                            print(
                                f"\t{ti+1}/{len(tasks)}: reuses {str(source_file)}. skipped ({n_skipped})"
                            )
                        continue

                    future = executor.submit(
                        extract_one_micrograph,
                        subset,
                        box_size,
                        fft_crop_size,
                        input_project_folder,
                        output_project_folder,
                        particle_dir,
                        sign,
                        fill_mode,
                        normalize,
                        fp16,
                        force,
                        plot_pdf,
                    )
                    futures.append(future)

                if args.verbose > 1 and len(n_skipped) > 0:
                    print(
                        f"\t{len(tasks):,} micrographs: {n_skipped:,} skipped, {len(futures):,} to go"
                    )

                if len(futures):
                    from tqdm import tqdm

                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="Extracting",
                        unit="micrograph",
                    ):
                        result = future.result()
                        results.append(result)

            from cryosparc.dataset import Dataset

            data = Dataset.append(*results, assert_same_fields=True)
            if args.verbose > 1:
                print(
                    f"\t{len(data):,} particles extracted from {n_micrographs:,} micrographs"
                )

            if output_job:
                output_job.save_output("extracted_particles", data)
                output_job.log(
                    f"Extracted {len(data):,} particles from {n_micrographs:,} micrographs\nJob completed",
                    level="text",
                )
                output_job.stop()
                data = None
            else:
                output_title += f"{'+'+micrograph_input if micrograph_input else ''}->extract particles"

    if data is None:
        return

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
            type=input_type,
            name=input_type,
            slots=list(output_slots),
            passthrough=(input_job.uid, input_group_name),
            title=f"{args.jobID}" + output_title,
            desc=f"{' '.join(sys.argv)}",
        )
        if args.verbose > 1:
            print(
                f"The results are saved to a new CryoSPARC external job: {args.projectID}/{args.workspaceID}/{new_jobID}"
            )


def extract_one_micrograph(
    subset,
    box_size,
    fft_crop_size,
    input_project_folder,
    output_project_folder,
    output_particle_foler,
    sign=-1,
    fill_mode="random",
    normalize=True,
    fp16=False,
    force=False,
    plot_pdf=False,
):
    micrograph_path = subset["location/micrograph_path"][0]
    micrograph_file = input_project_folder / subset["location/micrograph_path"][0]

    extracted_particles_filename = (
        f"{output_particle_foler}/{Path(micrograph_path).stem}.mrcs"
    )
    particle_file = output_project_folder / extracted_particles_filename

    apix = subset["location/micrograph_psize_A"][0] * box_size / fft_crop_size

    import mrcfile
    import numpy as np

    skip = False
    if not force and Path(particle_file).exists():
        with mrcfile.open(particle_file, header_only=True) as mrc:
            nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
            if nz == len(subset) and ny == nx and ny == fft_crop_size:
                skip = True

    if not skip:
        mic_shape = subset["location/micrograph_shape"][0]
        location_x = np.rint(subset["location/center_x_frac"] * mic_shape[1]).astype(
            np.int32
        )
        location_y = np.rint(subset["location/center_y_frac"] * mic_shape[0]).astype(
            np.int32
        )

        with mrcfile.mmap(str(micrograph_file), mode="r") as mrc_micrograph:
            micrograph = mrc_micrograph.data

            if plot_pdf:
                micrograph_lp = helicon.low_high_pass_filter(
                    data=micrograph, low_pass_fraction=0.05
                )
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.imshow(micrograph_lp, cmap="gray")
                ax.scatter(
                    location_x,
                    location_y,
                    color="red",
                    s=10,
                    label="Particle Locations",
                )
                ax.set_title("Micrograph with Particle Locations")
                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")
                ax.legend()
                pdf_filename = particle_file.with_suffix(".pdf")
                plt.savefig(pdf_filename, format="pdf")
                plt.close(fig)

            dtype = np.float16 if fp16 else np.float32
            particles = np.zeros(
                (len(subset), fft_crop_size, fft_crop_size), dtype=dtype
            )

            x0_offsets = location_x - box_size // 2
            y0_offsets = location_y - box_size // 2

            clip_buffer = np.zeros((box_size, box_size), dtype=np.float32)

            for i in range(len(subset)):
                x0, y0 = x0_offsets[i], y0_offsets[i]

                x_start, x_end = max(0, x0), min(mic_shape[1], x0 + box_size)
                y_start, y_end = max(0, y0), min(mic_shape[0], y0 + box_size)

                clip = micrograph[y_start:y_end, x_start:x_end]

                if clip.shape != (box_size, box_size):
                    clip_buffer.fill(0)
                    clip_buffer[
                        y_start - y0 : y_end - y0, x_start - x0 : x_end - x0
                    ] = clip.astype(clip_buffer.dtype)
                    clip = clip_buffer

                    if fill_mode is not None:
                        zeros = clip == 0
                        non_zeros = ~zeros
                        if fill_mode == "mean":
                            clip[zeros] = np.mean(clip[non_zeros])
                        elif fill_mode == "random":
                            non_zero_values = clip[non_zeros]
                            clip[zeros] = np.random.normal(
                                loc=np.mean(non_zero_values),
                                scale=np.std(non_zero_values),
                                size=np.count_nonzero(zeros),
                            )

                if fft_crop_size < box_size:
                    clip = helicon.fft_crop(
                        clip, output_size=(fft_crop_size, fft_crop_size)
                    )

                if sign < 0:
                    clip = np.max(clip) + np.min(clip) - clip

                if normalize:
                    std = np.std(clip)
                    if std:
                        mean = np.mean(clip)
                        clip = (clip - mean) / std

                particles[i] = clip.astype(dtype)

        with mrcfile.new(particle_file, overwrite=True) as mrc_output:
            mrc_output.set_data(particles)
            mrc_output.voxel_size = (apix, apix, apix)

    ret = subset.copy()
    ret["blob/path"] = str(extracted_particles_filename)
    ret["blob/idx"] = np.arange(len(ret))
    ret["blob/shape"] = [(fft_crop_size, fft_crop_size)] * len(ret)
    ret["blob/psize_A"] = apix
    ret["blob/sign"] = [sign] * len(ret)
    ret["blob/import_sig"] = [1] * len(ret)
    return ret


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
        "--groupIndex",
        type=int,
        metavar="<n>",
        help="the output group index of the input cryosparc job",
        default=0,
    )

    parser.add_argument(
        "--assignExposureGroupByBeamShift",
        type=str,
        metavar="0|1|xml_folder=path:min_micrographs_per_group=<n>",
        help="assign images to exposure groups according to the beam shifts, one group per beam shift position. disabled by default",
        default=None,
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
        help="assign images to exposure groups, one group per micrograph. disabled by default",
        default=0,
    )
    parser.add_argument(
        "--copyExposureGroupAssignments",
        type=str,
        metavar="<star file>",
        help="copy the optics group assignments from this star file. rlnMicrographMovieName and rlnOpticsGroup must be in this star file. disabled by default",
        default=0,
    )
    parser.add_argument(
        "--copyExposureGroupParameters",
        type=str,
        metavar="source_cs_file=<filename>|source_job_id=<Jxx>[:beam_tilt=<0|1>:trefoil=<0|1>:tetrafoil=<0|1>:cs=<0|1>:anisomag=<0|1>]",
        help="copy exposure group parameters (beam tilt, trefoil, tetrafoil, cs, anisotropic distortion, etc.). disabled by default",
        default="",
    )
    parser.add_argument(
        "--splitByMicrograph",
        type=bool,
        metavar="<0|1>",
        help="split the dataset by micrograph. disabled by default",
        default=0,
    )
    parser.add_argument(
        "--extractParticles",
        type=str,
        metavar="box_size=<n>:fft_crop_size=<n>[:recenter=<0|1>][replace_ctf=<0|1>][normalize=<0|1>][fill_mode=<mean|random>][sign=<-1|1>][n_micrographs=<-1|n>][fp16=<0|1>][:micrographs_cs_file=<filename>|micrographs_job_id=<JXXX>][reuse_job_id=<JXXX>]",
        help="split the dataset by micrograph. disabled by default",
        default="",
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
        not in "cpu groupIndex jobID projectID saveLocal verbose workspaceID".split()
    ]

    if (
        args.projectID or args.workspaceID or args.jobID or args.groupIndex
    ) and args.csFile:
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
