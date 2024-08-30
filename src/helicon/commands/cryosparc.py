#!/usr/bin/env python

'''A command line tool that interacts with a CryoSPARC server and performs image analysis tasks'''

import argparse, sys
from pathlib import Path
import numpy as np
import helicon

def main(args):
    helicon.log_command_line()

    if args.cpu<1: args.cpu = helicon.available_cpu()
    
    cs = helicon.connect_cryosparc()
    job = cs.find_job(args.project_id, args.job_id)
    data_orig = job.load_output("particles")
    data = data_orig.copy()

    if args.verbose>1:
        image_name = helicon.first_matched_atrr(data, attrs="location/micrograph_path blob/path".split())
        if image_name is None:
            helicon.color_print(f"\tERROR: location/micrograph_path or blob/path must be available")
            sys.exit(-1)        
        micrographs = np.unique(data[image_name])
        if args.verbose>1:
            print(f"{args.project_id}/{args.workspace_id}/{args.job_id}: {len(data)} particles from {len(micrographs)} micrographs")

    if args.verbose>10:
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
                
            image_name = helicon.first_matched_atrr(data, attrs="location/micrograph_path blob/path".split())
            if image_name is None:
                helicon.color_print(f"\tERROR: location/micrograph_path or blob/path must be available")
                sys.exit(-1)
                
            software = helicon.guess_data_collection_software(data[image_name][0])
            if software is None:
                helicon.color_print(f"\tWARNING: cannot detect the data collection software using {image_name}: {data[image_name][0]}\n\tI only know the filenames by {', '.join(sorted(helicon.movie_filename_patterns().keys()))}")
                sys.exit(-1)

            if software in ["EPU"]:
                extractBeamShift = helicon.extract_EPU_beamshift_pos
            elif software in ["serialEM_pncc"]:
                extractBeamShift = helicon.extract_serialEM_pncc_beamshift
            # split by beamshift groups
            def get_micrograph_path_2_beamshift_groups(micrographs):
                mapping = {m: extractBeamShift(m) for m in micrographs}
                mapping2 = {s: si+1  for si, s in enumerate(sorted(set(mapping.values())))}
                return {m: mapping2[mapping[m]] for m in micrographs}
                
            micrographs = np.unique(data[image_name])
            micrograph_path_2_beamshift_group = get_micrograph_path_2_beamshift_groups(micrographs)
            particle_group = [micrograph_path_2_beamshift_group[row[image_name]] for row in data.rows()]
            data["ctf/exp_group_id"] = np.array(particle_group)

            group_ids = np.sort(np.unique(data["ctf/exp_group_id"]))               
            for gi in group_ids:
                mask = np.where(data["ctf/exp_group_id"] == gi)
                for col in "ctf/cs_mm ctf/phase_shift_rad ctf/shift_A ctf/tilt_A ctf/trefoil_A ctf/tetra_A ctf/anisomag".split():
                    if col in data:
                        data[col][mask] = np.median(data[col][mask])

            output_slots.add("ctf")
            output_title += f"->{len(group_ids)} beamshift groups"

            if args.verbose>1:
                print(f"\t{len(group_ids_orig)} -> {len(group_ids)} exposure groups")

        elif option_name == "assignExposureGroupByTime" and abs(param) > 0:
            time_group_size = param
            
            group_ids_orig = np.sort(np.unique(data["ctf/exp_group_id"]))

            if time_group_size < 0 and len(group_ids_orig) > 1: # combine previous groups (if there are) into a single group first
                if args.verbose > 1:
                    print(f"\tCombining {len(group_ids_orig)} exposure groups into 1 group")
                data["ctf/exp_group_id"] = 1
                group_ids_orig = np.sort(np.unique(data["ctf/exp_group_id"]))
                time_group_size = abs(time_group_size)
                
            image_name = helicon.first_matched_atrr(data, attrs="location/micrograph_path blob/path".split())
            if image_name is None:
                helicon.color_print(f"\tERROR: location/micrograph_path or blob/path must be available")
                sys.exit(-1)
                
            software = helicon.guess_data_collection_software(data[image_name][0])
            if software is None:
                helicon.color_print(f"\tWARNING: cannot detect the data collection software using {image_name}: {data[image_name][0]}\n\tI only know the filenames by {', '.join(sorted(helicon.movie_filename_patterns().keys()))}")
                sys.exit(-1)
            elif software not in ["EPU", "EPU_old"]:
                helicon.color_print(f"\tWARNING: I can only detect data collection time for EPU-collected data. It appears that you used {software} to collect the data")
                sys.exit(-1)

            if software in ["EPU"]:
                extractDataCollectionTime = helicon.extract_EPU_data_collection_time
            elif software in ["EPU_old"]:
                extractDataCollectionTime = helicon.extract_EPU_old_data_collection_time

            micrographs = np.unique(data[image_name])
            micrograph_path_2_time = {m: extractDataCollectionTime(m) for m in micrographs}
            last_group_id = 0
            new_particle_group_ids = np.zeros(len(data))
            for gi in group_ids_orig:
                mask = np.where(data["ctf/exp_group_id"] == gi)
                group_micrographs = np.unique(data[image_name][mask])
                group_micrograph_time = [micrograph_path_2_time[m] for m in group_micrographs]
                group_time_2_subgroup = helicon.assign_to_groups(group_micrograph_time, time_group_size)
                group_particle_2_subgroup = [group_time_2_subgroup[micrograph_path_2_time[m]] for m in data[image_name][mask]]
                new_particle_group_ids[mask] = np.array(group_particle_2_subgroup) + last_group_id
                last_group_id = np.max(new_particle_group_ids)
            data["ctf/exp_group_id"] = new_particle_group_ids

            group_ids = np.sort(np.unique(data["ctf/exp_group_id"]))               
            for gi in group_ids:
                mask = np.where(data["ctf/exp_group_id"] == gi)
                for col in "ctf/cs_mm ctf/phase_shift_rad ctf/shift_A ctf/tilt_A ctf/trefoil_A ctf/tetra_A ctf/anisomag".split():
                    if col in data:
                        data[col][mask] = np.median(data[col][mask])
            
            output_slots.add("ctf")
            output_title += f"->{len(group_ids)} time groups"

            if args.verbose>1:
                print(f"\t{len(group_ids_orig)} -> {len(group_ids)} exposure groups")             

        elif option_name == "assignExposureGroupPerMicrograph" and param:
            group_ids_orig = np.sort(np.unique(data["ctf/exp_group_id"]))
                
            image_name = helicon.first_matched_atrr(data, attrs="location/micrograph_path blob/path".split())
            if image_name is None:
                helicon.color_print(f"\tERROR: location/micrograph_path or blob/path must be available")
                sys.exit(-1)
                
            micrographs = np.unique(data[image_name])
            for mi, m in enumerate(micrographs):
                mask = np.where(data[image_name] == m)
                data["ctf/exp_group_id"][mask] = mi+1
            
            output_slots.add("ctf")
            output_title += f"->{len(group_ids)} per-micrograph groups"

            if args.verbose>1:
                group_ids = np.sort(np.unique(data["ctf/exp_group_id"]))               
                print(f"\t{len(group_ids_orig)} -> {len(group_ids)} exposure groups")             
    
    if args.save_local:
        output_file = f"{args.project_id}_{args.workspace_id}_{args.job_id}" + output_title + ".cs"
        output_file = '-'.join(output_file.split())
        data.save(output_file)
        if args.verbose>1:
            print(f"The results are saved to {output_file}")
    else:
        project = cs.find_project(args.project_id)
        new_job_id = project.save_external_result(
            workspace_uid = args.workspace_id,
            dataset = data,
            type = "particle",
            name = "particles",
            slots = list(output_slots),
            passthrough = (job.uid, "particles"),
            title = f"{args.job_id}" + output_title
        )
        if args.verbose>1:
            print(f"The results are saved to a new CryoSPARC external job: {args.project_id}/{args.workspace_id}/{new_job_id}")

def add_args(parser):
    parser.add_argument("--project_id", type=str, metavar="<Pxx>", help="input cryosparc project id", required=True)
    parser.add_argument("--workspace_id", type=str, metavar="<Wx>", help="input cryosparc workspace id", required=True)
    parser.add_argument("--job_id", type=str, metavar="<Jxx>", help="input cryosparc job id", required=True)

    parser.add_argument("--assignExposureGroupByBeamShift", type=bool, metavar="<0|1>",
                        help="assign images to exposure groups according to the beam shifts, one group per beam shift position. default to 0", default=0)
    parser.add_argument("--assignExposureGroupByTime", type=int, metavar="<n>",
                        help="assign images to exposure groups according to data collection time, n movies per group. disabled by default", default=-1)
    parser.add_argument("--assignExposureGroupPerMicrograph", type=bool, metavar="<0|1>",
                        help="assign images to exposure groups, one group per micrograph. default to 0", default=0)
    parser.add_argument("--save_local", type=bool, metavar="<0|1>", help="save results to a local cs file instead of creating a new external job on the CryoSPARC server. default to 0", default=0)
    parser.add_argument("--verbose", type=int, metavar="<0|1>", help="verbose mode. default to 2", default=3)
    parser.add_argument("--cpu", type=int, metavar="<n>", help="number of cpus to use. default to 1", default=1)

    return parser

def check_args(args, parser):
    args.append_options = [a.dest for a in parser._actions if type(a) is argparse._AppendAction]
    all_options = helicon.get_option_list(sys.argv[1:])
    args.all_options = [o for o in all_options if o not in "cpu job_id project_id save_local verbose workspace_id".split()]
    
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())