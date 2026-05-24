"""Handler for the copyExposureGroupParameters option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np
from cryosparc.dataset import Dataset
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


option_name = "copyExposureGroupParameters"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the copyExposureGroupParameters option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--copyExposureGroupParameters",
        type=str,
        metavar="source_cs_file=<filename>|source_job_id=<Jxx>[:beam_tilt=<0|1>:trefoil=<0|1>:tetrafoil=<0|1>:cs=<0|1>:anisomag=<0|1>]",
        help="copy exposure group parameters (beam tilt, trefoil, tetrafoil, cs, anisotropic distortion, etc.). disabled by default",
        default="",
    )


def handle(
    data,
    args: argparse.Namespace,
    index_d: dict,
    param: object,
    output_title: str,
    output_slots: set,
    exp_group_id_name: str,
    micrograph_name: str,
    original_exp_group_ids: list,
):
    """Handle the copyExposureGroupParameters option.

    Parameters
    ----------
    data : Dataset
        The cryosparc Dataset.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.
    output_title : str
        Title for output filename construction.
    output_slots : set
        Output slot names.
    exp_group_id_name : str
        Name of the exposure group ID column.
    micrograph_name : str
        Name of the micrograph name column.
    original_exp_group_ids : list
        Original exposure group IDs.

    Returns
    -------
    tuple
        (data, output_title, output_slots, index_d) after processing.
    """
    if param:
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
            logger.warning("ignoring unknown parameters: %s", param_unsuppported)
        if args.verbose > 2:
            logger.info(f"\tCustom parameters: {param_changed}")

        if param_dict["source_cs_file"]:
            if param_dict["source_job_id"]:
                logger.warning(
                    "both source_cs_file and source_job_id are specified. I will use source_cs_file"
                )
            source_data_name = param_dict["source_cs_file"]
            source_data = Dataset.load(param_dict["source_cs_file"])
            source_exp_group_id_names_all = helicon.all_matched_attrs(
                source_data, query_str="exp_group_id"
            )
            if len(source_exp_group_id_names_all) == 0:
                raise HeliconError(
                    f"{param_dict['source_cs_file']} does not contain exp_group_id"
                )
        elif param_dict["source_job_id"]:
            source_job = cs.find_job(args.projectID, param_dict["source_job_id"])
            input_particle_group_name = None
            for g in source_job.doc["output_result_groups"]:
                if g["type"] in ["particle", "exposure"]:
                    input_particle_group_name = g["name"]
                    break
            if not input_particle_group_name:
                raise HeliconError(
                    f"{source_job} does not provide particles or exposures"
                )
            source_data_name = source_job.doc["uid"]
            source_data = source_job.load_output(input_particle_group_name)
            source_exp_group_id_names_all = helicon.all_matched_attrs(
                source_data, query_str="exp_group_id"
            )
            if len(source_exp_group_id_names_all) == 0:
                raise HeliconError(
                    f"{param_dict['source_job_id']} does not contain exp_group_id"
                )
        else:
            raise HeliconError(
                "either source_cs_file or source_job_id must be specified"
            )

        source_exp_group_id_name = helicon.first_matched_attr(
            source_data,
            attrs="ctf/exp_group_id location/exp_group_id mscope_params/exp_group_id".split(),
        )

        source_group_ids = np.unique(source_data[source_exp_group_id_name])

        source_micrograph_id_name = helicon.first_matched_attr(
            source_data,
            attrs="location/micrograph_uid uid".split(),
        )

        micrograph_id_name = helicon.first_matched_attr(
            data,
            attrs="location/micrograph_uid uid".split(),
        )

        mapping = {}
        for sgid in source_group_ids:
            source_mask = np.where(source_data[source_exp_group_id_name] == sgid)
            for uid in source_data[source_micrograph_id_name][source_mask]:
                mapping[uid] = int(sgid)
        unknown_egid = np.min(np.array(list(mapping.values()))) - 1

        mids = np.unique(data[micrograph_id_name])
        for mid in mids:
            mask = np.where(data[micrograph_id_name] == mid)
            data[exp_group_id_name][mask] = mapping.get(mid, unknown_egid)

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
            logger.warning(
                "No exposure group ctf parameters found in the source dataset. I will only copy the exposure group assignments"
            )

        group_ids = np.sort(np.unique(data[exp_group_id_name]))

        for group_id in group_ids:
            mask = np.where(data[exp_group_id_name] == group_id)
            if group_id in source_group_ids:
                source_mask = np.where(
                    source_data[source_exp_group_id_name] == group_id
                )
                for p in ctf_params_to_copy:
                    data[p][mask] = np.median(source_data[p][source_mask])
            else:
                for p in ctf_params_to_copy:
                    data[p][mask] = np.median(source_data[p])

        output_slots.add(exp_group_id_name.split("/")[0])
        output_slots.add("ctf")
        output_title += f"->copied params {' '.join(ctf_params_to_copy)} of {len(group_ids)} exposure groups from {source_data_name}"

        if args.verbose > 1:
            msg = f"{len(original_exp_group_ids)} -> {len(group_ids)} exposure groups"
            if len(ctf_params_to_copy):
                msg += f": {' '.join(ctf_params_to_copy)}"
            logger.info(f"\t{msg}")
    return data, output_title, output_slots, index_d
