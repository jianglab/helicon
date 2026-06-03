#!/usr/bin/env python

"""A command line tool that interacts with a CryoSPARC server and performs image analysis tasks"""

from __future__ import annotations
import argparse, sys, logging
from pathlib import Path
import numpy as np
import helicon
from cryosparc.dataset import Dataset
from helicon.lib.exceptions import HeliconError, HeliconValidationError

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Interact with a CryoSPARC server or cs file to perform exposure group ops.

    Loads data from CryoSPARC server jobs or local .cs files and applies
    operations like exposure group assignment, particle extraction, etc.

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

    if args.csFile:
        input_project_folder = [Path(f).resolve().parent.parent for f in args.csFile]
        if len(set(input_project_folder)) > 1:
            tmp = " ".join([str(p) for p in set(input_project_folder)])
            msg = f"you have specified input cs files in {len(input_project_folder)} projects ({tmp}). All input cs files must be from the same project!"
            raise HeliconError(msg)
        else:
            input_project_folder = input_project_folder[0]
        data_orig = []
        for f in args.csFile:
            tmp = Dataset.load(f)
            _passthrough_files = sorted(
                Path(f).parent.glob("*_passthrough_particles.cs")
            )
            if not _passthrough_files:
                _passthrough_files = sorted(
                    Path(f).parent.glob("*_passthrough_exposures.cs")
                )
            if _passthrough_files:
                if args.verbose > 1:
                    logger.debug(
                        "Merging passthrough file: %s", _passthrough_files[0].name
                    )
                _pdata = Dataset.load(str(_passthrough_files[0]))
                _pt_descrs = _pdata.descr()
                _to_add_names, _to_add_types = [], []
                for _item in _pt_descrs:
                    if _item[0] not in tmp:
                        _to_add_names.append(_item[0])
                        _to_add_types.append(_item[1] if len(_item) == 2 else _item[1:])
                if _to_add_names:
                    tmp.add_fields(_to_add_names, _to_add_types)
                    for _col in _to_add_names:
                        tmp[_col] = _pdata[_col]
            data_orig.append(tmp)
        input_type = ["particle" if "blob/path" in d else "exposure" for d in data_orig]
    else:
        cs = helicon.connect_cryosparc()
        project = cs.find_project(args.projectID)
        input_project_folder = project.dir()
        data_orig = []
        input_type = []
        for i, jobID in enumerate(args.jobID):
            input_job = cs.find_job(args.projectID, jobID)
            if len(input_job.doc["output_result_groups"]) < 1:
                logger.warning("%s does not have any output groups. Ignored", jobID)
                continue
            input_group = input_job.doc["output_result_groups"][args.groupIndex[i]]
            input_group_name = input_group["name"]
            data_orig.append(input_job.load_output(input_group_name))
            input_type.append(input_group["type"])
            if args.outputWorkspaceID is None:
                args.outputWorkspaceID = input_job.doc["workspace_uids"][-1]
        if len(data_orig) < 1:
            logger.warning("no input data. I am going to quit")
            raise HeliconError("no input data")

    if len(set(input_type)) > 1:
        msg = f"you have specified {len(input_type)} types of input {input_type}. All inputs should of the same type!"
        raise HeliconError(msg)
    else:
        input_type = input_type[0]

    if len(data_orig) > 1:
        from cryosparc.dataset import Dataset

        data_orig = Dataset.union(*data_orig)
    else:
        data_orig = data_orig[0]

    if data_orig is None or not len(data_orig):
        logger.warning("no data in the input. Nothing to do.")
        raise HeliconError("no data in the input")

    if args.saveLocal:
        output_project_folder = Path(".")
    else:
        output_project_folder = input_project_folder
        output_job = None

    args.input_project_folder = input_project_folder
    args.output_project_folder = output_project_folder

    data = data_orig.copy()

    attrs = "movie_blob/path micrograph_blob/path location/micrograph_path blob/path".split()
    micrograph_name = helicon.first_matched_attr(data, attrs=attrs)
    if micrograph_name is None:
        raise HeliconError(
            f"at least one of the {len(attrs)} parameters ({' '.join(attrs)}) must be available. You data are:\n{data}"
        )

    if args.verbose > 1:
        micrographs = np.unique(data[micrograph_name])
        if input_type == "particle":
            msg = f"{len(data):,} particles in {len(micrographs):,} micrographs"
        else:
            msg = f"{len(micrographs):,} micrographs"
        if args.csFile:
            msg += f" from {len(args.csFile)} cs files"
        else:
            msg += f" from {args.projectID}/{','.join(args.jobID)}"
        logger.info(msg)

    if args.verbose > 10:
        logger.debug(data)

    exp_group_id_names_all = helicon.all_matched_attrs(data, query_str="exp_group_id")
    exp_group_id_name = helicon.first_matched_attr(
        data,
        attrs="ctf/exp_group_id location/exp_group_id mscope_params/exp_group_id".split(),
    )
    if exp_group_id_name is None:
        exp_group_id_name = "mscope_params/exp_group_id"
        data.add_fields([exp_group_id_name], ["u4"])

    original_exp_group_ids = np.unique(data[exp_group_id_name])

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
            logger.info("%s: %s", option_name, param)

        from helicon.plugins.cryosparc import dispatch

        data, output_title, output_slots, index_d = dispatch(
            option_name,
            data,
            args,
            index_d,
            param,
            output_title,
            output_slots,
            exp_group_id_name,
            micrograph_name,
            original_exp_group_ids,
        )


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the cryosparc command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to attach arguments to.
    """
    # Infrastructure arguments
    parser.add_argument(
        "--csFile",
        type=str,
        nargs="+",
        metavar="<filename>",
        help="input local .cs file(s)",
        default=[],
    )
    parser.add_argument(
        "-p", "--projectID", type=str, metavar="<pid>", help="CryoSPARC project ID"
    )
    parser.add_argument(
        "-j",
        "--jobID",
        type=str,
        action="append",
        metavar="<jid>",
        help="CryoSPARC job ID(s)",
        default=[],
    )
    parser.add_argument(
        "-g",
        "--groupIndex",
        type=int,
        action="append",
        metavar="<n>",
        help="output group index for each job (default: 0)",
        default=[],
    )
    parser.add_argument(
        "-w",
        "--outputWorkspaceID",
        type=str,
        metavar="<wid>",
        help="output workspace ID (e.g. W1)",
    )
    parser.add_argument(
        "--saveLocal",
        type=int,
        metavar="<0|1>",
        help="save output data as local .cs file(s) instead of uploading to CryoSPARC server",
        default=0,
    )
    parser.add_argument(
        "--cpu",
        type=int,
        metavar="<n>",
        help="number of CPUs. default to -1 (all available)",
        default=-1,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        metavar="<0|1|2>",
        help="verbose mode. default to %(default)s",
        default=2,
    )

    from helicon.plugins.cryosparc import add_plugin_args

    add_plugin_args(parser)

    return parser


def check_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Validate cryosparc command arguments.

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
        not in "cpu groupIndex jobID projectID saveLocal verbose outputWorkspaceID".split()
    ]

    if (args.projectID or args.jobID or args.groupIndex) and args.csFile:
        msg = f"You should only specify options for CryoSPARC server (--projectID --jobID) or local file (--csFile), but not both"
        logger.error(msg)
        raise HeliconValidationError(msg)

    if not ((args.projectID and args.jobID) or args.csFile):
        msg = f"You should specify options for either CryoSPARC server (--projectID --jobID) or local file (--csFile)"
        logger.error(msg)
        raise HeliconValidationError(msg)

    if len(args.jobID):
        if len(args.groupIndex) not in [0, len(args.jobID)]:
            msg = f"You should specify {len(args.jobID)} --jobID options but {len(args.groupIndex)} --groupIndex options. You should specify either no --groupIndex option (i.e. default to 0) or {len(args.jobID)} --groupIndex options (i.e. the same number as that of --joibID options)"
            logger.error(msg)
            raise HeliconValidationError(msg)
        elif len(args.groupIndex) == 0:
            args.groupIndex = [0] * len(args.jobID)

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    args = check_args(args, parser)
    main(args)
