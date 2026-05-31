"""Handler for the process option."""

from __future__ import annotations
import helicon
import pandas as pd
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


option_name = "process"


def add_args(parser):
    parser.add_argument(
        "--process",
        metavar="processor_name:param1=value1:param2=value2",
        type=str,
        nargs="+",
        action="append",
        help="apply a processor named 'processorname' with all its parameters/values.",
    )


def handle(data, args, index_d, param):
    """Handle the process option.

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
    if param:
        process = param

        data_tmp = helicon.dataframe_convert(data, target="jspr")
        data_tmp = helicon.dataframe_jspr2dict(data_tmp)

        processors = []
        for p in process:
            processorname, param_dict = helicon.parse_param_str(p)
            if not param_dict:
                param_dict = {}
            if processorname in helicon.outplaceprocs:
                processors.append((processorname, param_dict, 1))
            else:
                processors.append((processorname, param_dict, 0))

        tag = args.tag if args.tag else "-".join([p[0] for p in processors])
        if tag:
            tag = "." + tag.strip(".")

        micrographNames = data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
        mgraphs = micrographNames.groupby(micrographNames, sort=False)

        mcount = 0
        d = helicon.EMData()

        for mgraphName, mgraphParticles in mgraphs:
            tmpdata = data.loc[mgraphParticles.index]
            filename = tmpdata["rlnImageName"].iloc[0].split("@")[-1]
            newfilename = str(Path(filename).with_suffix("")) + tag + ".mrcs"
            if not os.access(Path(newfilename).parent, os.W_OK):
                newfilename = Path(newfilename).name
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
                logger.info(
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
    return data, index_d
