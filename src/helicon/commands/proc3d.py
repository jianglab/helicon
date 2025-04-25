#!/usr/bin/env python

"""A command line tool that anaylzes/transforms 3D maps"""

import argparse, sys
from pathlib import Path
import numpy as np
import mrcfile
import helicon


def main(args):
    helicon.log_command_line()

    if args.cpu < 1:
        args.cpu = helicon.available_cpu()

    n_maps = len(args.inputMapFile)
    with mrcfile.open(args.inputMapFile[0], mode="r") as mrc:
        data = mrc.data
        nz, ny, nx = data.shape
        apix = round(
            float(mrc.voxel_size.x), 4
        )  # assuming equal sampling in x,y,z dimensions
        if args.verbose > 0:
            msg = f"Input map: {args.inputMapFile[0]}"
            msg += f"\n\tnx,ny,nz={nx},{ny},{nz} pixels\tsampling={apix} Å/pixel"
            print(msg)

    if args.verbose > 0:
        title = f"{args.inputMapFile[0]}: {nx}x{ny}x{nz} pixels | apix={apix}Å/pixel"
        helicon.display_map_orthoslices(data, title=title, hold=False)

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

        if option_name == "apix":
            apix = float(param)

        elif option_name == "helical_sym" and param:
            param_dict_default = dict(
                twist=0.0,  # °
                rise=0.0,  # Å
                csym=1,
                center_len=0.0,  # Å
                center_n_rise=0.0,
                center_fraction=0.0,
                new_apix=apix,
                new_nz=nz,
                new_nxy=nx,
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
            twist = float(param_dict["twist"])
            rise = float(param_dict["rise"])
            csym = int(param_dict.get("csym", 1))
            if rise <= 0:
                helicon.color_print("\tERROR: rise (>0) must be specified")
                sys.exit(-1)
            if csym < 1:
                helicon.color_print("\tERROR: csym (>0) must be specified")
                sys.exit(-1)
            new_apix = float(param_dict.get("new_apix", apix))
            new_nz = int(param_dict["new_nz"])
            new_nxy = int(param_dict["new_nxy"])
            center_len = float(param_dict["center_len"])
            center_n_rise = float(param_dict["center_n_rise"])
            center_fraction = float(param_dict["center_fraction"])
            tmp = (
                int(center_len > 0) + int(center_n_rise > 0) + int(center_fraction > 0)
            )
            if tmp != 1:
                if tmp <= 0:
                    msg = "\tERROR: center_len or center_n_rise or center_fraction must be specified"
                else:
                    msg = "\tERROR: only one of the these three options (center_len, center_n_rise, center_fraction) should be specified"
                helicon.color_print(msg)
                sys.exit(-1)
            if center_len > 0:
                if center_len < rise:
                    helicon.color_print(
                        f"\tERROR: center_len must be larger than rise (={rise} Å)"
                    )
                    sys.exit(-1)
                center_fraction = center_len / (nz * apix)
            elif center_n_rise > 0:
                center_fraction = center_n_rise * rise / (nz * apix)
            center_fraction = max(rise / (nz * apix), min(1.0, center_fraction))
            data = helicon.apply_helical_symmetry(
                data=data,
                apix=apix,
                twist_degree=twist,
                rise_angstrom=rise,
                csym=csym,
                fraction=center_fraction,
                new_size=(new_nz, new_nxy, new_nxy),
                new_apix=new_apix,
                cpu=args.cpu,
            )
            apix = new_apix
            nz, ny, nx = data.shape

        index_d[option_name] += 1

    if args.verbose > 1:
        msg = f"Output map: {str(args.outputMapFile)}"
        msg += f"\n\tnx,ny,nz={nx},{ny},{nz} pixels\tsampling={apix} Å/pixel"
        print(msg)

    with mrcfile.new(args.outputMapFile, data=data, overwrite=args.force) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = apix

    if args.verbose > 0:
        title = f"{str(args.outputMapFile)}: {nx}x{ny}x{nz} pixels | apix={apix}Å/pixel"
        helicon.display_map_orthoslices(data, title=title, hold=True)


def add_args(parser):
    parser.add_argument(
        "inputMapFile",
        type=str,
        metavar="<inputMapFile>",
        nargs="+",
        help="input 3D map file(s) in MRC format",
        default=[],
    )

    parser.add_argument(
        "--outputMapFile",
        type=str,
        metavar="<filename>",
        help="save output map to this file. default to %(default)s",
        default="",
    )

    parser.add_argument(
        "--apix",
        type=float,
        metavar="<Å>",
        help="set pixel size to this value. default to %(default)s",
        default=None,
    )

    parser.add_argument(
        "--helical_sym",
        type=str,
        metavar="twist=<°>:rise=<Å>[:csym=<n>:center_n_rise=<n>:center_len=<Å>:center_fraction=<val>:new_apix=<Å>:new_nz=<pixel>:new_nxy=<pixel>]",
        help="symmetrize the map with the specified helical parameters. default to %(default)s",
        default="",
    )

    parser.add_argument(
        "--force",
        type=int,
        metavar="<0|1>",
        help="force overwrite the output file. default to %(default)s",
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
        help="number of cpus to use. default to %(default)s and use all idle cpus",
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
        if o not in "cpu force inputMapFile outputMapFile verbose".split()
    ]

    if args.outputMapFile:
        args.outputMapFile = Path(args.outputMapFile)
    else:
        args.outputMapFile = Path(args.inputMapFile[0]).with_suffix(".proc3d.mrc")

    if args.outputMapFile.exists() and not args.force:
        helicon.color_print(
            f"ERROR: output file {str(args.outputMapFile)} already exists. Use --force to overwrite it"
        )
        sys.exit(-1)

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
