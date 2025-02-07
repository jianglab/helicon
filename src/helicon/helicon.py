"""Entry point for helicon"""

import argparse
import os, sys
from importlib import import_module
import helicon

shiny_commands = ["denovo3D", "helicalPitch", "helicalProjection", "whereIsMyClass"]
streamlit_commands = ["ctfSimulation", "helicalLattice", "hi3d", "hill", "procart"]


def _get_commands(cmd_dir: str, doc_str: str = "") -> None:
    parser = argparse.ArgumentParser(description=doc_str, allow_abbrev=True)
    parser.add_argument(
        "--version", action="version", version="helicon " + helicon.__version__
    )

    subparsers = parser.add_subparsers(title="Choose a command")
    subparsers.required = True

    dir_lbl = os.path.basename(cmd_dir)
    module_files = sorted(os.listdir(cmd_dir))
    for module_file in module_files:
        if module_file != "__init__.py" and module_file[-3:] == ".py":
            module_name = module_file[:-3]
            if module_name in shiny_commands and not helicon.has_shiny():
                continue
            elif module_name in streamlit_commands and not helicon.has_streamlit():
                continue

            module_name_full = ".".join(["helicon", dir_lbl, module_name])
            module = import_module(module_name_full)

            if hasattr(module, "add_args"):
                parsed_doc = module.__doc__.split("\n") if module.__doc__ else list()
                descr_txt = parsed_doc[0] if parsed_doc else ""
                epilog_txt = "" if len(parsed_doc) <= 1 else "\n".join(parsed_doc[1:])

                this_parser = subparsers.add_parser(
                    module_file[:-3],
                    help=descr_txt,
                    description=descr_txt,
                    epilog=epilog_txt,
                    allow_abbrev=True,
                )
                module.add_args(this_parser)
                this_parser.set_defaults(
                    main_function=module.main,
                    this_parser=this_parser,
                    check_args_function=None,
                )
                if hasattr(module, "check_args"):
                    this_parser.set_defaults(check_args_function=module.check_args)

    try:
        args = parser.parse_args()
        if args.check_args_function is not None:
            args = args.check_args_function(args, args.this_parser)
    except:
        subparser = sys.argv[1] if len(sys.argv) > 1 else None
        if subparser and subparser in subparsers.choices:
            subparsers.choices[subparser].print_help()

        sys.exit(-1)

    args.main_function(args)


def main():
    _get_commands(
        cmd_dir=os.path.join(os.path.dirname(__file__), "commands"),
        doc_str="helicon commands",
    )
