"""Entry point for helicon"""

import argparse
import logging
import os, sys
from importlib import import_module
import helicon
from helicon.lib.exceptions import HeliconError, HeliconExit

logger = logging.getLogger(__name__)

cli_commands = [
    "cryosparc",
    "images2star",
    "proc3d",
    "trueFSC",
]
napari_commands = [
    "display",
]
shiny_commands = [
    "webApps",
]
streamlit_commands = [
    "ctfSimulation",
    "map2seq",
    "procart",
]
temporary_commands = [
    "HOM_containerC",
]


class HeliconArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, f"{self.prog}: error: {message}\n")


def _maybe_reexec_macos_display() -> None:
    """Relaunch ``helicon display`` under a Helicon-named executable on macOS.

    Dock and menu-bar titles are derived from the process executable basename.
    Console-script launches inherit ``python3.14`` unless the process is
    re-exec'd from a real executable file whose basename is ``Helicon``.

    A symlink is not enough here: macOS resolves the symlink at ``exec`` time
    and the Dock reads the *resolved* executable path, so only the menu-bar
    title changes. To also fix the Dock hover tooltip we copy the running
    interpreter into a file named ``Helicon`` and re-exec it, teaching the
    copy where its standard library lives via ``PYTHONHOME`` (a copied conda
    binary otherwise fails to locate its site-packages).
    """
    if sys.platform != "darwin":
        return
    if os.environ.get("HELICON_MACOS_IDENTITY") == "1":
        return
    if "display" not in sys.argv:
        return

    try:
        import ctypes
        import ctypes.util
        import shutil
        import tempfile
        from pathlib import Path

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if hasattr(libc, "setprogname"):
            libc.setprogname(b"Helicon")

        bin_dir = Path(tempfile.gettempdir()) / "helicon_bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        helicon_bin = bin_dir / "Helicon"
        python_bin = Path(sys.executable).resolve()

        # A legacy launch may have left a symlink pointing at the real
        # interpreter. A symlink is useless here: macOS resolves it at exec
        # time, so the Dock would keep reporting ``python3.14``. Remove it so
        # we copy a real executable instead of exec'ing through the link (which
        # would also make ``shutil.copy2`` write onto the shared interpreter).
        if helicon_bin.is_symlink():
            helicon_bin.unlink()

        # Refresh the cached copy whenever the interpreter changes (upgrade,
        # different machine, cleared temp dir, etc.). Compare size + mtime so a
        # pristine cached copy is not rewritten on every launch. ``is_file()``
        # follows symlinks, so we only reach this check after the symlink has
        # been removed above.
        refresh = not helicon_bin.exists()
        if not refresh:
            try:
                src_stat = python_bin.stat()
                dst_stat = helicon_bin.stat()
                refresh = (
                    src_stat.st_size != dst_stat.st_size
                    or src_stat.st_mtime_ns != dst_stat.st_mtime_ns
                )
            except OSError:
                refresh = True
        if refresh:
            shutil.copy2(python_bin, helicon_bin)

        env = os.environ.copy()
        env["HELICON_MACOS_IDENTITY"] = "1"
        env["PYTHONHOME"] = str(Path(sys.prefix).resolve())
        cmd = [str(helicon_bin)] + sys.argv
        os.execve(str(helicon_bin), cmd, env)
    except Exception as exc:  # pragma: no cover - macOS only, env dependent
        # Keep the original process if the identity relaunch fails. This
        # leaves the Dock/menu name as the python executable, so surface the
        # failure rather than hiding it.
        try:
            logger.debug("macOS identity relaunch skipped: %s", exc)
        except Exception:
            pass
        return


def _get_commands(
    cli_commands: list,
    napari_commands: list,
    shiny_commands: list,
    streamlit_commands: list,
    doc_str: str = "",
) -> None:
    parser = HeliconArgumentParser(description=doc_str, allow_abbrev=True)
    parser.add_argument(
        "--version", action="version", version="helicon " + helicon.__version__
    )

    subparsers = parser.add_subparsers(
        title="Choose a command", parser_class=HeliconArgumentParser
    )
    subparsers.required = True

    for module_name in sorted(
        cli_commands + napari_commands + shiny_commands + streamlit_commands
    ):
        if module_name in napari_commands and not helicon.has_napari():
            continue
        elif module_name in shiny_commands and not helicon.has_shiny():
            continue
        elif module_name in streamlit_commands and not helicon.has_streamlit():
            continue

        module_name_full = ".".join(["helicon", "commands", module_name])
        module = import_module(module_name_full)

        if hasattr(module, "add_args"):
            parsed_doc = module.__doc__.split("\n") if module.__doc__ else list()
            descr_txt = parsed_doc[0] if parsed_doc else ""
            epilog_txt = "" if len(parsed_doc) <= 1 else "\n".join(parsed_doc[1:])

            this_parser = subparsers.add_parser(
                module_name,
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
    except SystemExit as e:
        if e.code != 0:
            subparser = sys.argv[1] if len(sys.argv) > 1 else None
            if subparser and subparser in subparsers.choices:
                subparsers.choices[subparser].print_help()
            else:
                parser.print_usage()
            sys.exit(-1)
        else:
            raise
    except HeliconError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)
    except Exception:
        subparser = sys.argv[1] if len(sys.argv) > 1 else None
        if subparser and subparser in subparsers.choices:
            subparsers.choices[subparser].print_help()
        else:
            parser.print_usage()

        sys.exit(-1)

    try:
        args.main_function(args)
    except HeliconExit:
        sys.exit(0)
    except HeliconError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR: {e}")
        if helicon.available_cpu() > 1:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def main():
    _maybe_reexec_macos_display()
    _get_commands(
        cli_commands=cli_commands,
        napari_commands=napari_commands,
        shiny_commands=shiny_commands,
        streamlit_commands=streamlit_commands,
        doc_str="helicon commands",
    )
