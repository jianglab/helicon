"""Option-handler plugins of the cryosparc, images2star and proc3d commands."""

from __future__ import annotations
import argparse
from collections.abc import Iterable


def plugin_options_in_argv(
    argv: list[str], parser: argparse.ArgumentParser, plugin_names: Iterable[str]
) -> list[str]:
    """List the plugin options given on the command line, in their given order.

    Infrastructure options (input/output files, ``--verbose``, ``--cpu``, ...)
    are left out, so the result can be dispatched to the plugin handlers.

    Parameters
    ----------
    argv : list of str
        Command-line arguments.
    parser : argparse.ArgumentParser
        The parser of the command, used to map option flags to their dest.
    plugin_names : Iterable of str
        Option names (dests) handled by plugins.

    Returns
    -------
    list of str
        The dest of each plugin option in ``argv``, repeated options included.
    """
    from helicon.lib.system import get_option_list

    flag_to_dest = {
        s.lstrip("-"): a.dest for a in parser._actions for s in a.option_strings
    }
    plugin_names = set(plugin_names)
    dests = [flag_to_dest.get(o, o) for o in get_option_list(argv)]
    return [d for d in dests if d in plugin_names]
