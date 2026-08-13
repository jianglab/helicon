"""Reusable pipeline engine behind the ``proc3d`` command.

The ``helicon proc3d`` CLI and the Helicon display "Proc3D" tools panel both
run 3D maps through this engine, so GUI behavior cannot drift from CLI
semantics. The engine transforms an in-memory NumPy volume; plugins that
write files do not exist for proc3d (all four current operations transform
the volume in memory), so the same option set is available on the command
line and in the GUI.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable

import numpy as np

from helicon.plugins.proc3d import dispatch

logger = logging.getLogger(__name__)


def apply_options(
    data: np.ndarray,
    apix: float,
    options: list[str],
    args: argparse.Namespace,
    append_options: Iterable[str] = (),
) -> tuple[np.ndarray, float]:
    """Apply ``proc3d`` plugin options to a volume in order.

    This is the ordered dispatch loop that used to be embedded in
    ``helicon.commands.proc3d.main``. Each option name is looked up in the
    auto-discovered plugin registry and its ``handle(data, args, index_d,
    param, apix, nx, ny, nz)`` is invoked with the current volume, pixel
    size, and dimensions, exactly like the CLI does. Dimensions are refreshed
    from ``data.shape`` after every option so later options see the
    transformed geometry (e.g. a ``clip`` following a ``flip_hand``).

    Parameters
    ----------
    data : np.ndarray
        Input 3D volume with shape ``(nz, ny, nx)``. The engine works on a
        copy, so the caller's array is never mutated in place.
    apix : float
        Pixel size in Angstroms.
    options : list of str
        Ordered option names with leading ``--`` stripped, e.g.
        ``["flip_hand", "clip"]``.
    args : argparse.Namespace
        Namespace holding the parsed option values; the same object passed
        to every plugin handler.
    append_options : Iterable of str, optional
        Dest names whose argparse action type is ``_AppendAction``. proc3d
        currently has none; kept for parity with the images2star engine.

    Returns
    -------
    tuple of (np.ndarray, float)
        The transformed volume and its (possibly updated) pixel size.

    Raises
    ------
    ValueError
        If an option in ``options`` has no registered plugin handler.
    """
    data = data.copy()
    nz, ny, nx = data.shape
    index_d = {option_name: 0 for option_name in options}
    append_set = set(append_options)
    for option_name in options:
        values = args.__dict__[option_name]
        if option_name in append_set:
            param = values[index_d[option_name]]
        else:
            param = values
        if getattr(args, "verbose", 0):
            logger.info("%s: %s", option_name, param)
        data, apix, nx, ny, nz = dispatch(
            option_name, data, args, index_d, param, apix, nx, ny, nz
        )
        index_d[option_name] += 1
    return data, apix


def operation_specs() -> dict[str, dict]:
    """Introspect every registered plugin into a generic option spec.

    Each spec describes the argparse action behind a plugin's ``--option`` so
    that a GUI (or any caller) can render a parameter form and build an
    ``argparse.Namespace`` without re-implementing option semantics.

    Returns
    -------
    dict of str to dict
        Maps option names (leading ``--`` stripped) to specs with keys:
        ``dest``, ``option_string``, ``metavar``, ``type``, ``nargs``,
        ``choices``, ``default``, ``help``, and ``append`` (True for
        argparse ``_AppendAction`` options, whose occurrences accumulate).
    """
    from helicon.plugins import proc3d as plugins

    specs = {}
    for name in sorted(plugins._plugins):
        mod = plugins._plugins[name]
        if not hasattr(mod, "add_args"):
            continue
        parser = argparse.ArgumentParser(add_help=False)
        try:
            mod.add_args(parser)
        except Exception:
            continue
        action = next((a for a in parser._actions if a.dest != "help"), None)
        if action is None:
            continue
        specs[name] = {
            "dest": action.dest,
            "option_string": (
                action.option_strings[0]
                if action.option_strings
                else f"--{action.dest}"
            ),
            "metavar": _metavar_text(action.metavar),
            "type": action.type if action.type is not None else str,
            "nargs": action.nargs,
            "choices": list(action.choices) if action.choices else None,
            "default": action.default,
            "help": action.help or "",
            "append": type(action) is argparse._AppendAction,
        }
    return specs


def _metavar_text(metavar) -> str:
    """Normalize an argparse metavar (str or tuple) to a single string."""
    if isinstance(metavar, (tuple, list)):
        return " ".join(str(m) for m in metavar)
    return str(metavar) if metavar is not None else ""


def gui_operation_specs() -> dict[str, dict]:
    """Return operation specs safe for the GUI stack.

    All proc3d plugins (``apix``, ``clip``, ``flip_hand``,
    ``z_moving_average``) transform the volume in memory, so the full set is
    available in the tools panel — there is no file-writing operation to
    exclude, unlike :func:`helicon.lib.images2star_engine.gui_operation_specs`.
    """
    return operation_specs()


def stack_to_namespace(
    stack: list[tuple[str, object]], specs: dict[str, dict], **extra
) -> argparse.Namespace:
    """Build an ``argparse.Namespace`` from an ordered operation stack.

    Every registered plugin option is seeded with its argparse default (plus
    the CLI infrastructure defaults handlers rely on), then each stacked
    operation overrides its option: append-style options accumulate their
    per-occurrence values in order, non-append options may appear at most
    once. The result is exactly what :func:`apply_options` expects.

    Parameters
    ----------
    stack : list of (str, object)
        Ordered ``(option_name, converted_param)`` entries.
    specs : dict of str to dict
        Operation specs from :func:`operation_specs`.
    **extra
        Extra attributes to set on the namespace (e.g. ``inputMapFile``).

    Returns
    -------
    argparse.Namespace
        Namespace ready for :func:`apply_options`.

    Raises
    ------
    ValueError
        If a non-append option appears more than once in the stack.
    """
    from helicon.commands import proc3d

    parser = argparse.ArgumentParser(add_help=False)
    proc3d.add_args(parser)
    args = argparse.Namespace()
    for action in parser._actions:
        if action.dest == "help":
            continue
        setattr(args, action.dest, action.default)
    # GUI runs stay quiet and single-threaded; the engine itself is
    # synchronous, so only the plugin handlers' ``verbose`` flag matters.
    args.verbose = 0
    args.cpu = 1
    args.force = 0
    args.inputMapFile = extra.pop("inputMapFile", "")
    args.outputMapFile = extra.pop("outputMapFile", None)

    values = {}
    for name, param in stack:
        if name not in specs:
            raise ValueError(f"unknown operation: {name}")
        if specs[name]["append"]:
            values.setdefault(name, []).append(param)
        else:
            if name in values:
                raise ValueError(f"option {name} can only be applied once")
            values[name] = param
    for name, param in values.items():
        setattr(args, name, param)
    for key, value in extra.items():
        setattr(args, key, value)
    return args
