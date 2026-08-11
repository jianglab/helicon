"""Reusable pipeline engine behind the ``images2star`` command.

The ``helicon images2star`` CLI and the Helicon display "Images2Star" tools
panel both run datasets through this engine, so GUI behavior cannot drift
from CLI semantics. The engine transforms a pandas DataFrame in memory;
plugins that write image files (``--createstack``, ``--process``, ...) run
only when the caller chooses to execute them.
"""

from __future__ import annotations

import argparse
import logging
import shlex
from collections.abc import Iterable

import pandas as pd

from helicon.plugins.images2star import dispatch

logger = logging.getLogger(__name__)

# Operations that write image/plot files or need CLI-only infrastructure.
# They stay available on the command line; the GUI operation stack only
# exposes pure in-memory transforms (Phase 1 scope).
GUI_EXCLUDED_OPERATIONS = frozenset(
    {
        # File-writing / artifact-producing operations.
        "averagePowerSpectra",
        "createStack",
        "denoiseCurvelet",
        "estimateHelicalAngleVariance",
        "extractHelices",
        "fullStack",
        "maskGold",
        "minStack",
        "process",
        "recoverFullFilaments",
        "splitByMicrograph",
        # Handled outside the engine (path conversion happens at save time).
        "path",
        # Legacy plugin without argparse registration.
        "sets",
    }
)


def apply_options(
    data: pd.DataFrame,
    options: list[str],
    args: argparse.Namespace,
    append_options: Iterable[str] = (),
) -> pd.DataFrame:
    """Apply ``images2star`` plugin options to ``data`` in order.

    This is the ordered dispatch loop that used to be embedded in
    ``helicon.commands.images2star.main``. Each option name is looked up in
    the auto-discovered plugin registry and its ``handle(data, args,
    index_d, param)`` is invoked with the current DataFrame, exactly like the
    CLI does. Option order is semantic, so callers must preserve it.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to transform. DataFrame ``attrs`` (such as ``optics``) are
        carried through to the result.
    options : list of str
        Ordered option names with leading ``--`` stripped, e.g.
        ``["select", "sortby"]``.
    args : argparse.Namespace
        Namespace holding the parsed option values; the same object passed
        to every plugin handler.
    append_options : Iterable of str, optional
        Dest names whose argparse action type is ``_AppendAction``. For these
        the engine consumes one parameter value per occurrence in ``options``,
        matching the CLI's index tracking.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame.

    Raises
    ------
    ValueError
        If an option in ``options`` has no registered plugin handler.
    """
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
        data, index_d = dispatch(option_name, data, args, index_d, param)
    return data


def operation_specs() -> dict[str, dict]:
    """Introspect every registered plugin into a generic option spec.

    Each spec describes the argparse action behind a plugin's ``--option`` so
    that a GUI (or any caller) can render a parameter form and build an
    ``argparse.Namespace`` without re-implementing option semantics.

    Returns
    -------
    dict of str to dict
        Maps option names (leading ``--`` stripped) to specs with keys:
        ``dest``, ``option_string``, ``type``, ``nargs``, ``choices``,
        ``default``, ``help``, and ``append`` (True for argparse
        ``_AppendAction`` options, whose occurrences accumulate).
    """
    from helicon.plugins import images2star as plugins

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
    """Return operation specs safe for the GUI stack (no file writes)."""
    return {
        name: spec
        for name, spec in operation_specs().items()
        if name not in GUI_EXCLUDED_OPERATIONS
    }


def _convert_token(token: str, typ) -> object:
    """Convert one CLI-style token with the option's argparse type."""
    if typ is bool:
        return token.strip().lower() in ("1", "true", "yes", "on")
    try:
        return typ(token)
    except Exception:
        raise ValueError(f"cannot parse {token!r} as {typ.__name__}")


def parse_operation_value(text: str, spec: dict) -> object:
    """Parse CLI-style parameter text into one occurrence's converted value.

    The text mirrors the command line: space-separated tokens, optionally
    quoted. The result is exactly the object the CLI's argparse would have
    produced for one occurrence of the option (a single value for
    ``nargs=None``, a list for fixed/``+``/``*`` nargs, and ``None`` for an
    empty optional ``nargs='?'``).

    Parameters
    ----------
    text : str
        The parameter text entered by the user.
    spec : dict
        An operation spec from :func:`operation_specs`.

    Returns
    -------
    object
        The converted parameter value.

    Raises
    ------
    ValueError
        If the text does not match the option's arity or a choice/type.
    """
    if spec["choices"] is not None:
        value = text.strip()
        if value not in spec["choices"]:
            raise ValueError(f"{value!r} is not one of {spec['choices']}")
        return value

    tokens = shlex.split(text)
    nargs = spec["nargs"]
    if nargs == 0:
        # argparse store_true / store_false actions take no parameter.
        if tokens:
            raise ValueError(f"expected no value, got {len(tokens)}")
        return True
    if nargs is None or nargs == 1:
        if len(tokens) != 1:
            raise ValueError(f"expected a single value, got {len(tokens)}")
        return _convert_token(tokens[0], spec["type"])
    if nargs == "?":
        if len(tokens) > 1:
            raise ValueError(f"expected at most one value, got {len(tokens)}")
        return _convert_token(tokens[0], spec["type"]) if tokens else None
    if isinstance(nargs, int):
        if len(tokens) != nargs:
            raise ValueError(f"expected {nargs} values, got {len(tokens)}")
        return [_convert_token(t, spec["type"]) for t in tokens]
    if nargs == "+":
        if not tokens:
            raise ValueError("expected at least one value")
        return [_convert_token(t, spec["type"]) for t in tokens]
    if nargs == "*":
        return [_convert_token(t, spec["type"]) for t in tokens]
    raise ValueError(f"unsupported nargs {nargs!r}")


def stack_to_namespace(
    stack: list[tuple[str, object]], specs: dict[str, dict], **extra
) -> argparse.Namespace:
    """Build an ``argparse.Namespace`` from an ordered operation stack.

    Every registered plugin option is seeded with its argparse default (plus
    the CLI infrastructure defaults handlers rely on), then each stacked
    operation overrides its option: append-style options accumulate their
    per-occurrence values in order, non-append options may appear at most
    once. The result is exactly what ``apply_options`` expects.

    Parameters
    ----------
    stack : list of (str, object)
        Ordered ``(option_name, converted_param)`` entries.
    specs : dict of str to dict
        Operation specs from :func:`operation_specs`.
    **extra
        Extra attributes to set on the namespace (e.g. ``output_starFile``).

    Returns
    -------
    argparse.Namespace
        Namespace ready for :func:`apply_options`.

    Raises
    ------
    ValueError
        If a non-append option appears more than once in the stack.
    """
    from helicon.commands import images2star

    parser = argparse.ArgumentParser(add_help=False)
    images2star.add_args(parser)
    args = argparse.Namespace()
    for action in parser._actions:
        if action.dest == "help":
            continue
        value = action.default
        setattr(args, action.dest, value)
    args.verbose = 0
    args.cpu = 1
    args.input_imageFiles = [extra.pop("input_imageFiles", "")]
    args.output_starFile = extra.pop("output_starFile", "")

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
