"""Shared machinery of the ordered plugin pipelines.

``proc3d``, ``images2star`` and ``cryosparc`` run their plugin options as an
ordered pipeline: the same option may be given more than once and every
occurrence carries its own value (see AGENTS.md, "Repeated options are a
pipeline"). Introspecting the registry and building the ``argparse.Namespace``
from an operation stack are the same for all of them, so they live here once
instead of in each engine, where the copies had drifted apart.
"""

from __future__ import annotations

import argparse

__all__ = ["metavar_text", "operation_specs", "stack_to_namespace"]


def metavar_text(metavar) -> str:
    """Normalize an argparse metavar (str or tuple) to a single string."""
    if isinstance(metavar, (tuple, list)):
        return " ".join(str(m) for m in metavar)
    return str(metavar) if metavar is not None else ""


def operation_specs(plugins: object) -> dict[str, dict]:
    """Introspect a plugin package's registry into generic option specs.

    Each spec describes the argparse action behind a plugin's ``--option`` so
    that a GUI (or any caller) can render a parameter form and build an
    ``argparse.Namespace`` without re-implementing option semantics.

    Parameters
    ----------
    plugins : module
        A ``helicon.plugins.<category>`` package; its ``_plugins`` maps an
        option name to the module that handles it.

    Returns
    -------
    dict of str to dict
        Maps option names (leading ``--`` stripped) to specs with keys:
        ``dest``, ``option_string``, ``metavar``, ``type``, ``nargs``,
        ``choices``, ``default``, ``help``, and ``append`` (True for argparse
        ``_AppendAction`` options, whose occurrences accumulate).
    """
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
            "metavar": metavar_text(action.metavar),
            "type": action.type if action.type is not None else str,
            "nargs": action.nargs,
            "choices": list(action.choices) if action.choices else None,
            "default": action.default,
            "help": action.help or "",
            "append": type(action) is argparse._AppendAction,
        }
    return specs


def stack_to_namespace(
    stack: list[tuple[str, object]],
    specs: dict[str, dict],
    add_args,
    seed_defaults,
    **extra,
) -> argparse.Namespace:
    """Build an ``argparse.Namespace`` from an ordered operation stack.

    Every registered plugin option is seeded with its argparse default, then
    ``seed_defaults(args, extra)`` adds the CLI infrastructure defaults the
    handlers rely on and pops its own keys from ``extra``. Each stacked
    operation then overrides its option: an append-style option accumulates its
    per-occurrence values in order, a non-append option may appear at most
    once. The result is exactly what the pipeline loop expects.

    Parameters
    ----------
    stack : list of (str, object)
        Ordered ``(option_name, converted_param)`` entries.
    specs : dict of str to dict
        Operation specs from :func:`operation_specs`.
    add_args : callable
        The command's ``add_args(parser)``.
    seed_defaults : callable
        ``seed_defaults(args, extra)``, which sets the extra defaults and pops
        the keys it consumed from ``extra``.
    **extra
        Extra attributes to set on the namespace (e.g. ``output_starFile``).

    Returns
    -------
    argparse.Namespace
        Namespace ready for the pipeline loop.

    Raises
    ------
    ValueError
        If a non-append option appears more than once in the stack.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_args(parser)
    args = argparse.Namespace()
    for action in parser._actions:
        if action.dest == "help":
            continue
        setattr(args, action.dest, action.default)
    seed_defaults(args, extra)

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
