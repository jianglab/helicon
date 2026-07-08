"""Auto-discovering handler package for cryosparc options."""

from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path


def _discover_plugins():
    """Auto-discover plugin modules and build the registry by option_name."""
    plugins = {}
    pkg_dir = Path(__file__).parent
    #for importer, modname, ispkg in pkgutil.iter_modules([pkg_dir]):
    for importer, modname, ispkg in pkgutil.iter_modules([str(pkg_dir)]):
        if modname.startswith("_") or ispkg:
            continue
        try:
            mod = importlib.import_module(f".{modname}", __package__)
        except ImportError:
            continue
        if hasattr(mod, "option_name") and hasattr(mod, "handle"):
            plugins[mod.option_name] = mod
    return plugins


_plugins = _discover_plugins()


def dispatch(
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
):
    """Dispatch to the handler for the given option name.

    Parameters
    ----------
    option_name : str
        The option name to dispatch.
    data : Dataset
        The cryosparc Dataset.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The resolved parameter value for this option.
    output_title : str
        Title for output filename construction.
    output_slots : set
        Output slot names.
    exp_group_id_name : str
        Name of the exposure group ID column.
    micrograph_name : str
        Name of the micrograph name column.
    original_exp_group_ids : np.ndarray
        Original exposure group IDs.

    Returns
    -------
    tuple
        (data, output_title, output_slots, index_d) after processing.
    """
    mod = _plugins.get(option_name)
    if mod is None:
        return data, output_title, output_slots, index_d
    return mod.handle(
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


def add_plugin_args(parser):
    """Call add_args on all discovered plugins."""
    for mod in _plugins.values():
        if hasattr(mod, "add_args"):
            mod.add_args(parser)


__all__ = ["dispatch", "add_plugin_args"]
