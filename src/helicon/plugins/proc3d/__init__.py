"""Auto-discovering handler package for proc3d options."""

from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path


def _discover_plugins():
    """Auto-discover plugin modules and build the registry by option_name."""
    plugins = {}
    pkg_dir = Path(__file__).parent
    for importer, modname, ispkg in pkgutil.iter_modules([pkg_dir]):
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


def dispatch(option_name, data, args, index_d, param, apix, nx, ny, nz):
    """Dispatch to the handler for the given option name.

    Parameters
    ----------
    option_name : str
        The option name to dispatch.
    data : np.ndarray
        3D volume data.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.
    apix : float
        Current pixel size.
    nx, ny, nz : int
        Current volume dimensions.

    Returns
    -------
    tuple
        (data, apix, nx, ny, nz) after handler processing.
    """
    mod = _plugins.get(option_name)
    if mod is None:
        raise ValueError(f"Unknown option: {option_name}")
    return mod.handle(data, args, index_d, param, apix, nx, ny, nz)


def add_plugin_args(parser):
    """Call add_args on all discovered plugins.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    for mod in _plugins.values():
        if hasattr(mod, "add_args"):
            mod.add_args(parser)


__all__ = ["dispatch", "add_plugin_args"]
