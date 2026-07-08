"""Auto-discovering handler package for images2star options."""

from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path


# Export helpers for backward compatibility
# (utility functions moved to lib/io.py and lib/analysis.py)


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


def dispatch(option_name, data, args, index_d, param):
    """Dispatch to the handler for the given option name."""
    mod = _plugins.get(option_name)
    if mod is None:
        raise ValueError(f"Unknown option: {option_name}")
    return mod.handle(data, args, index_d, param)


def add_plugin_args(parser):
    """Call add_args on all discovered plugins."""
    for mod in _plugins.values():
        if hasattr(mod, "add_args"):
            mod.add_args(parser)
