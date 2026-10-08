"""Qt GUI sub-package of ``helicon.lib``.

Houses all Qt widgets and display-command implementation. The
``helicon.commands.display`` command is a thin coordinator: the actual
implementation lives in these modules, which the coordinator re-exports so
that ``from helicon.commands.display import X`` keeps working for every
existing caller (file browser, gallery backends/widgets, proc3d/images2star
tools, and the tests).
"""

import importlib

# The submodules are imported when first used, not when this package is: every
# ``helicon.lib.gui.x`` import used to import all of them, including the dialogs
# of images2star and proc3d (with all their plugins, matplotlib among them) that
# ``helicon display`` needs only when one is opened. ``from helicon.lib.gui import
# x`` and ``helicon.lib.gui.x`` work as before.
__all__ = [
    "file_browser",
    "file_openers",
    "fsc",
    "gallery_backends",
    "gallery_widget",
    "images2star_widget",
    "macos",
    "proc3d_widget",
    "star_parsers",
    "theme",
    "trackers",
    "viewer",
    "webapps",
]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
