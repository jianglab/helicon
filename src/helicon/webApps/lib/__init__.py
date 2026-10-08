"""Compute libraries of the web apps.

The submodules are imported when first used, not when this package is: each tab
needs only a few of them, and they pull in SciPy, scikit-image and more (importing
all of them for every tab took about 2 s). ``from helicon.webApps.lib import x``
and ``helicon.webApps.lib.x`` both work as before.
"""

import importlib

__all__ = [
    "denovo3d_pipeline",
    "image_loader",
    "denovo3d_solver",
    "hi3d_core",
    "whereismyclass_compute",
    "helical_projection_utils",
    "helical_pitch_compute",
    "helical_projection_compute",
    "hill_compute",
]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
