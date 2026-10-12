"""The way the web app starts worker processes.

Forking a process that has threads running, or that has used the BLAS or a GUI
library, can hang or crash the child (on macOS the child dies in the first
``eigh``); spawning one re-imports helicon with numba in every worker, seconds
each. A fork server avoids both: the workers are forked from a small process of
its own that imported the modules the workers need once.
"""

from __future__ import annotations

# imported once by the fork server, so that a worker starts in milliseconds
PRELOAD = (
    "helicon.webApps.lib.layerline_index",
    "helicon.webApps.lib.helical_pitch_phase",
    "helicon.lib.layerlines",
    "scipy.special",
)


def pool_context():
    """The context to start worker processes with: a fork server with ``PRELOAD`` loaded.

    The fork server is started by the first pool that uses it and keeps the
    modules it had then; where there is no fork server (Windows) workers are
    spawned.

    Returns
    -------
    multiprocessing.context.BaseContext
    """
    import multiprocessing as mp

    try:
        context = mp.get_context("forkserver")
    except ValueError:
        return mp.get_context("spawn")
    context.set_forkserver_preload(list(PRELOAD))
    return context
