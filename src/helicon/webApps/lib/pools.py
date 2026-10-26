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
    "helicon.webApps.lib.denovo3d_pipeline",
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


def _limit_threads(threads):
    """Initialiser of a worker: at most ``threads`` threads in its BLAS and numba."""
    import os

    n = str(max(1, int(threads)))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = n
    # NUMBA_NUM_THREADS is left alone: numba reads it when it is imported (the fork
    # server preloads it), and a different value in the environment when the first
    # function is compiled makes every task fail. set_num_threads below limits it.
    try:
        from threadpoolctl import threadpool_limits

        global _LIMITS
        _LIMITS = threadpool_limits(limits=int(n))  # held for the life of the worker
    except Exception:  # pragma: no cover - the limits are an optimisation
        pass
    try:
        from helicon.lib.numba_compat import set_num_threads

        set_num_threads(int(n))
    except Exception:  # pragma: no cover
        pass


_LIMITS = None


class TaskPool:
    """Worker processes for tasks that a Stop has to end at once.

    A task running in a thread cannot be interrupted: after a Stop it ran on, took
    the CPUs, and was waited for. A task in a process can be ended by ending the
    process. The pool is for one run: after :meth:`stop` it is not used again.

    Parameters
    ----------
    max_workers : int
        Number of worker processes.
    threads : int, optional
        Threads each worker may use (in BLAS and numba), so that the workers
        together do not use more CPUs than there are. Defaults to 1.
    gb_per_worker : float, optional
        Memory a worker needs (its copy of the data); the pool has no more workers than the free
        memory holds, so that two users starting runs at once do not use it all. Defaults to 1.
    """

    def __init__(self, max_workers, threads=1, gb_per_worker=1.0):
        from concurrent.futures import ProcessPoolExecutor

        import helicon

        max_workers = min(
            int(max_workers), int(helicon.available_cpu(mem_gb_per_cpu=gb_per_worker))
        )
        self._executor = ProcessPoolExecutor(
            max_workers=max(1, max_workers),
            mp_context=pool_context(),
            initializer=_limit_threads,
            initargs=(threads,),
        )

    def submit(self, fn, *args):
        """A ``concurrent.futures.Future`` of ``fn(*args)`` run in a worker."""
        return self._executor.submit(fn, *args)

    def stop(self):
        """End the tasks at once: those waiting are dropped, the workers killed."""
        executor = self._executor
        workers = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for worker in workers:
            try:
                worker.kill()
            except Exception:  # already gone
                pass

    def shutdown(self):
        """Let the pool go without waiting for its tasks (a run that is done)."""
        self._executor.shutdown(wait=False, cancel_futures=True)
