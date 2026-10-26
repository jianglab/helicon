"""Worker processes that a Stop ends at once."""

import time
from pathlib import Path

from helicon.webApps.lib import pools


class TestTaskPool:
    def test_it_runs_tasks_in_worker_processes(self):
        pool = pools.TaskPool(2)
        try:
            assert pool.submit(pow, 2, 10).result(timeout=120) == 1024
            futures = [pool.submit(pow, 3, k) for k in range(6)]
            assert [f.result(timeout=120) for f in futures] == [3**k for k in range(6)]
        finally:
            pool.shutdown()

    def test_a_stop_ends_tasks_that_are_running_and_drops_those_waiting(self):
        pool = pools.TaskPool(2)
        # the pool never has more workers than the machine has CPUs (one on a small
        # CI runner), so ask it how many it has instead of assuming two
        n = pool._executor._max_workers
        futures = [pool.submit(time.sleep, 60) for _ in range(n + 2)]
        # wait until the workers are running
        deadline = time.time() + 60
        while time.time() < deadline:
            running = [p for p in (pool._executor._processes or {}).values()]
            if len(running) == n and sum(f.running() for f in futures) >= n:
                break
            time.sleep(0.05)
        workers = list(pool._executor._processes.values())
        assert len(workers) == n and all(w.is_alive() for w in workers)
        t0 = time.time()
        pool.stop()
        for w in workers:
            w.join(timeout=5)
        took = time.time() - t0
        assert not any(w.is_alive() for w in workers)
        assert took < 3.0  # not the minute the tasks would have taken
        # the pool settles the tasks that were waiting a moment later, in a thread of its
        # own (so how soon depends on how busy the machine is): wait for it
        deadline = time.time() + 30
        while time.time() < deadline and not all(f.done() for f in futures[n:]):
            time.sleep(0.05)
        assert all(f.cancelled() or f.done() for f in futures[n:])

    def test_the_workers_are_limited_to_their_share_of_the_threads(self):
        import os

        pool = pools.TaskPool(1, threads=3)
        try:
            assert (
                pool.submit(os.getenv, "OPENBLAS_NUM_THREADS").result(timeout=120)
                == "3"
            )
        finally:
            pool.shutdown()

    def test_a_worker_does_not_change_the_numba_thread_setting(self):
        # numba fails to compile once NUMBA_NUM_THREADS differs from what it read at
        # import, which made every task of a search fail
        import os

        pool = pools.TaskPool(1, threads=3)
        try:
            assert pool.submit(os.getenv, "NUMBA_NUM_THREADS").result(
                timeout=120
            ) == os.getenv("NUMBA_NUM_THREADS")
        finally:
            pool.shutdown()


class TestDenovo3dStop:
    @staticmethod
    def _source():
        from helicon.webApps.tabs import denovo3d_tab

        return Path(denovo3d_tab.__file__).read_text(encoding="utf-8")

    def test_the_search_runs_in_processes_and_a_stop_ends_them(self):
        src = self._source()
        assert src.count("pools.TaskPool(") == 2  # the search and the redraw
        assert src.count("pool.stop()") == 2
        # the wait is in steps that see the flag, not until a task has finished
        assert src.count("timeout=0.2, return_when=asyncio.FIRST_COMPLETED") >= 1
        assert "asyncio.as_completed" not in src

    def test_nothing_waits_for_a_thread_after_a_stop(self):
        src = self._source()
        # the matching of projections is not in a "with" block that would wait
        assert "with ThreadPoolExecutor" not in src
        assert "pool.shutdown(wait=False, cancel_futures=True)" in src

    def test_a_job_uses_the_threads_it_was_given(self):
        from helicon.webApps.lib import denovo3d_pipeline

        src = Path(denovo3d_pipeline.__file__).read_text(encoding="utf-8")
        assert "cpu = min(helicon.available_cpu(), max(1, int(n_cpu)))" in src


class TestMemoryCap:
    def test_workers_fit_in_the_free_memory(self):
        pool = pools.TaskPool(10_000, gb_per_worker=1.0)
        try:
            assert 1 <= pool._executor._max_workers < 10_000
        finally:
            pool.shutdown()
