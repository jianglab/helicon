"""Where real threads (and so real processes too) cannot be started -- as
under Pyodide, where ``threading.Thread.start()`` raises ``RuntimeError:
can't start new thread`` -- ``helicon.pool_executor`` hands back something
that behaves like a pool but runs every task immediately, in order, in the
calling thread.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pytest

from helicon.lib import system


@pytest.fixture
def no_real_threads(monkeypatch):
    monkeypatch.setattr(system, "_HAS_REAL_THREADS", False)


class TestHasRealThreads:
    def test_true_here(self):
        # this process is the real test runner, not Pyodide
        assert system.has_real_threads() is True

    def test_false_when_starting_one_raises(self, monkeypatch):
        import threading

        monkeypatch.setattr(system, "_HAS_REAL_THREADS", None)

        class _Unstartable:
            def __init__(self, target):
                pass

            def start(self):
                raise RuntimeError("can't start new thread")

        # has_real_threads() imports threading itself (same module object);
        # patching the stdlib module reaches it
        monkeypatch.setattr(threading, "Thread", _Unstartable)
        assert system.has_real_threads() is False

    def test_checked_once_and_cached(self, monkeypatch):
        monkeypatch.setattr(system, "_HAS_REAL_THREADS", "cached-sentinel")
        assert system.has_real_threads() == "cached-sentinel"


class TestPoolExecutorWithRealThreads:
    def test_it_is_the_real_executor_class(self):
        ex = system.pool_executor(ThreadPoolExecutor, max_workers=2)
        try:
            assert isinstance(ex, ThreadPoolExecutor)
        finally:
            ex.shutdown()


class TestSerialExecutor:
    def test_it_is_used_instead(self, no_real_threads):
        ex = system.pool_executor(ThreadPoolExecutor, max_workers=2)
        assert isinstance(ex, system.SerialExecutor)

    def test_it_also_stands_in_for_a_process_pool(self, no_real_threads):
        ex = system.pool_executor(ProcessPoolExecutor, max_workers=2)
        assert isinstance(ex, system.SerialExecutor)

    def test_results_and_as_completed_work_as_with_a_real_pool(self, no_real_threads):
        with system.pool_executor(ThreadPoolExecutor, max_workers=3) as ex:
            futures = {ex.submit(lambda n: n * n, n): n for n in range(5)}
            results = sorted(f.result() for f in as_completed(futures))
        assert results == [0, 1, 4, 9, 16]

    def test_arguments_and_keywords_both_reach_the_task(self, no_real_threads):
        with system.pool_executor(ThreadPoolExecutor) as ex:
            future = ex.submit(lambda a, b, c=0: a + b + c, 1, 2, c=3)
        assert future.result() == 6

    def test_an_exception_in_the_task_propagates_from_result(self, no_real_threads):
        with system.pool_executor(ThreadPoolExecutor) as ex:
            future = ex.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            future.result()

    def test_map_runs_every_item(self, no_real_threads):
        with system.pool_executor(ThreadPoolExecutor) as ex:
            assert list(ex.map(lambda n: n + 1, [1, 2, 3])) == [2, 3, 4]

    def test_shutdown_and_the_context_manager_are_harmless_no_ops(
        self, no_real_threads
    ):
        ex = system.pool_executor(ThreadPoolExecutor)
        with ex as entered:
            assert entered is ex
        ex.shutdown(wait=True, cancel_futures=True)  # must not raise
