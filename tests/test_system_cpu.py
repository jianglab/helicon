"""How many CPUs helicon may use (numba's thread limit without importing numba), and
how much memory."""

import os
import sys
from types import SimpleNamespace

import pytest

from helicon.lib import system


@pytest.fixture
def no_psutil(monkeypatch):
    """As under Pyodide, which has no psutil wheel: ``_psutil_or_none()`` is None."""
    monkeypatch.setattr(system, "_psutil_or_none", lambda: None)


@pytest.fixture
def numba_not_imported(monkeypatch):
    # numba may have been imported by another test; the limit must not need it
    monkeypatch.delitem(sys.modules, "numba", raising=False)


class TestNumbaThreadLimit:
    def test_an_imported_numba_is_asked(self, monkeypatch):
        fake = SimpleNamespace(config=SimpleNamespace(NUMBA_NUM_THREADS=3))
        monkeypatch.setitem(sys.modules, "numba", fake)
        assert system._numba_thread_limit() == 3

    def test_the_environment_variable_decides_without_importing_numba(
        self, monkeypatch, numba_not_imported
    ):
        monkeypatch.setenv("NUMBA_NUM_THREADS", "5")
        assert system._numba_thread_limit() == 5
        assert "numba" not in sys.modules

    def test_otherwise_it_is_the_cpus_the_process_may_run_on(
        self, monkeypatch, numba_not_imported
    ):
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False
        )
        assert system._numba_thread_limit() == 3
        assert "numba" not in sys.modules

    def test_without_an_affinity_call_it_is_every_cpu(
        self, monkeypatch, numba_not_imported
    ):
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        assert system._numba_thread_limit() == 8

    def test_without_numba_installed_nothing_is_limited(
        self, monkeypatch, numba_not_imported
    ):
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda n, *a: None if n == "numba" else real(n, *a),
        )
        assert system._numba_thread_limit() is None


class TestAvailableCpu:
    def test_numbas_limit_caps_it(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16")
        monkeypatch.setattr(system, "_numba_thread_limit", lambda: 4)
        assert system.available_cpu() == 4

    def test_no_limit_leaves_the_cpus_alone(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16")
        monkeypatch.setattr(system, "_numba_thread_limit", lambda: None)
        assert system.available_cpu() == 16

    def test_a_failing_lookup_does_not_stop_it(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16")

        def broken():
            raise RuntimeError("numba is in a strange state")

        monkeypatch.setattr(system, "_numba_thread_limit", broken)
        assert system.available_cpu() == 16

    def test_it_is_at_least_one(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "1")
        monkeypatch.setattr(system, "_numba_thread_limit", lambda: 0)
        assert system.available_cpu() == 1


class TestAvailableMemory:
    @pytest.fixture(autouse=True)
    def _requires_psutil(self):
        pytest.importorskip("psutil")

    def test_the_slurm_job_limit_caps_it(self, monkeypatch):
        import psutil

        used = psutil.Process().memory_info().rss
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(used // 2**20 + 100))
        assert system.available_memory() <= 101 * 2**20

    def test_the_limit_per_cpu_counts_every_cpu(self, monkeypatch):
        import psutil

        used = psutil.Process().memory_info().rss
        monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
        monkeypatch.setenv("SLURM_MEM_PER_CPU", str(used // 2**20 // 4 + 50))
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
        assert system.available_memory() <= 201 * 2**20

    def test_without_a_job_limit_it_is_at_most_the_free_memory(self, monkeypatch):
        import psutil

        for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
            monkeypatch.delenv(name, raising=False)
        assert 0 < system.available_memory() <= psutil.virtual_memory().total


class TestPsutilOrNone:
    def test_the_real_psutil_comes_back(self):
        psutil = pytest.importorskip("psutil")

        assert system._psutil_or_none() is psutil

    def test_none_when_it_cannot_be_imported(self, monkeypatch):
        # a None entry in sys.modules makes "import psutil" raise ImportError,
        # the same failure as it never having been installed at all
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert system._psutil_or_none() is None


class TestWithoutPsutil:
    """As under Pyodide, which has no psutil wheel."""

    def test_available_cpu_falls_back_to_os_cpu_count(self, monkeypatch, no_psutil):
        monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
        monkeypatch.setattr(system, "_numba_thread_limit", lambda: None)
        monkeypatch.setattr(os, "cpu_count", lambda: 6)
        assert system.available_cpu() == 6

    def test_available_cpu_with_no_cpu_count_is_still_at_least_one(
        self, monkeypatch, no_psutil
    ):
        monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
        monkeypatch.setattr(system, "_numba_thread_limit", lambda: None)
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        assert system.available_cpu() == 1

    def test_a_per_cpu_memory_budget_is_left_unenforced(self, monkeypatch, no_psutil):
        monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
        monkeypatch.setattr(system, "_numba_thread_limit", lambda: None)
        monkeypatch.setattr(os, "cpu_count", lambda: 6)
        # would cap to 0 (and then 1) if a memory figure were read; without
        # psutil there is none, so the cpu count found above is kept
        assert system.available_cpu(mem_gb_per_cpu=1_000_000) == 6

    def test_available_memory_is_a_conservative_constant(self, monkeypatch, no_psutil):
        assert system.available_memory() == 1024**3
