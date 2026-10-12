"""How many CPUs helicon may use: numba's thread limit without importing numba."""

import os
import sys
from types import SimpleNamespace

import pytest

from helicon.lib import system


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
