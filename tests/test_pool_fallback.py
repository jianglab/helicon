"""A killed worker process must not fail the pitch estimate."""

import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from joint_pixel import synthetic as syn  # noqa: E402

import helicon  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402


@pytest.fixture(scope="module")
def pairs():
    params = syn.class2d_table(400.0, 8, n_fil=40, length=600.0, seed=0)
    return ph.prepare_pairs(params, use_refined_path=False, max_sep=1500.0)


class _Dies:
    """A process pool whose workers are always killed."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, *a, **k):
        raise BrokenProcessPool(
            "A process in the process pool was terminated abruptly while the "
            "future was running or pending."
        )


class TestKilledWorker:
    def test_the_period_fits_are_repeated_one_at_a_time(self, pairs, monkeypatch):
        tasks = [dict(periods=np.arange(380.0, 420.0, 2.0)) for _ in range(3)]
        monkeypatch.setattr(helicon, "available_cpu", lambda *a, **k: 4)
        (
            ph._estimate_periods.__wrapped__(pairs, tasks)
            if hasattr(ph._estimate_periods, "__wrapped__")
            else None
        )
        import concurrent.futures as cf

        monkeypatch.setattr(cf, "ProcessPoolExecutor", _Dies)
        out = ph._estimate_periods(pairs, tasks)
        assert len(out) == 3
        assert all(abs(p / 400.0 - 1.0) < 0.05 for p in out)

    def test_the_pool_is_left_clean_for_the_next_estimate(self, pairs, monkeypatch):
        monkeypatch.setattr(helicon, "available_cpu", lambda *a, **k: 4)
        import concurrent.futures as cf

        monkeypatch.setattr(cf, "ProcessPoolExecutor", _Dies)
        ph._estimate_periods(pairs, [dict(periods=np.arange(380.0, 420.0, 2.0))] * 2)
        assert ph._POOL_PAIRS == []

    def test_the_registrations_are_repeated_one_at_a_time(self, monkeypatch):
        import concurrent.futures as cf

        a = syn.class_averages(400.0, 12, apix=4.0)[0]
        prepared, _ = ph._register_prepared(a, list(range(12)), 128)
        todo = [(i, j) for i in range(12) for j in range(i + 1, 12)]
        assert len(todo) >= 40  # enough for the pool to be used
        monkeypatch.setattr(helicon, "available_cpu", lambda *a, **k: 4)
        reference = ph._register_all(prepared, todo, [(False, False)], None)
        monkeypatch.setattr(cf, "ProcessPoolExecutor", _Dies)
        again = ph._register_all(prepared, todo, [(False, False)], None)
        assert [r["corr"] for r in again] == pytest.approx(
            [r["corr"] for r in reference]
        )
        assert ph._POOL_IMAGES == {}


class TestWorkersFitTheMemory:
    @staticmethod
    def _workers(monkeypatch, n_pairs, n_classes, machine_gb=16.0, cpus=8):
        class P:
            D = np.zeros(n_pairs)
            max_sep = 5700.0

        P.n_classes = n_classes
        asked = []

        def available(gb=None):
            if gb is None:
                return cpus
            asked.append(gb)
            return max(1, int(machine_gb / gb))

        monkeypatch.setattr(helicon, "available_cpu", available)
        return ph._pool_workers(P, 20), asked[-1]

    def test_many_classes_mean_fewer_workers(self, monkeypatch):
        few, gb_few = self._workers(monkeypatch, 1_000_000, 20)
        many, gb_many = self._workers(monkeypatch, 1_000_000, 200)
        # 200 classes: the histogram alone is 0.7 GB, with its copies several
        assert gb_many > gb_few and gb_many > 3.0
        assert many < few

    def test_the_estimate_covers_what_was_measured(self, monkeypatch):
        # 100 classes, 7.8 million pairs: the largest worker was 1.5 GB resident
        _, gb = self._workers(monkeypatch, 7_776_694, 100)
        assert gb >= 1.5
        # 40 classes, 2.3 million pairs: 0.66 GB
        _, gb = self._workers(monkeypatch, 2_291_236, 40)
        assert gb >= 0.66

    def test_a_big_machine_keeps_its_workers(self, monkeypatch):
        n, _ = self._workers(monkeypatch, 1_000_000, 50, machine_gb=256.0, cpus=8)
        assert n == 8


class TestWorkersReadTheSharedPairs:
    def test_the_pairs_come_back_from_the_files_unchanged(self, pairs, tmp_path):
        import dataclasses

        rest = ph._share_pairs(pairs, tmp_path)
        try:
            ph._init_pairs_worker(tmp_path, rest)
            got = ph._POOL_PAIRS[0]
            for f in dataclasses.fields(pairs):
                a, b = getattr(pairs, f.name), getattr(got, f.name)
                if isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b)
                else:
                    assert a == b, f.name
            assert not got.D.flags.writeable  # mapped read-only, shared
        finally:
            ph._POOL_PAIRS.clear()

    def test_worker_processes_give_the_result_of_one_process(
        self, pairs, monkeypatch, caplog
    ):
        tasks = [
            dict(periods=np.arange(380.0, 420.0, 2.0), length_scale=s)
            for s in (400.0, 800.0, 1600.0)
        ]
        monkeypatch.setattr(ph, "_pool_workers", lambda pairs, n: 1)
        serial = ph._estimate_periods(pairs, tasks)
        monkeypatch.setattr(ph, "_pool_workers", lambda pairs, n: 2)
        with caplog.at_level("WARNING"):
            parallel = ph._estimate_periods(pairs, tasks)
        assert "terminated" not in caplog.text  # the pool ran, it did not fall back
        assert parallel == pytest.approx(serial)
        assert ph._POOL_PAIRS == []
