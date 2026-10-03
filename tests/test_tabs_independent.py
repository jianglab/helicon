"""The web app's tabs share nothing: a value set in one never changes another,
and nothing one session computes reaches another's."""

import inspect
import re
import threading
from pathlib import Path

import numpy as np
import pytest

from helicon.webApps import app
from helicon.webApps.lib import helical_pitch_phase as ph

TABS = Path(app.__file__).parent / "tabs"


class TestNoSharedState:
    def test_no_shared_project_object(self):
        assert not (Path(app.__file__).parent / "lib" / "shared_state.py").exists()
        for f in TABS.glob("*_tab.py"):
            src = f.read_text()
            assert "shared_state" not in src and "ProjectState" not in src, f.name

    @pytest.mark.parametrize("tab", sorted(p.stem for p in TABS.glob("*_tab.py")))
    def test_each_tab_server_takes_only_its_own_session(self, tab):
        src = (TABS / f"{tab}.py").read_text()
        for args in re.findall(r"def \w+_tab_server\(([^)]*)\)", src):
            assert [a.strip() for a in args.split(",")] in (
                ["input", "output", "session"],
                ["input", "session"],
            ), (tab, args)

    def test_the_app_hands_the_tabs_nothing(self):
        src = inspect.getsource(app)
        for call in re.findall(r"\w+_tab_server\(([^)]*)\)", src):
            if call.strip() in ("input, output, session", "input, session"):
                continue
            assert re.fullmatch(r'\s*"\w+"\s*', call), call


class TestWorkerScratchIsNotShared:
    """Each session's pitch estimate runs in a thread of its own; the data its
    worker processes read must be its own."""

    def _pairs(self, period, seed):
        from tests.test_helical_pitch_phase import make_params

        return ph.prepare_pairs(make_params(n_fil=60, period=period, seed=seed))

    def test_two_estimates_at_once_each_get_their_own_pairs(self, monkeypatch):
        monkeypatch.setattr(ph, "_pool_workers", lambda pairs, n: 1)
        jobs = {300.0: self._pairs(300.0, 1), 420.0: self._pairs(420.0, 2)}
        tasks = [dict(length_scale=800.0)] * 3
        alone = {p: ph._estimate_periods(pairs, tasks) for p, pairs in jobs.items()}
        together, start = {}, threading.Barrier(2)

        def run(p):
            start.wait()
            together[p] = ph._estimate_periods(jobs[p], tasks)

        threads = [threading.Thread(target=run, args=(p,)) for p in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert together == alone
        for p, periods in together.items():
            assert np.allclose(periods, p, rtol=0.05)
