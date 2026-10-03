"""Long work in the web app tabs does not hold the event loop every session shares.

Shiny runs all sessions' reactive code on one event loop. Work that takes
seconds or minutes inside an effect -- or a loop that blocks while waiting for
threads -- stalls every visitor of a hosted copy for that long.
"""

import inspect
import re

import pytest

from helicon.webApps.tabs import (
    abinitio3d_tab,
    denovo3d_tab,
    helical_projection_tab,
)


def _src(module):
    return inspect.getsource(module)


class TestNoBlockingWaits:
    @pytest.mark.parametrize("module", [denovo3d_tab, abinitio3d_tab])
    def test_thread_results_are_awaited_not_waited_for(self, module):
        src = _src(module)
        # concurrent.futures.as_completed blocks; asyncio's yields
        assert not re.search(r"(?<!asyncio\.)\bas_completed\(", src)

    def test_a_stop_does_not_wait_for_running_tasks(self):
        src = _src(denovo3d_tab)
        # the search pools; the ranking's one-thread pool is awaited inside it
        assert "with ThreadPoolExecutor(max_workers=cpu)" not in src
        assert src.count("executor.shutdown(wait=False, cancel_futures=True)") >= 2


class TestLongWorkInBackgroundTasks:
    @pytest.mark.parametrize(
        "module, button, task",
        [
            (denovo3d_tab, "dn_auto_stitch", "auto_stitch_task"),
            (helical_projection_tab, "compare_projections", "compare_task"),
            (helical_projection_tab, "generate_xyz_projections", "xyz_task"),
        ],
    )
    def test_started_as_a_background_task(self, module, button, task):
        src = _src(module)
        assert re.search(
            rf'{task} = helicon\.shiny\.background_task\(\s*"{button}"', src
        )
        assert f"{task}.invoke(" in src
        # the button shows the task busy
        assert re.search(rf'ui\.input_task_button\(\s*"{button}"', src)

    def test_helical_projection_has_no_work_in_place(self):
        # a ui.Progress block in an effect is the sign of work done in place
        assert "ui.Progress(" not in _src(helical_projection_tab)
