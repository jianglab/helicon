"""A map that cannot be loaded costs that map, not the session.

Searching dozens of EMDB entries turns up withdrawn entries, files too large
for the connection, and dropped downloads; on a bad day a map that worked
yesterday fails today. None of that is exceptional enough to end a user's
session, so the rule here is: the map is skipped, the user is told, and
everything else carries on.
"""

import inspect

import numpy as np
import pytest

from helicon.webApps.lib import helical_projection_compute as compute


def _unloadable():
    """A map whose data cannot be obtained, without touching the network."""
    return compute.MapInfo(
        filename="/nonexistent/no-such-map.mrc",
        label="broken",
        twist=25.0,
        rise=6.0,
        csym=1,
    )


class TestTheComputeLayerReportsRatherThanRaises:
    def _query(self):
        rng = np.random.default_rng(0)
        image = np.zeros((32, 48), dtype=np.float32)
        image[12:20, 6:42] = 1.0
        return image + rng.normal(0, 0.05, image.shape).astype("f4")

    @pytest.mark.parametrize("method", ["volume", "gaussian"])
    def test_a_map_that_cannot_be_loaded_gives_no_result(self, method):
        map_info, result = compute.symmetrize_project_align_one_map(
            _unloadable(), self._query(), "q", 2.0, True, 1.2, False, 0.0, 0.05, method
        )
        assert result is None
        assert map_info.label == "broken"

    def test_the_xyz_projection_says_which_map_and_why(self):
        with pytest.raises(ValueError) as caught:
            compute.get_one_map_xyz_projects(
                map_info=compute.MapInfo(
                    url="https://example.invalid/emd_00000.map.gz",
                    label="broken",
                    twist=25.0,
                    rise=6.0,
                    csym=1,
                ),
                length_z=1,
                map_projection_xyz_choices=["z"],
            )
        # the message reaches the user, so it has to name the source
        assert "https://example.invalid/emd_00000.map.gz" in str(caught.value)

    def test_it_does_not_repeat_the_id_the_caller_already_shows(self):
        # the tab prints "<label>: <reason>", so a reason that names the map
        # again reads as "EMD-38069: ... from EMDB for EMD-38069"
        with pytest.raises(ValueError) as caught:
            compute.get_one_map_xyz_projects(
                map_info=compute.MapInfo(
                    emd_id="emd-00000",
                    label="EMD-00000",
                    twist=25.0,
                    rise=6.0,
                    csym=1,
                ),
                length_z=1,
                map_projection_xyz_choices=["z"],
            )
        assert "00000" not in str(caught.value)
        assert "EMDB" in str(caught.value)


class TestTheTabWarnsAndSurvives:
    """Wiring: the failure paths in the tab, which no unit test can enter."""

    def _source(self):
        from helicon.webApps.tabs import helical_projection_tab as tab

        return inspect.getsource(tab)

    def test_a_worker_that_raises_does_not_escape_the_effect(self):
        # f.result() re-raises whatever the worker raised, and an exception
        # escaping a reactive effect disconnects the browser
        source = self._source()
        assert "m_info, res = f.result()" in source
        body = source[source.index("for f in as_completed(futures):") :][:900]
        assert "try:" in body and "except Exception" in body
        assert "futures[f]" in body

    def test_failed_xyz_projections_are_collected_and_shown(self):
        source = self._source()
        body = source[source.index("def _get_map_xyz_projections") :][:2600]
        assert "failures.append" in body
        assert "warn(" in body

    def test_the_search_reports_failures_with_their_reason(self):
        source = self._source()
        body = source[source.index("def _compare_projections") :]
        assert "errors[m_info.label] = str(e)" in body
        assert "Some maps could not be searched" in body

    def test_the_warning_is_a_modal_the_user_can_dismiss(self):
        source = self._source()
        body = source[source.index("def warn(") :][:1400]
        assert "ui.modal_show" in body and "easy_close=True" in body
        # a search over dozens of maps can fail on dozens of maps
        assert "and %d more" in body


class TestMapsWithoutATwist:
    """A map with no twist is not a helix, so the search passes it by.

    EMDB entries frequently carry no helical parameters, and selecting a
    filtered table hands over hundreds of maps at once, so these are skipped
    quietly rather than raised as an error the user has to dismiss.
    """

    @pytest.mark.parametrize(
        "twist,expected",
        [
            (25.0, True),
            (-179.4, True),
            (0.0, False),
            (0.0005, False),
            (None, False),
            (float("nan"), False),
            ("", False),
        ],
    )
    def test_what_counts_as_a_twist(self, twist, expected):
        map_info = compute.MapInfo(data=np.zeros((2, 2, 2)), twist=twist)
        assert compute.has_twist(map_info) is expected

    def test_they_are_skipped_without_a_dialog(self):
        from helicon.webApps.tabs import helical_projection_tab as tab

        body = inspect.getsource(tab)
        body = body[body.index("def _compare_projections") :]
        body = body[: body.index("map_side_projections_with_alignments.set(good)")]
        assert "compute.has_twist" in body
        assert "ui.notification_show" in body
        assert "no helical twist" in body
        # the old blocking "Twist value error" dialog is gone
        assert "Twist value error" not in body

    def test_a_selection_with_no_twists_at_all_says_so(self):
        from helicon.webApps.tabs import helical_projection_tab as tab

        body = inspect.getsource(tab)
        body = body[body.index("def _compare_projections") :]
        assert "Nothing to search" in body
        assert "if not active_maps:" in body
