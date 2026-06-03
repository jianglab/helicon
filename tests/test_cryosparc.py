import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from helicon.commands import cryosparc
from helicon.lib.exceptions import HeliconValidationError


class MockCSDataset(dict):
    """Minimal mock of cryosparc.tools.Dataset for handler unit tests.

    Supports ``data[col]`` (returns np.ndarray), ``data[col] = vals``,
    ``len(data)``, ``data.keys()``, and ``data.rows()``.
    """

    def __init__(self, data):
        self._store = {}
        for k, v in data.items():
            self._store[k] = np.array(v)
        super().__init__()

    def __getitem__(self, key):
        if key in self._store:
            return self._store[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self._store[key] = (
            np.array(value) if not isinstance(value, np.ndarray) else value
        )

    def __contains__(self, key):
        return key in self._store

    def __len__(self):
        if self._store:
            return len(next(iter(self._store.values())))
        return 0

    def keys(self):
        return self._store.keys()

    def rows(self):
        """Iterate over rows as dict-like objects."""
        n = len(self)
        for i in range(n):
            yield {k: v[i] for k, v in self._store.items()}


class TestCryosparcArgs(object):
    def test_add_args_parser_has_expected_arguments(self):
        parser = argparse.ArgumentParser()
        cryosparc.add_args(parser)
        actions = {a.dest for a in parser._actions}
        expected = {
            "csFile",
            "projectID",
            "outputWorkspaceID",
            "jobID",
            "groupIndex",
            "assignExposureGroupByBeamShiftLabel",
            "assignExposureGroupByBeamShiftXY",
            "assignExposureGroupByTime",
            "assignExposureGroupPerMicrograph",
            "copyExposureGroupAssignments",
            "copyExposureGroupParameters",
            "splitByMicrograph",
            "changePixelSize",
            "extractParticles",
            "saveLocal",
            "verbose",
            "cpu",
        }
        missing = expected - actions
        assert not missing, f"Missing args: {missing}"

    def _make_args(self, **kwargs):
        defaults = dict(
            csFile=[],
            projectID=None,
            outputWorkspaceID=None,
            jobID=[],
            groupIndex=[],
            assignExposureGroupByBeamShift=None,
            assignExposureGroupByTime=-1,
            assignExposureGroupPerMicrograph=0,
            copyExposureGroupAssignments=0,
            copyExposureGroupParameters="",
            splitByMicrograph=0,
            changePixelSize=0,
            extractParticles="",
            saveLocal=0,
            verbose=3,
            cpu=-1,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("helicon.get_option_list", return_value=[])
    def test_check_args_no_inputs_raises_error(self, mock_getopt):
        args = self._make_args()
        parser = argparse.ArgumentParser()
        cryosparc.add_args(parser)

        with pytest.raises(HeliconValidationError):
            cryosparc.check_args(args, parser)

    @patch("helicon.get_option_list", return_value=[])
    def test_check_args_project_and_job_passes(self, mock_getopt):
        args = self._make_args(projectID="P407", jobID=["J100"])
        parser = argparse.ArgumentParser()
        cryosparc.add_args(parser)

        result = cryosparc.check_args(args, parser)
        assert result is args
        assert result.groupIndex == [0]

    @patch("helicon.get_option_list", return_value=[])
    def test_check_args_cs_file_passes(self, mock_getopt):
        args = self._make_args(csFile=["/path/to/file.cs"])
        parser = argparse.ArgumentParser()
        cryosparc.add_args(parser)

        result = cryosparc.check_args(args, parser)
        assert result is args

    @patch("helicon.get_option_list", return_value=[])
    def test_check_args_both_csfile_and_project_raises_error(self, mock_getopt):
        args = self._make_args(
            projectID="P407", jobID=["J100"], csFile=["/path/to/file.cs"]
        )
        parser = argparse.ArgumentParser()
        cryosparc.add_args(parser)

        with pytest.raises(HeliconValidationError):
            cryosparc.check_args(args, parser)

    @patch("helicon.get_option_list", return_value=[])
    def _make_xy_args(self, **kwargs):
        defaults = dict(
            csFile=[],
            projectID="P407",
            outputWorkspaceID="W1",
            jobID=["J100"],
            groupIndex=[0],
            assignExposureGroupByBeamShiftLabel=None,
            assignExposureGroupByBeamShiftXY="1",
            assignExposureGroupByTime=-1,
            assignExposureGroupPerMicrograph=0,
            copyExposureGroupAssignments=0,
            copyExposureGroupParameters="",
            splitByMicrograph=0,
            changePixelSize=0,
            extractParticles="",
            saveLocal=0,
            verbose=0,
            cpu=-1,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("helicon.get_option_list", return_value=[])
    def test_check_args_group_index_length_mismatch_raises_error(self, mock_getopt):
        args = self._make_args(
            projectID="P407",
            jobID=["J100", "J101"],
            groupIndex=[0],
        )
        parser = argparse.ArgumentParser()
        cryosparc.add_args(parser)

        with pytest.raises(HeliconValidationError):
            cryosparc.check_args(args, parser)
        args = self._make_args(
            projectID="P407",
            jobID=["J100", "J101"],
            groupIndex=[0, 1],
        )
        parser = argparse.ArgumentParser()
        cryosparc.add_args(parser)

        result = cryosparc.check_args(args, parser)
        assert result is args
        assert result.groupIndex == [0, 1]


class TestCryosparcPerMicrographHandler(object):
    """Tests for assignExposureGroupPerMicrograph handler."""

    def test_assigns_unique_group_per_micrograph(self):
        from helicon.plugins.cryosparc.assignexposuregrouppermicrograph import handle

        data = MockCSDataset(
            {
                "location/micrograph_name": ["a.mrc", "a.mrc", "b.mrc", "c.mrc"],
                "ctf/exp_group_id": [1, 1, 1, 1],
            }
        )

        with patch("helicon.sync_group_columns"):
            result, title, slots, idx = handle(
                data,
                argparse.Namespace(verbose=0),
                {},
                param=1,
                output_title="",
                output_slots=set(),
                exp_group_id_name="ctf/exp_group_id",
                micrograph_name="location/micrograph_name",
                original_exp_group_ids=[1],
            )

        group_ids = np.sort(np.unique(result["ctf/exp_group_id"]))
        assert list(group_ids) == [1, 2, 3]
        assert "per-micrograph" in title

    def test_returns_unchanged_when_param_off(self):
        from helicon.plugins.cryosparc.assignexposuregrouppermicrograph import handle

        data = MockCSDataset(
            {
                "location/micrograph_name": ["a.mrc"],
                "ctf/exp_group_id": [5],
            }
        )

        result, title, slots, idx = handle(
            data,
            argparse.Namespace(verbose=0),
            {},
            param=0,
            output_title="original",
            output_slots=set(),
            exp_group_id_name="ctf/exp_group_id",
            micrograph_name="location/micrograph_name",
            original_exp_group_ids=[5],
        )

        assert list(result["ctf/exp_group_id"]) == [5]
        assert title == "original"


class TestCryosparcResetHandler(object):
    """Tests for resetExposureGroups handler."""

    def test_resets_to_single_group(self):
        from helicon.plugins.cryosparc.resetexposuregroups import handle

        data = MockCSDataset(
            {
                "location/micrograph_name": ["a.mrc", "b.mrc", "c.mrc"],
                "ctf/exp_group_id": [1, 2, 3],
            }
        )

        with patch("helicon.sync_group_columns"):
            result, title, slots, idx = handle(
                data,
                argparse.Namespace(verbose=0),
                {},
                param=1,
                output_title="",
                output_slots=set(),
                exp_group_id_name="ctf/exp_group_id",
                micrograph_name="location/micrograph_name",
                original_exp_group_ids=[1, 2, 3],
            )

        assert list(np.unique(result["ctf/exp_group_id"])) == [1]


class TestCryosparcTimeHandler(object):
    """Tests for assignExposureGroupByTime handler."""

    def test_assigns_time_groups(self):
        from helicon.plugins.cryosparc.assignexposuregroupbytime import handle

        micros = ["mg1.mrc", "mg1.mrc", "mg2.mrc", "mg3.mrc"]
        data = MockCSDataset(
            {
                "location/micrograph_name": micros,
                "ctf/exp_group_id": [1, 1, 1, 1],
            }
        )

        with (
            patch("helicon.assign_time_groups") as mock_atg,
            patch("helicon.sync_group_columns"),
            patch("helicon.propagate_ctf_median"),
        ):
            mock_atg.return_value = (np.array([1, 1, 2, 3]), {}, {})
            result, title, slots, idx = handle(
                data,
                argparse.Namespace(verbose=0),
                {},
                param=2,
                output_title="",
                output_slots=set(),
                exp_group_id_name="ctf/exp_group_id",
                micrograph_name="location/micrograph_name",
                original_exp_group_ids=[1],
            )

        assert list(result["ctf/exp_group_id"]) == [1, 1, 2, 3]
        mock_atg.assert_called_once()

    def test_negative_param_merges_then_splits(self):
        from helicon.plugins.cryosparc.assignexposuregroupbytime import handle

        data = MockCSDataset(
            {
                "location/micrograph_name": ["mg1.mrc", "mg2.mrc", "mg3.mrc"],
                "ctf/exp_group_id": [1, 2, 3],
            }
        )

        with (
            patch("helicon.assign_time_groups") as mock_atg,
            patch("helicon.sync_group_columns"),
            patch("helicon.propagate_ctf_median"),
        ):
            mock_atg.return_value = (np.array([1, 2, 3]), {}, {})
            result, title, slots, idx = handle(
                data,
                argparse.Namespace(verbose=0),
                {},
                param=-1,
                output_title="",
                output_slots=set(),
                exp_group_id_name="ctf/exp_group_id",
                micrograph_name="location/micrograph_name",
                original_exp_group_ids=[1, 2, 3],
            )

        # Negative param should merge groups to 1 before splitting
        _, kwargs = mock_atg.call_args
        # source_group_ids should be [1] after merge
        assert list(kwargs["source_group_ids"]) == [1]
        assert kwargs["time_group_size"] == 1

    def test_passes_mtime_fallback_none(self):
        from helicon.plugins.cryosparc.assignexposuregroupbytime import handle

        data = MockCSDataset(
            {
                "location/micrograph_name": ["mg1.mrc", "mg2.mrc"],
                "ctf/exp_group_id": [1, 1],
            }
        )

        with (
            patch("helicon.assign_time_groups") as mock_atg,
            patch("helicon.sync_group_columns"),
            patch("helicon.propagate_ctf_median"),
        ):
            mock_atg.return_value = (np.array([1, 2]), {}, {})
            handle(
                data,
                argparse.Namespace(verbose=0),
                {},
                param=1,
                output_title="",
                output_slots=set(),
                exp_group_id_name="ctf/exp_group_id",
                micrograph_name="location/micrograph_name",
                original_exp_group_ids=[1],
            )

        _, kwargs = mock_atg.call_args
        assert kwargs["use_mtime_fallback"] is None


class TestCryosparcBeamShiftLabelHandler(object):
    """Tests for assignExposureGroupByBeamShiftLabel handler."""

    def test_beam_shift_label_groups(self):
        from helicon.plugins.cryosparc.assignexposuregroupbybeamshiftlabel import handle

        micros = [
            "FoilHole_1_Data_2_3_20240101_000000_fractions.mrc",
            "FoilHole_4_Data_5_6_20240101_000001_fractions.mrc",
        ]
        data = MockCSDataset(
            {
                "location/micrograph_name": micros,
                "ctf/exp_group_id": [1, 1],
            }
        )

        with (
            patch("helicon.guess_data_collection_software", return_value="EPU"),
            patch("helicon.assign_beamshift_groups") as mock_abs,
            patch("helicon.combine_groups") as mock_cg,
            patch("helicon.sync_group_columns"),
            patch("helicon.propagate_ctf_median"),
        ):
            mock_abs.return_value = {micros[0]: 1, micros[1]: 2}
            mock_cg.return_value = np.array([1, 2])
            result, title, slots, idx = handle(
                data,
                argparse.Namespace(verbose=0),
                {},
                param="1",
                output_title="",
                output_slots=set(),
                exp_group_id_name="ctf/exp_group_id",
                micrograph_name="location/micrograph_name",
                original_exp_group_ids=[1],
            )

        mock_abs.assert_called_once()
        mock_cg.assert_called_once()


class TestCryosparcCopyAssignmentsHandler(object):
    """Tests for copyExposureGroupAssignments handler."""

    def test_copies_assignments_from_star(self):
        from helicon.plugins.cryosparc.copyexposuregroupassignments import handle

        micros = ["/data/mg1.mrc", "/data/mg2.mrc"]
        data = MockCSDataset(
            {
                "location/micrograph_name": micros,
                "ctf/exp_group_id": [1, 1],
            }
        )

        from helicon.plugins.cryosparc.copyexposuregroupassignments import handle

        star_df = pd.DataFrame(
            {
                "rlnMicrographMovieName": ["/data/mg1.mrc", "/data/mg2.mrc"],
                "rlnOpticsGroup": [10, 20],
            }
        )

        with (
            patch("helicon.images2dataframe", return_value=star_df),
            patch("helicon.check_required_columns"),
            patch("helicon.sync_group_columns"),
        ):
            result, title, slots, idx = handle(
                data,
                argparse.Namespace(verbose=0),
                {},
                param="some.star",
                output_title="",
                output_slots=set(),
                exp_group_id_name="ctf/exp_group_id",
                micrograph_name="location/micrograph_name",
                original_exp_group_ids=[1],
            )

        # mg1 -> group 1 (10 - 10 + 1), mg2 -> group 11 (20 - 10 + 1)
        ids = list(result["ctf/exp_group_id"])
        assert ids == [1, 11]


class TestCryosparcBeamShiftXYHandler(object):
    """Tests for assignExposureGroupByBeamShiftXY handler."""

    def test_returns_unchanged_when_param_off(self):
        from helicon.plugins.cryosparc.assignexposuregroupbybeamshiftxy import handle

        data = MockCSDataset(
            {
                "location/micrograph_name": ["a.mrc"],
                "ctf/exp_group_id": [5],
            }
        )

        result, title, slots, idx = handle(
            data,
            argparse.Namespace(verbose=0),
            {},
            param=None,
            output_title="original",
            output_slots=set(),
            exp_group_id_name="ctf/exp_group_id",
            micrograph_name="location/micrograph_name",
            original_exp_group_ids=[5],
        )

        assert list(result["ctf/exp_group_id"]) == [5]
        assert title == "original"

    def test_returns_unchanged_when_param_is_zero(self):
        from helicon.plugins.cryosparc.assignexposuregroupbybeamshiftxy import handle

        data = MockCSDataset(
            {
                "location/micrograph_name": ["a.mrc"],
                "ctf/exp_group_id": [5],
            }
        )

        result, title, slots, idx = handle(
            data,
            argparse.Namespace(verbose=0),
            {},
            param="0",
            output_title="",
            output_slots=set(),
            exp_group_id_name="ctf/exp_group_id",
            micrograph_name="location/micrograph_name",
            original_exp_group_ids=[5],
        )

        assert list(result["ctf/exp_group_id"]) == [5]

    def test_uses_args_input_project_folder(self):
        from helicon.plugins.cryosparc.assignexposuregroupbybeamshiftxy import handle
        from helicon.lib.exceptions import HeliconIOError

        data = MockCSDataset(
            {
                "location/micrograph_name": ["a.mrc"],
                "ctf/exp_group_id": [1],
            }
        )

        args = argparse.Namespace(
            verbose=0,
            cpu=-1,
            input_project_folder=Path("/some/fake/path"),
            csFile=[],
        )

        with patch("helicon.check_foilhole_xml_files") as mock_check:
            mock_check.side_effect = HeliconIOError("no xml files")
            with pytest.raises(HeliconIOError):
                handle(
                    data,
                    args,
                    {},
                    param="1",
                    output_title="",
                    output_slots=set(),
                    exp_group_id_name="ctf/exp_group_id",
                    micrograph_name="location/micrograph_name",
                    original_exp_group_ids=[1],
                )

        mock_check.assert_called_once()

    def test_assigns_groups_by_beamshift_xy(self):
        from helicon.plugins.cryosparc.assignexposuregroupbybeamshiftxy import handle

        micros = ["mg1.mrc", "mg2.mrc", "mg3.mrc"]
        data = MockCSDataset(
            {
                "location/micrograph_name": micros,
                "ctf/exp_group_id": [1, 1, 1],
            }
        )

        args = argparse.Namespace(
            verbose=0,
            cpu=-1,
            input_project_folder=Path("/project"),
            csFile=["/project/J100/output.cs"],
            projectID="P407",
            outputWorkspaceID="W1",
            jobID=["J100"],
        )

        with (
            patch("helicon.check_foilhole_xml_files"),
            patch(
                "helicon.EPU_micrograph_path_2_movie_xml_path",
                side_effect=lambda micrograph_path, xml_folder="": f"/project/{Path(micrograph_path).stem}.xml",
            ),
            patch(
                "helicon.EPU_xml_2_beamshift",
                side_effect=lambda xml_file: {
                    "/project/mg1.xml": (1.0, 2.0),
                    "/project/mg2.xml": (3.0, 4.0),
                    "/project/mg3.xml": (1.1, 2.1),
                }[str(xml_file)],
            ),
            patch(
                "helicon.assign_beamshifts_to_cluster", return_value=np.array([1, 2, 1])
            ),
            patch("helicon.sync_group_columns"),
            patch("helicon.propagate_ctf_median"),
        ):
            result, title, slots, idx = handle(
                data,
                args,
                {},
                param="1",
                output_title="",
                output_slots=set(),
                exp_group_id_name="ctf/exp_group_id",
                micrograph_name="location/micrograph_name",
                original_exp_group_ids=[1],
            )

        assert len(np.unique(result["ctf/exp_group_id"])) > 1
        assert "beamshift XY groups" in title
