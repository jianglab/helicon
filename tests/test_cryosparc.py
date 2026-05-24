import argparse
import sys
import pytest
from unittest.mock import patch
from helicon.commands import cryosparc
from helicon.lib.exceptions import HeliconValidationError


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
