import pytest
import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from helicon.commands import HOM_containerC
from helicon.lib.exceptions import HeliconValidationError


class TestHOMcontainerCArgs(object):
    def test_add_args_parser_has_expected_arguments(self):
        parser = argparse.ArgumentParser()
        HOM_containerC.add_args(parser)
        actions = {a.dest for a in parser._actions}
        expected = {"input_star", "output_star", "verbose", "param", "force"}
        missing = expected - actions
        assert not missing, f"Missing args: {missing}"

    @patch("os.path.exists", return_value=True)
    def test_check_args_rejects_existing_output(self, mock_exists):
        parser = argparse.ArgumentParser()
        HOM_containerC.add_args(parser)
        args = parser.parse_args(["in.star", "out.star"])
        with pytest.raises(HeliconValidationError):
            HOM_containerC.check_args(args, parser)

    @patch("os.path.exists", return_value=True)
    def test_check_args_force_overwrites_existing(self, mock_exists):
        parser = argparse.ArgumentParser()
        HOM_containerC.add_args(parser)
        args = parser.parse_args(["in.star", "out.star", "--force", "1"])
        result = HOM_containerC.check_args(args, parser)
        assert result is args
        assert result.force == 1

    def test_add_args_requires_input_and_output(self):
        parser = argparse.ArgumentParser()
        HOM_containerC.add_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])
        with pytest.raises(SystemExit):
            parser.parse_args(["in.star"])

    @patch("helicon.commands.HOM_containerC._read_star")
    @patch("helicon.commands.HOM_containerC.HelicalSegmentConsistency")
    @patch("helicon.commands.HOM_containerC._write_star")
    def test_main_calls_through(self, mock_write, mock_hsc, mock_read):
        df_in = pd.DataFrame({"rlnImageName": ["1@test.mrc"]})
        df_out = df_in.copy()
        stats = {"note": "ok"}
        mock_read.return_value = df_in
        mock_hsc.return_value = (df_out, stats)

        parser = argparse.ArgumentParser()
        HOM_containerC.add_args(parser)
        args = parser.parse_args(["in.star", "out.star", "--verbose", "2"])
        HOM_containerC.main(args)

        mock_read.assert_called_once_with("in.star")
        mock_hsc.assert_called_once_with(
            df_in,
            convert_path_fn=None,
            verbose=2,
            input_star_path="in.star",
            output_star_path="out.star",
            param=None,
        )
        mock_write.assert_called_once_with(df_out, "out.star", like="in.star")


class TestHOMcontainerCHelpers(object):
    @patch("helicon.commands.HOM_containerC.starfile")
    def test_read_star_with_starfile_dict(self, mock_starfile):
        particles = pd.DataFrame({"rlnAngleRot": [1.0, 2.0]})
        mock_starfile.read.return_value = {"data_particles": particles}
        result = HOM_containerC._read_star("dummy.star")
        pd.testing.assert_frame_equal(result, particles)
        mock_starfile.read.assert_called_once_with("dummy.star")

    @patch("helicon.commands.HOM_containerC.starfile")
    def test_read_star_with_starfile_dataframe(self, mock_starfile):
        df = pd.DataFrame({"rlnAngleRot": [1.0, 2.0]})
        mock_starfile.read.return_value = df
        result = HOM_containerC._read_star("dummy.star")
        pd.testing.assert_frame_equal(result, df)

    @patch("helicon.commands.HOM_containerC.starfile")
    def test_read_star_falls_back_to_first_table(self, mock_starfile):
        mock_starfile.read.return_value = {
            "data_optics": pd.DataFrame({"rlnVoltage": [300.0]}),
        }
        result = HOM_containerC._read_star("dummy.star")
        assert isinstance(result, pd.DataFrame)
        assert "rlnVoltage" in result.columns

    @patch("helicon.commands.HOM_containerC.starfile")
    def test_read_star_raises_when_starfile_none(self, mock_starfile):
        with patch.object(HOM_containerC, "starfile", None):
            with pytest.raises(RuntimeError):
                HOM_containerC._read_star("dummy.star")

    @patch("helicon.commands.HOM_containerC.starfile")
    @patch("os.path.exists", return_value=True)
    def test_write_star_with_optics(self, mock_exists, mock_starfile):
        particles = pd.DataFrame({"rlnAngleRot": [1.0]})
        optics = pd.DataFrame({"rlnVoltage": [300.0]})
        mock_starfile.read.return_value = {"data_optics": optics}

        HOM_containerC._write_star(particles, "out.star", like="in.star")

        mock_starfile.read.assert_called_once_with("in.star")
        out_dict = mock_starfile.write.call_args[0][0]
        assert "data_optics" in out_dict
        assert "data_particles" in out_dict
        pd.testing.assert_frame_equal(out_dict["data_particles"], particles)
        assert mock_starfile.write.call_args[1]["overwrite"] == True

    @patch("helicon.commands.HOM_containerC.starfile")
    @patch("os.path.exists", return_value=False)
    def test_write_star_single_table(self, mock_exists, mock_starfile):
        df = pd.DataFrame({"rlnAngleRot": [1.0]})
        HOM_containerC._write_star(df, "out.star", like="in.star")
        mock_starfile.write.assert_called_once_with(df, "out.star", overwrite=True)

    @patch("helicon.commands.HOM_containerC.starfile")
    @patch("os.path.exists", return_value=True)
    def test_write_star_no_like_writes_single(self, mock_exists, mock_starfile):
        mock_starfile.read.return_value = pd.DataFrame({"a": [1]})
        df = pd.DataFrame({"rlnAngleRot": [1.0]})
        HOM_containerC._write_star(df, "out.star")
        mock_starfile.write.assert_called_once_with(df, "out.star", overwrite=True)

    @patch("helicon.commands.HOM_containerC.starfile")
    def test_write_star_raises_when_starfile_none(self, mock_starfile):
        with patch.object(HOM_containerC, "starfile", None):
            with pytest.raises(RuntimeError):
                HOM_containerC._write_star(pd.DataFrame(), "out.star")


class TestHelicalSegmentConsistency(object):
    def _make_minimal_dataframe(self, n_micrographs=2, segments_per_micrograph=30):
        rows = []
        for mi in range(n_micrographs):
            for si in range(segments_per_micrograph):
                particle_num = mi * segments_per_micrograph + si + 1
                rows.append(
                    {
                        "rlnImageName": f"{particle_num}@micrograph_{mi}.mrc",
                        "rlnAngleRot": float(si * 10),
                        "rlnAngleTilt": 30.0,
                        "rlnAnglePsi": 0.0,
                        "rlnClassNumber": 1,
                        "rlnHelicalTubeID": mi + 1,
                        "rlnHelicalTrackLengthAngst": float(si * 10),
                    }
                )
        return pd.DataFrame(rows)

    @patch("os.getcwd", return_value="/tmp/test_hom")
    @patch("os.mkdir")
    @patch("os.chdir")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.hist")
    @patch("matplotlib.pyplot.scatter")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("pathlib.Path.mkdir")
    def test_helical_segment_consistency_minimal_data(
        self,
        mock_mkdir,
        mock_tight,
        mock_subplots,
        mock_close,
        mock_savefig,
        mock_plot,
        mock_scatter,
        mock_hist,
        mock_figure,
        mock_chdir,
        mock_mkdir_os,
        mock_getcwd,
    ):
        mock_subplots.side_effect = [
            (MagicMock(), MagicMock()),
            (MagicMock(), (MagicMock(), MagicMock())),
        ]

        df = self._make_minimal_dataframe(n_micrographs=2, segments_per_micrograph=30)

        result, stats = HOM_containerC.HelicalSegmentConsistency(
            df,
            verbose=0,
            input_star_path="some_input_path.star",
            output_star_path="test_out.star",
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(stats, dict)
        expected_cols = {
            "rlnHelicalTubeAndMicIDGood",
            "rlnHelicalTubeAndMicIDGoodSegValue",
            "rlnPartNum",
            "rlnMicrographFromImageName",
            "rlnMicUniqId",
            "rlnHelicalTubeAndMicID",
        }
        assert expected_cols.issubset(set(result.columns))
        assert "rlnPartNum" in result.columns
        assert "rlnHelicalTubeAndMicID" in result.columns
        assert "rlnHelicalTubeAndMicIDGood" in result.columns
        assert "rlnHelicalTubeAndMicIDGoodSegValue" in result.columns
        assert len(result) == len(df)
        assert list(result["rlnMicrographFromImageName"].unique()) == [
            "micrograph_0.mrc",
            "micrograph_1.mrc",
        ]

    @patch("os.getcwd", return_value="/tmp/test_hom")
    @patch("os.mkdir")
    @patch("os.chdir")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.hist")
    @patch("matplotlib.pyplot.scatter")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("pathlib.Path.mkdir")
    def test_helical_segment_consistency_empty_data_raises(
        self,
        mock_mkdir,
        mock_tight,
        mock_subplots,
        mock_close,
        mock_savefig,
        mock_plot,
        mock_scatter,
        mock_hist,
        mock_figure,
        mock_chdir,
        mock_mkdir_os,
        mock_getcwd,
    ):
        mock_subplots.side_effect = [
            (MagicMock(), MagicMock()),
            (MagicMock(), (MagicMock(), MagicMock())),
        ]

        df = pd.DataFrame(
            columns=[
                "rlnImageName",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
                "rlnClassNumber",
                "rlnHelicalTubeID",
                "rlnHelicalTrackLengthAngst",
            ]
        )

        with pytest.raises((KeyError, ValueError)):
            HOM_containerC.HelicalSegmentConsistency(
                df,
                verbose=0,
                input_star_path="empty.star",
                output_star_path="empty_out.star",
            )

    @patch("os.getcwd", return_value="/tmp/test_hom")
    @patch("os.mkdir")
    @patch("os.chdir")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.hist")
    @patch("matplotlib.pyplot.scatter")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("pathlib.Path.mkdir")
    def test_helical_segment_consistency_with_param(
        self,
        mock_mkdir,
        mock_tight,
        mock_subplots,
        mock_close,
        mock_savefig,
        mock_plot,
        mock_scatter,
        mock_hist,
        mock_figure,
        mock_chdir,
        mock_mkdir_os,
        mock_getcwd,
    ):
        mock_subplots.side_effect = [
            (MagicMock(), MagicMock()),
            (MagicMock(), (MagicMock(), MagicMock())),
        ]

        df = self._make_minimal_dataframe(n_micrographs=2, segments_per_micrograph=30)

        result, stats = HOM_containerC.HelicalSegmentConsistency(
            df,
            verbose=0,
            input_star_path="some_input_path.star",
            output_star_path="test_out.star",
            param='{"key": "value"}',
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(stats, dict)
        expected_cols = {
            "rlnHelicalTubeAndMicIDGood",
            "rlnHelicalTubeAndMicIDGoodSegValue",
            "rlnHelicalTubeAndMicID",
        }
        assert expected_cols.issubset(set(result.columns))


class TestHOMcontainerCStandaloneEntryPoint(object):
    @patch("helicon.commands.HOM_containerC._read_star")
    @patch("helicon.commands.HOM_containerC.HelicalSegmentConsistency")
    @patch("helicon.commands.HOM_containerC._write_star")
    @patch("argparse.ArgumentParser.parse_args")
    def test_standalone_entry(self, mock_parse, mock_write, mock_hsc, mock_read):
        df_in = pd.DataFrame({"rlnImageName": ["1@t.mrc"]})
        df_out = df_in.copy()
        mock_read.return_value = df_in
        mock_hsc.return_value = (df_out, {"ok": True})
        mock_parse.return_value = argparse.Namespace(
            input_star="in.star",
            output_star="out.star",
            verbose=1,
            param=None,
            force=0,
        )
        with patch.object(
            HOM_containerC, "check_args", return_value=mock_parse.return_value
        ):
            HOM_containerC.main(mock_parse.return_value)

        mock_write.assert_called_once_with(df_out, "out.star", like="in.star")
