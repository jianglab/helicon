import argparse
import pytest
from helicon.commands import symmetry_mismatch


class TestSymmetryMismatchArgs(object):
    def test_add_args_parser_has_expected_arguments(self):
        parser = argparse.ArgumentParser()
        symmetry_mismatch.add_args(parser)
        actions = {a.dest for a in parser._actions}
        expected = {
            "projectID",
            "jobID1",
            "jobID2",
            "input1",
            "pass_through1",
            "input2",
            "pass_through2",
            "outputFile1",
            "outputFile2",
            "sym1",
            "sym2",
            "workspaceID",
            "dist_tol",
            "axis_tol",
            "verbose",
        }
        assert expected.issubset(actions), f"Missing: {expected - actions}"

    def _make_args(self, **kwargs):
        defaults = dict(
            projectID=None,
            jobID1=None,
            jobID2=None,
            input1=None,
            pass_through1=None,
            input2=None,
            pass_through2=None,
            outputFile1=None,
            outputFile2=None,
            sym1=None,
            sym2=None,
            workspaceID=None,
            dist_tol=50.0,
            axis_tol=5.0,
            verbose=2,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_check_args_no_inputs_raises_error(self):
        args = self._make_args()
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            symmetry_mismatch.check_args(args, parser)

    def test_check_args_input1_without_sym1_raises_error(self):
        args = self._make_args(input1="/path/to/file.cs")
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            symmetry_mismatch.check_args(args, parser)

    def test_check_args_input2_without_sym2_raises_error(self):
        args = self._make_args(
            input1="/path/to/file1.cs",
            sym1="C5",
            input2="/path/to/file2.cs",
        )
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            symmetry_mismatch.check_args(args, parser)

    def test_check_args_input1_without_input2_or_job2_raises_error(self):
        args = self._make_args(input1="/path/to/file1.cs", sym1="C5")
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            symmetry_mismatch.check_args(args, parser)

    def test_check_args_bad_output_extension_raises_error(self):
        args = self._make_args(
            input1="/path/to/file1.cs",
            sym1="C5",
            input2="/path/to/file2.cs",
            sym2="C12",
            outputFile1="/path/out.txt",
        )
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            symmetry_mismatch.check_args(args, parser)

    def test_check_args_negative_dist_tol_raises_error(self):
        args = self._make_args(
            input1="/path/to/file1.cs",
            sym1="C5",
            input2="/path/to/file2.cs",
            sym2="C12",
            dist_tol=-1.0,
        )
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            symmetry_mismatch.check_args(args, parser)

    def test_check_args_negative_axis_tol_raises_error(self):
        args = self._make_args(
            input1="/path/to/file1.cs",
            sym1="C5",
            input2="/path/to/file2.cs",
            sym2="C12",
            axis_tol=0.0,
        )
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            symmetry_mismatch.check_args(args, parser)

    def test_check_args_full_local_inputs_passes(self):
        args = self._make_args(
            input1="/path/to/file1.cs",
            sym1="C5",
            input2="/path/to/file2.cs",
            sym2="C12",
            outputFile1="/path/out1.cs",
            outputFile2="/path/out2.cs",
            dist_tol=30.0,
            axis_tol=3.0,
        )
        parser = argparse.ArgumentParser()
        result = symmetry_mismatch.check_args(args, parser)
        assert result is args
        assert result.dist_tol == 30.0
        assert result.axis_tol == 3.0

    def test_check_args_project_and_jobs_passes(self):
        args = self._make_args(
            projectID="P407",
            jobID1="J100",
            jobID2="J189",
        )
        parser = argparse.ArgumentParser()
        result = symmetry_mismatch.check_args(args, parser)
        assert result is args
