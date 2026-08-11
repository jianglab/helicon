"""Tests for the shared images2star option pipeline engine."""

import argparse
from unittest.mock import patch

import pandas as pd
import pytest

from helicon.lib.exceptions import HeliconError
from helicon.lib.images2star_engine import (
    apply_options,
    gui_operation_specs,
    parse_operation_value,
    stack_to_namespace,
)


def _make_optics():
    """Return a minimal RELION optics table."""
    return pd.DataFrame({"rlnOpticsGroup": [1]})


def _make_data():
    """Return a small in-memory dataset with attrs['optics'] set."""
    data = pd.DataFrame(
        {
            "rlnMicrographName": [
                "/data/A.mrc",
                "/data/A.mrc",
                "/data/B.mrc",
                "/data/B.mrc",
            ],
            "rlnHelicalTubeID": [1, 2, 1, 2],
            "rlnDefocusU": [2.0, 1.0, 2.0, 1.0],
            "rlnDefocusV": [10.0, 20.0, 5.0, 15.0],
        }
    )
    data.attrs["optics"] = _make_optics()
    return data


class TestApplyOptions(object):
    """Tests for the ordered plugin dispatch engine."""

    def test_empty_options_returns_data_unchanged(self):
        data = _make_data()
        args = argparse.Namespace(verbose=0)

        result = apply_options(data, [], args, [])

        assert result.equals(data)
        assert result.attrs["optics"] is data.attrs["optics"]

    def test_applies_options_in_order(self):
        """setParm then select keeps every row; the reverse excludes all."""
        data = _make_data()
        args = argparse.Namespace(
            verbose=0,
            setParm=[["rlnHelicalTubeID", "9"]],
            select=["rlnHelicalTubeID", "9"],
        )

        result = apply_options(data, ["setParm", "select"], args, ["setParm"])
        assert len(result) == 4
        assert (result["rlnHelicalTubeID"] == 9).all()

        # setParm mutates the input in place, so the reversed case needs a
        # fresh dataset: selecting a value that does not exist excludes all.
        reversed_data = _make_data()
        reversed_args = argparse.Namespace(
            verbose=0,
            setParm=[["rlnHelicalTubeID", "9"]],
            select=["rlnHelicalTubeID", "9"],
        )
        with pytest.raises(HeliconError):
            apply_options(
                reversed_data, ["select", "setParm"], reversed_args, ["setParm"]
            )

    def test_append_options_consumed_in_order(self):
        data = _make_data()
        args = argparse.Namespace(
            verbose=0,
            sortby=[["rlnDefocusU"], ["rlnDefocusV"]],
        )

        result = apply_options(data, ["sortby", "sortby"], args, ["sortby"])

        # The second sort is primary; the first breaks ties (stable sort).
        assert result["rlnDefocusV"].tolist() == [5.0, 10.0, 15.0, 20.0]
        assert result["rlnDefocusU"].tolist() == [2.0, 2.0, 1.0, 1.0]

    def test_unknown_option_raises_value_error(self):
        data = _make_data()
        args = argparse.Namespace(verbose=0, noSuchOption=None)

        with pytest.raises(ValueError, match="Unknown option"):
            apply_options(data, ["noSuchOption"], args, [])

    def test_psi_prior_180_missing_column_raises_helicon_error(self):
        data = _make_data()
        args = argparse.Namespace(verbose=0, psiPrior180=1)

        with pytest.raises(HeliconError, match="rlnAnglePsiPrior"):
            apply_options(data, ["psiPrior180"], args, [])

    def test_psi_prior_180_duplicates_with_optics_attrs(self):
        data = _make_data()
        data["rlnAnglePsiPrior"] = [0.0] * len(data)
        args = argparse.Namespace(verbose=0, psiPrior180=1)

        result = apply_options(data, ["psiPrior180"], args, [])

        assert len(result) == 8
        assert result["rlnAnglePsiPrior"].tolist() == [0.0] * 4 + [180.0] * 4
        assert result["rlnHelicalTubeID"].tolist() == [1, 2, 1, 2, 11, 12, 11, 12]
        assert result.attrs["optics"] is data.attrs["optics"]

    def test_add_parm_missing_column_raises_helicon_error(self):
        data = _make_data()
        args = argparse.Namespace(verbose=0, addParm=[["rlnNoSuchColumn", "1.5"]])

        with pytest.raises(HeliconError, match="rlnNoSuchColumn"):
            apply_options(data, ["addParm"], args, ["addParm"])

    def test_add_parm_adds_value_to_existing_column(self):
        data = _make_data()
        data["rlnDefocusU"] = [2.0, 1.0, 2.0, 1.0]
        args = argparse.Namespace(verbose=0, addParm=[["rlnDefocusU", "0.5"]])

        result = apply_options(data, ["addParm"], args, ["addParm"])

        assert result["rlnDefocusU"].tolist() == [2.5, 1.5, 2.5, 1.5]

    def test_mult_parm_missing_column_raises_helicon_error(self):
        data = _make_data()
        args = argparse.Namespace(verbose=0, multParm=[["rlnNoSuchColumn", "2"]])

        with pytest.raises(HeliconError, match="rlnNoSuchColumn"):
            apply_options(data, ["multParm"], args, ["multParm"])

    def test_mult_parm_multiplies_existing_column(self):
        data = _make_data()
        data["rlnDefocusU"] = [2.0, 1.0, 2.0, 1.0]
        args = argparse.Namespace(verbose=0, multParm=[["rlnDefocusU", "2"]])

        result = apply_options(data, ["multParm"], args, ["multParm"])

        assert result["rlnDefocusU"].tolist() == [4.0, 2.0, 4.0, 2.0]

    def test_matches_check_args_option_ordering(self, tmp_path):
        """CLI-parsed args drive the engine exactly as main() would."""
        from helicon.commands import images2star

        parser = argparse.ArgumentParser()
        images2star.add_args(parser)
        argv = [
            "in.star",
            str(tmp_path / "out.star"),
            "--select",
            "rlnDefocusV",
            "5,20",
            "--sortby",
            "rlnDefocusU",
        ]
        with patch("sys.argv", ["helicon", "images2star"] + argv):
            args = parser.parse_args(argv)
            args = images2star.check_args(args, parser)

        assert args.all_options == ["select", "sortby"]
        assert "sortby" in args.append_options

        result = apply_options(
            _make_data(), args.all_options, args, args.append_options
        )

        # Rows with rlnDefocusV in {5, 20}, sorted by rlnDefocusU.
        assert result["rlnDefocusU"].tolist() == [1.0, 2.0]
        assert result["rlnDefocusV"].tolist() == [20.0, 5.0]


class TestParseOperationValue(object):
    """Tests for CLI-style parameter parsing into converted values."""

    @staticmethod
    def _spec(nargs, typ=str, choices=None):
        return {"choices": choices, "nargs": nargs, "type": typ}

    def test_single_value_converted(self):
        assert parse_operation_value("42", self._spec(None, int)) == 42

    def test_quoted_tokens_preserved(self):
        assert parse_operation_value('"a b" c', self._spec("+")) == ["a b", "c"]

    def test_fixed_nargs_returns_list(self):
        assert parse_operation_value("1.5 2.5", self._spec(2, float)) == [1.5, 2.5]

    def test_choices_validated(self):
        spec = self._spec(None, choices=["graphene", "gold"])
        assert parse_operation_value("gold", spec) == "gold"
        with pytest.raises(ValueError, match="not one of"):
            parse_operation_value("silver", spec)

    def test_wrong_arity_raises(self):
        with pytest.raises(ValueError, match="expected 2 values"):
            parse_operation_value("only-one", self._spec(2))

    def test_store_true_takes_no_value(self):
        assert parse_operation_value("", self._spec(0)) is True
        with pytest.raises(ValueError, match="no value"):
            parse_operation_value("x", self._spec(0))


class TestStackToNamespace(object):
    """Tests for building a CLI-equivalent Namespace from a stack."""

    @staticmethod
    def _specs():
        return {
            "sortby": {
                "append": True,
                "nargs": "+",
                "type": str,
                "default": [],
                "dest": "sortby",
            },
            "select": {
                "append": False,
                "nargs": 2,
                "type": str,
                "default": [],
                "dest": "select",
            },
        }

    def test_append_options_accumulate_in_order(self):
        args = stack_to_namespace(
            [
                ("sortby", ["rlnDefocusU"]),
                ("select", ["rlnDefocusV", "5,20"]),
                ("sortby", ["rlnDefocusV"]),
            ],
            self._specs(),
        )
        assert args.sortby == [["rlnDefocusU"], ["rlnDefocusV"]]
        assert args.select == ["rlnDefocusV", "5,20"]

    def test_duplicate_non_append_raises(self):
        with pytest.raises(ValueError, match="only be applied once"):
            stack_to_namespace(
                [("select", ["a", "1"]), ("select", ["b", "2"])], self._specs()
            )

    def test_unknown_operation_raises(self):
        with pytest.raises(ValueError, match="unknown operation"):
            stack_to_namespace([("nope", None)], self._specs())

    def test_seeds_cli_infrastructure_defaults(self):
        args = stack_to_namespace([], self._specs())
        assert args.verbose == 0
        assert args.cpu == 1
        assert args.input_imageFiles == [""]
        assert args.output_starFile == ""


class TestGuiSpecs(object):
    """Tests for the GUI-safe operation registry."""

    def test_excludes_file_writing_and_cli_only_operations(self):
        specs = gui_operation_specs()
        for name in (
            "process",
            "createStack",
            "splitByMicrograph",
            "extractHelices",
            "path",
            "sets",
        ):
            assert name not in specs

    def test_every_spec_carries_required_keys(self):
        specs = gui_operation_specs()
        assert specs
        required = {
            "dest",
            "option_string",
            "metavar",
            "type",
            "nargs",
            "choices",
            "default",
            "help",
            "append",
        }
        for spec in specs.values():
            assert required <= set(spec)
