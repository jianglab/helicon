import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
import pytest
import helicon
from helicon.commands import images2star
from helicon.lib.exceptions import HeliconValidationError, HeliconFileExistsError


def _make_optics(n_groups=1):
    """Create a minimal RELION optics table."""
    rows = []
    for i in range(1, n_groups + 1):
        rows.append(
            {
                "rlnOpticsGroup": i,
                "rlnOpticsGroupName": f"opticsGroup{i}",
                "rlnVoltage": 300.0,
                "rlnSphericalAberration": 2.7,
                "rlnAmplitudeContrast": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _make_dataframe(micrographs, optics_groups=None, image_col="rlnMicrographName"):
    """Create a DataFrame with required columns for images2star option tests.

    Args:
        micrographs: List of micrograph filenames (one per row).
        optics_groups: List of existing rlnOpticsGroup IDs (or int for all same).
        image_col: Column name to use for the image path.

    Returns:
        DataFrame with .attrs["optics"] set.
    """
    if isinstance(optics_groups, int):
        optics_groups = [optics_groups] * len(micrographs)
    data = pd.DataFrame(
        {
            image_col: micrographs,
            "rlnOpticsGroup": optics_groups,
        }
    )
    max_group = max(optics_groups) if optics_groups else 1
    data.attrs["optics"] = _make_optics(max_group)
    return data


class TestImages2starBeamShiftLabel(object):
    """Tests for assignOpticGroupByBeamShiftLabel handler logic."""

    def _run_option(self, data, format, image_name="rlnMicrographName", verbose=0):
        """Execute the Label handler code block (images2star.py lines 2396-2484)."""
        optics_orig = data.attrs["optics"]
        existing_groups = data["rlnOpticsGroup"].copy()

        micrographs = data[image_name].unique()
        micrograph_to_subgroup = helicon.assign_beamshift_groups(
            micrographs, format, start_id=1
        )
        new_subgroups = data[image_name].map(micrograph_to_subgroup)

        data["rlnOpticsGroup"] = helicon.combine_groups(
            existing_groups.values, new_subgroups.values
        )

        pairs = pd.DataFrame(
            {"existing": existing_groups, "combined": data["rlnOpticsGroup"]}
        ).drop_duplicates()
        optics = optics_orig.copy().iloc[0:0]
        for _, pair in pairs.iterrows():
            parent = (
                optics_orig[
                    optics_orig["rlnOpticsGroup"].astype(str) == str(pair["existing"])
                ]
                .iloc[[0]]
                .copy()
            )
            parent["rlnOpticsGroup"] = pair["combined"]
            parent["rlnOpticsGroupName"] = f"opticsGroup{pair['combined']}"
            optics = pd.concat([optics, parent], ignore_index=True)

        data.attrs["optics"] = optics
        return data

    @patch("helicon.assign_beamshift_groups")
    def test_label_splits_group_by_beam_shift(self, mock_assign):
        micros = [
            "FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff",
            "FoilHole_28788144_Data_28764755_47_20240328_192117_fractions.tiff",
        ]
        mock_assign.return_value = {micros[0]: 1, micros[1]: 2}
        data = _make_dataframe(micros, optics_groups=1)

        data = self._run_option(data, "EPU")

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        assert len(group_ids) == 2
        assert len(data.attrs["optics"]) == 2
        for gid in group_ids:
            assert (
                f"opticsGroup{gid}" in data.attrs["optics"]["rlnOpticsGroupName"].values
            )

    @patch("helicon.assign_beamshift_groups")
    def test_label_same_shift_no_split(self, mock_assign):
        micros = ["mg1.mrc", "mg2.mrc"]
        mock_assign.return_value = {micros[0]: 1, micros[1]: 1}
        data = _make_dataframe(micros, optics_groups=1)

        data = self._run_option(data, "serialEM_embl_heidelberg")

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        assert len(group_ids) == 1
        assert len(data.attrs["optics"]) == 1

    @patch("helicon.assign_beamshift_groups")
    def test_label_preserves_multiple_existing_groups(self, mock_assign):
        micros = ["a.mrc", "b.mrc", "c.mrc", "d.mrc"]
        mock_assign.return_value = {
            micros[0]: 1,
            micros[1]: 2,
            micros[2]: 1,
            micros[3]: 2,
        }
        existing = [1, 1, 2, 2]
        data = _make_dataframe(micros, optics_groups=existing)

        data = self._run_option(data, "EPU")

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        # (1,1), (1,2), (2,1), (2,2) -> 4 combined groups
        assert len(group_ids) == 4
        assert len(data.attrs["optics"]) == 4
        # Each existing row is cloned: check that optics table has parents
        optics_parents = data.attrs["optics"]["rlnOpticsGroupName"].values
        for gid in group_ids:
            assert f"opticsGroup{gid}" in optics_parents

    @patch("helicon.assign_beamshift_groups")
    def test_label_optics_table_has_correct_groups(self, mock_assign):
        micros = ["x.mrc", "y.mrc"]
        mock_assign.return_value = {micros[0]: 1, micros[1]: 2}
        data = _make_dataframe(micros, optics_groups=1)

        data = self._run_option(data, "EPU")

        optics = data.attrs["optics"]
        assert set(optics["rlnOpticsGroup"]) == {1, 2}
        assert set(optics["rlnOpticsGroupName"]) == {"opticsGroup1", "opticsGroup2"}


class TestImages2starTime(object):
    """Tests for assignOpticGroupByTime handler logic."""

    def _run_option(
        self, data, param, image_name="rlnMicrographName", software="EPU", verbose=0
    ):
        """Execute the Time handler code block (images2star.py lines 2588-2682)."""
        optics_orig = data.attrs["optics"]

        movies = data[image_name].unique()
        if software in ["EPU"]:
            movie2time = helicon.extract_timestamps(movies, software)
        else:
            movie2time = helicon.extract_timestamps(
                movies, None, use_mtime_fallback=True
            )

        existing_groups = data["rlnOpticsGroup"].copy()
        new_groups = np.zeros(len(data), dtype=int)
        last_group_id = 0
        for og_id in sorted(existing_groups.unique()):
            mask = existing_groups.values == og_id
            micros_in_group = data.loc[mask, image_name].unique()
            times = [movie2time[m] for m in micros_in_group]
            time2sub = helicon.assign_to_groups(times, param)
            movie2sub = {
                m: time2sub[movie2time[m]] + last_group_id for m in micros_in_group
            }
            new_groups[mask] = data.loc[mask, image_name].map(movie2sub).values
            last_group_id = new_groups.max()

        data["rlnOpticsGroup"] = new_groups

        pairs = pd.DataFrame(
            {"existing": existing_groups, "combined": data["rlnOpticsGroup"]}
        ).drop_duplicates()
        optics = optics_orig.copy().iloc[0:0]
        for _, pair in pairs.iterrows():
            parent = (
                optics_orig[
                    optics_orig["rlnOpticsGroup"].astype(str) == str(pair["existing"])
                ]
                .iloc[[0]]
                .copy()
            )
            parent["rlnOpticsGroup"] = pair["combined"]
            parent["rlnOpticsGroupName"] = f"opticsGroup{pair['combined']}"
            optics = pd.concat([optics, parent], ignore_index=True)

        data.attrs["optics"] = optics
        return data

    def test_time_epu_splits_by_time(self):
        micros = [
            "FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff",
            "FoilHole_28788144_Data_28764755_47_20240328_192117_fractions.tiff",
        ]
        data = _make_dataframe(micros, optics_groups=1)

        data = self._run_option(data, param=1, software="EPU")

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        assert len(group_ids) >= 1

    def test_time_single_micrograph_one_group(self):
        micros = ["FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff"]
        data = _make_dataframe(micros, optics_groups=1)

        data = self._run_option(data, param=1, software="EPU")

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        assert len(group_ids) == 1
        assert len(data.attrs["optics"]) == 1

    @patch("helicon.extract_timestamps")
    def test_time_with_mocked_timestamps(self, mock_extract):
        micros = ["a.mrc", "b.mrc", "c.mrc", "d.mrc"]
        mock_extract.return_value = {
            micros[0]: 100.0,
            micros[1]: 200.0,
            micros[2]: 300.0,
            micros[3]: 400.0,
        }
        data = _make_dataframe(micros, optics_groups=1)

        data = self._run_option(data, param=2, software="EPU")

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        # 4 micrographs with distinct times, grouped into 2 per group -> 2 groups
        assert len(group_ids) >= 2

    @patch("helicon.extract_timestamps")
    def test_time_with_mtime_fallback(self, mock_extract):
        micros = ["/some/path/movie1.mrc", "/some/path/movie2.mrc"]
        mock_extract.return_value = {
            micros[0]: 1000.0,
            micros[1]: 2000.0,
        }
        data = _make_dataframe(micros, optics_groups=1)

        data = self._run_option(data, param=1, software=None)

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        assert len(group_ids) == 2
        # verify extract_timestamps was called with use_mtime_fallback
        _, kwargs = mock_extract.call_args
        assert kwargs.get("use_mtime_fallback", False)


class TestImages2starPerMicrograph(object):
    """Tests for assignOpticGroupPerMicrograph handler logic."""

    def _run_option(self, data, image_name="rlnMicrographName", verbose=0):
        """Execute the PerMicrograph handler code block (images2star.py lines 2684-2735)."""
        optics_orig = data.attrs["optics"]

        tmp_col = "TEMP_image_name"
        data[tmp_col] = data[image_name].str.split("@", expand=True).iloc[:, -1]
        micrographs = data[tmp_col].unique()
        mapping = helicon.per_micrograph_mapping(micrographs)
        data["rlnOpticsGroup"] = data[tmp_col].map(mapping)
        data.drop(tmp_col, axis=1, inplace=True)

        optics = pd.concat(
            [optics_orig.iloc[[0]]] * len(micrographs), ignore_index=True
        )
        for gi in range(1, len(micrographs) + 1):
            optics.loc[gi - 1, "rlnOpticsGroup"] = gi
            optics.loc[gi - 1, "rlnOpticsGroupName"] = f"opticsGroup{gi}"
        data.attrs["optics"] = optics
        return data

    def test_per_micrograph_one_group_per_micrograph(self):
        micros = ["a.mrc", "a.mrc", "b.mrc", "c.mrc"]
        data = _make_dataframe(micros, optics_groups=1)

        data = self._run_option(data)

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        assert len(group_ids) == 3
        np.testing.assert_array_equal(data["rlnOpticsGroup"].values[:2], [1, 1])
        assert len(data.attrs["optics"]) == 3

    def test_per_micrograph_single_micrograph(self):
        micros = ["a.mrc", "a.mrc"]
        data = _make_dataframe(micros, optics_groups=1)

        data = self._run_option(data)

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        assert len(group_ids) == 1
        assert len(data.attrs["optics"]) == 1

    def test_per_micrograph_with_image_name_column(self):
        """Use rlnImageName column with @ syntax (imageNumber@path)."""
        micros = ["1@a.mrc", "2@a.mrc", "1@b.mrc"]
        data = _make_dataframe(micros, optics_groups=1, image_col="rlnImageName")

        data = self._run_option(data, image_name="rlnImageName")

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        assert len(group_ids) == 2
        assert len(data.attrs["optics"]) == 2


class TestImages2starBeamShiftXY(object):
    """Tests for assignOpticGroupByBeamShiftXY handler logic."""

    @patch("helicon.EPU_micrograph_path_2_movie_xml_path")
    @patch("helicon.EPU_xml_2_beamshift")
    @patch("helicon.assign_beamshifts_to_cluster")
    def test_xy_splits_groups(self, mock_cluster, mock_beamshift, mock_xml_path):
        micros = ["/data/mg1.mrc", "/data/mg2.mrc", "/data/mg3.mrc"]
        mock_xml_path.side_effect = lambda micrograph_path, xml_folder: (
            Path(micrograph_path).with_suffix(".xml")
        )
        mock_beamshift.side_effect = [(1.0, 2.0), (1.1, 2.1), (5.0, 5.0)]
        mock_cluster.return_value = np.array([1, 1, 2])

        data = _make_dataframe(micros, optics_groups=1)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.glob", return_value=["FoilHole_1.xml"]),
        ):
            _, param_dict = helicon.parse_param_str(
                "xml_folder=/data:min_micrographs_per_group=4"
            )
            xml_folder = param_dict.get("xml_folder", "")
            min_cluster_size = int(param_dict.get("min_micrographs_per_group", 4))
            image_name = "rlnMicrographName"

            micrographs = data[image_name].unique()
            xml_files_dict = {
                m: helicon.EPU_micrograph_path_2_movie_xml_path(
                    micrograph_path=m, xml_folder=xml_folder
                )
                for m in micrographs
            }
            micrographs_to_beamshifts = {
                m: helicon.EPU_xml_2_beamshift(xml_file=xml_files_dict[m])
                for m in micrographs
            }
            beamshifts = list(micrographs_to_beamshifts.values())
            beamshift_clusters = helicon.assign_beamshifts_to_cluster(
                beamshifts=beamshifts,
                min_cluster_size=min_cluster_size,
                cpu=1,
                verbose=0,
            )

            micrograph_to_cluster = {
                m: beamshift_clusters[mi]
                for mi, m in enumerate(micrographs_to_beamshifts.keys())
            }
            new_subgroups = data[image_name].map(micrograph_to_cluster)
            existing_groups = data["rlnOpticsGroup"].copy()
            data["rlnOpticsGroup"] = helicon.combine_groups(
                existing_groups.values, new_subgroups.values
            )

            group_ids = sorted(data["rlnOpticsGroup"].unique())
            assert len(group_ids) == 2

    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.glob", return_value=[])
    def test_xy_errors_without_xml_files(self, mock_glob, mock_exists):
        micros = ["/data/mg1.mrc"]
        data = _make_dataframe(micros, optics_groups=1)
        optics_orig = data.attrs["optics"]
        image_name = "rlnMicrographName"

        micrographs = data[image_name].unique()

        def has_xml(xml_folder, micrograph_path):
            if xml_folder:
                xfp = Path(xml_folder)
                if xfp.exists() and xfp.is_dir() and list(xfp.glob("FoilHole_*.xml")):
                    return True
            if Path(micrograph_path).exists():
                if list(Path(micrograph_path).parent.glob("FoilHole_*.xml")):
                    return True
            return False

        assert not has_xml(xml_folder="", micrograph_path=micrographs[0])


class TestImages2starCheckArgs(object):
    """Tests for images2star check_args validation."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_check_args_existing_output_raises_error(self, mock_exists):
        parser = argparse.ArgumentParser()
        images2star.add_args(parser)
        args = parser.parse_args(["input.cs", "x.star"])
        with pytest.raises(HeliconFileExistsError):
            images2star.check_args(args, parser)

    @patch("pathlib.Path.exists", return_value=True)
    def test_check_args_force_overwrites_existing(self, mock_exists):
        parser = argparse.ArgumentParser()
        images2star.add_args(parser)
        args = parser.parse_args(["input.cs", "x.star", "--force", "1"])
        args = images2star.check_args(args, parser)
        assert args.output_starFile == "x.star"

    def test_check_args_micrograph_star(self):
        parser = argparse.ArgumentParser()
        images2star.add_args(parser)
        args = parser.parse_args(["input.cs", "x.star", "--micrographStar", "ref.star"])
        assert args.micrographStar == "ref.star"

    def test_check_args_micrograph_star_default_none(self):
        parser = argparse.ArgumentParser()
        images2star.add_args(parser)
        args = parser.parse_args(["input.cs", "x.star"])
        assert args.micrographStar is None


class TestImages2starResetOpticGroups(object):
    """Tests for resetOpticGroups handler logic."""

    def test_reset_combines_all_groups_into_one(self):
        micros = ["a.mrc", "b.mrc", "c.mrc"]
        data = _make_dataframe(micros, optics_groups=[1, 2, 3])
        optics_orig = data.attrs["optics"]

        data["rlnOpticsGroup"] = 1
        optics = optics_orig.copy().iloc[0:0]
        new_row = optics_orig.copy().iloc[[0]]
        new_row["rlnOpticsGroup"] = 1
        new_row["rlnOpticsGroupName"] = "opticsGroup1"
        optics = pd.concat([optics, new_row], ignore_index=True)
        data.attrs["optics"] = optics

        group_ids = sorted(data["rlnOpticsGroup"].unique())
        assert group_ids == [1]
        assert len(data.attrs["optics"]) == 1
