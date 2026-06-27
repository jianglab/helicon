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
    """Tests for assignOpticGroupByBeamShiftLabel handler."""

    def _make_args(self, verbose=0):
        return argparse.Namespace(verbose=verbose)

    def test_label_splits_group_by_beam_shift(self):
        from helicon.plugins.images2star.assignopticgroupbybeamshiftlabel import handle

        micros = [
            "FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff",
            "FoilHole_28788144_Data_28764755_47_20240328_192117_fractions.tiff",
        ]
        data = _make_dataframe(micros, optics_groups=1)

        result, idx = handle(data, self._make_args(), {}, param="EPU")

        group_ids = sorted(result["rlnOpticsGroup"].unique())
        assert len(group_ids) == 2
        assert len(result.attrs["optics"]) == 2
        for gid in group_ids:
            assert (
                f"opticsGroup{gid}"
                in result.attrs["optics"]["rlnOpticsGroupName"].values
            )

    def test_label_same_shift_no_split(self):
        from helicon.plugins.images2star.assignopticgroupbybeamshiftlabel import handle

        micros = [
            "250123_SF0431_01129_1-7.eer",
            "250123_SF0431_01130_1-7.eer",
        ]
        data = _make_dataframe(micros, optics_groups=1)

        result, idx = handle(
            data, self._make_args(), {}, param="serialEM_embl_heidelberg"
        )

        group_ids = sorted(result["rlnOpticsGroup"].unique())
        assert len(group_ids) == 1
        assert len(result.attrs["optics"]) == 1

    def test_label_preserves_multiple_existing_groups(self):
        from helicon.plugins.images2star.assignopticgroupbybeamshiftlabel import handle

        micros = [
            "FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff",
            "FoilHole_28788144_Data_28764755_47_20240328_192117_fractions.tiff",
            "FoilHole_28788144_Data_28764755_48_20240328_192118_fractions.tiff",
            "FoilHole_28788144_Data_28764755_49_20240328_192119_fractions.tiff",
        ]
        existing = [1, 1, 2, 2]
        data = _make_dataframe(micros, optics_groups=existing)

        result, idx = handle(data, self._make_args(), {}, param="EPU")

        group_ids = sorted(result["rlnOpticsGroup"].unique())
        # (1,46), (1,47), (2,48), (2,49) -> 4 combined groups
        assert len(group_ids) == 4
        assert len(result.attrs["optics"]) == 4
        for gid in group_ids:
            assert (
                f"opticsGroup{gid}"
                in result.attrs["optics"]["rlnOpticsGroupName"].values
            )

    def test_label_optics_table_has_correct_groups(self):
        from helicon.plugins.images2star.assignopticgroupbybeamshiftlabel import handle

        micros = [
            "FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff",
            "FoilHole_28788144_Data_28764755_47_20240328_192117_fractions.tiff",
        ]
        data = _make_dataframe(micros, optics_groups=1)

        result, idx = handle(data, self._make_args(), {}, param="EPU")

        optics = result.attrs["optics"]
        assert set(optics["rlnOpticsGroup"]) == {1, 2}
        assert set(optics["rlnOpticsGroupName"]) == {"opticsGroup1", "opticsGroup2"}

    def test_label_returns_unchanged_when_param_no(self):
        from helicon.plugins.images2star.assignopticgroupbybeamshiftlabel import handle

        micros = ["a.mrc", "b.mrc"]
        data = _make_dataframe(micros, optics_groups=[1, 2])

        result, idx = handle(data, self._make_args(), {}, param="no")

        assert list(result["rlnOpticsGroup"]) == [1, 2]


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


class TestImages2starTimeHandler(object):
    """Tests for assignOpticGroupByTime handler via actual handle()."""

    def _make_args(self, verbose=0):
        return argparse.Namespace(verbose=verbose)

    def test_time_with_micrograph_column_fallback(self):
        """Non-EPU data with rlnMicrographName (no rlnMicrographMovieName)."""
        from helicon.plugins.images2star.assignopticgroupbytime import handle

        micros = ["/data/movie1.mrc", "/data/movie2.mrc"]
        data = _make_dataframe(micros, optics_groups=1)

        with (
            patch(
                "helicon.guess_data_collection_software", return_value="serialEM_pncc"
            ),
            patch("helicon.assign_time_groups") as mock_atg,
        ):
            mock_atg.return_value = (
                np.array([1, 2]),
                {m: float(i) for i, m in enumerate(micros)},
                {m: f"time_{i}" for i, m in enumerate(micros)},
            )
            result, idx = handle(data, self._make_args(), {}, param=1)

        group_ids = sorted(result["rlnOpticsGroup"].unique())
        assert len(group_ids) >= 1
        assert "rlnMovieCollectionTime" in result

    def test_time_uses_movie_column_when_available(self):
        """Non-EPU data with rlnMicrographMovieName should use it."""
        from helicon.plugins.images2star.assignopticgroupbytime import handle

        micros = ["/data/movie1.mrc", "/data/movie2.mrc"]
        data = _make_dataframe(micros, optics_groups=1)
        data["rlnMicrographMovieName"] = micros

        with (
            patch(
                "helicon.guess_data_collection_software", return_value="serialEM_pncc"
            ),
            patch("helicon.assign_time_groups") as mock_atg,
        ):
            mock_atg.return_value = (
                np.array([1, 2]),
                {m: float(i) for i, m in enumerate(micros)},
                {m: f"time_{i}" for i, m in enumerate(micros)},
            )
            result, idx = handle(data, self._make_args(verbose=3), {}, param=1)

        group_ids = sorted(result["rlnOpticsGroup"].unique())
        assert len(group_ids) >= 1

    def test_time_returns_unchanged_when_param_negative(self):
        from helicon.plugins.images2star.assignopticgroupbytime import handle

        micros = ["a.mrc", "b.mrc"]
        data = _make_dataframe(micros, optics_groups=[1, 2])

        result, idx = handle(data, self._make_args(), {}, param=-1)

        assert list(result["rlnOpticsGroup"]) == [1, 2]


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


class TestImages2starResetOpticGroup(object):
    """Tests for resetOpticGroup handler."""

    def test_resets_all_groups_to_one(self):
        from helicon.plugins.images2star.resetopticgroup import handle

        micros = ["a.mrc"] * 3 + ["b.mrc"] * 2 + ["c.mrc"] * 2
        data = _make_dataframe(micros, optics_groups=[1, 1, 1, 2, 2, 3, 3])

        result, idx = handle(data, argparse.Namespace(verbose=0), {}, param=1)

        assert list(result["rlnOpticsGroup"].unique()) == [1]
        assert len(result["rlnOpticsGroup"]) == 7
        assert len(result.attrs["optics"]) == 1
        assert result.attrs["optics"].iloc[0]["rlnOpticsGroup"] == 1
        assert result.attrs["optics"].iloc[0]["rlnOpticsGroupName"] == "opticsGroup1"

    def test_returns_unchanged_when_param_is_false(self):
        from helicon.plugins.images2star.resetopticgroup import handle

        micros = ["a.mrc", "b.mrc", "c.mrc"]
        data = _make_dataframe(micros, optics_groups=[1, 2, 3])

        result, idx = handle(data, argparse.Namespace(verbose=0), {}, param=0)

        assert list(result["rlnOpticsGroup"].unique()) == [1, 2, 3]
        assert len(result.attrs["optics"]) == 3

    def test_requires_optics_block(self):
        from helicon.plugins.images2star.resetopticgroup import handle
        from helicon.lib.exceptions import HeliconError

        data = pd.DataFrame({"rlnOpticsGroup": [1, 2]})

        with pytest.raises(HeliconError, match="data_optics"):
            handle(data, argparse.Namespace(verbose=0), {}, param=1)


class TestImages2starBeamShiftXY(object):
    """Tests for assignOpticGroupByBeamShiftXY handler."""

    def _make_args(self, verbose=0, cpu=-1):
        return argparse.Namespace(verbose=verbose, cpu=cpu)

    def test_xy_returns_unchanged_when_param_zero(self):
        from helicon.plugins.images2star.assignopticgroupbybeamshiftxy import handle

        micros = ["a.mrc", "b.mrc"]
        data = _make_dataframe(micros, optics_groups=[1, 2])

        result, idx = handle(data, self._make_args(), {}, param="0")

        assert list(result["rlnOpticsGroup"]) == [1, 2]

    def test_xy_splits_groups(self):
        from helicon.plugins.images2star.assignopticgroupbybeamshiftxy import handle

        micros = ["/data/mg1.mrc", "/data/mg2.mrc", "/data/mg3.mrc"]
        data = _make_dataframe(micros, optics_groups=1)

        with (
            patch("helicon.check_foilhole_xml_files"),
            patch(
                "helicon.EPU_micrograph_path_2_movie_xml_path",
                side_effect=lambda micrograph_path, xml_folder="": Path(
                    micrograph_path
                ).with_suffix(".xml"),
            ),
            patch(
                "helicon.EPU_xml_2_beamshift",
                side_effect=[(1.0, 2.0), (1.1, 2.1), (5.0, 5.0)],
            ),
            patch(
                "helicon.assign_beamshifts_to_cluster", return_value=np.array([1, 1, 2])
            ),
        ):
            result, idx = handle(data, self._make_args(), {}, param="1")

        group_ids = sorted(result["rlnOpticsGroup"].unique())
        assert len(group_ids) == 2

    def test_xy_errors_without_xml_files(self):
        from helicon.plugins.images2star.assignopticgroupbybeamshiftxy import handle
        from helicon.lib.exceptions import HeliconIOError

        micros = ["/data/mg1.mrc"]
        data = _make_dataframe(micros, optics_groups=1)

        with patch("helicon.check_foilhole_xml_files") as mock_check:
            mock_check.side_effect = HeliconIOError("no xml files")
            with pytest.raises(HeliconIOError):
                handle(data, self._make_args(), {}, param="1")

        mock_check.assert_called_once()


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


class TestImages2starBreakFilaments(object):
    """Tests for breakFilaments handler."""

    def _make_args(self, verbose=0):
        return argparse.Namespace(verbose=verbose)

    def _make_data(self, n_micrographs=1, segments_per_filament=10, n_filaments=3):
        rows = []
        for mi in range(n_micrographs):
            for ti in range(n_filaments):
                for si in range(segments_per_filament):
                    rows.append(
                        {
                            "rlnMicrographName": f"/data/micrograph_{mi}.mrc",
                            "rlnHelicalTubeID": ti,
                            "rlnHelicalTrackLengthAngst": si * 10.0,
                            "rlnImageName": f"{si + mi * 1000:06d}@/data/particles.mrcs",
                        }
                    )
        df = pd.DataFrame(rows)
        df.attrs["optics"] = pd.DataFrame(
            {
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnVoltage": [300.0],
                "rlnSphericalAberration": [2.7],
                "rlnAmplitudeContrast": [0.1],
            }
        )
        return df

    def _make_args(self, verbose=0):
        return argparse.Namespace(verbose=verbose)

    def _make_index_d(self):
        return {"breakFilaments": 0}

    def _make_data(self, n_micrographs=1, segments_per_filament=10, n_filaments=3):
        rows = []
        for mi in range(n_micrographs):
            for ti in range(n_filaments):
                for si in range(segments_per_filament):
                    rows.append(
                        {
                            "rlnMicrographName": f"/data/micrograph_{mi}.mrc",
                            "rlnHelicalTubeID": ti,
                            "rlnHelicalTrackLengthAngst": si * 10.0,
                            "rlnImageName": f"{si + mi * 1000:06d}@/data/particles.mrcs",
                        }
                    )
        df = pd.DataFrame(rows)
        df.attrs["optics"] = pd.DataFrame(
            {
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnVoltage": [300.0],
                "rlnSphericalAberration": [2.7],
                "rlnAmplitudeContrast": [0.1],
            }
        )
        return df

    def test_breaks_long_filaments(self):
        from helicon.plugins.images2star.breakfilaments import handle

        data = self._make_data(
            n_micrographs=1, segments_per_filament=100, n_filaments=3
        )
        result, idx = handle(
            data,
            self._make_args(verbose=1),
            self._make_index_d(),
            param="maxSegments=30",
        )

        # 100 segments -> 4 groups (30+30+30+10) per filament, 3 filaments = 12 groups
        n_new_tubes = result["rlnHelicalTubeID"].nunique()
        assert n_new_tubes == 12
        assert result["rlnHelicalTubeIDOriginal"].nunique() == 3

    def test_preserves_short_filaments(self):
        from helicon.plugins.images2star.breakfilaments import handle

        data = self._make_data(n_micrographs=1, segments_per_filament=10, n_filaments=3)
        result, idx = handle(
            data, self._make_args(), self._make_index_d(), param="maxSegments=50"
        )

        assert result["rlnHelicalTubeID"].nunique() == 3
        # New IDs are assigned per group (each original filament becomes one new group)
        assert list(result["rlnHelicalTubeID"]) == [0] * 10 + [1] * 10 + [2] * 10

    def test_backs_up_original_tube_id(self):
        from helicon.plugins.images2star.breakfilaments import handle

        data = self._make_data(
            n_micrographs=1, segments_per_filament=100, n_filaments=2
        )
        result, idx = handle(
            data, self._make_args(), self._make_index_d(), param="maxSegments=30"
        )

        assert "rlnHelicalTubeIDOriginal" in result.columns
        assert set(result["rlnHelicalTubeIDOriginal"]) == {0, 1}

    def test_preserves_existing_original_tube_id(self):
        from helicon.plugins.images2star.breakfilaments import handle

        data = self._make_data(
            n_micrographs=1, segments_per_filament=100, n_filaments=2
        )
        data["rlnHelicalTubeIDOriginal"] = 99
        result, idx = handle(
            data, self._make_args(), self._make_index_d(), param="maxSegments=30"
        )

        assert list(result["rlnHelicalTubeIDOriginal"]) == [99] * 200

    def test_handles_multiple_micrographs(self):
        from helicon.plugins.images2star.breakfilaments import handle

        data = self._make_data(n_micrographs=3, segments_per_filament=75, n_filaments=2)
        result, idx = handle(
            data, self._make_args(), self._make_index_d(), param="maxSegments=20"
        )

        # 75 segments -> 4 groups (20+20+20+15) per filament, 2 filaments * 3 micrographs = 24 groups
        assert result["rlnHelicalTubeID"].nunique() == 24

    def test_maintains_segment_order_by_track_length(self):
        from helicon.plugins.images2star.breakfilaments import handle

        rows = []
        for si in range(99, -1, -1):
            rows.append(
                {
                    "rlnMicrographName": "/data/micrograph_0.mrc",
                    "rlnHelicalTubeID": 0,
                    "rlnHelicalTrackLengthAngst": float(si),
                    "rlnImageName": f"{si:06d}@/data/particles.mrcs",
                }
            )
        data = pd.DataFrame(rows)
        data.attrs["optics"] = pd.DataFrame(
            {
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnVoltage": [300.0],
                "rlnSphericalAberration": [2.7],
                "rlnAmplitudeContrast": [0.1],
            }
        )
        result, idx = handle(
            data, self._make_args(), self._make_index_d(), param="maxSegments=30"
        )

        # Each chunk should contain contiguous ranges of track lengths (sorted within chunk)
        chunk = result[result["rlnHelicalTubeID"] == 0]
        track_vals = sorted(chunk["rlnHelicalTrackLengthAngst"])
        assert track_vals == [float(i) for i in range(30)]

    def test_errors_without_helical_tube_id(self):
        from helicon.plugins.images2star.breakfilaments import handle
        from helicon.lib.exceptions import HeliconError

        data = pd.DataFrame({"rlnMicrographName": ["a.mrc", "b.mrc"]})
        with pytest.raises(HeliconError, match="rlnHelicalTubeID"):
            handle(
                data, self._make_args(), self._make_index_d(), param="maxSegments=50"
            )

    def test_errors_with_bad_max_segments(self):
        from helicon.plugins.images2star.breakfilaments import handle
        from helicon.lib.exceptions import HeliconError

        data = self._make_data()
        with pytest.raises(HeliconError, match="maxSegments"):
            handle(data, self._make_args(), self._make_index_d(), param="maxSegments=0")

    def test_registered_in_argparse(self):
        parser = argparse.ArgumentParser()
        images2star.add_args(parser)
        args = parser.parse_args(
            ["in.star", "out.star", "--breakFilaments", "maxSegments=40"]
        )
        assert args.breakFilaments == ["maxSegments=40"]

    def test_default_max_segments_is_50(self):
        from helicon.plugins.images2star.breakfilaments import handle

        data = self._make_data(
            n_micrographs=1, segments_per_filament=120, n_filaments=1
        )
        # No explicit maxSegments defaults to 50 -> 3 groups (50+50+20)
        result, idx = handle(
            data, self._make_args(), self._make_index_d(), param="anything"
        )

        assert result["rlnHelicalTubeID"].nunique() == 3
