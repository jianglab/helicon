import numpy as np
import pandas as pd
from pathlib import Path
from helicon.lib.groups import (
    combine_groups,
    extract_timestamps,
    per_micrograph_ids,
    per_micrograph_mapping,
    sync_group_columns,
)


class TestCombineGroups(object):
    def test_basic_split(self):
        existing = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        new = np.array([1, 1, 2, 2, 1, 1, 2, 2])
        result = combine_groups(existing, new)
        expected = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        np.testing.assert_array_equal(result, expected)

    def test_single_group_split(self):
        existing = np.array([1, 1, 1])
        new = np.array([1, 2, 3])
        result = combine_groups(existing, new)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_no_split(self):
        existing = np.array([1, 1, 2, 2])
        new = np.array([1, 1, 1, 1])
        result = combine_groups(existing, new)
        expected = np.array([1, 1, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_all_same(self):
        existing = np.array([1, 1, 1])
        new = np.array([1, 1, 1])
        result = combine_groups(existing, new)
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_1_indexed(self):
        existing = np.array([1, 1])
        new = np.array([5, 10])
        result = combine_groups(existing, new)
        assert result[0] == 1
        assert result[1] == 2

    def test_large_gap_in_existing_ids(self):
        existing = np.array([10, 10, 20, 20])
        new = np.array([1, 2, 1, 2])
        result = combine_groups(existing, new)
        # (10,1), (10,2), (20,1), (20,2) -> 4 unique pairs
        assert len(set(result)) == 4

    def test_empty(self):
        existing = np.array([], dtype=int)
        new = np.array([], dtype=int)
        result = combine_groups(existing, new)
        assert len(result) == 0


class TestExtractTimestamps(object):
    def test_epu_timestamp(self):
        micrographs = [
            "FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff"
        ]
        result = extract_timestamps(micrographs, "EPU")
        # Expect a reasonable Unix timestamp (within 24h of the displayed time)
        assert result[micrographs[0]] is not None
        assert result[micrographs[0]] != float("inf")

    def test_multiple_micrographs(self):
        micrographs = [
            "FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff",
            "FoilHole_28788144_Data_28764755_47_20240328_192117_fractions.tiff",
        ]
        result = extract_timestamps(micrographs, "EPU")
        assert len(result) == 2
        # Different beam shifts -> different timestamps -> different values
        assert result[micrographs[0]] != result[micrographs[1]]

    def test_serialEM_serial_number_fallback(self):
        micrographs = ["250123_SF0431_00592_1-14_patch_aligned_doseweighted.mrc"]
        result = extract_timestamps(micrographs, "serialEM_embl_heidelberg")
        # No timestamp group in this pattern, falls back to serial_number=00592
        assert abs(result[micrographs[0]] - 592.0) < 1e-7

    def test_unknown_software(self):
        micrographs = ["unknown_file.mrc"]
        result = extract_timestamps(micrographs, "nonexistent")
        assert result[micrographs[0]] == float("inf")

    def test_no_timestamp_no_serial(self):
        micrographs = ["FoilHole_28788144_Data_28764755_46_no_date.fractions.tiff"]
        result = extract_timestamps(micrographs, "EPU")
        # Doesn't match timestamp pattern, no serial_number group
        # Falls to inf
        assert result[micrographs[0]] == float("inf")

    def test_none_software(self):
        micrographs = ["test.mrc"]
        result = extract_timestamps(micrographs, None)
        assert result[micrographs[0]] == float("inf")

    def test_path_with_directory(self):
        micrographs = [
            "/path/to/J298/motioncorrected/010002802325146112512_250123_SF0431_00592_1-14_patch_aligned_doseweighted.mrc"
        ]
        result = extract_timestamps(micrographs, "serialEM_embl_heidelberg")
        # serial_number=00592 extracted from basename
        assert abs(result[micrographs[0]] - 592.0) < 1e-7

    def test_mtime_fallback_returns_mtime(self):
        # __file__ exists on disk, so use_mtime_fallback should return its mtime
        micrographs = [__file__]
        result = extract_timestamps(micrographs, "nonexistent", use_mtime_fallback=True)
        expected = Path(__file__).stat().st_mtime
        assert result[__file__] == expected

    def test_mtime_fallback_false_defaults_to_inf(self):
        micrographs = [__file__]
        result = extract_timestamps(
            micrographs, "nonexistent", use_mtime_fallback=False
        )
        assert result[__file__] == float("inf")

    def test_mtime_fallback_nonexistent_file(self):
        micrographs = ["/nonexistent/path/file.mrc"]
        result = extract_timestamps(micrographs, "nonexistent", use_mtime_fallback=True)
        assert result["/nonexistent/path/file.mrc"] == float("inf")


class TestPerMicrographMapping(object):
    def test_basic(self):
        micrographs = ["a.mrc", "b.mrc", "c.mrc"]
        result = per_micrograph_mapping(micrographs)
        assert result == {"a.mrc": 1, "b.mrc": 2, "c.mrc": 3}

    def test_custom_start_id(self):
        micrographs = ["x.mrc", "y.mrc"]
        result = per_micrograph_mapping(micrographs, start_id=5)
        assert result == {"x.mrc": 5, "y.mrc": 6}

    def test_empty(self):
        result = per_micrograph_mapping([])
        assert result == {}

    def test_preserves_order(self):
        micrographs = ["c.mrc", "a.mrc", "b.mrc"]
        result = per_micrograph_mapping(micrographs)
        assert result["c.mrc"] == 1
        assert result["a.mrc"] == 2
        assert result["b.mrc"] == 3


class TestPerMicrographIds(object):
    """Tests for per_micrograph_ids — used by cryosparc per-micrograph handler."""

    def test_basic(self):
        names = np.array(["a.mrc", "a.mrc", "b.mrc", "c.mrc"])
        result = per_micrograph_ids(names)
        # np.unique sorts: ["a.mrc", "b.mrc", "c.mrc"]
        expected = np.array([1, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_custom_start_id(self):
        names = np.array(["x", "y", "y", "x"])
        result = per_micrograph_ids(names, start_id=5)
        expected = np.array([5, 6, 6, 5])
        np.testing.assert_array_equal(result, expected)

    def test_empty(self):
        names = np.array([], dtype=str)
        result = per_micrograph_ids(names)
        assert len(result) == 0

    def test_single_unique(self):
        names = np.array(["a", "a", "a"])
        result = per_micrograph_ids(names)
        assert (result == 1).all()

    def test_all_unique(self):
        names = np.array(["a", "b", "c", "d"])
        result = per_micrograph_ids(names)
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected)

    def test_consistency_with_mapping(self):
        """per_micrograph_ids and per_micrograph_mapping produce equivalent
        ­group assignments (same unique IDs per micrograph, never zero)."""
        names = np.array(["z", "y", "x", "z", "y", "w"])
        ids_result = per_micrograph_ids(names)
        ids_mapping = per_micrograph_mapping(
            sorted(np.unique(names))
        )  # np.unique sorts
        expected = np.array([ids_mapping[n] for n in names])
        np.testing.assert_array_equal(ids_result, expected)


class TestSyncGroupColumns(object):
    """Tests for sync_group_columns — used by cryosparc handlers."""

    def test_dataframe_syncs_all_matching_columns(self):
        data = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "a_og": [10, 20, 30],
                "b_og": [40, 50, 60],
            }
        )
        sync_group_columns(data, "a", query_str="_og")
        assert data["a_og"].tolist() == [1, 2, 3]
        assert data["b_og"].tolist() == [1, 2, 3]

    def test_dataframe_keeps_primary_unchanged(self):
        data = pd.DataFrame(
            {
                "ctf/exp_group_id": [5, 5, 7],
                "location/exp_group_id": [1, 2, 3],
            }
        )
        sync_group_columns(data, "ctf/exp_group_id")
        assert data["ctf/exp_group_id"].tolist() == [5, 5, 7]
        assert data["location/exp_group_id"].tolist() == [5, 5, 7]

    def test_no_match_does_nothing(self):
        data = pd.DataFrame(
            {
                "x": [1, 2],
                "y": [3, 4],
            }
        )
        sync_group_columns(data, "x")
        assert data["y"].tolist() == [3, 4]

    def test_single_column_does_nothing(self):
        data = pd.DataFrame(
            {
                "a": [1, 2],
            }
        )
        sync_group_columns(data, "a")  # should not raise
        assert data["a"].tolist() == [1, 2]

    def test_custom_query_str(self):
        data = pd.DataFrame(
            {
                "group_id": [1, 2],
                "group_id_old": [10, 20],
                "other": [100, 200],
            }
        )
        sync_group_columns(data, "group_id", query_str="group_id")
        assert data["group_id_old"].tolist() == [1, 2]
        assert data["other"].tolist() == [100, 200]

    def test_empty_dataframe(self):
        data = pd.DataFrame()
        sync_group_columns(data, "x")  # should not raise
        assert len(data.columns) == 0
