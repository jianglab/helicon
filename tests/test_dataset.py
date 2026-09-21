import os
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from helicon.lib import dataset


class TestDataset(object):
    def setup_method(self, method):
        # Isolate from EMDB_MIRROR_DIR env var
        self._old_mirror = os.environ.pop("EMDB_MIRROR_DIR", None)
        # Reset EMDB singleton
        from helicon.lib.dataset import EMDB

        EMDB._instance = None

        # Mock the functions that fetch data from the network
        self._mock_get_entries_patcher = patch("helicon.lib.dataset.get_emd_entries")
        self._mock_update_patcher = patch(
            "helicon.lib.dataset.update_helical_parameters_from_curated_table"
        )
        mock_get_entries = self._mock_get_entries_patcher.start()
        mock_update = self._mock_update_patcher.start()
        mock_get_entries.return_value = pd.DataFrame(
            {
                "emd_id": ["1234", "5678"],
                "title": ["Test Entry 1", "Test Entry 2"],
                "method": ["helical", "singleParticle"],
                "resolution": [3.0, 4.0],
                "pdb": ["1abc", "1def"],
                "rise": [1.0, np.nan],
                "twist": [10.0, np.nan],
                "csym": ["C1", np.nan],
            }
        )
        mock_update.side_effect = lambda df: df

        # Create an instance of the EMDB class
        self.emdb = dataset.EMDB(cache_dir="test_cache")

    def teardown_method(self, method):
        self._mock_get_entries_patcher.stop()
        self._mock_update_patcher.stop()
        import shutil

        shutil.rmtree("test_cache", ignore_errors=True)
        # Restore EMDB_MIRROR_DIR
        if self._old_mirror is not None:
            os.environ["EMDB_MIRROR_DIR"] = self._old_mirror

    def test_emdb_initialization(self):
        assert len(self.emdb) == 2
        assert "1234" in self.emdb.emd_ids
        assert "5678" in self.emdb.emd_ids

    def test_get_info(self):
        info = self.emdb.get_info("1234")
        assert info.title == "Test Entry 1"
        assert info.resolution == 3.0

    def test_helical_structure_ids(self):
        ids = self.emdb.helical_structure_ids()
        assert ids == ["1234"]

    @patch("helicon.lib.dataset.get_amyloid_atlas")
    def test_amyloid_atlas_ids(self, mock_get_amyloid_atlas):
        mock_get_amyloid_atlas.return_value = pd.DataFrame(
            {
                "emd_id": ["EMD-1234", "EMD-9999"],
            }
        )
        ids = self.emdb.amyloid_atlas_ids()
        assert ids == ["1234"]

    @patch("helicon.download_file_from_url")
    def test_get_emdb_map_file(self, mock_download):
        mock_download.return_value = "test_cache/emd_1234.map.gz"
        map_file = self.emdb.get_emdb_map_file("1234")
        assert str(map_file) == "test_cache/emd_1234.map.gz"
        mock_download.assert_called_once()

    @patch("mrcfile.open")
    @patch("helicon.lib.dataset.EMDB.get_emdb_map_file")
    def test_read_emdb_map(self, mock_get_map_file, mock_mrc_open):
        mock_get_map_file.return_value = "test_cache/emd_1234.map.gz"

        mock_mrc = MagicMock()
        mock_mrc.voxel_size.x = 1.0
        mock_mrc.header.mapc = 1
        mock_mrc.header.mapr = 2
        mock_mrc.header.maps = 3
        mock_mrc.data = np.zeros((10, 10, 10))

        mock_mrc_open.return_value.__enter__.return_value = mock_mrc

        data, apix = self.emdb.read_emdb_map("1234")
        assert apix == 1.0
        assert data.shape == (10, 10, 10)


class TestContourLevel:
    """EMDB records the level its depositors recommend; reading it is free.

    The XML this parses is already downloaded and cached for other purposes,
    and the level is in the map's own units, which makes it a far better floor
    for "what counts as density" than any fraction of the map's maximum.
    """

    def _emdb(self, xml):
        from helicon.lib.dataset import EMDB

        emdb = EMDB.__new__(EMDB)
        emdb.read_emdb_xml = lambda emd_id: xml
        return emdb

    def test_it_finds_the_level_in_the_nested_metadata(self):
        xml = {
            "map": {
                "contour_list": {
                    "contour": {"level": "28.0", "primary": "true"},
                }
            }
        }
        assert self._emdb(xml).contour_level("emd-1427") == 28.0

    def test_several_contours_give_the_first(self):
        xml = {
            "map": {
                "contour_list": {
                    "contour": [
                        {"level": "0.032", "primary": "true"},
                        {"level": "0.05"},
                    ]
                }
            }
        }
        assert self._emdb(xml).contour_level("emd-0000") == 0.032

    def test_an_entry_without_one_gives_none(self):
        assert (
            self._emdb({"map": {"dimensions": {"col": "256"}}}).contour_level(
                "emd-0000"
            )
            is None
        )

    def test_unreadable_metadata_is_not_an_error(self):
        from helicon.lib.dataset import EMDB

        emdb = EMDB.__new__(EMDB)

        def boom(emd_id):
            raise OSError("no such file")

        emdb.read_emdb_xml = boom
        assert emdb.contour_level("emd-0000") is None
