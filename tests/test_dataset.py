import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from helicon.lib import dataset

class TestDataset(unittest.TestCase):
    @patch('helicon.lib.dataset.get_emd_entries')
    @patch('helicon.lib.dataset.update_helical_parameters_from_curated_table')
    def setUp(self, mock_update, mock_get_entries):
        # Mock the functions that fetch data from the network
        mock_get_entries.return_value = pd.DataFrame({
            'emd_id': ['1234', '5678'],
            'title': ['Test Entry 1', 'Test Entry 2'],
            'method': ['helical', 'singleParticle'],
            'resolution': [3.0, 4.0],
            'pdb': ['1abc', '1def'],
            'rise': [1.0, np.nan],
            'twist': [10.0, np.nan],
            'csym': ['C1', np.nan],
        })
        mock_update.side_effect = lambda df: df

        # Create an instance of the EMDB class
        self.emdb = dataset.EMDB(cache_dir='test_cache')

    def tearDown(self):
        import shutil
        shutil.rmtree('test_cache', ignore_errors=True)

    def test_emdb_initialization(self):
        self.assertEqual(len(self.emdb), 2)
        self.assertIn('1234', self.emdb.emd_ids)
        self.assertIn('5678', self.emdb.emd_ids)

    def test_get_info(self):
        info = self.emdb.get_info('1234')
        self.assertEqual(info.title, 'Test Entry 1')
        self.assertEqual(info.resolution, 3.0)

    def test_helical_structure_ids(self):
        ids = self.emdb.helical_structure_ids()
        self.assertEqual(ids, ['1234'])

    @patch('helicon.lib.dataset.get_amyloid_atlas')
    def test_amyloid_atlas_ids(self, mock_get_amyloid_atlas):
        mock_get_amyloid_atlas.return_value = pd.DataFrame({
            'emd_id': ['EMD-1234', 'EMD-9999'],
        })
        ids = self.emdb.amyloid_atlas_ids()
        self.assertEqual(ids, ['1234'])

    @patch('helicon.download_file_from_url')
    def test_get_emdb_map_file(self, mock_download):
        mock_download.return_value = 'test_cache/emd_1234.map.gz'
        map_file = self.emdb.get_emdb_map_file('1234')
        self.assertEqual(str(map_file), 'test_cache/emd_1234.map.gz')
        mock_download.assert_called_once()

    @patch('mrcfile.open')
    @patch('helicon.lib.dataset.EMDB.get_emdb_map_file')
    def test_read_emdb_map(self, mock_get_map_file, mock_mrc_open):
        mock_get_map_file.return_value = 'test_cache/emd_1234.map.gz'

        mock_mrc = MagicMock()
        mock_mrc.voxel_size.x = 1.0
        mock_mrc.header.mapc = 1
        mock_mrc.header.mapr = 2
        mock_mrc.header.maps = 3
        mock_mrc.data = np.zeros((10, 10, 10))

        mock_mrc_open.return_value.__enter__.return_value = mock_mrc

        data, apix = self.emdb.read_emdb_map('1234')
        self.assertEqual(apix, 1.0)
        self.assertEqual(data.shape, (10, 10, 10))

if __name__ == '__main__':
    unittest.main()
