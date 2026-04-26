import unittest
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from helicon.lib.dataset import EMDB


class TestEMDBMirror(unittest.TestCase):
    def setUp(self):
        self.workspace = Path("test_emdb_mirror_workspace").resolve()
        if self.workspace.exists():
            # Ensure workspace is writable before deleting
            for root, dirs, files in os.walk(self.workspace):
                os.chmod(root, 0o775)
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)
        self.cache_dir = self.workspace / "cache"
        self.mirror_dir = self.workspace / "mirror"
        self.cache_dir.mkdir()
        self.mirror_dir.mkdir()

    def tearDown(self):
        if self.workspace.exists():
            # Ensure workspace is writable before deleting
            for root, dirs, files in os.walk(self.workspace):
                os.chmod(root, 0o775)
            shutil.rmtree(self.workspace)

    @patch("helicon.lib.dataset.get_emd_entries")
    @patch("helicon.download_file_from_url")
    def test_mirror_priority_logic(self, mock_download, mock_get_entries):
        # Setup mocks
        mock_get_entries.return_value = pd.DataFrame(
            {"emdb_id": ["EMD-29999"], "emd_id": ["29999"]}
        )

        def download_side_effect(url, target_file_name=None, return_filename=False):
            with open(target_file_name, "w") as f:
                f.write("dummy content")
            return target_file_name if return_filename else None

        mock_download.side_effect = download_side_effect

        # Scenario 1: Mirror writable, file doesn't exist
        os.environ["EMDB_MIRROR_DIR"] = str(self.mirror_dir)
        emdb = EMDB(cache_dir=str(self.cache_dir), use_curated_helical_parameters=False)
        emdb.emd_ids = ["29999"]

        xml_file = emdb.get_emdb_xml_file("29999")
        self.assertTrue(xml_file.is_symlink())
        mirror_xml = self.mirror_dir / "structures/EMD-29999/header/emd-29999.xml"
        self.assertEqual(str(xml_file.resolve()), str(mirror_xml.resolve()))

        # Scenario 2: Mirror NOT writable
        # Clear mirror and cache
        if (self.mirror_dir / "structures").exists():
            shutil.rmtree(self.mirror_dir / "structures")
        for f in self.cache_dir.glob("*"):
            f.unlink()

        try:
            os.chmod(self.mirror_dir, 0o555)
            xml_file = emdb.get_emdb_xml_file("29999")
            self.assertFalse(xml_file.is_symlink())
            self.assertEqual(str(xml_file.parent), str(self.cache_dir))
        finally:
            os.chmod(self.mirror_dir, 0o775)

        # Scenario 3: File already in cache
        xml_cache = self.cache_dir / "emd_29999.xml"
        with open(xml_cache, "w") as f:
            f.write("dummy content")

        mock_download.reset_mock()
        xml_file = emdb.get_emdb_xml_file("29999")
        self.assertEqual(mock_download.call_count, 0)
        self.assertEqual(xml_file, xml_cache)


if __name__ == "__main__":
    unittest.main()
