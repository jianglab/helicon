import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from helicon.lib.dataset import EMDB


class TestEMDBMirror(object):
    def setup_method(self, method):
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

    def teardown_method(self, method):
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
        assert xml_file.is_symlink()
        mirror_xml = self.mirror_dir / "structures/EMD-29999/header/emd-29999.xml"
        assert str(xml_file.resolve()) == str(mirror_xml.resolve())

        # Scenario 2: Mirror NOT writable
        # Clear mirror and cache
        if (self.mirror_dir / "structures").exists():
            shutil.rmtree(self.mirror_dir / "structures")
        for f in self.cache_dir.glob("*"):
            f.unlink()

        try:
            os.chmod(self.mirror_dir, 0o555)
            xml_file = emdb.get_emdb_xml_file("29999")
            assert not xml_file.is_symlink()
            assert str(xml_file.parent) == str(self.cache_dir)
        finally:
            os.chmod(self.mirror_dir, 0o775)

        # Scenario 3: File already in cache
        xml_cache = self.cache_dir / "emd_29999.xml"
        with open(xml_cache, "w") as f:
            f.write("dummy content")

        mock_download.reset_mock()
        xml_file = emdb.get_emdb_xml_file("29999")
        assert mock_download.call_count == 0
        assert xml_file == xml_cache


import pytest
import requests

import helicon


class _FakeResponse:
    """A streamed download that can be told to break part-way through."""

    def __init__(self, chunks, fail_after=None):
        self.chunks, self.fail_after = chunks, fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=None):
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise requests.exceptions.ChunkedEncodingError("connection dropped")
            yield chunk


class TestDownloadsAreAtomic:
    """The final name only ever refers to a complete file.

    It used to be opened first and filled in place, so another user of a
    shared mirror -- or another thread fetching the same entry -- could read a
    half-written .map.gz, and an interrupted download left a truncated file
    that every later call trusted.
    """

    def test_a_complete_download_lands_under_its_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            requests, "get", lambda *a, **k: _FakeResponse([b"ab", b"cd"])
        )
        target = tmp_path / "emd_1.map.gz"
        got = helicon.download_file_from_url(
            "https://example.invalid/emd_1.map.gz",
            target_file_name=str(target),
            return_filename=True,
        )
        assert got == str(target)
        assert target.read_bytes() == b"abcd"
        assert list(tmp_path.iterdir()) == [target]

    def test_an_interrupted_download_leaves_nothing_behind(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            requests, "get", lambda *a, **k: _FakeResponse([b"ab", b"cd"], fail_after=1)
        )
        target = tmp_path / "emd_1.map.gz"
        with pytest.raises(IOError):
            helicon.download_file_from_url(
                "https://example.invalid/emd_1.map.gz", target_file_name=str(target)
            )
        assert list(tmp_path.iterdir()) == []

    def test_a_failed_refresh_keeps_the_complete_file(self, tmp_path, monkeypatch):
        target = tmp_path / "emd_1.map.gz"
        target.write_bytes(b"complete")
        monkeypatch.setattr(
            requests, "get", lambda *a, **k: _FakeResponse([b"xx", b"yy"], fail_after=1)
        )
        with pytest.raises(IOError):
            helicon.download_file_from_url(
                "https://example.invalid/emd_1.map.gz", target_file_name=str(target)
            )
        assert target.read_bytes() == b"complete"


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="permission checks do not apply to root",
)
class TestASharedMirror:
    """EMDB_MIRROR_DIR is used ahead of each user's own cache.

    An entry missing from it is added to it when this user can write there --
    at every level of the path, not just the root -- and cached for this user
    alone when they cannot.
    """

    RELPATH = "structures/EMD-29999/map/emd_29999.map.gz"

    def _emdb(self, tmp_path):
        emdb = object.__new__(EMDB)  # bypass the singleton and its network setup
        emdb.emd_ids = ["29999"]
        emdb.cache_dir = tmp_path / "cache"
        emdb.cache_dir.mkdir()
        emdb.local_emdb_mirror = tmp_path / "mirror"
        emdb.local_emdb_mirror.mkdir()
        return emdb

    def _fetch(self, emdb):
        return emdb._get_emdb_file(
            "29999",
            cache_filename="emd_29999.map.gz",
            mirror_relpath=self.RELPATH,
            url_method=lambda e: "https://example.invalid/emd_%s.map.gz" % e,
        )

    def _fake_download(self, calls):
        def download(url, target_file_name=None, return_filename=False):
            calls.append(target_file_name)
            Path(target_file_name).write_bytes(b"map")
            return target_file_name

        return download

    def test_a_missing_entry_is_added_to_the_mirror(self, tmp_path, monkeypatch):
        emdb, calls = self._emdb(tmp_path), []
        monkeypatch.setattr(
            helicon, "download_file_from_url", self._fake_download(calls)
        )
        got = self._fetch(emdb)
        mirrored = emdb.local_emdb_mirror / self.RELPATH
        assert calls == [str(mirrored)]
        assert got.is_symlink() and got.resolve() == mirrored.resolve()

    def test_an_entry_already_there_is_read_in_place(self, tmp_path, monkeypatch):
        emdb, calls = self._emdb(tmp_path), []
        mirrored = emdb.local_emdb_mirror / self.RELPATH
        mirrored.parent.mkdir(parents=True)
        mirrored.write_bytes(b"map")
        os.chmod(emdb.local_emdb_mirror, 0o555)
        monkeypatch.setattr(
            helicon, "download_file_from_url", self._fake_download(calls)
        )
        try:
            got = self._fetch(emdb)
        finally:
            os.chmod(emdb.local_emdb_mirror, 0o755)
        assert calls == []
        assert got.resolve() == mirrored.resolve()

    def test_a_subdirectory_another_user_made_read_only_falls_back(
        self, tmp_path, monkeypatch
    ):
        # the root is writable, but structures/ was created by someone whose
        # umask left it read-only to everyone else
        emdb, calls = self._emdb(tmp_path), []
        structures = emdb.local_emdb_mirror / "structures"
        structures.mkdir()
        os.chmod(structures, 0o555)
        monkeypatch.setattr(
            helicon, "download_file_from_url", self._fake_download(calls)
        )
        try:
            got = self._fetch(emdb)
        finally:
            os.chmod(structures, 0o755)
        assert got == emdb.cache_dir / "emd_29999.map.gz"
        assert not got.is_symlink()
        assert got.read_bytes() == b"map"

    def test_a_read_only_mirror_without_the_entry_falls_back(
        self, tmp_path, monkeypatch
    ):
        emdb, calls = self._emdb(tmp_path), []
        os.chmod(emdb.local_emdb_mirror, 0o555)
        monkeypatch.setattr(
            helicon, "download_file_from_url", self._fake_download(calls)
        )
        try:
            got = self._fetch(emdb)
        finally:
            os.chmod(emdb.local_emdb_mirror, 0o755)
        assert calls == [str(emdb.cache_dir / "emd_29999.map.gz")]
