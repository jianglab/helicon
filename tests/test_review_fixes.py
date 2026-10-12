"""Behaviour fixed after a review of the whole repository."""

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import helicon
from helicon.lib import cache as cache_mod
from helicon.lib import path_utils


class TestPathConversion:
    def _link(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        (real / "a.mrcs").write_bytes(b"x")
        (tmp_path / "link").symlink_to(real, target_is_directory=True)
        return pd.Series([str(tmp_path / "link" / "a.mrcs")])

    def test_absolute_keeps_a_symlink_and_real_follows_it(self, tmp_path):
        names = self._link(tmp_path)
        absolute = path_utils.convert_file_path(names, "absolute")[0]
        real = path_utils.convert_file_path(names, "real")[0]
        assert absolute == str(tmp_path / "link" / "a.mrcs")
        assert real == str((tmp_path / "real" / "a.mrcs").resolve())

    def test_absolute_folds_dots_without_following_links(self, tmp_path):
        self._link(tmp_path)
        sub = tmp_path / "link" / ".." / "link" / "a.mrcs"
        out = path_utils.convert_file_path(pd.Series([str(sub)]), "abs")[0]
        assert out == str(tmp_path / "link" / "a.mrcs")

    def test_a_relative_path_is_made_absolute_from_the_current_directory(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "a.mrcs").write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        out = path_utils.convert_file_path(pd.Series(["./a.mrcs"]), "abs")[0]
        assert out == str(tmp_path / "a.mrcs")

    def test_an_unknown_format_is_a_value_error_not_an_assert(self):
        with pytest.raises(ValueError, match="relativ"):
            path_utils.convert_file_path(pd.Series(["a"]), "relativ")

    def test_a_list_is_a_type_error(self):
        with pytest.raises(TypeError):
            path_utils.convert_file_path(["a"], "abs")


class TestLexicalAbsolute:
    def test_dots_are_folded_and_the_root_is_kept(self):
        assert path_utils._lexical_absolute(Path("/a/b/../c/./d")) == Path("/a/c/d")

    def test_going_above_the_root_stays_at_the_root(self):
        assert path_utils._lexical_absolute(Path("/../a")) == Path("/a")


class TestLogCommandLine:
    def test_it_returns_0_and_appends(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert helicon.log_command_line() == 0
        assert helicon.log_command_line() == 0
        lines = (tmp_path / ".helicon.txt").read_text().splitlines()
        assert len(lines) == 2 and all(" ".join(sys.argv) in l for l in lines)

    def test_it_returns_minus_1_when_the_file_cannot_be_written(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".helicon.txt").mkdir()  # a directory cannot be opened for writing
        assert helicon.log_command_line() == -1


class TestStarWriter:
    def _frame(self, **extra):
        return pd.DataFrame({"rlnImageName": ["1@a.mrcs", "2@a.mrcs"], **extra})

    def test_a_path_is_written_and_closed(self, tmp_path):
        out = tmp_path / "p.star"
        helicon.dataframe2star(self._frame(), str(out))
        text = out.read_text()
        assert "data_particles" in text and "1@a.mrcs" in text

    def test_a_file_object_is_left_open(self):
        buffer = io.StringIO()
        helicon.dataframe2star(self._frame(), buffer)
        assert not buffer.closed and "data_particles" in buffer.getvalue()

    def test_micrograph_rows_are_a_micrographs_block(self, tmp_path):
        out = tmp_path / "m.star"
        frame = pd.DataFrame({"rlnMicrographName": ["a.mrc", "b.mrc"]})
        helicon.dataframe2star(frame, str(out))
        assert "data_micrographs" in out.read_text()


class TestCacheSize:
    def test_a_symlink_is_not_counted_at_the_size_of_its_target(self, tmp_path):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"x" * 100_000)
        cache_dir = tmp_path / "cache"
        (cache_dir / "sub").mkdir(parents=True)
        (cache_dir / "a.bin").write_bytes(b"x" * 10)
        (cache_dir / "sub" / "b.bin").write_bytes(b"x" * 20)
        (cache_dir / "link.bin").symlink_to(outside)
        (cache_dir / "linkdir").symlink_to(tmp_path, target_is_directory=True)
        assert cache_mod._dir_size(cache_dir) == 30

    def test_an_empty_directory_is_0(self, tmp_path):
        assert cache_mod._dir_size(tmp_path) == 0


class TestPluginsStayLight:
    def test_importing_the_images2star_plugins_does_not_import_matplotlib(self):
        code = (
            "import sys; import helicon.plugins.images2star; "
            "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
        )
        assert subprocess.run([sys.executable, "-c", code]).returncode == 0

    def test_a_plugin_that_cannot_be_imported_is_reported(self, caplog, monkeypatch):
        import importlib

        from helicon.plugins import images2star as plugins

        real_import = importlib.import_module

        def fake(name, package=None):
            if name == ".minstack":
                raise ImportError("no such dependency")
            return real_import(name, package)

        monkeypatch.setattr(importlib, "import_module", fake)
        with caplog.at_level("WARNING"):
            found = plugins._discover_plugins()
        assert "minstack" in caplog.text and "no such dependency" in caplog.text
        assert "minStack" not in found and len(found) > 10


class TestSetParm:
    def _handle(self, data, *param):
        from types import SimpleNamespace

        from helicon.plugins.images2star import setparm

        index_d = {setparm.option_name: 0}
        return setparm.handle(data, SimpleNamespace(), index_d, list(param))

    def _data(self, optics=True):
        data = pd.DataFrame({"rlnImageName": ["1@a.mrcs"]})
        if optics:
            data.attrs["optics"] = pd.DataFrame(
                {"rlnOpticsGroup": [1], "rlnVoltage": [300.0]}
            )
        return data

    def test_an_optics_parameter_goes_to_the_optics_group(self):
        data, _ = self._handle(self._data(), "rlnVoltage", "200")
        assert data.attrs["optics"]["rlnVoltage"][0] == 200
        assert "rlnVoltage" not in data

    def test_without_an_optics_block_it_goes_to_the_particles(self):
        data, _ = self._handle(self._data(optics=False), "rlnVoltage", "200")
        assert data["rlnVoltage"][0] == 200

    def test_an_ordinary_parameter_goes_to_the_particles(self):
        data, _ = self._handle(self._data(), "rlnDefocusU", "15000")
        assert data["rlnDefocusU"][0] == 15000


class TestTraceBackHistory:
    def test_no_shared_default_list(self):
        import inspect

        from helicon.plugins.images2star import recoverfullfilaments

        source = inspect.getsource(recoverfullfilaments)
        assert "history=[]" not in source
