"""The cache root's environment variable, its lazy creation, and the lazy
``helicon.shiny`` attribute."""

import os
import subprocess
import sys

import pytest

from helicon.lib import cache as cache_mod


class TestCacheDirEnvironmentVariable:
    def test_helicon_cache_dir_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HELION_CACHE_DIR", raising=False)
        monkeypatch.setenv("HELICON_CACHE_DIR", str(tmp_path / "a"))
        assert cache_mod.setup_cache_dir() == tmp_path / "a"

    def test_old_misspelling_still_works(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HELICON_CACHE_DIR", raising=False)
        monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path / "old"))
        assert cache_mod.setup_cache_dir() == tmp_path / "old"

    def test_correct_spelling_wins_over_the_old_one(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HELICON_CACHE_DIR", str(tmp_path / "new"))
        monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path / "old"))
        assert cache_mod.setup_cache_dir() == tmp_path / "new"


class TestLazyCacheDirCreation:
    def test_create_false_does_not_make_the_directory(self, tmp_path, monkeypatch):
        target = tmp_path / "not" / "yet"
        monkeypatch.setenv("HELICON_CACHE_DIR", str(target))
        assert cache_mod.setup_cache_dir(create=False) == target
        assert not target.exists()

    def test_create_true_makes_the_directory(self, tmp_path, monkeypatch):
        target = tmp_path / "made"
        monkeypatch.setenv("HELICON_CACHE_DIR", str(target))
        assert cache_mod.setup_cache_dir() == target
        assert target.is_dir()

    def test_unwritable_location_falls_back_without_creating(
        self, tmp_path, monkeypatch
    ):
        blocker = tmp_path / "a_file"
        blocker.write_text("x")
        monkeypatch.setenv("HELICON_CACHE_DIR", str(blocker / "sub"))
        chosen = cache_mod.setup_cache_dir(create=False)
        assert chosen != blocker / "sub"
        assert chosen.name == "helicon_cache"

    def test_can_create_dir(self, tmp_path):
        assert cache_mod._can_create_dir(tmp_path)
        assert cache_mod._can_create_dir(tmp_path / "x" / "y")
        f = tmp_path / "f"
        f.write_text("")
        assert not cache_mod._can_create_dir(f)
        assert not cache_mod._can_create_dir(f / "child")

    def test_cache_decorator_creates_the_folder_on_use(self, tmp_path):
        target = tmp_path / "lazy_root" / "feature"

        @cache_mod.cache(cache_dir=str(target), expires_after=None)
        def double(x):
            return 2 * x

        assert double(3) == 6
        assert target.is_dir()

    def test_import_helicon_does_not_create_the_cache_root(self, tmp_path):
        target = tmp_path / "never_made"
        code = (
            "import os, helicon; from pathlib import Path; "
            "assert helicon.cache_dir == Path(os.environ['HELICON_CACHE_DIR']); "
            "assert not helicon.cache_dir.exists()"
        )
        env = {**os.environ, "HELICON_CACHE_DIR": str(target)}
        env.pop("HELION_CACHE_DIR", None)
        subprocess.run([sys.executable, "-c", code], env=env, check=True)


class TestLazyShinyAttribute:
    def test_import_helicon_does_not_import_shiny(self):
        code = (
            "import sys, helicon; "
            "assert 'helicon.lib.shiny' not in sys.modules; "
            "assert 'shiny' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_helicon_shiny_resolves_on_first_use(self):
        pytest.importorskip("shiny")
        import helicon
        import helicon.lib.shiny as lib_shiny

        assert helicon.shiny is lib_shiny
        assert callable(helicon.shiny.slider)
        from helicon import shiny

        assert shiny is lib_shiny

    def test_unknown_attribute_still_raises(self):
        import helicon

        with pytest.raises(AttributeError):
            helicon.no_such_attribute_here
