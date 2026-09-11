"""Tests for helicon.cache: expiry, disk pruning, and clear scope."""

import datetime
import time
from pathlib import Path

import pytest

import helicon
from helicon.lib import cache as cache_mod

# Captured before the autouse fixture shortens them for the tests.
DEFAULT_PRUNE_INTERVAL = cache_mod._PRUNE_INTERVAL
DEFAULT_PRUNE_INTERVAL_CAPPED = cache_mod._PRUNE_INTERVAL_CAPPED


def _dir_size_mb(path) -> float:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e6


def _n_entries(path) -> int:
    return len(list(Path(path).rglob("output.pkl")))


def _wait_for(predicate, timeout=10.0, interval=0.1) -> bool:
    """Poll until *predicate* holds; pruning runs on a background thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(autouse=True)
def _reset_prune_throttle():
    """Each test gets a clean throttle table, and a short sweep interval."""
    originals = (cache_mod._PRUNE_INTERVAL, cache_mod._PRUNE_INTERVAL_CAPPED)
    cache_mod._PRUNE_INTERVAL = datetime.timedelta(seconds=0)
    cache_mod._PRUNE_INTERVAL_CAPPED = datetime.timedelta(seconds=0)
    cache_mod._prune_next.clear()
    cache_mod._dir_bytes_limit.clear()
    yield
    cache_mod._PRUNE_INTERVAL, cache_mod._PRUNE_INTERVAL_CAPPED = originals
    cache_mod._prune_next.clear()
    cache_mod._dir_bytes_limit.clear()


class TestExpiry:
    def test_result_is_cached_before_expiry(self, tmp_path):
        calls = {"n": 0}

        @helicon.cache(expires_after=datetime.timedelta(seconds=60), cache_dir=tmp_path)
        def f(x):
            calls["n"] += 1
            return x * 2

        assert f(21) == 42
        assert f(21) == 42
        assert calls["n"] == 1

    def test_result_is_recomputed_after_expiry(self, tmp_path):
        calls = {"n": 0}

        @helicon.cache(expires_after=datetime.timedelta(seconds=1), cache_dir=tmp_path)
        def f(x):
            calls["n"] += 1
            return x * 2

        f(21)
        time.sleep(1.5)
        f(21)
        assert calls["n"] == 2

    def test_expiry_survives_a_new_decoration(self, tmp_path):
        """A restarted process re-decorates; expiry keys off the on-disk write time."""
        calls = {"n": 0}

        def build():
            @helicon.cache(
                expires_after=datetime.timedelta(seconds=1), cache_dir=tmp_path
            )
            def f(x):
                calls["n"] += 1
                return x * 2

            return f

        build()(21)
        assert calls["n"] == 1
        build()(21)  # fresh decoration, entry still valid
        assert calls["n"] == 1
        time.sleep(1.5)
        build()(21)  # fresh decoration, entry now stale
        assert calls["n"] == 2

    def test_none_never_expires(self, tmp_path):
        calls = {"n": 0}

        @helicon.cache(expires_after=None, cache_dir=tmp_path)
        def f(x):
            calls["n"] += 1
            return x

        f(1)
        time.sleep(1.1)
        f(1)
        assert calls["n"] == 1

    def test_int_is_interpreted_as_days(self, tmp_path):
        f = helicon.cache(expires_after=7, cache_dir=tmp_path)(lambda x: x)
        assert f.get_cache_info()["cache_period"] == datetime.timedelta(days=7)

    def test_rejects_other_types(self, tmp_path):
        with pytest.raises(TypeError):
            helicon.cache(expires_after="a week", cache_dir=tmp_path)


class TestPruning:
    def test_expired_entries_are_removed_from_disk(self, tmp_path):
        """Expiry alone only overwrites keys that are requested again."""

        @helicon.cache(expires_after=datetime.timedelta(seconds=1), cache_dir=tmp_path)
        def f(x):
            return b"x" * 50_000

        for i in range(10):
            f(i)
        assert _n_entries(tmp_path) == 10

        time.sleep(1.5)  # every entry is now expired
        cache_mod._prune_next.clear()
        f(0)  # any call triggers the sweep

        # Only the key just refreshed should survive.
        assert _wait_for(lambda: _n_entries(tmp_path) == 1), (
            f"expired entries were not pruned: {_n_entries(tmp_path)} left, "
            f"{_dir_size_mb(tmp_path):.2f} MB"
        )

    def test_unexpired_entries_are_kept(self, tmp_path):
        @helicon.cache(expires_after=datetime.timedelta(seconds=60), cache_dir=tmp_path)
        def f(x):
            return b"x" * 50_000

        for i in range(5):
            f(i)
        f(0)
        time.sleep(0.5)
        assert _n_entries(tmp_path) == 5

    def test_no_pruning_when_cache_never_expires(self, tmp_path):
        @helicon.cache(expires_after=None, cache_dir=tmp_path)
        def f(x):
            return b"x" * 50_000

        for i in range(5):
            f(i)
        f(0)
        time.sleep(0.5)
        assert _n_entries(tmp_path) == 5
        assert not (Path(tmp_path) / cache_mod._PRUNE_STAMP).exists()

    def test_sweep_is_throttled_by_the_stamp_file(self, tmp_path):
        """A recent stamp means another process already swept; skip the walk."""
        cache_mod._PRUNE_INTERVAL = datetime.timedelta(hours=1)

        @helicon.cache(expires_after=datetime.timedelta(seconds=1), cache_dir=tmp_path)
        def f(x):
            return b"x" * 50_000

        for i in range(5):
            f(i)
        stamp = Path(tmp_path) / cache_mod._PRUNE_STAMP
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()

        time.sleep(1.5)
        cache_mod._prune_next.clear()  # in-memory throttle cleared, stamp is fresh
        f(0)
        time.sleep(0.5)
        assert _n_entries(tmp_path) == 5, "swept despite a fresh stamp file"


class TestSizeCap:
    """A cap bounds total disk use, which age-based expiry cannot do."""

    def _make(self, tmp_path, expires_after=None):
        @helicon.cache(expires_after=expires_after, cache_dir=tmp_path)
        def f(x):
            return b"x" * 200_000  # 200 KB

        return f

    def test_cap_evicts_even_when_nothing_has_expired(self, tmp_path):
        f = self._make(tmp_path)  # expires_after=None: no entry is ever stale
        for i in range(30):
            f(i)
        assert _dir_size_mb(tmp_path) > 5

        helicon.set_cache_dir_limit(tmp_path, "2M")
        f(0)
        assert _wait_for(
            lambda: _dir_size_mb(tmp_path) <= 2.5
        ), f"cap not enforced: {_dir_size_mb(tmp_path):.2f} MB"

    def test_no_cap_means_unbounded(self, tmp_path):
        f = self._make(tmp_path)
        for i in range(30):
            f(i)
        f(0)
        time.sleep(0.5)
        assert _dir_size_mb(tmp_path) > 5

    def test_cap_can_be_removed(self, tmp_path):
        f = self._make(tmp_path)
        helicon.set_cache_dir_limit(tmp_path, "2M")
        for i in range(30):
            f(i)
        assert _wait_for(lambda: _dir_size_mb(tmp_path) <= 2.5)

        helicon.set_cache_dir_limit(tmp_path, None)
        for i in range(30, 60):
            f(i)
        time.sleep(0.5)
        assert _dir_size_mb(tmp_path) > 5

    def test_cap_applies_to_the_directory_not_one_function(self, tmp_path):
        """Two functions share a directory; the cap covers their combined size."""
        a = self._make(tmp_path)

        @helicon.cache(expires_after=None, cache_dir=tmp_path)
        def b(x):
            return b"y" * 200_000

        for i in range(15):
            a(i)
            b(i)
        assert _dir_size_mb(tmp_path) > 5

        helicon.set_cache_dir_limit(tmp_path, "2M")
        a(0)
        assert _wait_for(lambda: _dir_size_mb(tmp_path) <= 2.5)

    def test_capped_dirs_sweep_more_often(self):
        """A cap can be blown through in hours; a week-long TTL cannot."""
        assert DEFAULT_PRUNE_INTERVAL_CAPPED < DEFAULT_PRUNE_INTERVAL


class TestDenovo3DCap:
    """denovo3d_pipeline caps its cache directory at import time.

    The autouse fixture clears the registry, so rather than inspecting the
    import side effect these re-run the module's own registration statement.
    """

    def _register(self, limit):
        from helicon.webApps.lib import denovo3d_pipeline as pipeline

        helicon.set_cache_dir_limit(
            helicon.cache_dir / "denovo3D",
            limit if limit not in ("0", "") else None,
        )
        return str(helicon.cache_dir / "denovo3D"), pipeline

    def test_default_limit_is_registered(self):
        key, pipeline = self._register(pipeline_limit := "5G")
        assert cache_mod._dir_bytes_limit[key] == pipeline_limit
        assert pipeline.DENOVO3D_CACHE_LIMIT  # module defines one

    def test_zero_disables_the_cap(self):
        key, _ = self._register("0")
        assert key not in cache_mod._dir_bytes_limit

    def test_module_reads_the_env_override(self, monkeypatch):
        import importlib

        from helicon.webApps.lib import denovo3d_pipeline as pipeline

        monkeypatch.setenv("HELICON_DENOVO3D_CACHE_LIMIT", "12G")
        importlib.reload(pipeline)
        try:
            assert pipeline.DENOVO3D_CACHE_LIMIT == "12G"
            key = str(helicon.cache_dir / "denovo3D")
            assert cache_mod._dir_bytes_limit[key] == "12G"
        finally:
            monkeypatch.delenv("HELICON_DENOVO3D_CACHE_LIMIT", raising=False)
            importlib.reload(pipeline)


class TestClearScope:
    def test_clear_cache_only_clears_that_function(self, tmp_path):
        n = {"a": 0, "b": 0}

        @helicon.cache(expires_after=None, cache_dir=tmp_path)
        def a(x):
            n["a"] += 1
            return x

        @helicon.cache(expires_after=None, cache_dir=tmp_path)
        def b(x):
            n["b"] += 1
            return x

        a(1)
        b(1)
        a.clear_cache()
        a(1)
        b(1)
        assert n == {"a": 2, "b": 1}

    def test_clear_cache_dir_clears_the_whole_directory(self, tmp_path):
        n = {"a": 0, "b": 0}

        @helicon.cache(expires_after=None, cache_dir=tmp_path)
        def a(x):
            n["a"] += 1
            return x

        @helicon.cache(expires_after=None, cache_dir=tmp_path)
        def b(x):
            n["b"] += 1
            return x

        a(1)
        b(1)
        a.clear_cache_dir()
        a(1)
        b(1)
        assert n == {"a": 2, "b": 2}

    def test_clear_cache_is_safe_without_a_usable_cache_dir(self):
        """The DummyMemory fallback has no entries and must not raise."""
        f = helicon.cache(expires_after=1, cache_dir="/proc/nonexistent/helicon")(
            lambda x: x
        )
        f.clear_cache()
        f.clear_cache_dir()
