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
    originals = (
        cache_mod._PRUNE_INTERVAL,
        cache_mod._PRUNE_INTERVAL_CAPPED,
        cache_mod._PRUNE_INTERVAL_FILLING,
    )
    cache_mod._PRUNE_INTERVAL = datetime.timedelta(seconds=0)
    cache_mod._PRUNE_INTERVAL_CAPPED = datetime.timedelta(seconds=0)
    cache_mod._PRUNE_INTERVAL_FILLING = datetime.timedelta(seconds=0)
    cache_mod._prune_next.clear()
    cache_mod._prune_interval.clear()
    cache_mod._dir_bytes_limit.clear()
    yield
    (
        cache_mod._PRUNE_INTERVAL,
        cache_mod._PRUNE_INTERVAL_CAPPED,
        cache_mod._PRUNE_INTERVAL_FILLING,
    ) = originals
    cache_mod._prune_next.clear()
    cache_mod._prune_interval.clear()
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
        cache_mod._prune_interval.clear()
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
        """A cap can be blown through in minutes; a week-long TTL cannot."""
        assert DEFAULT_PRUNE_INTERVAL_CAPPED < DEFAULT_PRUNE_INTERVAL

    def test_the_baseline_interval_is_short_next_to_the_write_rate(self):
        """What a capped directory can reach is the interval times the rate.

        A denovo3D twist search writes about 50 MB per reconstruction and takes
        a few seconds over each, so the interval has to be short enough that a
        burst of them cannot pass unswept. At one hour -- the value this
        started at -- twelve such processes took the directory from 3 GB to
        40 GB against a 5 GB cap between two sweeps, and took the disk to 99%
        full. Overshoot is bounded, not eliminated: this is the bound.
        """
        writes_per_interval = (
            DEFAULT_PRUNE_INTERVAL_CAPPED.total_seconds() / 4.0  # ~4 s each
        )
        assert (
            writes_per_interval * 50e6 < 1e9
        ), "a capped directory can gain more than a gigabyte between sweeps"

    def test_filling_is_swept_sooner_than_baseline(self):
        assert (
            cache_mod._PRUNE_INTERVAL_FILLING.total_seconds()
            < DEFAULT_PRUNE_INTERVAL_CAPPED.total_seconds()
        )
        assert 0 < cache_mod._PRUNE_FILL_FRACTION < 1


class TestSweepAnswersToSize:
    """The interval a sweep asks for next depends on how full it left things.

    Enforcing a size cap on a fixed clock cannot bound a directory: the bound
    is the write rate times the interval, and neither of those is the cap. So a
    directory found near its cap must be looked at again sooner.
    """

    def _make(self, tmp_path):
        @helicon.cache(expires_after=None, cache_dir=tmp_path)
        def f(x):
            return b"x" * 200_000  # 200 KB

        return f

    def _recorded_interval(self, tmp_path):
        return cache_mod._stamp_interval(Path(tmp_path) / cache_mod._PRUNE_STAMP, -1.0)

    def test_a_directory_left_near_its_cap_asks_to_be_swept_sooner(self, tmp_path):
        # A baseline short enough that sweeps actually run during the writes,
        # and a filling interval far from it so the two cannot be confused.
        cache_mod._PRUNE_INTERVAL_CAPPED = datetime.timedelta(seconds=0.05)
        cache_mod._PRUNE_INTERVAL_FILLING = datetime.timedelta(seconds=7)
        f = self._make(tmp_path)
        helicon.set_cache_dir_limit(tmp_path, "2M")
        for i in range(40):  # 8 MB written against a 2 MB cap
            f(i)
            time.sleep(0.02)
        assert _wait_for(lambda: self._recorded_interval(tmp_path) == 7.0), (
            f"at {_dir_size_mb(tmp_path):.2f} MB of a 2 MB cap the sweep still "
            f"asked to wait {self._recorded_interval(tmp_path)}s"
        )
        assert cache_mod._prune_interval[str(tmp_path)] == 7.0

    def test_an_empty_directory_keeps_the_baseline_interval(self, tmp_path):
        cache_mod._PRUNE_INTERVAL_CAPPED = datetime.timedelta(seconds=600)
        cache_mod._PRUNE_INTERVAL_FILLING = datetime.timedelta(seconds=7)
        f = self._make(tmp_path)
        helicon.set_cache_dir_limit(tmp_path, "100M")
        f(0)  # 200 KB against a 100 MB cap
        assert _wait_for(lambda: self._recorded_interval(tmp_path) == 600.0)

    def test_a_process_that_never_swept_honours_the_stamp(self, tmp_path):
        """A fresh worker in a pool must not fall back to the long interval."""
        stamp = Path(tmp_path) / cache_mod._PRUNE_STAMP
        stamp.write_text("7.0")
        assert cache_mod._stamp_interval(stamp, 600.0) == 7.0

    def test_an_unreadable_stamp_falls_back(self, tmp_path):
        stamp = Path(tmp_path) / cache_mod._PRUNE_STAMP
        stamp.write_text("")  # the empty stamp older versions wrote
        assert cache_mod._stamp_interval(stamp, 600.0) == 600.0
        stamp.write_text("not a number")
        assert cache_mod._stamp_interval(stamp, 600.0) == 600.0
        assert cache_mod._stamp_interval(Path(tmp_path) / "absent", 600.0) == 600.0

    def test_changing_the_cap_forgets_the_learned_interval(self, tmp_path):
        cache_mod._prune_interval[str(tmp_path)] = 7.0
        helicon.set_cache_dir_limit(tmp_path, "9M")
        assert str(tmp_path) not in cache_mod._prune_interval

    def test_dir_size_counts_nested_files(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one").write_bytes(b"x" * 1000)
        (tmp_path / "two").write_bytes(b"y" * 500)
        assert cache_mod._dir_size(tmp_path) == 1500


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


class TestEverythingUsesTheOneCacheRoot:
    """Every cache and log location has to come from ``setup_cache_dir``.

    Spelling out ``~/.cache/helicon`` picks one of the four places that function
    may choose and ignores the rest -- HELION_CACHE_DIR and the ``/fast-scratch``
    preference both move the cache, and anything that hardcoded the home path
    stayed behind, which is exactly where a user would not look for it.
    """

    def test_no_source_file_hardcodes_the_home_cache_path(self):
        import helicon.lib.cache as cache_mod

        root = Path(cache_mod.__file__).resolve().parents[2] / "helicon"
        offenders = []
        for path in root.rglob("*.py"):
            if path.name == "cache.py":
                continue  # the resolver itself is allowed to name the default
            text = path.read_text(encoding="utf-8", errors="ignore")
            for n, line in enumerate(text.splitlines(), 1):
                if '".cache"' in line and "#" not in line.split('".cache"')[0]:
                    offenders.append(f"{path.name}:{n}")
        assert not offenders, f"hardcoded cache paths: {offenders}"

    def test_the_denovo3d_log_follows_the_cache_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path / "elsewhere"))
        import importlib

        import helicon.lib.cache as cache_mod

        importlib.reload(cache_mod)
        assert cache_mod.setup_cache_dir() == tmp_path / "elsewhere"

    def test_cached_functions_all_name_a_subfolder(self):
        """One folder per feature under the root, so a user can clear just the
        one they mean -- and so a size cap set on one cannot evict another's."""
        import helicon.lib.cache as cache_mod

        root = Path(cache_mod.__file__).resolve().parents[2] / "helicon"
        bare = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "cache_dir=str(helicon.cache_dir)" in text:
                bare.append(path.name)
        assert not bare, f"cache the root directly: {bare}"


class TestDamagedCacheEntries:
    """A cache entry that cannot be read must not take the caller down with it.

    ``joblib.memory.expires_after`` indexes ``metadata["time"]`` directly, so a
    metadata.json that was truncated -- by a disk filling mid-write, or by
    eviction running alongside a write -- raised KeyError out of the cached
    call. Observed killing nine tasks of a sixty-task denovo3D search.
    """

    def _validate(self, seconds=3600):
        import joblib

        return cache_mod._tolerate_unreadable_metadata(
            joblib.memory.expires_after(seconds=seconds)
        )

    def test_metadata_without_a_time_is_a_miss(self):
        assert self._validate()({}) is False

    def test_metadata_of_the_wrong_shape_is_a_miss(self):
        for junk in (None, "not a dict", [1, 2, 3], {"time": "yesterday"}):
            assert self._validate()(junk) is False

    def test_a_fresh_entry_is_still_valid(self):
        assert self._validate()({"time": time.time()}) is True

    def test_an_expired_entry_is_still_expired(self):
        assert self._validate(seconds=1)({"time": time.time() - 10}) is False

    def test_a_damaged_entry_is_recomputed_rather_than_raising(self, tmp_path):
        """End to end: damage the metadata of a real cached call and make sure
        the next call returns a value instead of raising."""
        calls = []

        @cache_mod.cache(cache_dir=str(tmp_path), expires_after=7, verbose=0)
        def add(a, b):
            calls.append((a, b))
            return a + b

        assert add(2, 3) == 5
        assert len(calls) == 1
        add(2, 3)
        assert len(calls) == 1  # served from cache

        for meta in tmp_path.rglob("metadata.json"):
            meta.write_text("{}")  # what a truncated write leaves behind

        assert add(2, 3) == 5  # recomputed, not raised
        assert len(calls) == 2
