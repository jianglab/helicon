from __future__ import annotations

import logging, sys, os, time, datetime, threading
from pathlib import Path
from typing import Any, Optional, List

logger = logging.getLogger(__name__)

__all__ = [
    "setup_cache_dir",
    "import_with_auto_install",
    "DummyMemory",
    "cache",
    "set_cache_dir_limit",
]


def setup_cache_dir() -> Path:
    """Set up and return a writable cache directory.

    Checks the HELION_CACHE_DIR environment variable first, then
    /fast-scratch, and falls back to ~/.cache/helicon or a temp directory.

    Returns
    -------
    Path
        Path to the cache directory.
    """
    import getpass, tempfile

    if "HELION_CACHE_DIR" in os.environ:
        cache_dir = Path(os.getenv("HELION_CACHE_DIR"))
    elif Path("/fast-scratch").exists():
        cache_dir = Path("/fast-scratch") / getpass.getuser() / "helicon_cache"
    else:
        cache_dir = Path.home() / ".cache" / "helicon"

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_dir = Path(tempfile.gettempdir()) / getpass.getuser() / "helicon_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def import_with_auto_install(
    packages: str | list[str], scope: dict | None = None
) -> None:
    """Import one or more packages, with a helpful error if not found.

    Package names may include a colon to specify a pip name different from
    the import name (e.g. ``"sklearn:scikit-learn"``).

    Parameters
    ----------
    packages : str or list of str
        Package name(s) to import.
    scope : dict, optional
        Namespace to inject the imported module into.
        Defaults to the caller's local scope.

    Raises
    ------
    ImportError
        With a message suggesting the pip install command.
    """
    scope = scope or {}
    if isinstance(packages, str):
        packages = [packages]
    for package in packages:
        if ":" in package:
            package_import_name, package_pip_name = package.split(":")
        else:
            package_import_name, package_pip_name = package, package
        try:
            scope[package_import_name] = __import__(package_import_name)
        except ImportError:
            raise ImportError(
                f"Package '{package_pip_name}' is required but not installed.\n"
                f"  pip install {package_pip_name}"
            )


class DummyMemory:
    """Dummy joblib.Memory"""

    def __init__(
        self, location: str | None = None, bytes_limit: int = -1, verbose: int = 0
    ) -> None:
        """Initialize a dummy cache that does not persist results.

        Parameters
        ----------
        location : str, optional
            Ignored; kept for API compatibility.
        bytes_limit : int, optional
            Ignored; kept for API compatibility.
        verbose : int, optional
            Ignored; kept for API compatibility.
        """
        self.location = location
        self.verbose = verbose

    def cache(self, func: Any = None, **kwargs: Any) -> Any:
        """Return a decorator that calls the decorated function without caching.

        Parameters
        ----------
        func : callable, optional
            Function to decorate.
        **kwargs :
            Ignored; kept for API compatibility.

        Returns
        -------
        callable
            Decorated function or a decorator.
        """

        def decorator(f):
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        if func is None:
            return decorator
        else:
            return decorator(func)


# How often a cache directory is swept for expired entries.  The sweep walks
# the whole directory, so it is throttled: once per process, and no more often
# than this across processes (tracked by a stamp file inside the directory).
_PRUNE_INTERVAL = datetime.timedelta(days=1)
# A size cap needs enforcing far more often than an age limit: expiry only has
# to catch up with a week-long TTL, whereas a directory under a cap can blow
# past it in hours (denovo3D reached 7.6 GB in an afternoon).  Directories with
# a cap are therefore swept on this shorter interval.
_PRUNE_INTERVAL_CAPPED = datetime.timedelta(hours=1)
_PRUNE_STAMP = ".helicon_last_prune"
# cache_dir -> earliest monotonic time it may be swept again.  Checking this
# first keeps the per-call cost to a dict lookup rather than a stat().
_prune_next: dict = {}
_prune_lock = threading.Lock()
# cache_dir -> total size cap, enforced by the same sweep that drops expired
# entries.  The cap belongs to the directory rather than to any one function:
# several cached functions usually share a directory, and joblib's reduce_size
# operates on the directory as a whole.
_dir_bytes_limit: dict = {}


def set_cache_dir_limit(cache_dir: Any, bytes_limit: Any) -> None:
    """Cap the total size of a cache directory.

    Age-based expiry bounds how *stale* an entry may be, not how much disk the
    cache uses: a directory whose entries are all recent can still grow without
    limit.  This cap is enforced by the periodic sweep, whichever cached
    function in the directory happens to trigger it, and joblib discards
    least-recently-used entries first.

    Parameters
    ----------
    cache_dir : str or Path
        Directory the cap applies to.
    bytes_limit : int or str or None
        Size limit, either a number of bytes or a string with a K, M or G
        suffix (e.g. ``"5G"``).  ``None`` removes the cap.
    """
    key = str(cache_dir)
    with _prune_lock:
        if bytes_limit is None:
            _dir_bytes_limit.pop(key, None)
        else:
            _dir_bytes_limit[key] = bytes_limit
        # Drop the in-memory throttle so the next call reconsiders this
        # directory under its new limit.  The on-disk stamp still applies, so
        # the cap takes effect within one sweep interval rather than instantly.
        _prune_next.pop(key, None)


def _prune_expired(memory: Any, cache_dir: Any, age_limit: Any) -> None:
    """Sweep *cache_dir*: drop entries older than *age_limit* and enforce its cap.

    joblib's ``cache_validation_callback`` only recomputes an expired entry
    when that exact key is requested again; entries never requested again stay
    on disk forever, and nothing else reclaims them, so a cache grows without
    bound.  ``Memory.reduce_size`` applies both the age rule and any size cap
    registered by :func:`set_cache_dir_limit` to the whole directory, which is
    what actually frees the space.

    The sweep runs on a daemon thread -- it is pure housekeeping and must not
    delay the call that triggered it -- and is skipped when the directory has
    neither an expiry nor a cap.  Failures are logged and ignored: losing a
    sweep only costs disk space, whereas raising here would break an otherwise
    good cache hit.
    """
    if not hasattr(memory, "reduce_size"):
        return

    key = str(cache_dir)
    bytes_limit = _dir_bytes_limit.get(key)
    if age_limit is None and bytes_limit is None:
        return

    interval = (
        _PRUNE_INTERVAL_CAPPED if bytes_limit is not None else _PRUNE_INTERVAL
    ).total_seconds()

    # Claim the next slot up front, so concurrent calls take the cheap path and
    # only one sweep thread per directory is ever in flight.  The throttle is
    # purely time-based: a "once per process" guard would sweep only at start-up,
    # when nothing has expired yet, and never again in a long-running app.
    with _prune_lock:
        if time.monotonic() < _prune_next.get(key, 0.0):
            return
        _prune_next[key] = time.monotonic() + interval

    stamp = Path(cache_dir) / _PRUNE_STAMP
    try:
        if stamp.exists() and (time.time() - stamp.stat().st_mtime) < interval:
            return  # another process swept this directory recently
    except OSError:
        pass

    def _run():
        try:
            memory.reduce_size(age_limit=age_limit, bytes_limit=bytes_limit)
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.touch()
        except Exception:
            logger.debug("could not prune expired cache in %s", key, exc_info=True)

    threading.Thread(target=_run, name="helicon-cache-prune", daemon=True).start()


def _clear_function_cache(cached_func: Any) -> None:
    """Empty one function's cache entries, leaving the rest of the directory.

    Several functions typically share a cache directory (every
    ``hill_compute`` helper shares ``cache_dir/"hill"``), so the whole-directory
    ``Memory.clear()`` would discard their entries too.
    """
    clear = getattr(cached_func, "clear", None)
    if clear is None:
        return  # DummyMemory: nothing was ever cached
    clear(warn=False)


def cache(
    expires_after=datetime.timedelta(weeks=1),
    cache_dir: Optional[str] = None,
    ignore: Optional[List] = None,
    verbose: int = 0,
):
    """Decorator that caches function results with expiry using joblib.Memory.

    After the period expires, the cache is invalidated and the function is
    recomputed. If ``expires_after`` is None, the cache never expires.

    Expired entries are also swept off disk in the background, at most once a
    day per cache directory; without that, an expired entry is only ever
    overwritten if its exact key is requested again, so keys that fall out of
    use accumulate indefinitely.

    The decorated function gains two methods: ``clear_cache()`` empties just
    that function's entries, and ``clear_cache_dir()`` empties the whole
    directory, which is usually shared with other cached functions.

    Parameters
    ----------
    expires_after : timedelta or None, optional
        Time period to keep cache valid. Defaults to 1 week. If None, cache
        does not expire.
    cache_dir : str, optional
        Directory to store cache files.
    ignore : list, optional
        List of argument names to ignore for cache key.
    verbose : int, optional
        Verbosity level for joblib.Memory. Defaults to 0.

    Examples
    --------
    @cache(expires_after=timedelta(days=3))
    @cache(expires_after=timedelta(hours=12))
    @cache(expires_after=timedelta(weeks=2))
    @cache(expires_after=None)
    """
    import joblib
    import functools

    if isinstance(expires_after, (int, float)):
        expires_after = datetime.timedelta(days=expires_after)
    elif expires_after is not None and not isinstance(
        expires_after, datetime.timedelta
    ):
        raise TypeError(
            "'expires_after' must be a timedelta object, a number of days, or None"
        )

    ignore = ignore or []

    if expires_after is None:
        cache_validation_callback = lambda x: True
    else:
        cache_validation_callback = joblib.memory.expires_after(
            seconds=expires_after.total_seconds()
        )

    if cache_dir is None:
        cache_dir = setup_cache_dir()

    try:
        memory = joblib.Memory(cache_dir, verbose=verbose)
    except Exception:
        logger.warning(
            "cannot create the cache folder %s. Please make sure that you have write permission in the folder (%s)",
            cache_dir,
            str(Path(cache_dir).parent.absolute()),
        )
        memory = DummyMemory()

    def decorator(func):
        cached_func = memory.cache(
            func, ignore=ignore, cache_validation_callback=cache_validation_callback
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _prune_expired(memory, cache_dir, expires_after)
            return cached_func(*args, **kwargs)

        wrapper.clear_cache = lambda: _clear_function_cache(cached_func)
        wrapper.clear_cache_dir = lambda: getattr(memory, "clear", lambda: None)()
        wrapper.get_cache_info = lambda: {
            "cache_dir": cache_dir,
            "cache_period": expires_after,
            "function_name": func.__name__,
        }
        return wrapper

    return decorator
