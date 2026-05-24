from __future__ import annotations

import logging, sys, os, time, datetime
from pathlib import Path
from typing import Any, Optional, List

logger = logging.getLogger(__name__)

__all__ = [
    "setup_cache_dir",
    "import_with_auto_install",
    "DummyMemory",
    "cache",
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


def cache(
    expires_after=datetime.timedelta(weeks=1),
    cache_dir: Optional[str] = None,
    ignore: Optional[List] = None,
    verbose: int = 0,
):
    """Decorator that caches function results with expiry using joblib.Memory.

    After the period expires, the cache is invalidated and the function is
    recomputed. If ``expires_after`` is None, the cache never expires.

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
            return cached_func(*args, **kwargs)

        wrapper.clear_cache = lambda: memory.clear()
        wrapper.get_cache_info = lambda: {
            "cache_dir": cache_dir,
            "cache_period": expires_after,
            "function_name": func.__name__,
        }
        return wrapper

    return decorator
