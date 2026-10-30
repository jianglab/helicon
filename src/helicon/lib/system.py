from __future__ import annotations

import logging, sys, os
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


__all__ = [
    "get_option_list",
    "parse_param_str",
    "validate_param_dict",
    "has_shiny",
    "has_streamlit",
    "available_cpu",
    "available_memory",
    "omp_get_max_threads",
    "omp_set_num_threads",
    "get_terminal_size",
    "bytes2units",
    "ceil_power_of_10",
    "encode_numpy",
    "encode_PIL_Image",
]


def get_option_list(argv: list[str]) -> list[str]:
    """Extract option names from command-line arguments.

    Parameters
    ----------
    argv : list of str
        Command-line argument list.

    Returns
    -------
    list of str
        Option names (leading ``--`` stripped).
    """
    optionlist = []
    for arg1 in argv:
        if arg1[:2] == "--":
            argname = arg1.split("=")
            optionlist.append(argname[0].lstrip("-"))
    return optionlist


def parse_param_str(param_str: str) -> tuple[str | None, dict[str, Any]]:
    """Parse a parameter string into an optional name and dictionary.

    Format: ``[opt:]a=b:c=d,e`` becomes ``(opt, {'a': b, 'c': 'd,e'})``.

    Parameters
    ----------
    param_str : str
        Parameter string to parse.

    Returns
    -------
    tuple of (str or None, dict)
        Optional name and dictionary of parsed key-value pairs.
    """
    params = param_str.split(":")

    name = None
    d = {}
    for pi, p in enumerate(params):
        try:
            k, v = p.split("=", 1)
            if v.lower() == "true":
                v = 1
            elif v.lower() == "false":
                v = 0
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        if len(v) > 2 and v[0] == '"' and v[-1] == '"':
                            v = v[1:-1]
            d[k] = v
        except ValueError:
            if pi == 0:
                name = p
            else:
                logger.error("failed to parse parameter %s. Ignored", p)
    return (name, d)


def validate_param_dict(
    param: dict[str, Any], param_ref: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate and convert a parameter dict against a reference.

    Parameters
    ----------
    param : dict
        Input parameter dictionary.
    param_ref : dict
        Reference dictionary with default values and types.

    Returns
    -------
    tuple
        (final_param, changed, unsupported)
        - final_param (dict): Parameters with types coerced to match param_ref.
        - changed (dict): Parameters whose values differ from defaults.
        - unsupported (dict): Keys in param that are not in param_ref.
    """
    unsupported = {k: param[k] for k in param if k not in param_ref}
    final_param = {
        k: (type(param_ref[k])(param[k]) if k in param else param_ref[k])
        for k in param_ref
    }
    changed = {k: final_param[k] for k in final_param if final_param[k] != param_ref[k]}
    return final_param, changed, unsupported


def has_shiny() -> bool:
    """Check whether the packages helicon's web apps need are installed.

    Looks the packages up without importing them, which would cost seconds
    on every ``helicon`` command.

    Returns
    -------
    bool
        True if shiny and shinywidgets are installed, False otherwise.
    """
    from importlib.util import find_spec

    return all(find_spec(name) is not None for name in ("shiny", "shinywidgets"))


def has_streamlit() -> bool:
    """Check whether streamlit is installed.

    Looks it up without importing it: importing streamlit took 2.5 s (and pulled in
    plotly) on every ``helicon --help``.

    Returns
    -------
    bool
        True if streamlit is installed, False otherwise.
    """
    from importlib.util import find_spec

    return find_spec("streamlit") is not None


def has_napari() -> bool:
    """Check whether napari is available.

    Returns
    -------
    bool
        True if napari can be imported, False otherwise.
    """
    import importlib.util

    return importlib.util.find_spec("napari") is not None


def _numba_thread_limit() -> int | None:
    """The number of threads numba will run, without importing numba for it.

    Importing numba takes a second or more. numba's own rule is the number of
    CPUs the process may run on (its affinity), unless ``NUMBA_NUM_THREADS`` says
    otherwise; when numba has already been imported its answer is used.

    Returns
    -------
    int or None
        None when numba is not installed, so that it limits nothing.
    """
    import importlib.util
    import sys

    if "numba" in sys.modules:
        return int(sys.modules["numba"].config.NUMBA_NUM_THREADS)
    if importlib.util.find_spec("numba") is None:
        return None
    if os.environ.get("NUMBA_NUM_THREADS"):
        return int(os.environ["NUMBA_NUM_THREADS"])
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS and Windows have no affinity call
        return max(1, os.cpu_count() or 1)


def available_cpu(mem_gb_per_cpu: float | None = None) -> int:
    """Return the number of available CPUs accounting for SLURM, numba, and memory.

    Parameters
    ----------
    mem_gb_per_cpu : float, optional
        If provided, further constrain CPU
        count so that each CPU has at least this many GB of available memory.

    Returns
    -------
    int
        Number of usable CPUs (at least 1).
    """

    if "SLURM_CPUS_ON_NODE" in os.environ:
        cpu = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        import psutil

        cpu = max(1, int(psutil.cpu_count() * (1 - psutil.cpu_percent() / 100)))
    try:
        limit = _numba_thread_limit()
        if limit is not None:
            cpu = min(cpu, limit)
    except Exception:
        pass

    if mem_gb_per_cpu is not None:
        import psutil

        mem = psutil.virtual_memory()
        cpu = min(cpu, int(mem.available / 1024**3 / mem_gb_per_cpu))

    return max(1, cpu)


def available_memory() -> int:
    """Return the bytes of memory this process can still use.

    The node's available memory, capped by the Slurm job's ``--mem`` (less what
    this process uses) and by every memory cgroup the process is in: a job is
    killed at its own limit long before the node runs out. A cgroup's file cache
    counts as free, since it is given back before anything is killed.

    Returns
    -------
    int
        Available memory in bytes.
    """
    from pathlib import Path

    import psutil

    available = psutil.virtual_memory().available
    # Slurm may enforce --mem by watching the job, without a cgroup limit
    if "SLURM_MEM_PER_NODE" in os.environ:
        limit_mb = int(os.environ["SLURM_MEM_PER_NODE"])
    elif "SLURM_MEM_PER_CPU" in os.environ:
        limit_mb = int(os.environ["SLURM_MEM_PER_CPU"]) * int(
            os.environ.get("SLURM_CPUS_ON_NODE", 1)
        )
    else:
        limit_mb = 0
    if limit_mb > 0:
        used = psutil.Process().memory_info().rss
        available = min(available, limit_mb * 2**20 - used)
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return available
    root = Path("/sys/fs/cgroup")
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # cgroup v2: the limit may be on any ancestor
            group, names = root / path.lstrip("/"), ("memory.max", "memory.current")
        elif "memory" in controllers.split(","):  # cgroup v1
            group = root / "memory" / path.lstrip("/")
            names = ("memory.limit_in_bytes", "memory.usage_in_bytes")
        else:
            continue
        while root in group.parents or group == root:
            try:
                limit = (group / names[0]).read_text().strip()
                used = int((group / names[1]).read_text())
                stat = (group / "memory.stat").read_text().split()
                cache = dict(zip(stat[::2], stat[1::2]))
                used -= int(cache.get("file", cache.get("total_cache", 0)))
                if limit != "max" and int(limit) < 2**60:
                    available = min(available, int(limit) - max(used, 0))
            except (OSError, ValueError):
                pass
            if group == root:
                break
            group = group.parent
    return max(0, available)


def _load_omp_library():
    """Load the OpenMP runtime library in a cross-platform way.

    Returns
    -------
    ctypes.CDLL or None
        Loaded OpenMP library, or None if not found.
    """
    import ctypes
    import ctypes.util

    lib_names = []
    if sys.platform == "darwin":
        lib_names = ["libomp.dylib", "libgomp.dylib", "libiomp5.dylib"]
    elif sys.platform == "linux":
        lib_names = ["libgomp.so.1", "libgomp.so"]
    elif sys.platform == "win32":
        lib_names = ["libgomp-1.dll", "libiomp5md.dll"]

    for name in lib_names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            path = ctypes.util.find_library(name)
            if path:
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
    return None


_omp_lib = None


def omp_get_max_threads() -> int:
    """Get the maximum number of OpenMP threads.

    Returns
    -------
    int
        Maximum thread count, or 1 if OpenMP library is unavailable.
    """
    global _omp_lib
    if _omp_lib is None:
        _omp_lib = _load_omp_library()
    if _omp_lib is None:
        return 1
    return _omp_lib.omp_get_max_threads()


def omp_set_num_threads(n: int) -> None:
    """Set the number of OpenMP threads.

    If ``n <= 0``, resets to the maximum thread count.
    No-op if OpenMP library is unavailable.

    Parameters
    ----------
    n : int
        Desired thread count. Non-positive values reset to max.
    """
    global _omp_lib
    if _omp_lib is None:
        _omp_lib = _load_omp_library()
    if _omp_lib is None:
        return
    if n <= 0:
        max_n = omp_get_max_threads()
        _omp_lib.omp_set_num_threads(max_n)
    else:
        _omp_lib.omp_set_num_threads(n)


def get_terminal_size() -> tuple[int, int]:
    """Get the current terminal size.

    Returns
    -------
    tuple
        (rows, columns) of the terminal.
    """
    import shutil

    size = shutil.get_terminal_size()
    return (size.rows, size.columns)


def bytes2units(
    bytes: float | int, to: str | None = None, bsize: int = 1024
) -> tuple[float, str]:
    """Convert a byte count to a human-readable unit string.

    Parameters
    ----------
    bytes : int or float
        Number of bytes.
    to : str, optional
        Target unit (``"k"``, ``"m"``, ``"g"``, ``"t"``,
        ``"p"``, ``"e"``). If None, auto-select the largest suitable unit.
    bsize : int, optional
        Block size (default 1024).

    Returns
    -------
    tuple
        (value, unit_string) such as ``(1.5, "GB")``.
    """
    units = {"k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6}
    unitStr = {"k": "kB", "m": "MB", "g": "GB", "t": "TB", "p": "PB", "e": "EB"}
    if to is None:
        for u in units:
            x = bytes / (bsize ** units[u])
            if x < bsize:
                break
    else:
        u = to
        x = bytes / (bsize ** units[to])
    return (x, unitStr[u])


def ceil_power_of_10(n: float | int) -> int:
    """Round a positive number up to the nearest power of 10.

    Parameters
    ----------
    n : int or float
        Positive number.

    Returns
    -------
    int
        Smallest power of 10 >= n.

    Raises
    ------
    ValueError
        If n < 0.
    """
    if n < 0:
        raise ValueError(f"n={n} while n>0 is required")
    if n <= 1:
        return 10
    from math import ceil, log

    exp = log(n, 10)
    exp = ceil(exp)
    return 10**exp


def encode_numpy(img: np.ndarray, hflip: bool = False, vflip: bool = False) -> str:
    """Encode a numpy array as a base64 JPEG data URL.

    If the array is not uint8, it is normalized to [0, 255].

    Parameters
    ----------
    img : np.ndarray
        Image array.
    hflip : bool, optional
        Horizontal flip. Defaults to False.
    vflip : bool, optional
        Vertical flip. Defaults to False.

    Returns
    -------
    str
        Base64 data URL string.
    """
    if img.dtype != np.dtype("uint8"):
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            tmp = (255 * (img - vmin) / (vmax - vmin)).astype(np.uint8)
        else:
            tmp = np.zeros_like(img, dtype=np.uint8)
    else:
        tmp = img
    if hflip:
        tmp = tmp[:, ::-1]
    if vflip:
        tmp = tmp[::-1, :]
    from PIL import Image

    pil_img = Image.fromarray(tmp)
    return encode_PIL_Image(pil_img)


def encode_PIL_Image(img, hflip: bool = False, vflip: bool = False) -> str:
    """Encode a PIL Image as a base64 JPEG data URL.

    Parameters
    ----------
    img : PIL.Image
        Image to encode.
    hflip : bool, optional
        Horizontal flip. Defaults to False.
    vflip : bool, optional
        Vertical flip. Defaults to False.

    Returns
    -------
    str
        Base64 data URL string.
    """
    import io, base64

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64, {encoded}"
