from __future__ import annotations

import sys, os, datetime, logging
from pathlib import Path

__all__ = [
    "color_print",
    "getLogger",
    "log_command_line",
    "get_context_function_name",
    "timedelta2string",
    "Timer",
]


def color_print(*args, **kargs) -> None:
    """Print text with color using the rich library.

    Parameters
    ----------
    *args :
        Values to print.
    **kargs :
        Keyword arguments. Supports ``color`` (str, default "red")
        and ``end`` (str, default "\\n").
    """
    color = "red"
    if "color" in kargs:
        color = str(kargs["color"]).lower()
        kargs.pop("color")
    end = "\n"
    if "end" in kargs:
        end = kargs["end"]
        kargs.pop("end")
    from rich.console import Console

    console = Console()
    console.print(*args, style=color, end=end, **kargs)


def getLogger(logfile: str = "", verbose: int = 0) -> logging.Logger:
    """Create and configure a logger with file and stream handlers.

    Parameters
    ----------
    logfile : str, optional
        Path to the log file. If empty, derives a
        name from the script name with a ``.log`` extension.
    verbose : int, optional
        Verbosity level (0=errors only,
        1=warnings, 2=info, 3+=debug). Defaults to 0.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    import logging

    if not logfile:
        logfile = Path(sys.argv[0]).stem + ".log"

    logger = logging.getLogger(logfile)
    logger.setLevel(logging.DEBUG)

    # save to the log file (plain text, no color)
    fh = logging.FileHandler(logfile, mode="at")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

    # print to screen (colored by level)
    from rich.logging import RichHandler

    ch = RichHandler(
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    if verbose <= 0:
        ch.setLevel(logging.ERROR)
    elif verbose == 1:
        ch.setLevel(logging.WARNING)
    elif verbose == 2:
        ch.setLevel(logging.INFO)
    elif verbose > 2:
        ch.setLevel(logging.DEBUG)

    logger.addHandler(ch)
    logger.addHandler(fh)
    if Path(logfile).stat().st_size > 0:
        logger.info("%s" % ("#" * 128))
    return logger


def log_command_line() -> int:
    """Append the current command line and timestamp to ``.helicon.txt``.

    Returns
    -------
    int
        0 on success, -1 on failure.
    """
    try:
        hist = open(".helicon.txt", "r+")
        hist.seek(0, os.SEEK_END)
    except OSError:
        try:
            hist = open(".helicon.txt", "w")
        except OSError:
            return -1
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{current_time}: {' '.join(sys.argv)}\n"
    hist.write(msg)
    hist.close()


def get_context_function_name() -> str:
    """Return the name of the function that called this one.

    Returns
    -------
    str
        Calling function name.
    """
    import inspect

    return inspect.stack()[1].function


def timedelta2string(total_seconds: float | int) -> str:
    """Convert a duration in seconds to a human-readable string.

    Parameters
    ----------
    total_seconds : int or float
        Duration in seconds.

    Returns
    -------
    str
        Human-readable string (e.g. ``"2 hours, 30 minutes"``).
    """
    years = int(total_seconds // (60 * 60 * 24 * 365))
    tmp = total_seconds - years * (60 * 60 * 2 * 365)
    days = int(tmp // (60 * 60 * 24))
    tmp -= days * (60 * 60 * 24)
    hours = int(tmp // (60 * 60))
    tmp -= hours * (60 * 60)
    minutes = int(tmp // 60)
    seconds = int(tmp - minutes * 60 + 0.5)

    s = []
    if years:
        s += [f"{years} years"]
    if days:
        s += [f"{days} days"]
    if hours:
        s += [f"{hours} hours"]
    if minutes:
        s += [f"{minutes} minutes"]
    if seconds:
        s += [f"{seconds} seconds"]
    return ", ".join(s)


class Timer:
    """Context manager for timing code blocks.

    Logs start/end times and duration upon entering and exiting.

    Parameters
    ----------
    info : str, optional
        Label for the timer. Defaults to ``"Timer"``.
    verbose : int, optional
        If non-zero, log timing info (via logger or print).
        Defaults to 1.
    logger : logging.Logger, optional
        Logger to write timing messages to. Uses ``print()`` if None.
    """

    def __init__(self, info="Timer", verbose=1, logger=None):
        self.info = info
        self.verbose = verbose
        self.logger = logger

    def _log(self, msg):
        if self.logger is not None:
            self.logger.debug(msg)
        elif self.verbose:
            print(msg)

    def __enter__(self):
        """Start the timer and optionally log a start message.

        Returns
        -------
        Timer
            The timer instance for use in a ``with`` block.
        """
        from timeit import default_timer

        self.start = default_timer()
        if self.verbose:
            self._log(f"{self.info}: started at {datetime.datetime.now()}")
        return self

    def __exit__(self, *args):
        """Stop the timer and optionally log the elapsed time."""
        from timeit import default_timer

        self.end = default_timer()
        self.interval = self.end - self.start
        if self.verbose:
            self._log(
                f"{self.info}: ended at {datetime.datetime.now()}, duration={self.interval} seconds"
            )
