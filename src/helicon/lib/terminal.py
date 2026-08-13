"""Launching the host OS terminal from the Helicon display app.

The functions here start a native terminal (Terminal.app on macOS, Windows
Terminal/``cmd.exe`` on Windows, a desktop emulator on Linux/WSL) detached
from the Helicon process, and snapshot the running Helicon environment to a
temp file so the spawned shell resolves the same Python that launched the app.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _is_wsl() -> bool:
    """Return True when running inside Windows Subsystem for Linux."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _dbus_name_has_owner(name: str) -> bool:
    """Return True if *name* is currently owned on the session bus."""
    try:
        result = subprocess.run(
            [
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.freedesktop.DBus",
                "--object-path",
                "/org/freedesktop/DBus",
                "--method",
                "org.freedesktop.DBus.NameHasOwner",
                name,
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "(true," in (result.stdout or "").replace(" ", "")


def _gnome_terminal_usable() -> bool:
    """Return True when ``gnome-terminal`` is safe to launch.

    ``gnome-terminal`` talks to a D-Bus service.  On bare SSH/X11-forwarded
    Linux sessions that service is often listed as activatable but fails
    after a long timeout (~120s), which made Apps → Terminal appear to
    do nothing.  Only try it when the service is already up or the desktop
    looks like a real GNOME session that can activate it.
    """
    if not shutil.which("gnome-terminal"):
        return False
    if _dbus_name_has_owner("org.gnome.Terminal"):
        return True
    desktop = (os.environ.get("XDG_CURRENT_DESKTOP") or "").lower()
    session = (os.environ.get("DESKTOP_SESSION") or "").lower()
    return "gnome" in desktop or "gnome" in session


def _resolves_to_gnome_terminal(exe: str) -> bool:
    """Return True if *exe* ultimately invokes gnome-terminal."""
    path = shutil.which(exe)
    if not path:
        return False
    name = Path(path).name.lower()
    if "gnome-terminal" in name:
        return True
    try:
        real = Path(path).resolve().name.lower()
    except OSError:
        return False
    return "gnome-terminal" in real


def _linux_terminal_candidates(target: str) -> list[tuple[str, list[str]]]:
    """Build ordered ``(executable, args)`` launch candidates for Linux.

    Order: ``$TERMINAL`` if set, then lightweight X11 emulators, then
    desktop ones.  ``gnome-terminal`` is included only when
    :func:`_gnome_terminal_usable` is true.  ``x-terminal-emulator`` is
    last because on Debian it often wraps gnome-terminal and inherits the
    same D-Bus hang over SSH/X11.
    """
    shell = os.environ.get("SHELL") or "/bin/sh"
    # xterm/uxterm run an embedded shell command (rather than accepting a
    # working-directory flag), so reproduce the inherited environment here
    # too -- sourcing the shared env snapshot keeps the same Python that
    # launched Helicon (desktop emulators that take ``--working-directory``
    # already inherit it through the subprocess env).
    shell_cd = f"cd {shlex.quote(target)} && {_source_env_command()} ; exec {shell}"
    # Prefer ``--flag=value``: lxterminal (and some others) treat a separate
    # ``--working-directory DIR`` as unknown and print usage then exit 0.
    workdir = [f"--working-directory={target}"]

    candidates: list[tuple[str, list[str]]] = []

    preferred = (os.environ.get("TERMINAL") or "").strip()
    if preferred and shutil.which(preferred):
        candidates.append((preferred, list(workdir)))

    known: list[tuple[str, list[str]]] = [
        ("lxterminal", list(workdir)),
        ("xfce4-terminal", list(workdir)),
        ("mate-terminal", list(workdir)),
        ("konsole", [f"--workdir={target}"]),
        ("tilix", list(workdir)),
        ("terminator", list(workdir)),
        ("kitty", ["--directory", target]),
        ("alacritty", ["--working-directory", target]),
        ("wezterm", ["start", "--cwd", target]),
        ("foot", ["--working-directory", target]),
        ("qterminal", list(workdir)),
        ("xterm", ["-e", shell, "-c", shell_cd]),
        ("uxterm", ["-e", shell, "-c", shell_cd]),
        ("x-terminal-emulator", list(workdir)),
    ]
    if _gnome_terminal_usable():
        known.insert(0, ("gnome-terminal", list(workdir)))
    else:
        # Drop the Debian alternatives wrapper when it points at
        # gnome-terminal; otherwise we mark spawn "success" while the
        # child hangs ~120s on a dead D-Bus activation.
        known = [
            (exe, args)
            for exe, args in known
            if exe != "x-terminal-emulator" or not _resolves_to_gnome_terminal(exe)
        ]

    seen = {c[0] for c in candidates}
    for exe, args in known:
        if exe in seen:
            continue
        if shutil.which(exe):
            candidates.append((exe, args))
            seen.add(exe)
    return candidates


def _open_terminal(folder: str | None = None) -> None:
    """Launch the host OS native terminal, optionally in *folder*.

    The terminal is started detached so it keeps running when Helicon quits.
    Platform-specific launch strategies are tried in order; the first one
    that actually starts wins.  On Linux, candidates that exit immediately
    (missing flags, bad args) are skipped so a working emulator is used.

    The spawned terminal receives the corrected running environment from
    :func:`_terminal_env` so the shell uses the same Python that launched
    Helicon (see that function for why the environment is adjusted rather than
    assuming conda/venv).
    """
    target = folder if folder else os.getcwd()
    system = platform.system()
    env = _terminal_env()

    if system == "Darwin":
        _open_terminal_macos(target, env)
        return

    if system == "Windows":
        # Prefer the modern Windows Terminal, then fall back to cmd.exe.
        if shutil.which("wt"):
            _spawn_detached(["wt", "-d", target], env=env)
            return
        # ``/d`` skips cmd's AutoRun; ``/k`` keeps the window open.
        _spawn_detached(["cmd.exe", "/d", "/k", "cd", "/d", target], env=env)
        return

    if _is_wsl():
        # WSL distros usually lack a working X terminal (xterm breaks under
        # WSLg), so launch the Windows side through interop instead.
        if shutil.which("wt.exe"):
            if _spawn_detached(["wt.exe", "wsl", "--cd", target], env=env):
                return
        if shutil.which("wsl.exe"):
            if _spawn_detached(["wsl.exe", "--cd", target], env=env):
                return

    for exe, args in _linux_terminal_candidates(target):
        if _spawn_detached([exe, *args], check_early_exit=True, env=env):
            return

    shell = os.environ.get("SHELL") or "/bin/sh"
    _spawn_detached(
        [
            shell,
            "-c",
            f"cd {shlex.quote(target)} && {_source_env_command()} ; exec {shell}",
        ],
        env=env,
    )


def _open_terminal_macos(target: str, env: dict) -> None:
    """Launch Terminal.app on macOS with the corrected environment applied.

    GUI apps started through LaunchServices run from the launchd environment,
    so ``open -a Terminal <dir>`` drops the ``env`` passed to the subprocess
    and the shell loses the Helicon Python.  AppleScript ``do script`` can
    inject it, but typing many ``export`` lines would echo visibly.  Instead we
    snapshot the environment to a temp file and source it with a single line:
    the file is read into the already-initialised interactive shell, so rc-files
    (which re-activate conda/virtualenv and would clobber ``PATH``) do not run
    again.  The window is left at a prompt rooted in *target*.
    """
    _write_env_file(env)
    command = f"cd {shlex.quote(target)} && {_source_env_command()}"
    script_dir = Path(tempfile.gettempdir()) / "helicon_terminal"
    script_dir.mkdir(parents=True, exist_ok=True)
    osascript = (
        'tell application "Terminal" to do script ' f'"{_applescript_escape(command)}"'
    )
    _spawn_detached(["osascript", "-e", osascript])


def _applescript_escape(s: str) -> str:
    """Escape *s* for embedding inside a double-quoted AppleScript string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _env_file() -> Path:
    """Return the persistent path of the generated environment snapshot."""
    name = "helicon-terminal-env.bat" if os.name == "nt" else "helicon-terminal-env.sh"
    return Path(tempfile.gettempdir()) / "helicon_terminal" / name


def _env_pairs(env: dict) -> list[tuple[str, str]]:
    """Return ``(name, value)`` pairs captured for the opened terminal."""
    pairs: list[tuple[str, str]] = [("PATH", env["PATH"])]
    if env.get("PYTHONPATH"):
        pairs.append(("PYTHONPATH", env["PYTHONPATH"]))
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        if env.get(key):
            pairs.append((key, env[key]))
    return pairs


def _env_export_lines(env: dict) -> list[str]:
    """Return POSIX ``export`` lines reproducing the corrected environment."""
    return [f"export {k}={shlex.quote(v)}" for k, v in _env_pairs(env)]


def _write_env_file(env: dict) -> Path:
    """Snapshot *env* to a file the opened terminal can source on any platform.

    POSIX shells get ``export`` lines; Windows ``cmd.exe`` gets ``set`` lines.
    Reusing the same temp file each launch keeps a single source/call line.
    """
    path = _env_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        body = "\r\n".join(f'set "{k}={v}"' for k, v in _env_pairs(env)) + "\r\n"
    else:
        body = "\n".join(_env_export_lines(env)) + "\n"
    path.write_text(body)
    return path


def _source_env_command() -> str:
    """Return the platform-appropriate way to apply the persisted env.

    Returns ``source <file>`` for POSIX shells and ``call <file>`` for
    ``cmd.exe``.  The file is written by :func:`_write_env_file`; launching the
    terminal twice is unsupported, so callers always write before sourcing.
    """
    envfile = _env_file()
    if os.name == "nt":
        return f'call "{envfile}"'
    return f"source {shlex.quote(str(envfile))}"


def _spawn_detached(
    cmd: list[str],
    check_early_exit: bool = False,
    env: dict | None = None,
) -> bool:
    """Start ``cmd`` detached so it outlives the Helicon process.

    Parameters
    ----------
    cmd : list of str
        Command and arguments to launch.
    check_early_exit : bool, optional
        If True, wait briefly and treat a non-zero exit as failure so the
        caller can try the next candidate.  Defaults to False.
    env : dict, optional
        Environment for the child process.  Defaults to the current process
        environment.

    Returns
    -------
    bool
        True if the process appears to have started successfully.
    """
    try:
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | (
                subprocess.DETACHED_PROCESS
                if hasattr(subprocess, "DETACHED_PROCESS")
                else 0
            )
        proc = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
            **kwargs,
        )
    except Exception as exc:
        print(f"[helicon] failed to launch terminal: {exc}")
        return False

    if not check_early_exit:
        return True

    try:
        code = proc.wait(timeout=0.4)
    except subprocess.TimeoutExpired:
        return True
    # Exit 0 can mean a healthy hand-off (gnome-terminal → server).
    return code == 0


def _terminal_env() -> dict:
    """Return the running Helicon environment, corrected for the terminal.

    Rather than assuming the user installed Helicon through conda (they may
    have used ``module load``, a venv, or system Python), we derive the
    active interpreter's bin directory from ``sys.prefix`` and prepend it to
    ``PATH``.  That way the spawned shell resolves the same Python that launched
    Helicon regardless of packaging.

    When Helicon is running inside a non-base Python (``sys.prefix !=
    sys.base_prefix``), we also sync helper variables (``VIRTUAL_ENV``,
    ``CONDA_PREFIX``, ``CONDA_DEFAULT_ENV``) so rc-files / shells that respect
    an already-active environment don't try to re-activate base.
    """
    env = os.environ.copy()
    bin_dir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    path = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    path = [str(bin_dir)] + [p for p in path if p != str(bin_dir)]
    env["PATH"] = os.pathsep.join(path)

    if sys.prefix != sys.base_prefix:
        env["VIRTUAL_ENV"] = sys.prefix
        env["CONDA_PREFIX"] = sys.prefix
        env["CONDA_DEFAULT_ENV"] = Path(sys.prefix).name
    return env
