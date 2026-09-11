"""Launch RELION independently of Helicon's Python/conda environment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def is_relion_project(folder: str | Path) -> bool:
    return (Path(folder) / "default_pipeline.star").is_file()


def relion_environment(environ: dict[str, str], *, isolated: bool) -> dict[str, str]:
    """Keep desktop/SSH credentials, but rebuild software state in the launcher.

    In particular, do not inherit conda, Python, MPI, loader, Lmod bookkeeping,
    exported shell functions or shell startup hooks in isolated mode.
    """
    if not isolated:
        return dict(environ)
    keep = {
        "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TZ", "TERM", "TMPDIR",
        "DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS", "SSH_AUTH_SOCK", "SSH_AGENT_PID",
        "KRB5CCNAME", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
    }
    env = {
        key: value for key, value in environ.items()
        if key in keep or key.startswith("LC_")
    }
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"
    return env


def launch_relion(folder: str | Path, launcher: str = "", *, startup: str = "clean"):
    """Return (detached Popen, log path) for the selected startup environment.

    The script must initialize its software environment and end in ``exec
    relion``. It receives no interpolated commands or project-supplied code.
    ``clean`` runs a script isolated, or uses the current PATH with no script.
    ``user-shell`` inherits the environment, reads interactive Bash login
    profiles, then sources the script in that shell to retain setup functions.
    With no script in that mode, the initialized shell resolves ``relion``.
    """
    if os.name == "nt":
        raise OSError("Run helicon display inside Linux/WSL to launch RELION.")
    folder = Path(folder).resolve()
    if not is_relion_project(folder):
        raise ValueError(f"No default_pipeline.star file in {folder}")
    if startup not in {"clean", "user-shell"}:
        raise ValueError("RELION startup must be 'clean' or 'user-shell'.")
    env = relion_environment(
        dict(os.environ), isolated=bool(launcher) and startup == "clean"
    )
    script = None
    if launcher:
        script = Path(launcher).expanduser()
        if not script.is_absolute() or not script.is_file():
            raise ValueError("The RELION launcher must be an absolute path to a Bash script.")
    if script is not None or startup == "user-shell":
        bash = shutil.which("bash", path=env.get("PATH", os.defpath))
        if not bash:
            raise FileNotFoundError("Bash is required for the RELION launcher.")
        if startup == "user-shell":
            # Startup files can change directory. Restore the project after
            # startup; pass paths as arguments, never interpolate shell code.
            # Sourcing keeps non-exported module/conda functions available.
            action = 'source "$2"' if script is not None else "exec relion"
            command = [
                bash, "--login", "-i", "-c", f'cd -- "$1" && {action}',
                "helicon-relion", str(folder),
            ]
            if script is not None:
                command.append(str(script))
        else:
            command = [bash, "--noprofile", "--norc", str(script)]
    else:
        executable = shutil.which("relion", path=env.get("PATH", os.defpath))
        if not executable:
            raise FileNotFoundError(
                "RELION was not found on PATH. Use Apps → Configure RELION… "
                "to select a launcher script for your module or conda environment."
            )
        command = [executable]
    fd, log_name = tempfile.mkstemp(prefix="helicon-relion-", suffix=".log")
    log = Path(log_name)
    try:
        with os.fdopen(fd, "wb") as output:
            process = subprocess.Popen(
                command, cwd=folder, env=env, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
    except Exception:
        log.unlink(missing_ok=True)
        raise
    return process, log
