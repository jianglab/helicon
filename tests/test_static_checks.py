"""The static checks that find real mistakes stay clean (``ruff check``, see pyproject.toml).

They found a name that did not exist in four places that nobody had run: a CryoSPARC ``.csv``
reader that was never written, a ``Path`` that was never imported (curvelet denoising into a
folder), a name used before it is assigned (extracting particles from a micrographs file), and
a name of an ``except`` clause used after it (the message of a tab that cannot be loaded).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _ruff():
    exe = shutil.which("ruff")
    if exe:
        return [exe]
    probe = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"], capture_output=True
    )
    return [sys.executable, "-m", "ruff"] if probe.returncode == 0 else None


def test_ruff_finds_nothing():
    command = _ruff()
    if command is None:
        pytest.skip("ruff is not installed")
    result = subprocess.run(
        [*command, "check", "src", "tests", "--output-format", "concise"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:]


def test_a_cryosparc_csv_file_is_refused_with_a_message_not_a_name_error(tmp_path):
    from helicon.lib.exceptions import HeliconIOError
    from helicon.lib.io import image2dataframe

    path = tmp_path / "particles.csv"
    path.write_text("x\n1\n")
    with pytest.raises(HeliconIOError, match="CryoSPARC v0/v1"):
        image2dataframe(str(path))
