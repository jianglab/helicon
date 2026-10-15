"""``helicon --help`` and ``import helicon`` stay quick: nothing heavy is imported for them.

A budget in seconds would fail on a busy machine; what makes them slow is importing the heavy
packages, so that is what is checked, in a process of its own.
"""

import json
import subprocess
import sys

HEAVY = (
    "numba",
    "pandas",
    "scipy",
    "matplotlib",
    "sklearn",
    "torch",
    "napari",
    "shiny",
    "streamlit",
    "plotly",
    "bokeh",
    "PySide6",
    "h5py",
    "skimage",
    "mrcfile",
    "starfile",
)


def _loaded(code: str) -> list[str]:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            code
            + f"\nimport json, sys; print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_importing_helicon_imports_nothing_heavy():
    assert _loaded("import helicon") == []


def test_the_help_of_the_command_imports_nothing_heavy():
    code = (
        "import sys\n"
        "sys.argv = ['helicon', '--help']\n"
        "from helicon.helicon import main\n"
        "try:\n"
        "    main()\n"
        "except SystemExit:\n"
        "    pass\n"
    )
    assert _loaded(code) == []


def test_asking_whether_streamlit_is_there_does_not_import_it():
    assert _loaded("import helicon; helicon.has_streamlit(); helicon.has_shiny()") == []
