"""torch is not imported with helicon, or with the web app.

torch bundles its own OpenMP runtime; a second libomp in a process next to the
one numpy/scipy/numba use segfaults it inside __kmp_fork_barrier (the web app's
HILL tab on macOS, where pip's torch ships libomp.dylib). The Gaussian classes
that need torch are exported lazily instead.
"""

import subprocess
import sys

import pytest


def _modules_after(code):
    out = subprocess.run(
        [sys.executable, "-c", code + "\nimport sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


class TestNoTorch:
    def test_import_helicon(self):
        assert _modules_after("import helicon") == "False"

    def test_import_the_web_app(self):
        assert _modules_after("import helicon.webApps.app") == "False"

    def test_the_gaussian_classes_still_resolve(self):
        pytest.importorskip("torch")
        code = "import helicon; print(helicon.IsotropicGaussian.__name__)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        assert out.returncode == 0 and "IsotropicGaussian" in out.stdout

    def test_an_unknown_name_is_still_an_attribute_error(self):
        import helicon

        with pytest.raises(AttributeError):
            helicon.no_such_thing_here
