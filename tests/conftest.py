import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def star_df():
    """A small DataFrame with RELION-style columns suitable for STAR I/O tests."""
    return pd.DataFrame(
        {
            "rlnVoltage": [300.0, 300.0],
            "rlnDefocusU": [10000, 11000],
            "rlnDefocusV": [10000, 11000],
            "rlnDefocusAngle": [0.0, 0.0],
            "rlnSphericalAberration": [2.7, 2.7],
            "rlnAmplitudeContrast": [0.1, 0.1],
            "rlnImageName": ["001@test.mrcs", "002@test.mrcs"],
            "rlnMicrographOriginalPixelSize": [1.0, 1.0],
            "rlnImageSize": [64, 64],
            "rlnImagePixelSize": [1.0, 1.0],
        }
    )


@pytest.fixture
def cs_array():
    """A numpy structured array with CryoSPARC-style fields."""
    return np.array(
        [
            (300.0, 10000, 10000, 0.0, 2.7, 0.1, b"/path/to/test.mrc", 0),
            (300.0, 11000, 11000, 0.0, 2.7, 0.1, b"/path/to/test.mrc", 1),
        ],
        dtype=[
            ("ctf/accel_kv", "<f8"),
            ("ctf/df1_A", "<i8"),
            ("ctf/df2_A", "<i8"),
            ("ctf/df_angle_rad", "<f8"),
            ("ctf/cs_mm", "<f8"),
            ("ctf/amp_contrast", "<f8"),
            ("blob/path", "S128"),
            ("blob/idx", "<i8"),
        ],
    )


@pytest.fixture
def star_file(tmp_path, star_df):
    """Write a temporary STAR file and return its path."""
    import starfile

    path = tmp_path / "test.star"
    starfile.write({"particles": star_df}, str(path), overwrite=True)
    return path


@pytest.fixture
def cs_file(tmp_path, cs_array):
    """Write a temporary .cs file and return its path."""
    path = tmp_path / "test.cs"
    np.save(str(path), cs_array)
    return path


@pytest.fixture
def clean_tmp_path(tmp_path):
    """Provide a clean temp directory (pytest's tmp_path already does this)."""
    return tmp_path


# Fixtures of pytest-playwright. Its sync API starts one Playwright per session
# and leaves an asyncio event loop marked as running on the main thread, so any
# later ``asyncio.run()`` -- the HILL tab tests use it -- fails with "cannot be
# called from a running event loop". Running the browser tests last keeps the
# whole suite runnable in one pytest invocation.
_PLAYWRIGHT_FIXTURES = frozenset(
    {
        "playwright",
        "browser_type",
        "browser",
        "context",
        "new_context",
        "page",
        "launch_browser",
    }
)


def _uses_playwright(item) -> bool:
    """Whether a collected test asks for any pytest-playwright fixture."""
    return not _PLAYWRIGHT_FIXTURES.isdisjoint(getattr(item, "fixturenames", ()))


def pytest_collection_modifyitems(config, items):
    """Move the tests that drive a browser to the end, keeping relative order."""
    browser = [item for item in items if _uses_playwright(item)]
    if browser:
        items[:] = [item for item in items if not _uses_playwright(item)] + browser
