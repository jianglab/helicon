"""UI-level tests for the whereIsMyClass tab in the unified Helicon app.

These tests exercise the real launch flow used by the "Show in WhereIsMyClass"
command (``helicon.commands.display._launch_whereismyclass``): the app is
served as a module (``helicon.webApps.app:app``) and a bookmark-format URL
(``_inputs_``/``_values_``/``p``) restores the "url" input mode pointing at a
local star file. Clicking Run loads the class data into the helix table and
class gallery; selecting a helix row renders its micrograph with the class
checkboxes populated.
"""

import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
from playwright.sync_api import Page, expect
from shiny.playwright import controller

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


class _WimcServer:
    """Running app server: base URL plus captured stdout (for diagnosis)."""

    def __init__(self, base_url: str, output: list):
        self.base_url = base_url
        self.output = output


@pytest.fixture(scope="module")
def wimc_app():
    """Serve the unified Helicon app the same way the CLI does.

    Mirrors ``launch_shiny_app`` (src/helicon/lib/shiny.py): spawns
    ``shiny run --no-dev-mode --host 127.0.0.1 --port 0 helicon.webApps.app:app``
    and parses the random port from the server output.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "shiny",
            "run",
            "--no-dev-mode",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "helicon.webApps.app:app",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    port = []
    output = []

    def _reader():
        for line in proc.stdout:
            output.append(line)
            match = re.search(r"Uvicorn running on http://[\d.]+:(\d+)", line)
            if match:
                port.append(int(match.group(1)))

    threading.Thread(target=_reader, daemon=True).start()

    deadline = time.time() + 60
    while not port and time.time() < deadline:
        time.sleep(0.1)

    if not port:
        proc.terminate()
        pytest.fail(
            "Shiny app server did not start within 60s.\n--- server output ---\n"
            + "".join(output)
        )

    base_url = f"http://127.0.0.1:{port[0]}/"
    yield _WimcServer(base_url, output)

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def wimc_data(tmp_path_factory):
    """Create a synthetic Class2D job: star params, class averages, micrograph."""
    root = tmp_path_factory.mktemp("wimc_data")
    job_dir = root / "Class2D" / "job123"
    micrograph_dir = root / "Micrographs"
    job_dir.mkdir(parents=True)
    micrograph_dir.mkdir(parents=True)

    # 128x128 micrograph with non-uniform signal (1 A/px)
    y, x = np.mgrid[0:128, 0:128]
    micrograph = (np.sin(x / 8.0) * 100 + np.cos(y / 10.0) * 80).astype(np.float32)
    micrograph_path = micrograph_dir / "mic.mrc"
    with mrcfile.new(str(micrograph_path), data=micrograph, overwrite=True) as mrc:
        mrc.voxel_size = 1.0

    # two 64x64 class averages
    gy, gx = np.mgrid[0:64, 0:64]
    image0 = np.exp(-((gx - 32) ** 2 + (gy - 20) ** 2) / (2 * 6.0**2)).astype(
        np.float32
    )
    image1 = np.exp(-((gx - 30) ** 2 + (gy - 44) ** 2) / (2 * 8.0**2)).astype(
        np.float32
    )
    classes_path = job_dir / "run_it020_classes.mrcs"
    with mrcfile.new(
        str(classes_path), data=np.stack([image0, image1]), overwrite=True
    ) as mrc:
        mrc.voxel_size = 1.0

    # 3 segments: helix 1 has 2 (classes 1 and 2), helix 2 has 1 (class 1)
    star_path = job_dir / "run_it020_data.star"
    df = pd.DataFrame(
        {
            "rlnMicrographName": ["Micrographs/mic.mrc"] * 3,
            "rlnClassNumber": [1, 2, 1],
            "rlnHelicalTubeID": [1, 1, 2],
            "rlnHelicalTrackLengthAngst": [0.0, 50.0, 10.0],
            "rlnCoordinateX": [64.0, 70.0, 60.0],
            "rlnCoordinateY": [64.0, 58.0, 66.0],
            "rlnAnglePsi": [0.0, 90.0, 180.0],
            "rlnImageName": ["1@run_it020_classes.mrcs"] * 3,
        }
    )
    starfile.write({"particles": df}, str(star_path), overwrite=True)

    return {"star": star_path, "classes": classes_path, "micrograph": micrograph_path}


def _bookmark_url(base_url, star_path):
    """Build the bookmark URL the CLI would open for a local star file.

    Uses the real ``_make_bookmark_query`` and encodes it exactly like
    ``launch_shiny_app``: bare keys for empty values, otherwise
    ``urllib.parse.quote(v, safe='')``.
    """
    from helicon.commands.display import _make_bookmark_query

    params = _make_bookmark_query(
        "WhereIsMyClass",
        {"input_mode": "url", "url_star": str(Path(star_path).resolve())},
    )
    parts = []
    for key, value in params.items():
        if value == "":
            parts.append(key)
        else:
            parts.append(f"{key}={urllib.parse.quote(str(value), safe='')}")
    return base_url + "?" + "&".join(parts)


def _open_wimc(page, server, star_path):
    page.goto(_bookmark_url(server.base_url, star_path), timeout=60000)


def _open_wimc_restored(page, server, star_path):
    """Open the app with the bookmark URL and wait for the restore to land.

    The restore messages (``send_input_message`` from ``on_restore``) update the
    inputs client-side; the server only sees the new values after the client
    echoes them back over the websocket. Waiting on the client-side state first
    guarantees the echo precedes any later click on the same connection.
    """
    _open_wimc(page, server, star_path)
    controller.InputRadioButtons(
        page, "where_is_my_class-wimc_input_mode"
    ).expect_selected("url", timeout=30000)
    controller.InputText(page, "where_is_my_class-wimc_url_star").expect_value(
        str(Path(star_path).resolve()), timeout=30000
    )


def test_bookmark_restore_activates_url_mode(page: Page, wimc_app, wimc_data):
    # The radio defaults to "file selector"; a restored bookmark must switch
    # it to "url" and prefill the star path.
    _open_wimc_restored(page, wimc_app, wimc_data["star"])


def test_run_loads_helix_table_and_class_gallery(page: Page, wimc_app, wimc_data):
    _open_wimc_restored(page, wimc_app, wimc_data["star"])
    controller.InputTaskButton(page, "where_is_my_class-wimc_run").click(timeout=30000)

    table = controller.OutputDataFrame(page, "where_is_my_class-wimc_helices_dataframe")
    table.expect_nrow(2, timeout=30000)
    # sorted by length descending: helix 1 has length 50
    expect(table.cell_locator(0, 1)).to_have_text(re.compile(r"^50"))

    gallery_images = page.locator(
        '[id^="where_is_my_class-wimc_classes_gallery_image_"]'
    )
    expect(gallery_images).to_have_count(2, timeout=30000)


def test_select_helix_renders_micrograph_and_class_choices(
    page: Page, wimc_app, wimc_data
):
    _open_wimc_restored(page, wimc_app, wimc_data["star"])
    controller.InputTaskButton(page, "where_is_my_class-wimc_run").click(timeout=30000)

    table = controller.OutputDataFrame(page, "where_is_my_class-wimc_helices_dataframe")
    table.expect_nrow(2, timeout=30000)

    # select the longest helix (first row)
    row = table.loc_body.locator("tr[data-index='0']")
    checkbox = row.locator("input[type='checkbox']")
    if checkbox.count():
        checkbox.check()
    else:
        row.click()

    classes = controller.InputCheckboxGroup(
        page, "where_is_my_class-wimc_marked_helices_classes"
    )
    classes.expect_choices(["1", "2"], timeout=30000)
    classes.expect_selected(["1", "2"], timeout=30000)

    # The class-selection panel must become visible. Shiny for Python
    # namespaces input/output bindings but leaves plain ui.div(id=...)
    # attributes untouched, so the panel's id is NOT namespaced.
    panel = page.locator("#wimc_div_marked_classes")
    expect(panel).to_be_visible(timeout=30000)

    plot = page.locator("#where_is_my_class-wimc_display_micrograph .js-plotly-plot")
    expect(plot).to_have_count(1, timeout=30000)
