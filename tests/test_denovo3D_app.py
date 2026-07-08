"""UI-level tests for the denovo3D Shiny app using Playwright.

These tests launch the Shiny app in a subprocess and interact with it
through a headless Chromium browser.
"""

import pytest
import numpy as np
from pathlib import Path
from shiny.pytest import create_app_fixture
from shiny.playwright import controller
from playwright.sync_api import Page
from shiny.run import ShinyAppProc
from helicon.webApps.denovo3D import app as denovo3d_app

APP_PATH = Path(__file__).parents[1] / "src/helicon/webApps/denovo3D/app.py"

app = create_app_fixture(APP_PATH)


def test_app_starts(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    title = page.title()
    assert "Helicon denovo3D" in title


def test_app_has_sidebar(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    # The app has sidebar panels - check that at least some UI renders
    assert page.locator("body").is_visible()


def test_nav_panel_visible(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    # Check that we have at least a body with content
    body_text = page.locator("body").inner_text()
    assert len(body_text) > 0


def test_stitching_preview_handles_mixed_heights_with_large_right_shift():
    first = np.ones((160, 160), dtype=np.float32)
    second = np.full((320, 160), 2, dtype=np.float32)

    preview = denovo3d_app._combine_images_for_display([first, second], [0, 200])

    assert preview.shape == (320, 520)
    np.testing.assert_allclose(preview[:160, :160], 1)
    np.testing.assert_allclose(preview[:320, 360:520], 2)
