"""UI-level tests for the denovo3D tab in the unified Helicon Lab app."""

import pytest
import numpy as np
from pathlib import Path
from shiny.pytest import create_app_fixture
from shiny.playwright import controller
from playwright.sync_api import Page
from shiny.run import ShinyAppProc
from helicon.webApps.lib.helical_projection_utils import _combine_images_for_display

APP_PATH = Path(__file__).parents[1] / "src/helicon/webApps/app.py"

app = create_app_fixture(APP_PATH)


def test_app_starts(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    title = page.title()
    assert "Helicon" in title


def test_app_has_body(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    assert page.locator("body").is_visible()
