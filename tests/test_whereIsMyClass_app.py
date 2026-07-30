"""UI-level tests for the whereIsMyClass tab in the unified Helicon Lab app."""

from pathlib import Path
from shiny.pytest import create_app_fixture
from playwright.sync_api import Page
from shiny.run import ShinyAppProc

APP_PATH = Path(__file__).parents[1] / "src/helicon/webApps/app.py"

app = create_app_fixture(APP_PATH)


def test_app_starts(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    title = page.title()
    assert "Helicon" in title


def test_app_has_body(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    assert page.locator("body").is_visible()
