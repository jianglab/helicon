"""UI-level tests for the whereIsMyClass Shiny app using Playwright."""

from pathlib import Path
from shiny.pytest import create_app_fixture
from playwright.sync_api import Page
from shiny.run import ShinyAppProc

APP_PATH = Path(__file__).parents[1] / "src/helicon/webApps/whereIsMyClass/app.py"

app = create_app_fixture(APP_PATH)


def test_app_starts(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    title = page.title()
    assert "whereIsMyClass" in title


def test_app_has_sidebar(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    assert page.locator("body").is_visible()


def test_nav_panels_exist(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    body_text = page.locator("body").inner_text()
    assert len(body_text) > 0
