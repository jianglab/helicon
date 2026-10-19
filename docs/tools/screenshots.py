"""Take the screenshots of the user documentation from the current web apps.

    python docs/tools/screenshots.py [name ...]      # all of them, or only the named ones

Starts the web apps on a free port, opens each tab in a headless browser (Playwright:
``pip install playwright && playwright install chromium``) on its own example data (the public
EMPIAR-10940 2D classes, so the machine needs access to the EBI server), waits for the page to
show what the screenshot is about, and writes ``docs/images/<name>.png``. Run it again after a
change to an app, so that the pictures in the docs are the ones a user sees.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parent / "images"
APP = HERE.parent.parent / "src" / "helicon" / "webApps" / "app.py"
VIEWPORT = {"width": 1500, "height": 1000}


def _wait_text(page, text: str, seconds: float = 240.0) -> None:
    """Wait until ``text`` shows on the page."""
    page.get_by_text(text, exact=False).first.wait_for(timeout=seconds * 1000)


def _filled(path: Path) -> float:
    """The share of the main panel (right of the sidebar) that is not its background colour."""
    import numpy as np
    from PIL import Image

    pixels = np.asarray(Image.open(path).convert("RGB"))[60:, 520:].reshape(-1, 3)
    colours, counts = np.unique(pixels, axis=0, return_counts=True)
    return 1.0 - counts.max() / len(pixels)


def _settle(page, seconds: float = 3.0) -> None:
    """Let the plots finish drawing after the text that tells they are coming."""
    page.wait_for_load_state("networkidle")
    time.sleep(seconds)


def home(page, url):
    page.goto(f"{url}?tab=Home")
    _settle(page, 4)
    return "home"


def helicalpitch(page, url):
    page.goto(f"{url}?tab=HelicalPitch")
    _wait_text(page, "Viewing rows")  # the table of filaments, after the plots
    page.locator(".js-plotly-plot").nth(1).wait_for(timeout=120_000)
    _settle(page, 6)
    return "helicalpitch"


def denovo3d(page, url):
    # the page before the search: the transform and the search range. (A click on "Search
    # Parameters" did not start a search in the headless browser within six minutes, so the
    # results are not taken here.)
    page.goto(f"{url}?tab=Denovo3D")
    _wait_text(page, "Selected image")
    _settle(page)
    return "denovo3d"


def abinitio3d(page, url):
    page.goto(f"{url}?tab=AbInitio3D")
    _wait_text(page, "Selected classes")
    _settle(page)
    page.screenshot(path=IMAGES / "abinitio3d-input.png")
    # one click picks a class, a shift-click adds one more; the pitch needs at least three
    ids = page.locator("[id^='abinitio3d-select_classes_inner_image_']").evaluate_all(
        "els => els.map(e => e.id)"
    )
    page.locator("#" + ids[0]).click()
    for i in ids[1:8]:
        page.locator("#" + i).click(modifiers=["Shift"])
        time.sleep(0.5)
    page.get_by_role("button", name="Estimate pitch").click()
    page.locator("#abinitio3d-phase_circle_plot").wait_for(timeout=600_000)
    _settle(page, 15)
    page.mouse.move(700, 960)  # away from the buttons, so that no tip is showing
    time.sleep(1)
    return "abinitio3d"


def hi3d(page, url):
    page.goto(f"{url}?tab=HI3D")
    _wait_text(page, "resolution=")  # the map has been read
    # the plots are drawn in canvases (their titles are not text of the page)
    page.locator("canvas").nth(2).wait_for(state="attached", timeout=240_000)
    _settle(page, 15)
    return "hi3d"


def helicalprojection(page, url):
    page.goto(f"{url}?tab=HelicalProjection")
    _wait_text(page, "Selected image")
    _settle(page)
    return "helicalprojection"


def helicallattice(page, url):
    page.goto(f"{url}?tab=HelicalLattice")
    _wait_text(page, "unrolled into a 2D lattice")
    _settle(page, 4)
    return "helicallattice"


STAR = (
    "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/"
    "run_it020_data.star"
)


def whereismyclass(page, url):
    page.goto(f"{url}?tab=WhereIsMyClass")
    _wait_text(page, "RELION star or cryoSPARC cs file")
    page.locator("#where_is_my_class-wimc_input_mode input[value=url]").check()
    page.locator("#where_is_my_class-wimc_url_star").fill(STAR)
    page.get_by_role("button", name="Run").click()
    _wait_text(page, "Viewing rows", 600)  # the table of filaments
    # select the longest filament: its micrograph is downloaded from the project of the star file
    row = page.locator("#where_is_my_class-wimc_helices_dataframe tr[data-index='0']")
    checkbox = row.locator("input[type='checkbox']")
    if checkbox.count():
        checkbox.check()
    else:
        row.click()
    page.locator("#where_is_my_class-wimc_display_micrograph .js-plotly-plot").wait_for(
        timeout=300_000
    )
    _settle(page, 10)
    page.mouse.move(700, 990)
    return "whereismyclass"


SHOTS = {
    f.__name__: f
    for f in (
        home,
        helicalpitch,
        denovo3d,
        abinitio3d,
        hi3d,
        helicalprojection,
        helicallattice,
        whereismyclass,
    )
}


def main(names: list[str]) -> int:
    from playwright.sync_api import sync_playwright
    from shiny.run import run_shiny_app

    unknown = [n for n in names if n not in SHOTS]
    if unknown:
        print(f"unknown {unknown}; the screenshots are {list(SHOTS)}")
        return 2
    IMAGES.mkdir(exist_ok=True)
    with run_shiny_app(APP, wait_for_start=True, timeout_secs=180) as proc:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for name in names or list(SHOTS):
                page = browser.new_page(viewport=VIEWPORT)
                try:
                    stem = SHOTS[name](page, proc.url)
                    target = IMAGES / f"{stem}.png"
                    page.screenshot(path=target)
                    share = _filled(target)
                    # a picture of a page that had not finished drawing is mostly background
                    note = (
                        "" if share > 0.03 else "   WARNING: the main panel looks empty"
                    )
                    print(f"{name}: docs/images/{stem}.png{note}", flush=True)
                except (
                    Exception
                ) as error:  # keep going: one slow tab must not lose the rest
                    print(f"{name}: FAILED {type(error).__name__}: {str(error)[:200]}")
                finally:
                    page.close()
            browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
