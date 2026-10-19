"""Take the figures of the HILL tutorial from the current web app.

    python docs/tools/screenshots_hill.py [name ...]      # all of them, or only the named ones

Like ``screenshots.py``: starts the web apps, opens the HILL tab in a headless browser (Playwright)
with the settings of each figure in the address (the bookmark of the tab), and writes
``docs/images/hill/<name>.png``. The examples are the public 2D classes behind the tab's default
image set, and the EMDB entries named in the tutorial, so the machine needs network access.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parent / "images" / "hill"
APP = HERE.parent.parent / "src" / "helicon" / "webApps" / "app.py"
VIEWPORT = {"width": 1700, "height": 1500}
SIDEBAR = 430  # the width of the left panel


def _open(page, url, query="", emdb=None, image=None):
    """Open the HILL tab, with a bookmark ``query``; load an EMDB entry or pick an image."""
    page.goto(f"{url}?tab=HILL{query}")
    page.locator("#hill-hill_main_plots").wait_for(state="attached", timeout=180_000)
    plots = page.locator("#hill-hill_main_plots canvas")
    plots.nth(2).wait_for(state="attached", timeout=240_000)
    time.sleep(4)
    if emdb is not None:
        box = page.locator("#hill-hill_img_file_emd_id")
        box.fill(emdb)
        box.press("Enter")
        time.sleep(15)
        plots.nth(2).wait_for(state="attached", timeout=240_000)
    if image is not None:
        page.locator(f"#hill-hill_select_image_image_{image}").click()
    time.sleep(8)
    page.mouse.move(5, 5)


def _clip(page, left, top, right, bottom, path):
    page.screenshot(
        path=path,
        clip={"x": left, "y": top, "width": right - left, "height": bottom - top},
    )


def _main(page, path):
    """The plots and the sliders above them."""
    box = page.locator("#hill-hill_main_plots").bounding_box()
    _clip(page, SIDEBAR, 100, 1700, box["y"] + box["height"], path)


def _plots_only(page, path):
    box = page.locator("#hill-hill_main_plots").bounding_box()
    _clip(page, SIDEBAR, box["y"], 1700, box["y"] + box["height"], path)


def overview(page, url):
    _open(page, url)
    page.screenshot(path=IMAGES / "overview.png")


def input_panel(page, url):
    _open(page, url)
    _clip(page, 0, 100, SIDEBAR, 1010, IMAGES / "input.png")


def image_parameters(page, url):
    _open(page, url, image=2)
    page.get_by_text("Image Parameters").first.click()
    time.sleep(3)
    top = page.get_by_text("Image Parameters").first.bounding_box()["y"]
    bottom = page.locator("#hill-hill_display_transformed").bounding_box()
    _clip(
        page,
        0,
        top - 10,
        SIDEBAR,
        bottom["y"] + bottom["height"] + 40,
        IMAGES / "image-parameters.png",
    )


def slanted(page, url):
    # the rotation set 25 degrees off: the layer lines are slanted
    _open(page, url, image=1)
    page.evaluate("Shiny.setInputValue('hill-hill_angle', 96)")
    time.sleep(10)
    _main(page, IMAGES / "slanted.png")


def straight(page, url):
    _open(page, url, image=1)
    _main(page, IMAGES / "straight.png")


def example1(page, url):
    _open(page, url, image=2)
    _main(page, IMAGES / "example1.png")


def radial_profile(page, url):
    _open(page, url, image=2)
    page.set_viewport_size(
        {"width": 1700, "height": 2400}
    )  # the whole left panel at once
    time.sleep(4)
    top = page.locator("#hill-hill_display_transformed").bounding_box()
    profile = page.locator("#hill-hill_plot_radial_profile").bounding_box()
    diam = page.locator("#hill-hill_diameter").bounding_box()
    bottom = max(diam["y"] + diam["height"], profile["y"] + profile["height"]) + 30
    _clip(page, 0, top["y"], SIDEBAR, bottom, IMAGES / "radial-profile.png")


def rise(page, url):
    _open(page, url, "&rise=24", image=2)
    _main(page, IMAGES / "rise.png")


def pitch(page, url):
    _open(page, url, "&twist=24", image=2)
    _main(page, IMAGES / "pitch.png")


def example2(page, url):
    _open(page, url, "&input_mode=3", emdb="emd-26987")
    _main(page, IMAGES / "example2.png")


def example3(page, url):
    _open(page, url, "&input_mode=3&cutoff_res_x=5&cutoff_res_y=5", emdb="emd-10129")
    _main(page, IMAGES / "example3.png")


def example4(page, url):
    # the long pitch puts the labels of many layer lines on top of each other: only the first
    # order, as ellipses
    _open(page, url, "&input_mode=3&m_max=1", emdb="emd-0259")
    page.evaluate("Shiny.setInputValue('hill-hill_LLText', false)")
    time.sleep(10)
    _main(page, IMAGES / "example4.png")


def example5(page, url):
    # the long pitch puts the labels of many layer lines on top of each other: only the first
    # order, as ellipses
    _open(page, url, "&input_mode=3&m_max=1", emdb="emd-0260")
    page.evaluate("Shiny.setInputValue('hill-hill_LLText', false)")
    time.sleep(10)
    _main(page, IMAGES / "example5.png")


def contact(page, url):
    """Every image of the set, to choose from (not part of the docs)."""
    for i in range(1, 9):
        _open(page, url, image=i)
        _plots_only(page, HERE / f"_contact_{i}.png")


SHOTS = {
    f.__name__: f
    for f in (
        overview,
        input_panel,
        image_parameters,
        slanted,
        straight,
        example1,
        radial_profile,
        rise,
        pitch,
        example2,
        example3,
        example4,
        example5,
    )
}


def main(names: list[str]) -> int:
    from playwright.sync_api import sync_playwright
    from shiny.run import run_shiny_app

    everything = dict(SHOTS, contact=contact)
    unknown = [n for n in names if n not in everything]
    if unknown:
        print(f"unknown {unknown}; the figures are {list(SHOTS)}")
        return 2
    IMAGES.mkdir(parents=True, exist_ok=True)
    with run_shiny_app(APP, wait_for_start=True, timeout_secs=180) as proc:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for name in names or list(SHOTS):
                page = browser.new_page(viewport=VIEWPORT)
                try:
                    everything[name](page, proc.url)
                    print(f"{name}: done", flush=True)
                except Exception as error:
                    print(
                        f"{name}: FAILED {type(error).__name__}: {str(error)[:200]}",
                        flush=True,
                    )
                finally:
                    page.close()
            browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
