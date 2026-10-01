"""The range slider whose numbers can be typed in place (``helicon.shiny.range_slider``)."""

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page

import helicon

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

APP = """
from shiny import App, reactive, render, ui
import helicon

app_ui = ui.page_fluid(
    helicon.shiny.range_slider("band", "Band", min=0, max=1000, value=(200, 300), step=1),
    helicon.shiny.range_slider("other", "Other", min=0, max=10, value=(2, 8), step=1),
    helicon.shiny.slider("level", "Level", min=0, max=100, value=40, step=1),
    helicon.shiny.range_slider("same", "Same", min=0, max=100, value=(40, 40), step=1),
    helicon.shiny.slider("live", "Live", min=0, max=100, value=40, step=1, emit_while_sliding=True),
    ui.input_action_button("jump", "Jump"),
    ui.output_text("shown"),
    ui.output_text("emitted"),
)

def server(input, output, session):
    counts = reactive.value({"level": 0, "live": 0})

    def bump(name):
        with reactive.isolate():
            c = dict(counts())
            c[name] += 1
            counts.set(c)

    @reactive.effect
    @reactive.event(input.level, ignore_init=True)
    def _level():
        bump("level")

    @reactive.effect
    @reactive.event(input.live, ignore_init=True)
    def _live():
        bump("live")

    @reactive.effect
    @reactive.event(input.jump)
    def _jump():
        ui.update_slider("level", value=10)

    @render.text
    def emitted():
        return f"emitted level={counts()['level']} live={counts()['live']}"

    @render.text
    def shown():
        return (
            f"band={input.band()[0]:g},{input.band()[1]:g} "
            f"other={input.other()[0]:g},{input.other()[1]:g} level={input.level():g} "
            f"same={input.same()[0]:g},{input.same()[1]:g}"
        )

app = App(app_ui, server)
"""


class TestMarkup:
    def test_the_slider_and_the_editor_script_are_returned_together(self):
        html = str(
            helicon.shiny.range_slider("x", "X", min=0, max=10, value=(1, 2), step=1)
        )
        assert 'id="x"' in html
        assert "helicon-editable-slider" in html
        assert html.count("__heliconRangeEdit") >= 1

    def test_it_is_exported(self):
        assert "range_slider" in helicon.shiny.__all__
        assert "slider" in helicon.shiny.__all__

    def test_the_single_slider_shares_the_editor(self):
        html = str(helicon.shiny.slider("y", "Y", min=0, max=10, value=3, step=1))
        assert 'id="y"' in html and "helicon-editable-slider" in html


@pytest.fixture(scope="module")
def slider_app(tmp_path_factory):
    folder = tmp_path_factory.mktemp("range_slider_app")
    (folder / "app.py").write_text(APP)
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
            str(folder / "app.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    port, output = [], []

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
        pytest.fail("server did not start\n" + "".join(output))
    yield f"http://127.0.0.1:{port[0]}/"
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _shown(page):
    return page.locator("#shown").inner_text()


def _edit(page, slider, side, text):
    box = page.locator(f"#{slider}").locator(
        "xpath=ancestor::div[contains(@class,'helicon-editable-slider')]"
    )
    single = box.locator(".irs-single")
    if single.is_visible():
        b = single.bounding_box()
        x = b["x"] + b["width"] * (0.25 if side == "from" else 0.75)
        page.mouse.dblclick(x, b["y"] + b["height"] / 2)
    else:
        box.locator(".irs-from" if side == "from" else ".irs-to").dblclick()
    page.keyboard.press("Control+A")
    page.keyboard.type(text)


class TestEditing:
    def test_typing_a_number_in_place_sets_the_range(self, page: Page, slider_app):
        page.goto(slider_app)
        page.wait_for_selector("#shown:has-text('band=')")
        assert _shown(page) == "band=200,300 other=2,8 level=40 same=40,40"
        _edit(page, "band", "from", "250")
        page.keyboard.press("Enter")
        page.wait_for_function(
            "document.querySelector('#shown').innerText.includes('band=250,300')"
        )
        _edit(page, "band", "to", "400")
        page.keyboard.press("Enter")
        page.wait_for_function(
            "document.querySelector('#shown').innerText.includes('band=250,400')"
        )
        # the other slider on the page is untouched
        assert "other=2,8 level=40 same=40,40" in _shown(page)

    def test_escape_leaves_the_range_alone(self, page: Page, slider_app):
        page.goto(slider_app)
        page.wait_for_selector("#shown:has-text('band=')")
        _edit(page, "band", "from", "999")
        page.keyboard.press("Escape")
        time.sleep(0.5)
        assert _shown(page) == "band=200,300 other=2,8 level=40 same=40,40"
        assert page.locator("body > input[type=text]").count() == 0

    def test_a_lower_number_above_the_upper_one_carries_it_along(
        self, page: Page, slider_app
    ):
        page.goto(slider_app)
        page.wait_for_selector("#shown:has-text('band=')")
        _edit(page, "band", "from", "500")
        page.keyboard.press("Enter")
        page.wait_for_function(
            "document.querySelector('#shown').innerText.includes('band=500,500')"
        )

    def test_numbers_beyond_the_ends_are_limited_to_them(self, page: Page, slider_app):
        page.goto(slider_app)
        page.wait_for_selector("#shown:has-text('band=')")
        _edit(page, "other", "to", "50")
        page.keyboard.press("Enter")
        page.wait_for_function(
            "document.querySelector('#shown').innerText.includes('other=2,10')"
        )

    def test_a_single_slider_can_be_typed_into_too(self, page: Page, slider_app):
        page.goto(slider_app)
        page.wait_for_selector("#shown:has-text('level=')")
        box = page.locator("#level").locator(
            "xpath=ancestor::div[contains(@class,'helicon-editable-slider')]"
        )
        box.locator(".irs-single").dblclick()
        page.keyboard.press("Control+A")
        page.keyboard.type("75")
        page.keyboard.press("Enter")
        page.wait_for_function(
            "document.querySelector('#shown').innerText.includes('level=75')"
        )
        assert "band=200,300 other=2,8" in _shown(page)
        # beyond the end
        box.locator(".irs-single").dblclick()
        page.keyboard.press("Control+A")
        page.keyboard.type("500")
        page.keyboard.press("Enter")
        page.wait_for_function(
            "document.querySelector('#shown').innerText.includes('level=100')"
        )
        # Escape leaves it
        box.locator(".irs-single").dblclick()
        page.keyboard.press("Control+A")
        page.keyboard.type("5")
        page.keyboard.press("Escape")
        time.sleep(0.5)
        assert "level=100" in _shown(page)

    def test_equal_ends_are_one_label_whose_halves_are_the_two_numbers(
        self, page: Page, slider_app
    ):
        page.goto(slider_app)
        page.wait_for_selector("#shown:has-text('same=')")
        box = page.locator("#same").locator(
            "xpath=ancestor::div[contains(@class,'helicon-editable-slider')]"
        )

        def edit(fraction, text):
            label = box.locator(".irs-from")
            b = label.bounding_box()
            page.mouse.dblclick(
                b["x"] + b["width"] * fraction, b["y"] + b["height"] / 2
            )
            page.keyboard.press("Control+A")
            page.keyboard.type(text)
            page.keyboard.press("Enter")

        edit(0.85, "60")  # the right half: the upper number
        page.wait_for_function(
            "document.querySelector('#shown').innerText.includes('same=40,60')"
        )
        # now they differ and are shown apart: the lower number is its own label
        box.locator(".irs-from").dblclick()
        page.keyboard.press("Control+A")
        page.keyboard.type("10")
        page.keyboard.press("Enter")
        page.wait_for_function(
            "document.querySelector('#shown').innerText.includes('same=10,60')"
        )


def _handle_centre(page, slider_id):
    handle = (
        page.locator(f"#{slider_id}")
        .locator("xpath=ancestor::div[contains(@class,'helicon-editable-slider')]")
        .locator(".irs-handle")
        .first
    )
    handle.scroll_into_view_if_needed()
    b = handle.bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def _emitted(page, name):
    text = page.locator("#emitted").inner_text()
    return int(re.search(rf"{name}=(\d+)", text).group(1))


class TestEmission:
    def test_a_drag_sends_once_when_it_ends_unless_asked_to_send_along_the_way(
        self, page: Page, slider_app
    ):
        page.goto(slider_app)
        page.wait_for_selector("#emitted:has-text('emitted')")
        time.sleep(1.5)  # the script has taken hold of the sliders
        slow0, live0 = _emitted(page, "level"), _emitted(page, "live")
        for slider_id in ("level", "live"):
            x, y = _handle_centre(page, slider_id)
            page.mouse.move(x, y)
            page.mouse.down()
            for k in range(1, 7):
                page.mouse.move(x + 14 * k, y)
                time.sleep(0.45)  # longer than Shiny's own delay
            if slider_id == "level":
                # still holding the handle: the value on the page has moved
                # (the number on the slider follows) and nothing was sent
                assert _emitted(page, "level") == slow0
            page.mouse.up()
            time.sleep(1.0)
        assert _emitted(page, "level") == slow0 + 1
        assert _emitted(page, "live") >= live0 + 3  # along the way, and at the end

    def test_a_click_on_the_bar_a_typed_value_and_the_server_send_at_once(
        self, page: Page, slider_app
    ):
        page.goto(slider_app)
        page.wait_for_selector("#shown:has-text('level=')")
        time.sleep(1.5)
        page.locator("#jump").click()
        page.wait_for_function(
            "document.querySelector('#shown').innerText.includes('level=10')"
        )
        box = page.locator("#level").locator(
            "xpath=ancestor::div[contains(@class,'helicon-editable-slider')]"
        )
        line = box.locator(".irs-line").bounding_box()
        page.mouse.click(
            line["x"] + line["width"] * 0.8, line["y"] + line["height"] / 2
        )
        page.wait_for_function(
            "!document.querySelector('#shown').innerText.includes('level=10 ')"
        )
        assert "level=10 " not in _shown(page)


class TestEndsEditing:
    def test_the_smallest_and_largest_values_can_be_typed_too(
        self, page: Page, slider_app
    ):
        page.goto(slider_app)
        page.wait_for_selector("#shown:has-text('level=')")
        time.sleep(1.0)
        box = page.locator("#level").locator(
            "xpath=ancestor::div[contains(@class,'helicon-editable-slider')]"
        )

        def ends():
            return page.evaluate(
                "() => { const s = $('#level').data('ionRangeSlider'); return [s.options.min, s.options.max] }"
            )

        assert ends() == [0, 100]
        box.locator(".irs-max").dblclick()
        page.keyboard.press("Control+A")
        page.keyboard.type("250")
        page.keyboard.press("Enter")
        assert ends() == [0, 250]
        box.locator(".irs-min").dblclick()
        page.keyboard.press("Control+A")
        page.keyboard.type("20")
        page.keyboard.press("Enter")
        assert ends() == [20, 250]
        # an end past the other one is refused
        box.locator(".irs-max").dblclick()
        page.keyboard.press("Control+A")
        page.keyboard.type("5")
        page.keyboard.press("Enter")
        assert ends() == [20, 250]
        # and the value in between is still what it was
        assert "level=40" in _shown(page)
