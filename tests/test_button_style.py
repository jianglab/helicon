"""Buttons are blue-filled, but those with a colour of their own keep it."""

from helicon.webApps import app


class TestButtonStyle:
    def test_plain_and_outline_buttons_take_the_primary_colours(self):
        css = app._BUTTON_CSS
        for cls in (
            ".btn-default",
            ".btn-outline-primary",
            ".btn-outline-secondary",
            ".btn-secondary",
            ".btn-light",
        ):
            assert f".btn{cls}" in css
        # the colours of this app's btn-primary
        for value in ("#007bc2", "#0069a5", "#00629b", "#005c92", "38, 143, 203"):
            assert value in css

    def test_the_colours_of_a_warning_are_left_alone(self):
        css = app._BUTTON_CSS
        for cls in ("danger", "warning", "success", "info"):
            assert f"btn-{cls}" not in css and f"btn-outline-{cls}" not in css

    def test_the_rule_is_in_the_page(self):
        import inspect

        assert "ui.tags.style(_BUTTON_CSS)" in inspect.getsource(app)
