"""The manual transform controls, for one selected image and for several."""

import inspect

from helicon.webApps.tabs import helical_projection_tab as tab


def _source():
    return inspect.getsource(tab)


class TestThePerImageTransformCard:
    """Editing one of several images should feel like editing one.

    The single-image card has four sliders; the per-image card used numeric
    boxes, so switching from one selected image to several changed how the
    same four parameters are set. They are sliders in both now, with the
    ranges taken from the image the card belongs to.
    """

    def _card(self):
        body = _source()
        body = body[body.index("def hp_per_image_transform_ui") :]
        return body[: body.index("def _apply_auto_transform")]

    def test_every_control_is_a_slider(self):
        card = self._card()
        assert "ui.input_numeric" not in card
        for kind in ("rot", "threshold", "vcrop", "dy"):
            assert '_pi_id("%s", i)' % kind in card
        assert card.count("ui.input_slider") == 4

    def test_the_four_match_the_single_image_card(self):
        shared = _source()
        shared = shared[shared.index("def hp_shared_transform_ui") :]
        shared = shared[: shared.index("def hp_per_image_transform_ui")]
        card = self._card()
        for label in (
            '"Rotation (°)"',
            '"Threshold"',
            '"Vertical crop size (px)"',
            '"Vertical shift (px)"',
        ):
            assert label in shared and label in card

    def test_the_ranges_come_from_that_image(self):
        card = self._card()
        # the crop cannot exceed the image's own height, and a threshold
        # outside its own range of values means nothing
        assert "ny_i = int(img.shape[0])" in card
        assert "v_min = float(np.min(img))" in card
        assert "max=ny_i" in card
        assert "min=round(v_min, 3)" in card

    def test_the_effects_update_sliders_not_boxes(self):
        source = _source()
        for effect in ("_apply_auto_transform", "_reset_transform"):
            body = source[source.index("def %s" % effect) :][:1400]
            assert "update_numeric" not in body
            assert "ui.update_slider(_pi_id(" in body


class TestTheCardStaysOnTheImageBeingEdited:
    """Moving a slider must not throw the user back to the first image.

    The selected-images gallery shows the *transformed* images, so every
    slider move re-renders it -- and a gallery re-render performs a real click
    on its initial selection, which drove the card switcher back to image 1 in
    the middle of editing image 5.
    """

    def test_the_gallery_reselects_the_active_image(self):
        source = _source()
        body = source[source.index("def display_selected_image_gallery") :][:1400]
        assert "initial_selected_indices=reactive.value([active_selected_image()])" in (
            body
        )

    def test_clicking_an_image_is_remembered(self):
        source = _source()
        body = source[source.index("def _remember_active_selected_image") :][:700]
        assert "input.display_selected_image()" in body
        assert "active_selected_image.set(index)" in body
        # an out-of-range or unparseable value must not be stored
        assert "except (TypeError, ValueError)" in body
        assert "0 <= index < len(selected_images_original())" in body

    def test_a_new_selection_starts_at_the_first_image(self):
        source = _source()
        body = source[source.index("def _reset_active_selected_image") :][:400]
        assert "active_selected_image.set(0)" in body

    def test_switching_cards_does_not_re_render_them(self):
        # the cards are switched with CSS precisely so that each image keeps
        # what was typed into it; re-rendering on switch would reset them all
        source = _source()
        decorator = source[: source.index("def hp_per_image_transform_ui")]
        decorator = decorator[decorator.rindex("@render.ui") :]
        assert "active_selected_image" not in decorator
