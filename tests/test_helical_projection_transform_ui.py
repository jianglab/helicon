"""The manual transform controls in helicalProjection.

Every selected image has its own transform, held by the server and keyed by
the image's label, and its own card of four sliders -- one card for a single
image, one per image for several. The pure rule for carrying transforms across
a change of selection lives in helix_transform.reconcile_transforms and is
tested there; these pin how the tab uses it.
"""

import inspect

from helicon.webApps.tabs import helical_projection_tab as tab


def _source():
    return inspect.getsource(tab)


def _body(name, until):
    source = _source()
    body = source[source.index("def %s" % name) :]
    return body[: body.index(until)]


class TestOneKindOfCard:
    def test_every_control_is_a_slider(self):
        card = _body("hp_per_image_transform_ui", "def _auto_transform")
        assert "ui.input_numeric" not in card
        assert card.count("ui.input_slider") == 4
        for kind in ("rot", "threshold", "vcrop", "dy"):
            assert '_pi_id("%s", key, t.generation)' % kind in card

    def test_there_are_no_separate_single_image_sliders(self):
        # the shared sliders were hidden in multi-image mode, yet a removed
        # control keeps its last value on the server, and they were still
        # added on top of every image as a "nudge": the first image's rotation
        # was silently copied onto each image selected after it
        source = _source()
        assert "hp_shared_transform_ui" not in source
        drawing = _body("_transform_crop_images", "# -- Map loading --")
        for shared in ('"pre_rotation"', '"shift_y"', '"vertical_crop_size"'):
            assert shared not in drawing

    def test_the_ranges_come_from_that_image(self):
        card = _body("hp_per_image_transform_ui", "def _auto_transform")
        assert "ny_i = int(img.shape[0])" in card
        assert "v_min = float(np.min(img))" in card
        assert "max=ny_i" in card
        assert "min=round(v_min, 3)" in card


class TestTransformsFollowTheImageNotItsPosition:
    def test_controls_are_keyed_by_label_and_generation(self):
        body = _body("_pi_id", "def _input_or")
        assert "generation" in body
        # the key is the image's label; a position would let one image's
        # values land on another whenever the order shifted
        selected = _body("_selected_by_key", "def _auto_transforms")
        assert "selected_images_labels()" in selected

    def test_a_selection_change_goes_through_the_reconciliation(self):
        body = _body("_reconcile_transforms_with_selection", "@render.ui")
        assert "helix_transform.reconcile_transforms(" in body
        # existing images are carried over as their controls hold them now,
        # manual edits included
        assert "_as_controls_hold(previous)" in body
        # and the card shown is the one the rule picks: the image just added
        assert "active_selected_image.set(active)" in body

    def test_only_new_images_are_measured(self):
        body = _body("_reconcile_transforms_with_selection", "@render.ui")
        assert "lambda new_keys: _auto_transforms(new_keys, images_by_key)" in body

    def test_auto_and_reset_replace_values_with_fresh_generations(self):
        # values replaced from outside the controls must not be read back
        # from a control that still holds the old ones
        auto = _body("_auto_transforms", "def _as_controls_hold")
        assert "generation=_next_generation()" in auto
        reset = _body("_reset_transform", "# A plain effect")
        assert "generation=_next_generation()" in reset
        assert "update_numeric" not in auto + reset


class TestTheCardStaysOnTheImageBeingEdited:
    """Moving a slider must not throw the user back to the first image.

    The selected-images gallery shows the *transformed* images, so every
    slider move re-renders it -- and a gallery re-render performs a real click
    on its initial selection.
    """

    def test_the_gallery_reselects_the_active_image(self):
        body = _source()
        body = body[body.index("def display_selected_image_gallery") :][:1600]
        assert "initial_selected_indices=reactive.value([active_selected_image()])" in (
            body
        )

    def test_clicking_an_image_is_remembered(self):
        body = _body("_remember_active_selected_image", "@reactive.effect")
        assert "input.display_selected_image()" in body
        assert "active_selected_image.set(index)" in body
        assert "except (TypeError, ValueError)" in body
        assert "0 <= index < len(selected_images_original())" in body

    def test_the_active_card_is_drawn_visible(self):
        card = _body("hp_per_image_transform_ui", "def _auto_transform")
        assert "shown = active_selected_image()" in card
        assert '"" if i == shown else " display: none;"' in card

    def test_switching_cards_does_not_re_render_them(self):
        # the cards are switched with CSS precisely so each image keeps what
        # was set on it; re-rendering on a switch would reset them all
        source = _source()
        decorator = source[: source.index("def hp_per_image_transform_ui")]
        decorator = decorator[decorator.rindex("@render.ui") :]
        assert "active_selected_image" not in decorator
